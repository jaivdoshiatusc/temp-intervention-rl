from stable_baselines3 import PPO
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.utils import obs_as_tensor

from intervention_rl.blocker.mc_blocker_trainer import MCBlockerTrainer
from intervention_rl.blocker.mc_blocker_heuristic import MCBlockerHeuristic

class PPO_HIRL(PPO):
    def __init__(
        self,
        policy,
        env,
        learning_rate=0.001,
        n_steps=5,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=1,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=False,
        ent_coef=0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        stats_window_size=100,
        tensorboard_log=None,
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device='auto',
        _init_setup_model=True,
        target_kl=None,

        env_name="MountainCar-v0",
        exp_type="none",
        pretrained_blocker=None,
        blocker_switch_time=100000,
        new_action=0,  # Typically action 0 is "push left"
        alpha=0.01,
        beta=0.01,

        blocker_clearance=-1.15,
        catastrophe_clearance=-1.15,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            n_steps,
            batch_size,
            n_epochs,
            gamma,
            gae_lambda,
            clip_range,
            clip_range_vf,
            normalize_advantage,
            ent_coef,
            vf_coef,
            max_grad_norm,
            use_sde,
            sde_sample_freq,
            rollout_buffer_class,
            rollout_buffer_kwargs,
            target_kl,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model
        )

        # Initialize blocker variables
        self.env_name = env_name
        self.exp_type = exp_type  # Type of methods (none, expert, ours, hirl)
        self.pretrained_blocker = pretrained_blocker
        self.blocker_switch_time = blocker_switch_time
        self.new_action = new_action
        self.alpha = alpha
        self.beta = beta
        self.catastrophe_clearance = catastrophe_clearance
        self.blocker_clearance = blocker_clearance
        self.pretrained_blocker_switch = False

        # Initialize cumulative variables
        self.cum_catastrophe = 0
        self.cum_env_intervention = 0
        self.cum_exp_intervention = 0
        self.cum_disagreement = 0

        # Initialize after blocker cumulative variables
        self.blocker_cum_catastrophe = 0
        self.blocker_cum_env_intervention = 0
        self.blocker_cum_exp_intervention = 0
        self.blocker_cum_disagreement = 0

        # Initialize moving average variables
        self.mean_bonus = 0.0
        self.mean_model_entropy = 0.0
        self.mean_disagreement_prob = 0.0
        self.ema_alpha = 0.1  # Adjust alpha as needed (smoothing factor)

        # Initialize episode variables
        self._current_episode_reward = np.zeros(self.env.num_envs)
        self._current_episode_length = np.zeros(self.env.num_envs)
        self._current_episode_catastrophe = np.zeros(self.env.num_envs)
        self._current_episode_total_bonus = np.zeros(self.env.num_envs)

        if "MountainCar" in self.env_name:
            self.blocker_heuristic = MCBlockerHeuristic(self.catastrophe_clearance)
            if self.exp_type in ["ours", "hirl"]:
                self.blocker_model = MCBlockerTrainer(action_size=env.action_space.n, device=self.device)
    
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        total_bonus = 0
        total_model_entropy = 0
        total_disagreement_prob = 0
        num_env_steps = 0

        if not hasattr(self, 'custom_ep_info_buffer'):
            self.custom_ep_info_buffer = []
        else:
            self.custom_ep_info_buffer.clear()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            modified_actions = clipped_actions.copy()

            # Intervention logic
            if self.exp_type != "none":
                # Use the current observations directly
                full_obs_all = [self._last_obs[i] for i in range(env.num_envs)]

                model_entropy_arr = []
                disagreement_prob_arr = []

                for i in range(env.num_envs):
                    obs = full_obs_all[i]
                    action = clipped_actions[i]

                    if isinstance(action, np.ndarray):
                        action = action.item()

                    blocker_heuristic_decision = False
                    blocker_model_decision = False

                    if self.exp_type in ["ours", "hirl"]:
                        # Human Oversight Phase (Training the Blocker)
                        if self.num_timesteps <= self.blocker_switch_time:
                            blocker_heuristic_decision = self.blocker_heuristic.should_block(obs, action)
                            blocker_model_decision, model_entropy, disagreement_prob = self.blocker_model.should_block(
                                obs,
                                action,
                                blocker_heuristic_decision
                            )
                            model_entropy_arr.append(model_entropy)
                            disagreement_prob_arr.append(disagreement_prob)

                            total_model_entropy += model_entropy
                            total_disagreement_prob += disagreement_prob

                            self.blocker_model.store(obs, action, blocker_heuristic_decision)

                            if blocker_heuristic_decision:
                                modified_actions[i] = self.new_action

                                self.cum_env_intervention += 1
                                self.cum_exp_intervention += 1
                            if blocker_heuristic_decision != blocker_model_decision:
                                self.cum_disagreement += 1

                        # Blocker Phase
                        else:
                            if self.pretrained_blocker and not self.pretrained_blocker_switch:
                                self.blocker_model.load_weights(self.pretrained_blocker)
                                self.pretrained_blocker_switch = True

                            blocker_heuristic_decision = self.blocker_heuristic.should_block(obs, action)
                            blocker_model_decision, model_entropy, disagreement_prob = self.blocker_model.should_block(
                                obs,
                                action,
                                blocker_heuristic_decision
                            )
                            model_entropy_arr.append(model_entropy)
                            disagreement_prob_arr.append(disagreement_prob)

                            total_model_entropy += model_entropy
                            total_disagreement_prob += disagreement_prob

                            if blocker_model_decision:
                                modified_actions[i] = self.new_action
                                self.cum_env_intervention += 1
                                self.blocker_cum_env_intervention += 1
                            if blocker_heuristic_decision:
                                self.cum_exp_intervention += 1
                                self.blocker_cum_exp_intervention += 1
                            if blocker_heuristic_decision != blocker_model_decision:
                                self.cum_disagreement += 1
                                self.blocker_cum_disagreement += 1

                    elif self.exp_type == "expert":
                        blocker_heuristic_decision = self.blocker_heuristic.should_block(obs, action)

                        if blocker_heuristic_decision:
                            modified_actions[i] = self.new_action
                            self.cum_env_intervention += 1
                            self.cum_exp_intervention += 1

            # Step environment
            if self.exp_type in ["expert", "ours", "hirl"]:
                new_obs, rewards, dones, infos = env.step(modified_actions)
            else:
                new_obs, rewards, dones, infos = env.step(clipped_actions)

            # Increment time step
            self.num_timesteps += env.num_envs
            num_env_steps += env.num_envs

            # Initialize the list to track catastrophe occurrence at current step
            catastrophe_occurred_this_step = [False] * env.num_envs

            # Process new observations for catastrophes
            for i in range(env.num_envs):
                obs = new_obs[i]
                if self.blocker_heuristic.is_catastrophe(obs):
                    self.cum_catastrophe += 1
                    if self.num_timesteps > self.blocker_switch_time:
                        self.blocker_cum_catastrophe += 1
                    self._current_episode_catastrophe[i] += 1
                    catastrophe_occurred_this_step[i] = True
                else:
                    catastrophe_occurred_this_step[i] = False

            # Apply bonus only if a catastrophe occurred at this step
            if self.exp_type == "ours" and self.num_timesteps <= self.blocker_switch_time:
                for i in range(env.num_envs):
                    if catastrophe_occurred_this_step[i]:
                        bonus = self.alpha
                        rewards[i] += bonus
                        total_bonus += bonus
                        self._current_episode_total_bonus[i] += bonus

            # Update cumulative episode reward and length
            self._current_episode_reward += rewards
            self._current_episode_length += 1

            # Check for episode ends and update ep_info_buffer
            for i in range(env.num_envs):
                if dones[i]:
                    # Episode is done, store episode info with modified rewards
                    ep_info = {
                        "r": self._current_episode_reward[i],
                        "l": self._current_episode_length[i],
                        "catastrophes": self._current_episode_catastrophe[i],
                        "total_bonus": self._current_episode_total_bonus[i],
                    }
                    self.custom_ep_info_buffer.append(ep_info)
                    self._current_episode_reward[i] = 0
                    self._current_episode_length[i] = 0
                    self._current_episode_catastrophe[i] = 0
                    self._current_episode_total_bonus[i] = 0

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                clipped_actions = clipped_actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                clipped_actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        # Avoid division by zero
        if num_env_steps == 0:
            num_env_steps = 1

        # Compute current mean values over the rollout
        current_mean_bonus = total_bonus / num_env_steps
        current_mean_model_entropy = total_model_entropy / num_env_steps
        current_mean_disagreement_prob = total_disagreement_prob / num_env_steps

        # Update rolling averages using EMA
        self.mean_bonus = (self.ema_alpha * current_mean_bonus) + (1 - self.ema_alpha) * self.mean_bonus
        self.mean_model_entropy = (self.ema_alpha * current_mean_model_entropy) + (1 - self.ema_alpha) * self.mean_model_entropy
        self.mean_disagreement_prob = (self.ema_alpha * current_mean_disagreement_prob) + (1 - self.ema_alpha) * self.mean_disagreement_prob

        blocker_switch = int(self.num_timesteps <= self.blocker_switch_time)

        # Assuming you have methods to get labels from your blocker model
        positive_labels, negative_labels = self.blocker_model.get_labels()
        total_labels = positive_labels + negative_labels
        positive_proportion = positive_labels / total_labels if total_labels > 0 else 0
        negative_proportion = negative_labels / total_labels if total_labels > 0 else 0

        # Log variables
        self.logger.record('rollout/cum_catastrophe', self.cum_catastrophe)
        self.logger.record('rollout/cum_env_intervention', self.cum_env_intervention)
        self.logger.record('rollout/cum_exp_intervention', self.cum_exp_intervention)
        self.logger.record('rollout/cum_disagreement', self.cum_disagreement)
        self.logger.record('rollout/mean_bonus', self.mean_bonus)
        self.logger.record('rollout/mean_model_entropy', self.mean_model_entropy)
        self.logger.record('rollout/mean_disagreement_prob', self.mean_disagreement_prob)
        self.logger.record('rollout/blocker_switch', blocker_switch)

        self.logger.record('rollout/blocker_cum_catastrophe', self.blocker_cum_catastrophe)
        self.logger.record('rollout/blocker_cum_env_intervention', self.blocker_cum_env_intervention)
        self.logger.record('rollout/blocker_cum_exp_intervention', self.blocker_cum_exp_intervention)
        self.logger.record('rollout/blocker_cum_disagreement', self.blocker_cum_disagreement)

        self.logger.record('blocker/positive_labels', positive_labels)
        self.logger.record('blocker/negative_labels', negative_labels)
        self.logger.record('blocker/positive_proportion', positive_proportion)
        self.logger.record('blocker/negative_proportion', negative_proportion)
        self.logger.record('blocker/total_labels', total_labels)

        if len(self.custom_ep_info_buffer) > 0:
            ep_rew_mean = np.mean([ep_info["r"] for ep_info in self.custom_ep_info_buffer])
            ep_len_mean = np.mean([ep_info["l"] for ep_info in self.custom_ep_info_buffer])
            ep_catastrophe_mean = np.mean([ep_info["catastrophes"] for ep_info in self.custom_ep_info_buffer])
            ep_total_bonus_mean = np.mean([ep_info["total_bonus"] for ep_info in self.custom_ep_info_buffer])
            self.logger.record('rollout/ep_rew_mean', ep_rew_mean)
            self.logger.record('rollout/ep_len_mean', ep_len_mean)
            self.logger.record('rollout/ep_catastrophe_mean', ep_catastrophe_mean)
            self.logger.record('rollout/ep_total_bonus_mean', ep_total_bonus_mean)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

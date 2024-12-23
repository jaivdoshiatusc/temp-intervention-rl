from stable_baselines3 import PPO
import numpy as np
import torch as th
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.utils import obs_as_tensor

from intervention_rl.blocker.pong_blocker_heuristic import PongBlockerHeuristic
from intervention_rl.blocker.pong_blocker_trainer import PongBlockerTrainer

from intervention_rl.blocker.mc_blocker_heuristic import MCBlockerHeuristic
from intervention_rl.blocker.mc_blocker_trainer import MCBlockerTrainer

from intervention_rl.blocker.lander_blocker_heuristic import LunarLanderBlockerHeuristic
from intervention_rl.blocker.lander_blocker_trainer import LunarLanderBlockerTrainer

from intervention_rl.blocker.bo_blocker_heuristic import BreakoutBlockerHeuristic
from intervention_rl.blocker.bo_blocker_trainer import BreakoutBlockerTrainer

from intervention_rl.blocker.ast_blocker_heuristic import AsteroidsBlockerHeuristic
from intervention_rl.blocker.ast_blocker_trainer import AsteroidsBlockerTrainer

from intervention_rl.blocker.space_blocker_heuristic import SpaceInvadersBlockerHeuristic
from intervention_rl.blocker.space_blocker_trainer import SpaceInvadersBlockerTrainer

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

        env_name="PongNoFrameskip-v4",
        exp_type="none",
        pretrained_blocker=None,
        blocker_switch_time=100000,
        new_action=2,

        bonus_type="catastrophe",
        alpha=0.01,
        alpha_increase=0.01,
        max_alpha=500,
        beta=0.01,
        beta_increase=0.01,
        max_beta=500,
        iota=0.50, 
        penalty_type="none",
        penalty=0.0,     

        blocker_clearance = 8,
        catastrophe_clearance = 8,
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

        # Initialize environment variables
        self.policy_type = policy

        # Initialize blocker variables
        self.env_name = env_name
        self.exp_type = exp_type # Type of methods (none, expert, ours, hirl)
        self.pretrained_blocker = pretrained_blocker
        self.blocker_switch_time = blocker_switch_time
        self.new_action = new_action

        self.bonus_type = bonus_type
        self.alpha = alpha
        self.alpha_increase = alpha_increase
        self.beta = beta
        self.beta_increase = beta_increase
        self.iota = iota
        self.penalty_type = penalty_type
        self.penalty = penalty

        self.max_alpha = max_alpha
        self.max_beta = max_beta

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
        self._current_episode_total_penalty = np.zeros(self.env.num_envs)
        self._current_episode_env_intervention = np.zeros(self.env.num_envs)
        self._current_episode_exp_intervention = np.zeros(self.env.num_envs)

        if "Pong" in self.env_name:
            self.blocker_heuristic = PongBlockerHeuristic(self.catastrophe_clearance, self.blocker_clearance)
            if self.exp_type in ["ours", "hirl"]:
                self.blocker_heuristic = PongBlockerHeuristic(self.catastrophe_clearance, self.blocker_clearance)
                self.blocker_model = PongBlockerTrainer(action_size=env.action_space.n, device=self.device)
        elif "MountainCar" in self.env_name:
            self.blocker_heuristic = MCBlockerHeuristic(self.catastrophe_clearance)
            if self.exp_type in ["ours", "hirl"]:
                self.blocker_heuristic = MCBlockerHeuristic(self.catastrophe_clearance)
                self.blocker_model = MCBlockerTrainer(action_size=env.action_space.n, device=self.device)
        elif "LunarLander" in self.env_name:
            self.blocker_heuristic = LunarLanderBlockerHeuristic(self.catastrophe_clearance)
            if self.exp_type in ["ours", "hirl"]:
                self.blocker_heuristic = LunarLanderBlockerHeuristic()
                self.blocker_model = LunarLanderBlockerTrainer(action_size=env.action_space.n, device=self.device)
        elif "Breakout" in self.env_name:
            self.blocker_heuristic = BreakoutBlockerHeuristic(self.catastrophe_clearance)
            if self.exp_type in ["ours", "hirl"]:
                self.blocker_heuristic = BreakoutBlockerHeuristic(self.catastrophe_clearance)
                self.blocker_model = BreakoutBlockerTrainer(action_size=env.action_space.n, device=self.device)
        elif "Asteroids" in self.env_name:
            self.blocker_heuristic = AsteroidsBlockerHeuristic(self.catastrophe_clearance)
            if self.exp_type in ["ours", "hirl"]:
                self.blocker_heuristic = AsteroidsBlockerHeuristic(self.catastrophe_clearance)
                self.blocker_model = AsteroidsBlockerTrainer(action_size=env.action_space.n, device=self.device)
        elif "Space" in self.env_name:
            self.blocker_heuristic = SpaceInvadersBlockerHeuristic(self.catastrophe_clearance)
            if self.exp_type in ["ours", "hirl"]:
                self.blocker_heuristic = SpaceInvadersBlockerHeuristic(self.catastrophe_clearance)
                self.blocker_model = SpaceInvadersBlockerTrainer(action_size=env.action_space.n, device=self.device)

    def get_env_state(self, env, env_name, env_idx, policy_type, saved_obs):
        """
        Extracts the state or observation for a given environment and index.
        Returns an 8-dimensional vector for LunarLander.
        """
        if "Pong" in env_name:
            return env.venv.envs[env_idx].unwrapped.render()
        elif "Breakout" in env_name:
            return env.venv.envs[env_idx].unwrapped.render()
        elif "Asteroids" in env_name:
            return env.venv.envs[env_idx].unwrapped.render()
        elif "Space" in env_name:
            return env.venv.envs[env_idx].unwrapped.render()
        elif "LunarLander" in env_name:
            return saved_obs[env_idx]
        elif "MountainCar" in env_name:
            return saved_obs[env_idx]

        # Default to environment's observation
        return env.envs[env_idx].unwrapped.state
    
    def get_blocker_env_state(self, env, env_name, env_idx, policy_type, saved_obs):
        if "Space" in env_name or "Asteroids" in env_name:
            return env.venv.envs[env_idx].unwrapped.ale.getRAM()
        else:
            return self.get_env_state(env, env_name, env_idx, policy_type, saved_obs)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0 # Controls the number of iterations of the while loop in the collect_rollouts method.
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        total_bonus = 0
        total_model_entropy = 0
        total_disagreement_prob = 0
        num_env_steps = 0  # Tracks the total number of environment steps taken during the current rollout across all environments.

        if not hasattr(self, 'custom_ep_info_buffer'):
            self.custom_ep_info_buffer = []
        else:
            self.custom_ep_info_buffer.clear()


        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            modified_actions = clipped_actions.copy()
            catastrophe_occurred_this_step = [False] * env.num_envs
            exp_intervention_occurred_this_step = [False] * env.num_envs
            env_intervention_occurred_this_step = [False] * env.num_envs

            # Intervention logic
            if self.exp_type != "none":
                agent_obs_all = [self.get_env_state(env, self.env_name, i, self.policy_type, self._last_obs) for i in range(env.num_envs)] 
                blocker_heuristic_obs_all = [self.get_blocker_env_state(env, self.env_name, i, self.policy_type, self._last_obs) for i in range(env.num_envs)]               
                model_entropy_arr = []
                disagreement_prob_arr = []

                for i in range(env.num_envs):
                    agent_obs = agent_obs_all[i]
                    blocker_heuristic_obs = blocker_heuristic_obs_all[i]
                    action = clipped_actions[i]

                    if isinstance(action, np.ndarray):
                        action = action.item()

                    blocker_heuristic_decision = False
                    blocker_model_decision = False

                    if self.exp_type in ["ours", "hirl"]:
                        # Human Oversight Phase (Training the Blocker)
                        if self.num_timesteps <= self.blocker_switch_time:
                            blocker_heuristic_decision = self.blocker_heuristic.should_block(blocker_heuristic_obs, action)
                            blocker_model_decision, model_entropy, disagreement_prob = self.blocker_model.should_block(
                                agent_obs,
                                action,
                                blocker_heuristic_decision
                            )
                            model_entropy_arr.append(model_entropy)
                            disagreement_prob_arr.append(disagreement_prob)

                            # Track total_model_entropy, total_disagreement_prob
                            total_model_entropy += model_entropy
                            total_disagreement_prob += disagreement_prob

                            self.blocker_model.store(agent_obs, action, blocker_heuristic_decision)

                            if blocker_heuristic_decision:
                                modified_actions[i] = self.new_action

                                # Track cum_env_intervention, cum_exp_interventions
                                self.cum_env_intervention += 1
                                self.cum_exp_intervention += 1
                                self._current_episode_env_intervention[i] += 1
                                self._current_episode_exp_intervention[i] += 1
                                exp_intervention_occurred_this_step[i] = True
                                env_intervention_occurred_this_step[i] = True
                            if blocker_heuristic_decision != blocker_model_decision:
                                # Track cum_disagreement
                                self.cum_disagreement += 1
                        
                        # Blocker Phase
                        else:
                            if self.pretrained_blocker and not self.pretrained_blocker_switch:
                                self.blocker_model.load_weights(self.pretrained_blocker)
                                self.pretrained_blocker_switch = True
                            
                            blocker_heuristic_decision = self.blocker_heuristic.should_block(blocker_heuristic_obs, action)
                            # is action shape correct?
                            blocker_model_decision, model_entropy, disagreement_prob = self.blocker_model.should_block(
                                agent_obs,
                                action,
                                blocker_heuristic_decision
                            )
                            model_entropy_arr.append(model_entropy)
                            disagreement_prob_arr.append(disagreement_prob)

                            # Track total_model_entropy, total_disagreement_prob
                            total_model_entropy += model_entropy
                            total_disagreement_prob += disagreement_prob

                            if blocker_model_decision:
                                modified_actions[i] = self.new_action
                                # Track cum_env_intervention
                                self.cum_env_intervention += 1
                                self._current_episode_env_intervention[i] += 1
                                self.blocker_cum_env_intervention += 1
                                env_intervention_occurred_this_step[i] = True
                            if blocker_heuristic_decision:
                                # Track cum_exp_intervention
                                self.cum_exp_intervention += 1
                                self._current_episode_exp_intervention[i] += 1
                                self.blocker_cum_exp_intervention += 1
                                exp_intervention_occurred_this_step[i] = True
                            if blocker_heuristic_decision != blocker_model_decision:
                                # Track cum_disagreement
                                self.cum_disagreement += 1
                                self.blocker_cum_disagreement += 1

                    elif self.exp_type == "expert":
                        blocker_heuristic_decision = self.blocker_heuristic.should_block(blocker_heuristic_obs, action)

                        if blocker_heuristic_decision:
                                modified_actions[i] = self.new_action
                                # Track cum_env_intervention, cum_exp_interventions
                                self.cum_env_intervention += 1
                                self.cum_exp_intervention += 1
                                self._current_episode_env_intervention[i] += 1
                                self._current_episode_exp_intervention[i] += 1
                                exp_intervention_occurred_this_step[i] = True
                                env_intervention_occurred_this_step[i] = True

            # Step environment
            if self.exp_type in ["expert", "ours", "hirl"]:
                new_obs, rewards, dones, infos = env.step(modified_actions)
            else:
                new_obs, rewards, dones, infos = env.step(clipped_actions)

            # Increment time step
            self.num_timesteps += env.num_envs
            num_env_steps += env.num_envs

            # Process new observations for catastrophes
            for i in range(env.num_envs):
                blocker_heuristic_obs = self.get_blocker_env_state(env, self.env_name, i, self.policy_type, self._last_obs)
                if self.blocker_heuristic.is_catastrophe(blocker_heuristic_obs):
                    self.cum_catastrophe += 1
                    if self.num_timesteps > self.blocker_switch_time:
                        self.blocker_cum_catastrophe += 1
                    self._current_episode_catastrophe[i] += 1
                    catastrophe_occurred_this_step[i] = True
                else:
                    catastrophe_occurred_this_step[i] = False
            
            if self.penalty_type == "all":
                for i in range(env.num_envs):
                    if env_intervention_occurred_this_step[i]:
                        rewards[i] -= self.penalty
                        self._current_episode_total_penalty[i] += self.penalty

            if self.penalty_type == "catastrophe":
                for i in range(env.num_envs):
                    if catastrophe_occurred_this_step[i]:
                        rewards[i] -= self.penalty
                        self._current_episode_total_penalty[i] += self.penalty
            
            if self.penalty_type == "blocker" and self.num_timesteps > self.blocker_switch_time:
                for i in range(env.num_envs):
                    if env_intervention_occurred_this_step[i]:
                        rewards[i] -= self.penalty
                        self._current_episode_total_penalty[i] += self.penalty

            # Apply bonus based on bonus_type
            if self.exp_type == "ours" and self.num_timesteps <= self.blocker_switch_time:
                for i in range(env.num_envs):
                    if self.bonus_type == 'catastrophe':
                        if catastrophe_occurred_this_step[i]:
                            bonus = self.iota
                            rewards[i] += bonus
                            total_bonus += bonus
                            self._current_episode_total_bonus[i] += bonus
                    elif self.bonus_type == 'intervention':
                        if exp_intervention_occurred_this_step[i]:
                            bonus = self.iota
                            rewards[i] += bonus
                            total_bonus += bonus
                            self._current_episode_total_bonus[i] += bonus
                    elif self.bonus_type == 'blocker':
                        # Ensure model_entropy_arr and disagreement_prob_arr are available
                        if len(model_entropy_arr) > i and len(disagreement_prob_arr) > i:
                            bonus = (self.alpha * model_entropy_arr[i]) + (self.beta * disagreement_prob_arr[i])
                            rewards[i] += bonus
                            total_bonus += bonus
                            self._current_episode_total_bonus[i] += bonus

                        # Increment alpha and beta
                        self.alpha = min(self.alpha + self.alpha_increase, self.max_alpha)
                        self.beta = min(self.beta + self.beta_increase, self.max_beta)

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
                        "total_penalty": self._current_episode_total_penalty[i],
                        "env_interventions": self._current_episode_env_intervention[i],
                        "exp_interventions": self._current_episode_exp_intervention[i],
                    }
                    self.custom_ep_info_buffer.append(ep_info)
                    self._current_episode_reward[i] = 0
                    self._current_episode_length[i] = 0
                    self._current_episode_catastrophe[i] = 0
                    self._current_episode_total_bonus[i] = 0
                    self._current_episode_total_penalty[i] = 0
                    self._current_episode_env_intervention[i] = 0
                    self._current_episode_exp_intervention[i] = 0

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                clipped_actions = clipped_actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
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
            # Compute value for the last timestep
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

        # Get blocker labels
        if self.exp_type in ["ours", "hirl"]:
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

        if self.exp_type in ["ours", "hirl"]:
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
            ep_total_penalty_mean = np.mean([ep_info["total_penalty"] for ep_info in self.custom_ep_info_buffer])
            ep_env_intervention_mean = np.mean([ep_info["env_interventions"] for ep_info in self.custom_ep_info_buffer])
            ep_exp_intervention_mean = np.mean([ep_info["exp_interventions"] for ep_info in self.custom_ep_info_buffer])
            self.logger.record('rollout/ep_rew_mean', ep_rew_mean)
            self.logger.record('rollout/ep_len_mean', ep_len_mean)
            self.logger.record('rollout/ep_catastrophe_mean', ep_catastrophe_mean)
            self.logger.record('rollout/ep_total_bonus_mean', ep_total_bonus_mean)
            self.logger.record('rollout/ep_total_penalty_mean', ep_total_penalty_mean)
            self.logger.record('rollout/ep_env_intervention_mean', ep_env_intervention_mean)
            self.logger.record('rollout/ep_exp_intervention_mean', ep_exp_intervention_mean)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

from stable_baselines3 import A2C
import torch as th
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.utils import obs_as_tensor
# from intervention_rl.utils.my_pong_blocker import Blocker

class A2C_HIRL(A2C):
    def __init__(self, 
                 policy, 
                 env, 
                 learning_rate = 0.0007, 
                 n_steps = 5, 
                 gamma = 0.99, 
                 gae_lambda = 1, 
                 ent_coef = 0, 
                 vf_coef = 0.5, 
                 max_grad_norm = 0.5, 
                 rms_prop_eps = 0.00001, 
                 use_rms_prop = True, 
                 use_sde = False, 
                 sde_sample_freq = -1, 
                 rollout_buffer_class = None, 
                 rollout_buffer_kwargs = None, 
                 normalize_advantage = False, 
                 stats_window_size = 100, 
                 tensorboard_log = None, 
                 policy_kwargs = None, 
                 verbose = 0, 
                 seed = None, 
                 device = "auto", 
                 _init_setup_model = True,
                 ):
        super().__init__(policy, 
                         env, 
                         learning_rate, 
                         n_steps, 
                         gamma, 
                         gae_lambda, 
                         ent_coef, 
                         vf_coef, 
                         max_grad_norm, 
                         rms_prop_eps, 
                         use_rms_prop, 
                         use_sde, 
                         sde_sample_freq, 
                         rollout_buffer_class, 
                         rollout_buffer_kwargs, 
                         normalize_advantage, 
                         stats_window_size, 
                         tensorboard_log, 
                         policy_kwargs, 
                         verbose, 
                         seed, 
                         device, 
                         _init_setup_model)

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

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

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

            ## PSEUDO-CODE ##
            # check if the action is catastrophic, either using the blocker model or heuristics
            # if it is, replace it with the correct action 
            # you can also store catastrophic state-action pairs for training the blocker model later
            ##

            """
            edge cases:
            - I don't use a blocker
            - I only use an heuristic blocker (+ no training)
            - I only use a model blocker
            - I use both a model blocker and an heuristic blocker
            - I use weights of another model blocker (+ no training)

            in the below pseudocode, use_cnn determines whether I use a model blocker or not.
            the flag flips to be True after I pass N time steps.
            I only train the blocker model for the first N time steps.
            """

            """
            robot_action = clipped_actions
            # some modification to obs_tensor to get obs

            if use_cnn:
                cnn_block_decision = blocker.predict(obs, action)
                block_decision = cnn_block_decision
                action = 2 if block_decision == 1 else robot_action
                # mofification to the log_probs
            else:
                block_decision = blocker.should_block(obs, action)
                cnn_block_decision = blocker.predict(obs, action)

                action = 2 if block_decision == 1 else robot_action
                # mofification to the log_probs

                blocker.store(obs, action, block_decision)

                # calculate the uncertainty of the blocker model
                # check whether the blocker model and the heuristic agree
            """ 

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            """
            if use_cnn:
                modified_reward = reward
            else:
                modified_reward = reward + (args.alpha * cnn_entropy) + (args.beta if disagreement else 0)
            """

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

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
                actions,
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

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

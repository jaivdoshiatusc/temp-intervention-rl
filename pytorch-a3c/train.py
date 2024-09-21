import torch
import torch.nn.functional as F
import torch.optim as optim
import blocker
from blocker_model import CNNClassifier

from envs import create_atari_env
from model import ActorCritic

import random

def ensure_shared_grads(model, shared_model):
    """Copy gradients from model to shared_model."""
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def one_hot_encode_action(action, num_actions):
    """Convert an action into a one-hot encoded vector."""
    one_hot = torch.zeros(num_actions)
    one_hot[action] = 1.0
    return one_hot

def train(rank, args, shared_model, shared_cnn_model, counter, lock, optimizer=None, cnn_optimizer=None, all_observations=None, all_labels=None):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    if hasattr(env, 'get_wrapper_attr'):
        env.get_wrapper_attr('seed')(args.seed + rank)
    else:
        env.unwrapped.seed(args.seed + rank)

    # Create the ActorCritic model and CNN classifier
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    cnn_model = CNNClassifier(env.action_space.n)
    
    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
    if cnn_optimizer is None:
        cnn_optimizer = optim.Adam(shared_cnn_model.parameters(), lr=args.cnn_lr)

    model.train()

    state = env.reset()[0]
    state = torch.from_numpy(state)
    done = True

    episode_length = 0 # Track episode length
    observations = []  # Store observations for CNN training
    labels = []        # Store labels for CNN training (1 for block, 0 for no block)
    use_cnn = False    # Initially, use should_block()

    while True:
        with lock:
            if counter.value >= args.max_steps: break

        model.load_state_dict(shared_model.state_dict())
        cnn_model.load_state_dict(shared_cnn_model.state_dict())
        
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            episode_length += 1
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach().item()            
            robot_action = action

            full_obs = env.get_wrapper_attr('get_full_obs')()
            cnn_input = env.get_wrapper_attr('get_cropped_obs')()

            one_hot_action = one_hot_encode_action(robot_action, env.action_space.n) 

            cnn_input_obs = torch.tensor(cnn_input, dtype=torch.float32).permute(0, 2, 1).unsqueeze(0)
            cnn_input_action = one_hot_action.unsqueeze(0)
            cnn_output = cnn_model(cnn_input_obs, cnn_input_action)
            cnn_block_decision = torch.argmax(cnn_output, dim=1).item()

            # After N steps, switch to CNN for blocking decisions
            with lock:
                if counter.value >= args.max_steps_for_cnn and not use_cnn:
                    print(f"Process {rank} | Switching to CNN for blocking decisions after {counter.value} steps.")
                    use_cnn = True

            if use_cnn:
                block_decision = cnn_block_decision
                action = 2 if block_decision else robot_action
                # Learn on Expert Human Action
                log_prob = log_prob.gather(1, torch.tensor([[action]]))
            else:
                block_decision = blocker.should_block(full_obs, robot_action)
                action = 2 if block_decision else robot_action
                # Learn on Expert Human Action
                log_prob = log_prob.gather(1, torch.tensor([[action]]))

                # Store observation and label for CNN training
                all_observations.append((cnn_input, one_hot_action))
                all_labels.append(block_decision)

                # Calculate uncertainty (entropy) of CNN output
                cnn_prob = F.softmax(cnn_output, dim=-1)
                cnn_prob = torch.clamp(cnn_prob, min=1e-6)
                cnn_entropy = -(cnn_prob * cnn_prob.log()).sum().item()

                # Log the probabilities for disagreement checking
                if block_decision == 1:  # Human says block
                    disagreement = cnn_prob[0, 0]  # Probability of model saying "do not block"
                else:  # Human says do not block
                    disagreement = cnn_prob[0, 1]  # Probability of model saying "block"

            # Step environment
            state, reward, done, _, _ = env.step(action)
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)  # Clip rewards

            if use_cnn:
                modified_reward = reward
            else:
                modified_reward = reward + (args.alpha * cnn_entropy) + (args.beta if disagreement else 0)

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()[0]

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(modified_reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        total_loss = policy_loss + args.value_loss_coef * value_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
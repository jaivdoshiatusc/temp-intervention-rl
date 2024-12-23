import gymnasium as gym
import cv2
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def main():
    # Model path and configuration
    model_path = './ppo_model_2400000_steps.zip'
    env_name = "MountainCar-v0"

    # Create vectorized environment
    env = make_vec_env(env_name, n_envs=1, seed=42)

    # Load the PPO model
    model = PPO.load(model_path)

    obs = env.reset()
    total_steps = 1000
    frames = []

    for step in range(total_steps):
        # Predict the action using the loaded model
        action, _ = model.predict(obs)

        # Step in the environment
        obs, rewards, dones, info = env.step(action)

        # Extract state information
        state = obs[0]  # Assuming single environment
        car_position = state[0]
        car_velocity = state[1]

        # Render the current frame
        render_obs = env.envs[0].render()

        # Convert to BGR for OpenCV processing
        img = cv2.cvtColor(render_obs, cv2.COLOR_RGB2BGR)

        # Annotate the frame with car position and velocity
        cv2.putText(img, f'Position: {car_position:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, f'Velocity: {car_velocity:.2f}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Convert back to RGB for saving the frame
        render_obs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(render_obs)

        # Reset if the episode is done
        if dones[0]:
            obs = env.reset()

    # Save the collected frames as a GIF
    imageio.mimsave('mountain_car_rollout.gif', frames, fps=30)

if __name__ == '__main__':
    main()

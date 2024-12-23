import gymnasium as gym
import cv2
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage

VIEWPORT_WIDTH = 600  # VIEWPORT_W
VIEWPORT_HEIGHT = 400  # VIEWPORT_H
SCALE = 30.0  # SCALE

def obs_to_pixels(obs):
    # Scale factors as per the environment's specification
    scale_factors = {'x': 10, 'y': 6.666}

    # Extract position from the observation and apply scale factors
    x_pos_scaled = obs[0] * scale_factors['x']
    y_pos_scaled = obs[1] * scale_factors['y']

    # Convert x position: Center of the screen is 0 in the environment's coordinates
    x_pixel = (x_pos_scaled + VIEWPORT_WIDTH / SCALE / 2) * SCALE

    # Convert y position: Invert y axis for rendering (screen's 0,0 is at top-left)
    y_pixel = VIEWPORT_HEIGHT - (y_pos_scaled + VIEWPORT_HEIGHT / SCALE / 4) * SCALE

    return x_pixel, y_pixel

def create_frame(frame, obs):
    # Draw specified range (green box)
    green_color = (0, 255, 0)
    cv2.rectangle(
        frame,
        (240, 180),  # Top-left corner
        (360, 240),  # Bottom-right corner
        green_color,
        thickness=2  # Box border thickness
    )

    # LANDER POSITION MARKER
    # Draw a small green box to indicate the lander's position
    lander_x, lander_y = obs_to_pixels(obs)
    lander_x, lander_y = int(lander_x), int(lander_y)
    frame[max(lander_y-2, 0):min(lander_y+3, frame.shape[0]),
          max(lander_x-2, 0):min(lander_x+3, frame.shape[1])] = green_color

    return frame

def detect_catastrophe(state):
    """
    Check if the lander is in a catastrophic state based on render coordinates.
    """
    x, y, vel_x, vel_y, angle, angular_velocity, leg_1_contact, leg_2_contact = state
    adj_x, adj_y = obs_to_pixels(state)

    # Catastrophe conditions
    in_box = (240 <= adj_x <= 360) and (120 <= adj_y <= 240)  # Render coordinate bounds
    angle_condition = -0.25 <= angle <= 0.25
    velocity_condition = vel_y >= -0.50

    return in_box and velocity_condition and angle_condition

def draw_catastrophe(frame):
    """
    Annotate the frame to indicate catastrophe.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 0, 255)  # Red color
    font_scale = 1
    line_type = 2
    text_position = (10, 100)  # Top-left corner of the frame

    # Put "CATASTROPHE" on the frame
    cv2.putText(frame, 'CATASTROPHE', text_position, font, font_scale, font_color, line_type)
    return frame

def main():
    # Model path and configuration
    model_path = './ppo_model_1200000_steps.zip'
    env_name = "LunarLander-v2"

    # Create vectorized environment
    env = make_vec_env(env_name, n_envs=1, seed=42)
    # env = VecMonitor(env)  # Add monitoring
    # env = VecTransposeImage(env)  # Ensure correct observation format

    # Load the PPO model
    model = PPO.load(model_path)

    obs = env.reset()
    total_steps = 100
    frames = []

    for step in range(total_steps):
        # Predict the action using the loaded model
        # import ipdb; ipdb.set_trace()
        action, _ = model.predict(obs)

        # Step in the environment
        obs, rewards, dones, info = env.step(action)

        # Extract state information
        state = obs[0]  # Assuming single environment
        lander_x = state[0]
        lander_y = state[1]
        vel_x = state[2]
        vel_y = state[3]
        angle = state[4]
        angular_velocity = state[5]
        leg_1_contact = bool(state[6])
        leg_2_contact = bool(state[7])

        # Render the current frame
        render_obs = env.envs[0].render()

        # Convert to BGR for OpenCV processing
        img = cv2.cvtColor(render_obs, cv2.COLOR_RGB2BGR)
        adj_x, adj_y = obs_to_pixels(state)

        # Annotate the frame
        img = create_frame(img, state)

        # Check for catastrophe and annotate if detected
        if detect_catastrophe(state):
            img = draw_catastrophe(img)

        # Annotate the frame
        cv2.putText(img, f'Lander X: {adj_x:.2f}, Y: {adj_y:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f'Vel X: {vel_x:.2f}, Vel Y: {vel_y:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f'Angle: {angle:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Convert back to RGB for saving the frame
        render_obs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(render_obs)

        # Reset if the episode is done
        if dones[0]:
            obs = env.reset()

    # Save the collected frames as a GIF
    imageio.mimsave('lunar_lander_rollout.gif', frames, fps=30)

if __name__ == '__main__':
    main()

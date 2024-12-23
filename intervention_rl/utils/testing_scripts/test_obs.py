import numpy as np
import pickle
import random
import os
from PIL import Image

def save_random_observation_as_jpeg(data_path):
    # Load the dataset
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    observations = data['observations']
    actions = data['actions']
    labels = data['labels']

    # Randomly select an index
    total_samples = len(observations)
    random_index = random.randint(0, total_samples - 1)

    # Get the corresponding observation
    observation = observations[random_index]
    action = actions[random_index]
    label = labels[random_index]

    # Reverse the preprocessing steps to get the image back
    # Currently, the observation is in shape (C, H, W) and normalized [0,1]
    # We need to convert it back to shape (H, W, C) and pixel values [0,255]

    # Multiply by 255 to get pixel values
    observation = observation * 255.0
    # Clip values to [0, 255] range in case of rounding errors
    observation = np.clip(observation, 0, 255)
    # Convert to uint8
    observation = observation.astype(np.uint8)
    # Transpose back to (H, W, C)
    observation = np.transpose(observation, (1, 2, 0))

    # Save the image using PIL
    image = Image.fromarray(observation)
    # Build the filename
    image_filename = f'observation_{random_index}_action_{action}_label_{label}.jpeg'
    # Save the image in the same directory as the data
    data_dir = os.path.dirname(data_path)
    image_path = os.path.join(data_dir, image_filename)
    image.save(image_path)

    print(f"Saved random observation as {image_path}")

if __name__ == '__main__':
    data_path = 'collected_data/blocker_dataset.pkl'  # Update with your data file path
    save_random_observation_as_jpeg(data_path)

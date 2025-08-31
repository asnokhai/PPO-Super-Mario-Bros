import gym_super_mario_bros
import numpy as np
import matplotlib.pyplot as plt

from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from matplotlib import image as mpimg
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO

from wrappers import apply_wrappers
import os

import zipfile


# Parameter to change
FOLDER_PATH = "data/saved_models/PPO-8-11-001"



with zipfile.ZipFile("best_model.zip", "r") as f:
    print(f.namelist())

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v2')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = apply_wrappers(env)
env = FrameStack(env, 4)

def get_run_pos_data(run_name):
    model = PPO.load("data/saved_models/PPO-8-11-001/" + run_name)
    obs = env.reset()
    done = False

    x_pos_list = []
    y_pos_list = []

    while not done:

        obs = np.squeeze(obs, axis=-1)
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        x_pos_list.append(info['x_pos'])
        y_pos_list.append(info['y_pos'])

    return x_pos_list, y_pos_list


# Get all entries (files + subfolders)
entries = os.listdir(FOLDER_PATH)

# Filter for zip files
zip_files = [
    f for f in os.listdir(FOLDER_PATH)
    if f.endswith(".zip") and os.path.isfile(os.path.join(FOLDER_PATH, f))
]

# Remove '.zip' from name
zip_files = [s[:-4] for s in zip_files]

# Remove best_model entry
zip_files.remove("best_model")

# Sort numerically
zip_files = sorted(zip_files, key=int)
print(zip_files)


# Load the level image (screenshot, tileset, or pre-rendered map)
level_img = mpimg.imread("Figures/SuperMarioBrosMap1-1.png")  # make sure this file exists

# Plot the level background
fig, ax = plt.subplots(figsize=(10, 5))

level_img_flipped = np.flipud(level_img)

# extent defines the (x_min, x_max, y_min, y_max) of the image in your coordinate system
ax.imshow(level_img_flipped, extent=[0, level_img_flipped.shape[1], 0, level_img_flipped.shape[0]], origin="lower")

cmap = plt.cm.coolwarm  # blue → red
num_paths = len(zip_files)   # total runs
i = 0

for zip_file in zip_files:
    print(f"Running run {i}/{num_paths}:", zip_file)
    x_pos_list, y_pos_list = get_run_pos_data(zip_file)

    # shift y-values up by 200 to match image
    y_pos_list = [y + 200 for y in y_pos_list]

    # normalize index to [0,1] for colormap
    color = cmap(i / max(1, num_paths - 1))

    ax.plot(x_pos_list, y_pos_list, color=color, alpha=0.8, linewidth=1)

    i += 1
env.close()


# optional: add a colorbar legend
timestep_max = 10321920

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, timestep_max))
plt.colorbar(sm, label="Timestep")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("RL Paths: Blue=Early, Red=Late")
plt.show()

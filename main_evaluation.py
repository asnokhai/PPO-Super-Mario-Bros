import gym_super_mario_bros
import numpy as np
import time

from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO

from wrappers import apply_wrappers

import zipfile
with zipfile.ZipFile("best_model.zip", "r") as f:
    print(f.namelist())


# Parameters to change
MODEL_PATH = "data/saved_models/SA-LR-1e-5/2048000.zip"
LEVEL = "SuperMarioBros-1-1-v2"

# Load the model
model = PPO.load(MODEL_PATH)

env = gym_super_mario_bros.make(LEVEL)
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = apply_wrappers(env)
env = FrameStack(env, 4)


x = 30  # target iterations per second
delay = 1 / x  # seconds per iteration


obs = env.reset()
done = False

reward = 0

while not done:
    start_time = time.time()

    obs = np.squeeze(obs, axis=-1)
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

    reward += rewards

    if done:
        print(info['time'])

    elapsed = time.time() - start_time
    time_to_sleep = delay - elapsed

    if time_to_sleep > 0:
        time.sleep(time_to_sleep)

env.close()
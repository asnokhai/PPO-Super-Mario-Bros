import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor
import os
import re

from wrappers import apply_wrappers
from callback import SaveOnBestTrainingRewardCallback

def get_latest_checkpoint(directory):
    pattern = re.compile(r"(\d+)\.zip$")
    max_step = -1
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            step = int(match.group(1))
            max_step = max(max_step, step)
    return max_step if max_step != -1 else None


def make_env(env_name, seed=0):
    def _init():

        env = gym_super_mario_bros.make(env_name)
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        env = apply_wrappers(env)

        return env

    set_random_seed(seed)
    return _init

if __name__ == "__main__":

    # Parameters to change
    RUN_NAME = "PPO-31-8-001"
    LR = 1e-5
    NUM_ENVS = 8 # number of parallel environments. Handy to make it equal to the number of CPU cores.
    env_id = "SuperMarioBros-1-1-v2"

    TIMESTEPS = 2048*30  # Number of timesteps between each saved copy of the model. Should be a multiple of 2048
    ITERATIONS = 1 # Number of copies that will be trained. E.g. 10 means that it will train until 10 new copies are saved


    ###############################
    ###############################

    new_model = True
    if os.path.exists(f"data/saved_models/{RUN_NAME}"):
        new_model = False

    # Create log dir
    log_dir = "data/tmp/" + RUN_NAME
    os.makedirs(log_dir, exist_ok=True)

    env = VecMonitor(SubprocVecEnv([make_env(env_id, i) for i in range(NUM_ENVS)]), "data/tmp/" + RUN_NAME + "/TestMonitor") # Create the vectorized environment
    env = VecTransposeImage(env) # Transpose to (C, H, W)
    env = VecFrameStack(env, n_stack=4, channels_order='first') # Stack frames after transpose: output will be (C * n_stack, H, W)

    if new_model:
        model = PPO('CnnPolicy', env, verbose=0, tensorboard_log="./data/board/", learning_rate=LR)
        checkpoint = 0

        print("Creating new model...")
    else:
        checkpoint = get_latest_checkpoint(f"data/saved_models/{RUN_NAME}")
        model = PPO.load(f"data/saved_models/{RUN_NAME}/{checkpoint}.zip", env=env)


        print(f"Continuing training on existing model from timestep {checkpoint}")

    print("------------- Start Learning -------------")

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    for j in range(1, ITERATIONS+1):

        print(f"Number of timesteps of the model: {model.num_timesteps}")

        model.learn(total_timesteps=TIMESTEPS, callback=callback, reset_num_timesteps=False, tb_log_name=RUN_NAME)
        model.save(f"data/saved_models/{RUN_NAME}/{(TIMESTEPS*j)+checkpoint}")

    print("------------- Done Learning -------------")


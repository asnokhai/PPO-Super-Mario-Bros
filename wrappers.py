from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, TimeLimit
from gym import RewardWrapper

class TimePenaltyWrapper(RewardWrapper):
    def __init__(self, env, penalty=-0.1):
        super().__init__(env)
        self.penalty = penalty

    def reward(self, reward):
        return reward + self.penalty  # subtract 0.1 each step


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False

        for _ in range(self.skip):
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward

            if done:
                break

        return next_state, total_reward, done, info

class FinishReward(RewardWrapper):
    def __init__(self, env, finish_reward=15):
        super().__init__(env)
        self.finish_reward = finish_reward

    def step(self, action):
        state, reward, done, info = self.env.step(action)


        if info.get('flag_get', False):
            reward += self.finish_reward

        return state, reward, done, info


def apply_wrappers(env):
    #env = TimePenaltyWrapper(env, penalty=-0.1)
    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env, keep_dim=True)
    env = TimeLimit(env, max_episode_steps=1000)  # or lower
    env = FinishReward(env, 15)

    return env


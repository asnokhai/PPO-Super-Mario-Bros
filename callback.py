import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback that saves the best model based on mean reward,
    and logs extra Mario-specific metrics: flag reached, max x_pos, time to flag.
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.episode_idx = 0

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # === Logging additional info per episode ===
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                self.episode_idx += 1

                # TensorBoard logging (0/1)
                flag_reached = int(info.get('flag_get', False))
                x_pos = info.get('x_pos', 0)
                time_steps = info.get('time', 0)

                self.logger.record("mario/flag_reached", flag_reached)
                self.logger.record("mario/max_x_pos", x_pos)

                if flag_reached:
                    self.logger.record("mario/time_to_flag", time_steps)

        # === Save best model logic ===
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True

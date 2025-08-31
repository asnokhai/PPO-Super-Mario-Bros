from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# Change these parameters
LOG_NAME = "PPO-8-16-004_0"
VARIABLE_NAME = "rollout/ep_rew_mean"

def get_values(run_name, var_name):

    # Path to your TensorBoard event file
    log_path = "data/board/" + run_name
    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()

    # List available scalar keys
    print(ea.Tags()["scalars"])

    # Example: extract rewards
    reward_steps = ea.Scalars(var_name)

    # Convert to lists
    steps = [s.step for s in reward_steps]
    values = [s.value for s in reward_steps]

    return steps, values

### Plot single run and variable

# steps, values = get_values(LOG_NAME, VARIABLE_NAME)
# plt.plot(steps, values)
#
# plt.grid()
# plt.xlabel("Total Timesteps")
# plt.ylabel("Mean Episode Reward")
# plt.title("Training on 8 Levels")
# plt.legend()
# plt.show()


### Sensitivity Analysis

sens_anal_list = ['SA-LR-1e-6_0',
                    'SA-LR-3e-6_0',
                    'SA-LR-1e-5_0']
                    # 'SA-LR-3e-5_0',
                    # 'SA-LR-1e-4_0']

for run in sens_anal_list:
    steps, values = get_values(run, "rollout/ep_rew_mean")
    plt.plot(steps[:216], values[:216], label=run[-6:-2])

plt.grid()
plt.xlabel("Total Timesteps")
plt.ylabel("Mean Episode Reward")
plt.title("Effect of Learning Rate on Reward")
plt.legend()
plt.show()
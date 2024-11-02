import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags, logging
from rl.chapter02.k_armed_bandit_testbed import KArmedBanditTestbed

flags.DEFINE_integer("steps", 1000, "The number of steps to run.")
flags.DEFINE_integer("seed", 1, "The random seed.")
flags.DEFINE_spaceseplist("epsilon", "0.0 0.01 0.1", "The probability of taking a random action.")


# This is a naive implementation that does not compute averages in a computationally efficient way.
def main(argv):
    plt.figure(figsize=(12, 8))
    epsilon = [float(e) for e in flags.FLAGS.epsilon]
    colors = plt.get_cmap("tab10")
    for i, e in enumerate(epsilon):
        logging.info(f"Running with epsilon = {e}")
        testbed = KArmedBanditTestbed(10, seed=flags.FLAGS.seed)
        logging.debug(f"Mean rewards: {testbed.mean_rewards}")
        # A record of the rewards for pulling each arm. In this naive implementation, we need this
        # to compute the average reward for each arm at each step.
        rewards = [[] for _ in range(testbed.k)]
        # The average reward for each arm. We need this to take the greedy action.
        avg_rewards = np.zeros(testbed.k)
        # The reward for each step. We need this to plot the average reward over time.
        step_rewards = np.zeros(flags.FLAGS.steps)

        rng = np.random.default_rng(flags.FLAGS.seed)
        # Whether to take the greedy action or a random action.
        greedy_or_not = np.array([True, False])
        # Take a random action with probability `e`.
        p = np.array([1 - e, e])
        # Take the specified number of steps.
        for step in range(flags.FLAGS.steps):
            if step == 0:
                # For the first step, take a random action.
                action = rng.integers(0, testbed.k)
            else:
                # Take the action with the highest average reward with probability `1 - e`,
                # otherwise a random action.
                if rng.choice(greedy_or_not, p=p):
                    action = np.argmax(avg_rewards)
                    logging.debug(f"Taking greedy action: {action}.")
                else:
                    action = rng.integers(0, testbed.k)
                    logging.debug(f"Taking random action: {action}.")
            # Take the action and write down the reward.
            step_rewards[step] = testbed.pull_arm(action)
            rewards[action].append(step_rewards[step])
            # Update the average reward for this action.
            avg_rewards[action] = np.mean(rewards[action])

        # Compute and plot the average reward over time.
        avg_step_rewards = np.cumsum(step_rewards) / np.arange(1, flags.FLAGS.steps + 1)
        plt.plot(avg_step_rewards, color=colors(i), label=f"e = {e}")

    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.savefig("/tmp/my_plot.png")


if __name__ == "__main__":
    app.run(main)

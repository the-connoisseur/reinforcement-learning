import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags, logging
from rl.chapter02.k_armed_bandit_testbed import EpsilonGreedyBandit

flags.DEFINE_integer("steps", 10000, "The number of steps to run.")
flags.DEFINE_integer("seed", 1, "The random seed.")
flags.DEFINE_float("epsilon", 0.1, "The probability of taking a random action.")


# This application plots the average reward over time for the stationary and
# non-stationary bandit problems, for the same epsilon.
def main(argv):
    _, ax = plt.subplots()
    steps = np.arange(1, flags.FLAGS.steps + 1)
    colors = plt.get_cmap("tab10")

    for i, non_stationary in enumerate([False, True]):
        label = "Non-stationary" if non_stationary else "Stationary"
        logging.info(label)
        bandit = EpsilonGreedyBandit(
            10,
            flags.FLAGS.epsilon,
            np.random.default_rng(flags.FLAGS.seed),
            non_stationary,
        )
        step_rewards = np.zeros(flags.FLAGS.steps)

        for step in range(len(step_rewards)):
            step_rewards[step] = bandit.step().item()
        logging.info(f"Estimated action values: {bandit.q}")
        logging.info(f"Number of times each action was chosen: {bandit.n}")

        # Compute and plot the average reward over time.
        avg_step_rewards = np.cumsum(step_rewards) / np.arange(1, flags.FLAGS.steps + 1)
        ax.plot(steps, avg_step_rewards, color=colors(i), label=label)

    ax.legend()
    ax.set_xlabel("Steps")
    ax.set_ylabel("Average reward")
    plot_file = "/tmp/my_plot.png"
    plt.savefig(plot_file)
    logging.info(f"Plot saved to {plot_file}")


if __name__ == "__main__":
    app.run(main)

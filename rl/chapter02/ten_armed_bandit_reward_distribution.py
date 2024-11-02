import numpy as np
import matplotlib.pyplot as plt
from rl.chapter02.k_armed_bandit_testbed import KArmedBanditTestbed


def main():
    testbed = KArmedBanditTestbed(10)
    num_samples = 2000
    rewards = [testbed.pull_arm(i, num_samples) for i in range(testbed.k)]
    plt.figure(figsize=(12, 8))
    for i, sample in enumerate(rewards):
        plt.hist(sample, bins=30, density=True, alpha=0.5, label=f"Arm {i}")
    plt.title(f"Histograms of 10-armed bandit sampled {num_samples} times")
    plt.xlabel("Reward")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.savefig("/tmp/my_plot.png")


if __name__ == "__main__":
    main()

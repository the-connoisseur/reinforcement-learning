import numpy as np


class KArmedBanditTestbed:
    """A testbed for the k-armed bandit problem

    It randomly initializes the mean reward of each of the arms from a normal
    distribution with zero mean and standard deviation 1. When an arm is pulled, it
    randomly samples a reward from a normal distribution with the predetermined mean for
    that arm and unit variance.
    """

    def __init__(self, k, seed=1):
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.mean_rewards = self.rng.normal(0.0, 1.0, self.k)

    def pull_arm(self, arm, count=None):
        if arm < 0 or arm >= len(self.mean_rewards):
            raise ValueError(f"Invalid arm index, expected value in the range [0, {k})")
        return self.rng.normal(self.mean_rewards[arm], 1.0, count)

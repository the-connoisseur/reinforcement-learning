import numpy as np
from enum import Enum, unique


class KArmedBanditTestbed:
    """A testbed for the k-armed bandit problem

    It randomly initializes the mean reward for each of the actions from a normal
    distribution with zero mean and unit variance. When an action is taken, it randomly
    samples a reward from a normal distribution with the predetermined mean for that
    action and unit variance.
    """

    def __init__(self, k: int, rng: np.random.Generator):
        self.k: int = k
        self.rng: np.random.Generator = rng
        # Randomly assign the mean reward for each action.
        self.mean_rewards: np.ndarray = self.rng.normal(0.0, 1.0, self.k)

    def take_action(self, action: int, count: int = None) -> np.ndarray | float:
        """Takes the specified action `count` times.

        Returns the reward(s) for all actions taken. This is a scalar if `count` is
        None, otherwise a numpy array of shape (count,).
        """
        if action < 0 or action >= len(self.mean_rewards):
            raise ValueError(
                f"Invalid action index, expected value in the range [0, {k})"
            )
        return self.rng.normal(self.mean_rewards[action], 1.0, count)


class NonStationaryKArmedBanditTestbed(KArmedBanditTestbed):
    """A testbed for the k-armed bandit problem with non-stationary rewards

    After every action, the mean reward for each action is randomly updated.
    """

    def __init__(self, k: int, rng: np.random.Generator):
        super().__init__(k, rng)

    def take_action(self, action: int, count: int = None) -> np.ndarray | float:
        reward = super().take_action(action, count)
        mean_rewards_increment = self.rng.normal(0.0, 0.01, count)
        self.mean_rewards += mean_rewards_increment
        return reward


class EpsilonGreedyBandit:
    """Implements an epsilon-greedy policy for the k-armed bandit problem.

    `epsilon` is the probability of taking a random action.
    """

    @unique
    class Action(Enum):
        RANDOM = 0
        GREEDY = 1

    def __init__(
        self,
        k: int,
        epsilon: float,
        rng: np.random.Generator,
        non_stationary: bool = False,
    ):
        if non_stationary:
            self.testbed = NonStationaryKArmedBanditTestbed(k, rng)
        else:
            self.testbed = KArmedBanditTestbed(k, rng)
        self.rng: np.random.Generator = rng

        # The probabilities with which to take a random or greedy action.
        self.actions: np.ndarray = np.array([self.Action.RANDOM, self.Action.GREEDY])
        self.p: np.ndarray = np.array([epsilon, 1.0 - epsilon])

        # Estimate of action values.
        self.q: np.ndarray = np.zeros(k)
        # Number of times each action was chosen.
        self.n: np.ndarray = np.zeros(k, dtype=int)

    def step(self, count: int = 1) -> np.ndarray:
        """Steps through `count` actions, updating the action values.

        The actions are sampled from the epsilon-greedy policy.

        Returns a numpy ndarray of rewards with shape (count,).
        """
        reward: np.ndarray = np.zeros(count)
        for i in range(count):
            # If no actions have been taken yet, take a random action.
            random: bool = not np.any(self.n)
            action: int = self._next_action(random)
            reward[i] = self.testbed.take_action(action)
            self.n[action] += 1
            self.q[action] += (reward[i] - self.q[action]) / self.n[action]

        return reward

    def _next_action(self, random: bool):
        """Returns the next action to take, a value in the range [0, k).

        When `random` is False, a random action is selected with probability `epsilon`,
        otherwise the greedy action is selected.

        When `random` is True, the random action is always selected.
        """
        if not random and self.rng.choice(self.actions, p=self.p) == self.Action.GREEDY:
            # TODO: When there is a tie, choose randomly. This currently chooses the
            # first.
            return np.argmax(self.q)
        else:
            return self.rng.integers(0, self.testbed.k)

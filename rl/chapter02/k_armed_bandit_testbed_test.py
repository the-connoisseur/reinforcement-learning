import numpy as np
import unittest
from rl.chapter02.k_armed_bandit_testbed import KArmedBanditTestbed


class KArmedBanditTestbedTest(unittest.TestCase):
    def setUp(self):
        self.testbed = KArmedBanditTestbed(10)

    def test_pull_arm_default(self):
        """Test that we can pull all arms and get the expected reward."""
        reward = 0.0
        for i in range(100):
            for j in range(self.testbed.k):
                reward += self.testbed.pull_arm(j)
        self.assertAlmostEqual(reward, 167.38037556944178)

    def test_pull_arm_custom_count(self):
        """Test that we can pull an arm `count` times with one call."""
        reward = self.testbed.pull_arm(0, 100)
        self.assertIsInstance(reward, np.ndarray)
        self.assertEqual(reward.shape, (100,))


if __name__ == "__main__":
    unittest.main()

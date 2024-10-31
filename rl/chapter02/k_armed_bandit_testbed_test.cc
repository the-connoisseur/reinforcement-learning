#include "rl/chapter02/k_armed_bandit_testbed.h"

#include "absl/log/log.h"
#include "gtest/gtest.h"

#include "rl/testing/random_seed.h"

namespace rl::test {

// A simple test to make sure that KArmedBanditTestbed works as intended.
TEST(KArmedBanditTestbedTest, TenArmedBandit) {
  KArmedBanditTestbed<10> testbed{testing::RandomSeed()};
  double reward{0.0};
  for (int i = 0; i < 100; ++i) {
    for (int j = 0; j < 10; ++j) {
      reward += testbed.PullArm(j);
    }
  }
  LOG(INFO) << "Total reward: " << reward;
}

} // namespace rl::test

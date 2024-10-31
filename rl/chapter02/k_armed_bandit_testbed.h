#ifndef RL_CHAPTER02_K_ARMED_BANDIT_TESTBED_H_
#define RL_CHAPTER02_K_ARMED_BANDIT_TESTBED_H_

#include <array>
#include <random>

#include "absl/log/check.h"

namespace rl {

// This class implements a testbed for the K-armed bandit problem. It randomly
// initializes the mean reward of each of the arms from a normal distribution
// with zero mean and unit variance, and when an action is taken, randomly
// samples a reward from a normal distribution with the predetermined mean for
// that action and unit variance.
template <size_t N> class KArmedBanditTestbed {
public:
  KArmedBanditTestbed(int seed)
      : rng_{static_cast<std::mt19937::result_type>(seed)} {
    // We will pick our mean reward for each arm from a normal distribution with
    // zero mean and unit variance.
    std::normal_distribution<double> distribution{0.0, 1.0};
    for (size_t i = 0; i < N; ++i) {
      mean_rewards_[i] = distribution(rng_);
    }
  }

  // Selects an action. `arm` is 0-indexed, in the range [0, N).
  double PullArm(size_t arm) {
    CHECK_LT(arm, N);
    // We will pick our reward for the given arm from a normal distribution with
    // the predetermined mean and unit variance.
    std::normal_distribution<double> distribution(mean_rewards_[arm], 1.0);
    return distribution(rng_);
  }

private:
  std::mt19937 rng_;
  std::array<double, N> mean_rewards_;
};

} // namespace rl

#endif // RL_CHAPTER02_K_ARMED_BANDIT_TESTBED_H_

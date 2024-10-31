#ifndef RL_TESTING_RANDOM_SEED_H_
#define RL_TESTING_RANDOM_SEED_H_

namespace rl::testing {

// Returns the random seed to use for testing.
//
// This is ${TEST_RANDOM_SEED} if it is set or 1.
int RandomSeed();

} // namespace rl::testing

#endif // RL_TESTING_RANDOM_SEED_H_

#include "rl/testing/random_seed.h"

#include <cstdlib>

namespace rl::testing {

int RandomSeed() {
  const char *from_environment = std::getenv("TEST_RANDOM_SEED");
  if (from_environment != nullptr) {
    return std::atoi(from_environment);
  }
  return 1;
}

} // namespace rl::testing

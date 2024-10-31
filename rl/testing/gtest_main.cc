#include "gtest/gtest.h"

#include "rl/util/init.h"

GTEST_API_ int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  rl::util::InitLoggingAndFlags(&argc, &argv);
  return RUN_ALL_TESTS();
}

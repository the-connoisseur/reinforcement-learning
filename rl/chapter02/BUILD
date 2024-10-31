cc_library(
    name = "k_armed_bandit_testbed",
    hdrs = ["k_armed_bandit_testbed.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/log:check",
    ],
)

cc_test(
    name = "k_armed_bandit_testbed_test",
    srcs = ["k_armed_bandit_testbed_test.cc"],
    deps = [
        ":k_armed_bandit_testbed",
        "//rl/testing:googletest",
        "//rl/testing:random_seed",
        "@abseil-cpp//absl/log",
    ],
)

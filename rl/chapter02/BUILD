load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "k_armed_bandit_testbed",
    srcs = ["k_armed_bandit_testbed.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("numpy"),
    ],
)

py_test(
    name = "k_armed_bandit_testbed_test",
    size = "small",
    srcs = ["k_armed_bandit_testbed_test.py"],
    deps = [
        ":k_armed_bandit_testbed",
        requirement("numpy"),
    ],
)

py_binary(
    name = "ten_armed_bandit_reward_distribution",
    srcs = ["ten_armed_bandit_reward_distribution.py"],
    deps = [
        ":k_armed_bandit_testbed",
        requirement("matplotlib"),
        requirement("numpy"),
    ],
)

py_binary(
    name = "ten_armed_bandit_epsilon_greedy",
    srcs = ["ten_armed_bandit_epsilon_greedy.py"],
    deps = [
        ":k_armed_bandit_testbed",
        requirement("absl-py"),
        requirement("matplotlib"),
        requirement("numpy"),
    ],
)

py_binary(
    name = "non_stationary_epsilon_greedy_bandit",
    srcs = ["non_stationary_epsilon_greedy_bandit.py"],
    deps = [
        ":k_armed_bandit_testbed",
        requirement("absl-py"),
        requirement("matplotlib"),
        requirement("numpy"),
    ],
)

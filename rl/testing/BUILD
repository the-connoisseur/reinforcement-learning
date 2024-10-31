cc_library(
    name = "googletest",
    testonly = True,
    srcs = ["gtest_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//rl/util:init",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "random_seed",
    testonly = True,
    srcs = ["random_seed.cc"],
    hdrs = ["random_seed.h"],
    visibility = ["//visibility:public"],
)

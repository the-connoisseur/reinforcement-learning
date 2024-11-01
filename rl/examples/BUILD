load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "hello_world_py",
    srcs = ["hello_world.py"],
    main = "hello_world.py",
    deps = [
        requirement("absl-py"),
    ],
)

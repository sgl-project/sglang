"""Shared Bazel macros for the non-hermetic SGLang Python overlay."""

load("@rules_python//python:defs.bzl", "py_library", "py_test")

_SGLANG_LIB = "//python:sglang"

_HW_TAGS = {
    "cpu": "hw-cpu",
    "cuda": "hw-cuda",
    "npu": "hw-npu",
    "rocm": "hw-rocm",
    "xpu": "hw-xpu",
}

_PROFILE_TAGS = {
    "cpu": "profile-cpu",
    "cuda": "profile-cuda",
    "npu": "profile-npu",
    "other": "profile-other",
    "xpu": "profile-xpu",
}

_ARCH_CONSTRAINTS = {
    "any": [],
    "aarch64": ["@platforms//cpu:aarch64"],
    "x86_64": ["@platforms//cpu:x86_64"],
}

_ARCH_TAGS = {
    "any": "arch-any",
    "aarch64": "arch-aarch64",
    "x86_64": "arch-x86_64",
}


def _est_time_to_timeout(est_time):
    if est_time <= 60:
        return "short", "small"
    if est_time <= 300:
        return "moderate", "medium"
    if est_time <= 900:
        return "long", "large"
    return "eternal", "enormous"


def _sglang_dep_for(name):
    if native.package_name() == "python" and name == "sglang":
        return []
    return [_SGLANG_LIB]


def sgl_py_library(
        name,
        srcs = None,
        deps = [],
        data = [],
        visibility = None,
        **kwargs):
    """py_library with SGLang defaults for the Python overlay."""
    py_library(
        name = name,
        srcs = srcs if srcs != None else native.glob(["**/*.py"]),
        data = data,
        deps = _sglang_dep_for(name) + deps,
        visibility = visibility or ["//visibility:public"],
        **kwargs
    )


def sgl_registered_test(
        name,
        srcs,
        suite,
        est_time,
        hw = "cuda",
        profile = "cuda",
        arch = "any",
        deps = [],
        data = [],
        tags = [],
        env = {},
        target_compatible_with = [],
        **kwargs):
    """Mirror register_*_ci metadata as Bazel tags and platform constraints."""
    timeout, size = _est_time_to_timeout(est_time)
    py_test(
        name = name,
        srcs = srcs,
        data = data,
        deps = [_SGLANG_LIB] + deps,
        env = env,
        size = size,
        tags = [
            "suite-" + suite,
            _HW_TAGS[hw],
            _PROFILE_TAGS[profile],
            _ARCH_TAGS[arch],
        ] + tags,
        target_compatible_with = _ARCH_CONSTRAINTS[arch] + target_compatible_with,
        timeout = timeout,
        **kwargs
    )


def sgl_cpu_test(name, srcs, suite, est_time, arch = "any", **kwargs):
    sgl_registered_test(
        name = name,
        srcs = srcs,
        suite = suite,
        est_time = est_time,
        hw = "cpu",
        profile = "cpu",
        arch = arch,
        **kwargs
    )


def sgl_cuda_test(name, srcs, suite, est_time, arch = "x86_64", **kwargs):
    sgl_registered_test(
        name = name,
        srcs = srcs,
        suite = suite,
        est_time = est_time,
        hw = "cuda",
        profile = "cuda",
        arch = arch,
        **kwargs
    )


def sgl_npu_test(name, srcs, suite, est_time, arch = "aarch64", **kwargs):
    sgl_registered_test(
        name = name,
        srcs = srcs,
        suite = suite,
        est_time = est_time,
        hw = "npu",
        profile = "npu",
        arch = arch,
        **kwargs
    )


def sgl_xpu_test(name, srcs, suite, est_time, arch = "x86_64", **kwargs):
    sgl_registered_test(
        name = name,
        srcs = srcs,
        suite = suite,
        est_time = est_time,
        hw = "xpu",
        profile = "xpu",
        arch = arch,
        **kwargs
    )


def sgl_test_suite(name, hw = None, profile = None, suite = None, tests = []):
    tags = []
    if hw:
        tags.append(_HW_TAGS[hw])
    if profile:
        tags.append(_PROFILE_TAGS[profile])
    if suite:
        tags.append("suite-" + suite)
    native.test_suite(
        name = name,
        tags = tags,
        tests = tests,
    )

# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Test that runtime_base in pyproject_other.toml remains torch-free.

This prevents accidental introduction of packages that transitively pull
torch/triton into the srt_empty install target.
"""

from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

# Packages known to transitively depend on torch or triton.
# If a new package is added to runtime_base and it pulls torch,
# add it here and move it to runtime_common instead.
TORCH_PULLING_PACKAGES = frozenset(
    {
        "torch",
        "torchao",
        "timm",
        "xgrammar",
        "compressed-tensors",
        "outlines",
        "flashinfer",
        "sgl-kernel",
    }
)


def _parse_runtime_base() -> set:
    """Parse runtime_base package names from pyproject_other.toml."""
    # Try tomllib (3.11+) or tomli
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    toml_path = Path(__file__).resolve().parents[3] / "python" / "pyproject_other.toml"
    if not toml_path.exists():
        pytest.skip(f"pyproject_other.toml not found at {toml_path}")

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    runtime_base = data["project"]["optional-dependencies"]["runtime_base"]

    # Extract bare package names (strip version specifiers and extras)
    pkg_names = set()
    for dep in runtime_base:
        # "package[extra]>=1.0,<2.0" -> "package"
        name = (
            dep.split("[")[0]
            .split(">")[0]
            .split("<")[0]
            .split("=")[0]
            .split("!")[0]
            .split(";")[0]
            .strip()
        )
        pkg_names.add(name.lower())

    return pkg_names


def test_runtime_base_no_torch_deps():
    """runtime_base must not contain packages that pull in torch."""
    pkg_names = _parse_runtime_base()
    violations = pkg_names & TORCH_PULLING_PACKAGES
    assert not violations, (
        f"runtime_base contains torch-pulling packages: {sorted(violations)}. "
        f"Move them to runtime_common to keep srt_empty torch-free."
    )


def test_runtime_base_not_empty():
    """Sanity check: runtime_base should have a reasonable number of packages."""
    pkg_names = _parse_runtime_base()
    assert len(pkg_names) >= 20, (
        f"runtime_base only has {len(pkg_names)} packages, expected >= 20. "
        f"Did the toml structure change?"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

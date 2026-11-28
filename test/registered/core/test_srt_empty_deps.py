# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Test that runtime_base in pyproject_other.toml remains torch-free.

This prevents accidental introduction of packages that transitively pull
torch/triton into the srt_empty install target.
"""

from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

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


def _load_optional_dependencies() -> dict:
    """Load the optional-dependencies table from pyproject_other.toml."""
    import tomllib

    toml_path = Path(__file__).resolve().parents[3] / "python" / "pyproject_other.toml"
    if not toml_path.exists():
        pytest.skip(f"pyproject_other.toml not found at {toml_path}")

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    return data["project"]["optional-dependencies"]


def _parse_runtime_base() -> set:
    """Parse runtime_base package names from pyproject_other.toml."""
    runtime_base = _load_optional_dependencies()["runtime_base"]

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


def _transformers_requirements(deps: list) -> list:
    """Return the parsed transformers requirements in one extra's dependency list."""
    from packaging.requirements import Requirement

    return [
        Requirement(dep)
        for dep in deps
        if dep.split("[")[0].split(";")[0].strip().lower().startswith("transformers")
        and Requirement(dep).name.lower() == "transformers"
    ]


def test_runtime_base_transformers_is_a_range():
    """runtime_base must leave the exact transformers version to the platform extras."""
    requirements = _transformers_requirements(
        _load_optional_dependencies()["runtime_base"]
    )
    assert len(requirements) == 1, (
        f"runtime_base must list transformers exactly once, found {len(requirements)}."
    )
    assert "==" not in str(requirements[0].specifier), (
        f"runtime_base pins transformers exactly ({requirements[0]}). "
        f"Keep a range here and pin the exact version in each srt_* platform extra."
    )


def test_platform_extras_pin_transformers_exactly():
    """Every srt_* platform extra must pin one transformers version inside the runtime_base range."""
    optional_dependencies = _load_optional_dependencies()
    base_range = _transformers_requirements(optional_dependencies["runtime_base"])[
        0
    ].specifier

    platform_extras = {
        name: deps
        for name, deps in optional_dependencies.items()
        if name.startswith("srt_") and "sglang[runtime_common]" in deps
    }
    assert platform_extras, "no srt_* extra includes sglang[runtime_common]"

    for name, deps in platform_extras.items():
        requirements = _transformers_requirements(deps)
        assert len(requirements) == 1, (
            f"{name} must pin transformers exactly once, "
            f"found {[str(r) for r in requirements]}."
        )
        specifiers = list(requirements[0].specifier)
        assert len(specifiers) == 1 and specifiers[0].operator == "==", (
            f"{name} must pin transformers with a single '==', got {requirements[0]}."
        )
        assert base_range.contains(specifiers[0].version), (
            f"{name} pins transformers=={specifiers[0].version}, "
            f"outside the runtime_base range '{base_range}'."
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

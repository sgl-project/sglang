#!/usr/bin/env python3
"""Check that Dockerfiles building pinned wheels from source use the pinned versions.

docker/Dockerfile.cu134 builds sgl-deep-gemm and sgl-deep-ep from source, strips
their pins from python/pyproject.toml and installs the wheels with --no-deps, so a
pin bump that misses the Dockerfile ships the old version without any install error.
Every Dockerfile under docker/ that sets a default for one of the ARGs below is
checked, so a renamed or newly added CUDA-specific Dockerfile stays covered.
"""

import glob
import re
import sys
import tomllib
from collections import defaultdict

PYPROJECT = "python/pyproject.toml"

# Package pinned in pyproject -> {Dockerfile ARG: prefix}; pin X needs ARG=<prefix>X.
PINNED_ARGS = {
    "sgl-deep-gemm": {"DEEPGEMM_REF": "v", "SGL_DEEP_GEMM_VERSION": ""},
    "sgl-deep-ep": {"SGL_DEEP_EP_VERSION": ""},
}

# `package==X`: the Dockerfiles sed-replace this exact line, so no other form counts.
_EXACT_PIN = re.compile(r"([\w.-]+)==(\S+)")
# `ARG NAME=default`, possibly indented.
_ARG_DEFAULT = re.compile(r"^\s*ARG\s+(\w+)=(.*)$", re.MULTILINE)


def arg_defaults() -> dict[str, list[tuple[str, str]]]:
    """ARG name -> (path, default) for every ``ARG NAME=default`` under docker/."""
    defaults = defaultdict(list)
    for path in sorted(glob.glob("docker/**/*Dockerfile*", recursive=True)):
        with open(path, encoding="utf-8") as f:
            for name, value in _ARG_DEFAULT.findall(f.read()):
                defaults[name].append((path, value.strip()))
    return defaults


def main() -> int:
    with open(PYPROJECT, "rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    pins = defaultdict(list)  # package -> X of every `package==X` dependency
    for dep in deps:
        match = _EXACT_PIN.fullmatch(dep)
        if match:
            pins[match.group(1)].append(match.group(2))
    defaults = arg_defaults()

    errors = []
    for package, args in PINNED_ARGS.items():
        if len(pins[package]) != 1:
            errors.append(f"{PYPROJECT}: expected one exact '{package}==X' pin")
            continue
        version = pins[package][0]
        for arg, prefix in args.items():
            if not defaults[arg]:
                errors.append(f"no Dockerfile sets ARG {arg}=...; this check is dead")
            for path, value in defaults[arg]:
                if value != prefix + version:
                    errors.append(
                        f"{path}: ARG {arg}={value}, expected {prefix}{version} "
                        f"to match {package}=={version} in {PYPROJECT}"
                    )

    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

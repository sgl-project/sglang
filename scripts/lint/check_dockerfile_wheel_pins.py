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

PYPROJECT = "python/pyproject.toml"

# Package pinned in pyproject -> {Dockerfile ARG: its value for pin X}.
PINNED_ARGS = {
    "sgl-deep-gemm": {"DEEPGEMM_REF": "v{}", "SGL_DEEP_GEMM_VERSION": "{}"},
    "sgl-deep-ep": {"SGL_DEEP_EP_VERSION": "{}"},
}


def main() -> int:
    with open(PYPROJECT, "rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    dockerfiles = {}
    for path in sorted(glob.glob("docker/**/*Dockerfile*", recursive=True)):
        with open(path, encoding="utf-8") as f:
            dockerfiles[path] = f.read()

    errors = []
    for package, args in PINNED_ARGS.items():
        # The Dockerfiles sed-replace this exact line, so require the exact form.
        pins = [m[1] for d in deps if (m := re.fullmatch(rf"{package}==(\S+)", d))]
        if len(pins) != 1:
            errors.append(f"{PYPROJECT}: expected one exact '{package}==X' pin")
            continue
        for arg, template in args.items():
            expected = template.format(pins[0])
            values = [
                (path, value.strip())
                for path, text in dockerfiles.items()
                for value in re.findall(rf"^\s*ARG\s+{arg}=(.*)$", text, re.MULTILINE)
            ]
            if not values:
                errors.append(f"no Dockerfile sets ARG {arg}=...; this check is dead")
            for path, value in values:
                if value != expected:
                    errors.append(
                        f"{path}: ARG {arg}={value}, expected {expected} "
                        f"to match {package}=={pins[0]} in {PYPROJECT}"
                    )

    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

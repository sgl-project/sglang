#!/usr/bin/env python3
"""Install missing SGLang dependencies without changing numerical packages."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement

try:
    import tomllib
except ImportError:
    import tomli as tomllib


# Packages that must remain exactly as supplied by the baseline image. Most are
# numerical; ``kernels`` is also preserved because its SGLang-pinned release
# targets Transformers 5 and changes the import path of baseline Transformers
# 4.56 even though MinWM never uses that LLM integration. ``sglang-kernel`` is
# also preserved because the SGLang-pinned wheel is compiled against SGLang's
# pinned Torch and may be ABI-incompatible with the baseline Torch.
PRESERVED_PACKAGES = {
    "diffusers",
    "flash-attn-4",
    "flashinfer-cubin",
    "flashinfer-python",
    "kernels",
    "kernels-data",
    "sgl-deep-gemm",
    "sglang-kernel",
    "st-attn",
    "tilelang",
    "tokenspeed-mla",
    "torch",
    "torchaudio",
    "torchcodec",
    "torch-memory-saver",
    "transformers",
    "torchvision",
    "vsa",
}


def normalize(name: str) -> str:
    return name.lower().replace("_", "-")


def is_missing(requirement: Requirement) -> bool:
    if normalize(requirement.name) in PRESERVED_PACKAGES:
        return False
    if requirement.marker and not requirement.marker.evaluate():
        return False
    try:
        importlib.metadata.version(requirement.name)
    except importlib.metadata.PackageNotFoundError:
        return True
    return False


def install_no_deps(specifications: list[str]) -> None:
    if not specifications:
        return
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--root-user-action=ignore",
            *specifications,
        ]
    )


def install_missing(pyproject_path: Path) -> None:
    pyproject = tomllib.loads(pyproject_path.read_text())
    specifications = [
        *pyproject["project"]["dependencies"],
        *pyproject["project"]["optional-dependencies"]["diffusion"],
    ]
    missing = [
        specification
        for specification in specifications
        if is_missing(Requirement(specification))
    ]
    install_no_deps(missing)

    pending = {Requirement(value).name for value in missing}
    visited: set[str] = set()
    while pending:
        package = pending.pop()
        normalized = normalize(package)
        if normalized in visited or normalized in PRESERVED_PACKAGES:
            continue
        visited.add(normalized)
        try:
            distribution = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        transitive = [
            dependency
            for dependency in distribution.requires or ()
            if is_missing(Requirement(dependency))
        ]
        install_no_deps(transitive)
        pending.update(Requirement(value).name for value in transitive)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", required=True, type=Path)
    args = parser.parse_args()
    install_missing(args.pyproject)


if __name__ == "__main__":
    main()

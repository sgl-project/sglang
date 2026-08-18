#!/usr/bin/env python3
"""Validate the ROCm 7.2.4 torch stack that actually ships.

Runs after AITER's Triton swap, unlike the earlier in-build check: that one runs
before the swap, so it cannot catch a broken Triton wiring, and the components
installed in between are free to move torch underneath us.

Checks torch's own recorded requirements rather than enforcing `pip check` over
the whole environment. TileLang publishes metadata this image cannot satisfy (it
requires torch-c-dlpack-ext, and an apache-tvm-ffi range that its own build pins
outside of), so a whole-environment check fails here for reasons that have
nothing to do with the torch stack. `pip check` is still reported, and the
earlier gate enforces it, since that one sits before the TileLang build.

The caller guards this to *-rocm724; nothing here is flavor-conditional.
"""

import importlib.metadata as metadata
import re
import subprocess
import sys

import torch
from packaging.requirements import Requirement


def unmet_torch_requirements(installed: dict[str, str]) -> list[str]:
    unmet = []
    for spec in metadata.distribution("torch").requires or []:
        requirement = Requirement(spec)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        version = installed.get(re.sub(r"[-_.]+", "-", requirement.name).lower())
        if version is None:
            unmet.append(f"{requirement} -> not installed")
        elif not requirement.specifier.contains(version, prereleases=True):
            unmet.append(f"{requirement} -> {version}")
    return unmet


def main(argv: list[str]) -> int:
    if subprocess.run([sys.executable, "-m", "pip", "check"]).returncode:
        print("[Final] pip check reported the above; only the torch stack is enforced")

    assert torch.__version__.startswith("2.11."), torch.__version__
    assert torch.version.hip, torch.__version__

    installed = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            installed[re.sub(r"[-_.]+", "-", name).lower()] = dist.version

    unmet = unmet_torch_requirements(installed)
    assert not unmet, f"torch requirements are unsatisfied: {unmet}"

    cuda = sorted(
        name for name in installed if re.fullmatch(r"nvidia-.*-cu[0-9]+", name)
    )
    assert not cuda, f"NVIDIA CUDA runtime packages in a ROCm image: {cuda}"

    print(
        "[Final] torch",
        torch.__version__,
        "HIP",
        torch.version.hip,
        "triton",
        metadata.version("triton"),
        "triton-kernels",
        metadata.version("triton-kernels"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

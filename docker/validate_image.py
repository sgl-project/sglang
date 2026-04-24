"""Post-build validator for SGLang CUDA Docker images.

Run inside a built image by the docker release workflow:

    docker run --rm -e VALIDATE_CUDA_VERSION=12.9.1 \\
        lmsysorg/sglang@<digest> python3 /usr/local/bin/validate_image.py

Fails the release if:
- torch's CUDA build doesn't match the image's CUDA_VERSION
- bundled NVIDIA libs (cudnn, nccl) diverge from torch's compiled-in versions
- NVIDIA packages we hard-pin (cuda-python, nvidia-cublas) got silently bumped
- critical Python imports fail (sglang, sgl_kernel, flashinfer, etc.)
"""

from __future__ import annotations

import os
import re
import sys
from importlib.metadata import PackageNotFoundError, version

# Source of truth: Dockerfile's "Patching packages for CUDA 12/13 compatibility"
# block. Update HARD_PINS whenever that block changes pinned versions.
HARD_PINS: dict[str, dict[str, str]] = {
    "12": {
        "cuda-python": "12.9",
    },
    "13": {
        "cuda-python": "13.2.0",
        "nvidia-cublas": "13.1.0.3",
    },
}

_CUDA_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _cudnn_int_to_str(n: int) -> str:
    """torch.backends.cudnn.version() returns packed int, e.g. 91600 -> '9.16.0'."""
    return f"{n // 10000}.{(n // 100) % 100}.{n % 100}"


def _version_matches(installed: str, want: str) -> bool:
    # PyPI metadata may normalize "12.9" to "12.9.0", so accept exact match OR
    # "installed starts with want followed by a version separator".
    return installed == want or installed.startswith(want + ".")


def _check(label: str, fn):
    # Deliberately narrow: only AssertionError is a "validation failure".
    # Any other exception indicates a bug in the validator or a missing
    # prerequisite (e.g. NCCL absent) and should crash with a traceback.
    try:
        fn()
    except AssertionError as e:
        print(f"FAIL: {label}: {e}", file=sys.stderr)
        return 1
    print(f"ok:   {label}")
    return 0


def main() -> int:
    cuda_version = os.environ.get("VALIDATE_CUDA_VERSION", "")
    if not _CUDA_VERSION_RE.match(cuda_version):
        print(
            f"ERROR: VALIDATE_CUDA_VERSION must match N.N.N (got {cuda_version!r}; e.g. '12.9.1')",
            file=sys.stderr,
        )
        return 2
    expected_torch_cuda = cuda_version.rsplit(".", 1)[0]  # "12.9.1" -> "12.9"
    cuda_major = cuda_version.split(".", 1)[0]  # "12.9.1" -> "12"

    failures = 0

    # 1. torch CUDA variant must match the image's CUDA_VERSION.
    def _check_torch_cuda():
        import torch

        print(f"      torch={torch.__version__} cuda={torch.version.cuda}")
        assert (
            torch.version.cuda == expected_torch_cuda
        ), f"torch.version.cuda={torch.version.cuda} != expected {expected_torch_cuda}"

    failures += _check("torch CUDA variant", _check_torch_cuda)

    # 2. Critical native imports (detects ABI breaks, missing .so).
    # Split imports so the error message names the specific failing module.
    def _check_import_torchaudio():
        import torchaudio

        print(f"      torchaudio={torchaudio.__version__}")

    failures += _check("import torchaudio", _check_import_torchaudio)

    def _check_import_torchvision():
        import torchvision

        print(f"      torchvision={torchvision.__version__}")

    failures += _check("import torchvision", _check_import_torchvision)

    def _check_import_sglang():
        import sglang  # noqa: F401

    failures += _check("import sglang", _check_import_sglang)

    def _check_import_sgl_kernel():
        import sgl_kernel  # noqa: F401

    failures += _check("import sgl_kernel", _check_import_sgl_kernel)

    def _check_import_flashinfer():
        import flashinfer  # noqa: F401

    failures += _check("import flashinfer", _check_import_flashinfer)

    # 3. Torch-bundled NVIDIA libs: pypi wheel version must startswith the
    # version torch was compiled against. Catches silent wheel downgrades.
    def _check_torch_bundled_nccl():
        import torch

        torch_nccl = ".".join(str(v) for v in torch.cuda.nccl.version())
        pkg = f"nvidia-nccl-cu{cuda_major}"
        installed = version(pkg)
        print(f"      {pkg}={installed}, torch nccl={torch_nccl}")
        assert _version_matches(
            installed, torch_nccl
        ), f"{pkg}={installed} does not match torch's linked nccl {torch_nccl}"

    failures += _check("torch <-> nvidia-nccl cross-check", _check_torch_bundled_nccl)

    def _check_torch_bundled_cudnn():
        import torch

        torch_cudnn = _cudnn_int_to_str(torch.backends.cudnn.version())
        pkg = f"nvidia-cudnn-cu{cuda_major}"
        installed = version(pkg)
        print(f"      {pkg}={installed}, torch cudnn={torch_cudnn}")
        assert _version_matches(
            installed, torch_cudnn
        ), f"{pkg}={installed} does not match torch's linked cudnn {torch_cudnn}"

    failures += _check("torch <-> nvidia-cudnn cross-check", _check_torch_bundled_cudnn)

    # 4. Hard-pinned non-torch-bundled packages.
    for pkg, want in HARD_PINS.get(cuda_major, {}).items():

        def _check_hard_pin(pkg=pkg, want=want):
            try:
                got = version(pkg)
            except PackageNotFoundError:
                raise AssertionError(f"{pkg} not installed; expected {want}")
            print(f"      {pkg}={got}")
            assert _version_matches(got, want), f"{pkg}={got} expected {want}"

        failures += _check(f"hard-pin {pkg}=={want}", _check_hard_pin)

    if failures:
        print(f"\n{failures} validation check(s) failed", file=sys.stderr)
        return 1
    print("\nAll validation checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

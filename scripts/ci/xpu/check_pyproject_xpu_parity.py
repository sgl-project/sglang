"""Fail when pyproject.toml gains a dependency that pyproject_xpu.toml lacks.

XPU CI installs from pyproject_xpu.toml, so a package added only to
pyproject.toml is silently missing on XPU. Each such package must either be
added to pyproject_xpu.toml or listed in XPU_EXCLUDED with a reason.
Compares `dependencies` and every optional-dependency group present in both.
"""

import re
import sys
import tomllib
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[3] / "python"

# Packages in pyproject.toml that are intentionally absent on XPU.
CUDA_ONLY = "CUDA/NVIDIA-only"
UNREVIEWED = "pre-existing drift, not yet reviewed for XPU"
XPU_EXCLUDED = {
    "cuda-python": CUDA_ONLY,
    "cuda-tile": CUDA_ONLY,
    "flash-attn-4": CUDA_ONLY,
    "flashinfer-python": CUDA_ONLY,
    "humming-kernels": CUDA_ONLY,
    "nvidia-cutlass-dsl": CUDA_ONLY,
    "nvidia-mathdx": CUDA_ONLY,
    "nvidia-ml-py": CUDA_ONLY,
    "nvidia-modelopt": CUDA_ONLY,
    "nvshmem4py-cu13": CUDA_ONLY,
    "quack-kernels": CUDA_ONLY,
    "sgl-deep-ep": CUDA_ONLY,
    "sgl-deep-gemm": CUDA_ONLY,
    "sglang-kernel": "replaced by sglang-kernel-xpu",
    "tilelang": CUDA_ONLY,
    "tokenspeed-mla": CUDA_ONLY,
    "torch-memory-saver": CUDA_ONLY,
    "vsa": CUDA_ONLY,
    "decord2": "aarch64/arm-only marker",
    "tomli": "python<3.11 only",
    "xgrammar": "installed separately with --no-deps in XPU CI",
    # NgramCorpus JIT needs it; test_ngram_corpus.py is disabled on XPU until added.
    "apache-tvm-ffi": UNREVIEWED,
    "distro": UNREVIEWED,
    "kernels": UNREVIEWED,
    "numba": UNREVIEWED,
    "tokenizers": UNREVIEWED,
    "watchfiles": UNREVIEWED,
    "zstandard": UNREVIEWED,
    "msgpack": UNREVIEWED,
    "opencv-python-headless": UNREVIEWED,
    "websockets": UNREVIEWED,
    "addict": UNREVIEWED,
    "antlr4-python3-runtime": UNREVIEWED,
    "auto-round": UNREVIEWED,
    "av": UNREVIEWED,
    "diff-cover": UNREVIEWED,
    "granian": UNREVIEWED,
    "pytest-cov": UNREVIEWED,
    "sgl-eval": UNREVIEWED,
    "sglang": "self-reference in extras",
}


def pkg_name(req):
    name = re.split(r"[\s<>=!~;\[@(]", req.strip(), maxsplit=1)[0]
    return name.lower().replace("_", "-")


def dep_groups(path):
    project = tomllib.loads(path.read_text())["project"]
    groups = {"dependencies": project.get("dependencies", [])}
    groups.update(project.get("optional-dependencies", {}))
    return {g: {pkg_name(r) for r in reqs} for g, reqs in groups.items()}


def main():
    main_groups = dep_groups(PYTHON_DIR / "pyproject.toml")
    xpu_groups = dep_groups(PYTHON_DIR / "pyproject_xpu.toml")

    missing, used = [], set()
    for group, xpu_pkgs in xpu_groups.items():
        for pkg in sorted(main_groups.get(group, set()) - xpu_pkgs):
            if pkg in XPU_EXCLUDED:
                used.add(pkg)
            else:
                missing.append((group, pkg))

    for pkg in sorted(set(XPU_EXCLUDED) - used):
        print(f"::warning::XPU_EXCLUDED entry '{pkg}' is stale; remove it.")
    for group, pkg in missing:
        print(
            f"::error::'{pkg}' is in pyproject.toml [{group}] but not in "
            "pyproject_xpu.toml. Add it there, or to XPU_EXCLUDED in "
            f"{Path(__file__).name}."
        )
    if missing:
        sys.exit(1)
    print("pyproject_xpu.toml dependency parity OK")


if __name__ == "__main__":
    main()

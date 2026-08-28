# SPDX-License-Identifier: Apache-2.0

# KDA provenance: this kernel was automatically optimized by the Humanize2
# workflow (https://github.com/PolyArch/humanize) and Kernel Design Agents
# (https://github.com/mit-han-lab/kernel-design-agents).
# Source: https://github.com/BBuf/KDA-Pilot/pull/195 @
# 516c976cee824a236679adf6eb525275a0a9a120.
"""Candidate-owned CUTLASS SM120 NVFP4 GEMM extension."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parent
_FLASHINFER = importlib.util.find_spec("flashinfer")
if _FLASHINFER is None or _FLASHINFER.origin is None:
    raise ImportError("flashinfer's CUTLASS headers are required")
_DATA = Path(_FLASHINFER.origin).resolve().parent / "data"
_LOCAL_SOURCES = (_ROOT / "cutlass_fp4.cpp", _ROOT / "cutlass_fp4_cuda.cu")
_SOURCE_HASH = hashlib.sha256()
for _source in _LOCAL_SOURCES:
    _SOURCE_HASH.update(_source.name.encode())
    _SOURCE_HASH.update(_source.read_bytes())

_extension = load(
    # Reconstructed submissions share the host extension cache. Key it by the
    # local sources so the loaded kernel always matches this candidate.
    name=f"sglang_kda_qwen38_cutlass_fp4_sm120_{_SOURCE_HASH.hexdigest()[:12]}",
    sources=[str(source) for source in _LOCAL_SOURCES],
    extra_include_paths=[
        str(_DATA / "cutlass/include"),
        str(_DATA / "cutlass/tools/util/include"),
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "-lineinfo",
        "-gencode=arch=compute_120f,code=sm_120f",
        "--expt-relaxed-constexpr",
        "-static-global-template-stub=false",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ],
)

cutlass_fp4_gemm = _extension.cutlass_fp4_gemm

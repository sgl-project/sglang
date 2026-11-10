#!/usr/bin/env python3
"""Build the optional LMDeploy/1Cat TurboMind SM70 extension.

Run this with the same Python/Torch environment used to serve SGLang:

    conda run -n sglang-v100 python scripts/build_sm70_turbomind.py

Only the explicitly attributed TurboMind and MoE source subtrees are needed.
The resulting private extension is compiled against the active Torch ABI and
installed beside the SGLang JIT loaders; no vLLM or attention package from the
source repository is installed or imported.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from torch.utils.cpp_extension import load

repo = Path(__file__).resolve().parents[1]
source_root = Path(
    os.environ.get(
        "SGLANG_TURBOMIND_SM70_ROOT",
        Path.home() / ".cache/sglang-v100-sources/turbomind-sm70-source",
    )
).expanduser()
tm_root = source_root / "csrc" / "sm70_turbomind"
lmdeploy = tm_root / "lmdeploy"
cutlass_root = Path(
    os.environ.get(
        "SGLANG_TURBOMIND_CUTLASS_ROOT",
        Path.home() / ".cache/sglang-v100-sources/cutlass-turbomind",
    )
).expanduser()
if not lmdeploy.is_dir():
    raise SystemExit(
        f"TurboMind sources not found under {tm_root}. Set "
        "SGLANG_TURBOMIND_SM70_ROOT to the attributed source subset."
    )
if not (cutlass_root / "include" / "cutlass" / "cutlass.h").is_file():
    raise SystemExit(
        f"CUTLASS headers not found under {cutlass_root}. Set "
        "SGLANG_TURBOMIND_CUTLASS_ROOT to the pinned CUTLASS checkout."
    )

lmdeploy_sources = [
    "src/turbomind/core/check.cc",
    "src/turbomind/core/layout.cc",
    "src/turbomind/core/context.cc",
    "src/turbomind/core/allocator.cc",
    "src/turbomind/core/buffer.cc",
    "src/turbomind/core/stream.cc",
    "src/turbomind/utils/logger.cc",
    "src/turbomind/utils/parser.cc",
    "src/turbomind/kernels/gemm/gemm.cu",
    "src/turbomind/kernels/gemm/kernel.cu",
    "src/turbomind/kernels/gemm/dispatch_cache.cu",
    "src/turbomind/kernels/gemm/context.cu",
    "src/turbomind/kernels/gemm/convert_v3.cu",
    "src/turbomind/kernels/gemm/cast.cu",
    "src/turbomind/kernels/gemm/unpack.cu",
    "src/turbomind/kernels/gemm/tuner/cache_utils.cu",
    "src/turbomind/kernels/gemm/tuner/measurer.cu",
    "src/turbomind/kernels/gemm/tuner/sampler.cu",
    "src/turbomind/kernels/gemm/tuner/stopping_criterion.cc",
    "src/turbomind/kernels/gemm/tuner/params.cc",
    "src/turbomind/kernels/gemm/kernel/sm70_884_4.cu",
    "src/turbomind/kernels/gemm/kernel/sm70_884_8.cu",
    "src/turbomind/kernels/gemm/kernel/sm70_884_16.cu",
]
sources = [
    str(repo / "python/sglang/kernels/jit/csrc/sm70_turbomind_bindings.cpp"),
    str(repo / "python/sglang/kernels/jit/csrc/sm70_fp8_e5m2_cache.cu"),
    str(repo / "python/sglang/kernels/jit/csrc/sm70_fp16_moe_gemm.cu"),
    *(str(lmdeploy / path) for path in lmdeploy_sources),
    str(tm_root / "ops/tm_registry_sm70.cu"),
    str(tm_root / "ops/awq_sm70_gemm.cu"),
    str(source_root / "csrc/moe/moe_permute_unpermute_op.cu"),
    str(
        source_root
        / "csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu"
    ),
]

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
extension = load(
    name="sglang_sm70_turbomind",
    sources=sources,
    extra_include_paths=[
        str(lmdeploy),
        str(source_root / "csrc"),
        str(source_root / "csrc/moe"),
        str(cutlass_root / "include"),
        str(cutlass_root / "tools" / "util" / "include"),
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-DENABLE_SM70_TURBOMIND=1",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ],
    is_python_module=False,
    verbose=True,
)
destination = repo / "python/sglang/kernels/prebuilt/_sm70_turbomind_v100.so"
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(extension, destination)
print(f"Installed {destination}")

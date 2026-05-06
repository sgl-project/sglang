"""CUDA kernels for HISA stage 4 — sparse block MQA on Hopper.

Built on first import via torch.utils.cpp_extension.load. The compiled .so
is cached in TORCH_EXTENSIONS_DIR (default ~/.cache/torch_extensions).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_DIR = Path(__file__).parent
_BUILD_DIR = Path(os.environ.get("HISA_CUDA_BUILD_DIR", _DIR / "_build"))
_BUILD_DIR.mkdir(exist_ok=True)


_module = None


_DG_INC = "/usr/local/lib/python3.12/dist-packages/deep_gemm/include"


def _ensure_built():
    global _module
    if _module is not None:
        return _module
    _module = load(
        name="hisa_block_sparse_sm90",
        sources=[str(_DIR / "block_sparse_mqa_sm90.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-std=c++17",
        ],
        extra_cflags=["-O3"],
        build_directory=str(_BUILD_DIR),
        verbose=False,
    )
    return _module


_module_v1 = None


def _ensure_built_v1():
    global _module_v1
    if _module_v1 is not None:
        return _module_v1
    _module_v1 = load(
        name="hisa_block_sparse_sm90_v1",
        sources=[str(_DIR / "block_sparse_mqa_sm90_v1.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            f"-I{_DG_INC}",
            "-DCUTE_ARCH_MMA_SM90A_ENABLED",
            "-DCUTE_ARCH_TMA_SM90_ENABLED",
        ],
        extra_cflags=["-O3", f"-I{_DG_INC}"],
        build_directory=str(_BUILD_DIR / "v1"),
        verbose=False,
    )
    (_BUILD_DIR / "v1").mkdir(exist_ok=True)
    return _module_v1


def block_sparse_mqa_cuda_v1(
    q_fp8, k_fp8, k_scale, topk_block_index,
    weights, cu_seqlen_ks, cu_seqlen_ke, kv_block_size=8,
):
    assert kv_block_size == 8
    assert q_fp8.shape[1] == 64 and q_fp8.shape[2] == 128
    seq = q_fp8.shape[0]
    topk = topk_block_index.shape[-1]
    if topk_block_index.dtype != torch.int32:
        topk_block_index = topk_block_index.to(torch.int32)
    logits = torch.empty((seq, topk * kv_block_size), device=q_fp8.device, dtype=torch.float32)
    mod = _ensure_built_v1()
    mod.block_sparse_mqa_sm90_v1(
        q_fp8, k_fp8, k_scale, topk_block_index, logits, weights, cu_seqlen_ks, cu_seqlen_ke,
    )
    return logits


_module_v2 = None


def _ensure_built_v2():
    global _module_v2
    if _module_v2 is not None:
        return _module_v2
    (_BUILD_DIR / "v2").mkdir(exist_ok=True)
    _module_v2 = load(
        name="hisa_block_sparse_sm90_v2",
        sources=[str(_DIR / "block_sparse_mqa_sm90_v2.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            f"-I{_DG_INC}",
            "-DCUTE_ARCH_MMA_SM90A_ENABLED",
            "-DCUTE_ARCH_TMA_SM90_ENABLED",
        ],
        extra_cflags=["-O3", f"-I{_DG_INC}"],
        build_directory=str(_BUILD_DIR / "v2"),
        verbose=False,
    )
    return _module_v2


def block_sparse_mqa_cuda_v2(
    q_fp8, k_fp8, k_scale, topk_block_index,
    weights, cu_seqlen_ks, cu_seqlen_ke, kv_block_size=8,
):
    assert kv_block_size == 8
    assert q_fp8.shape[1] == 64 and q_fp8.shape[2] == 128
    seq = q_fp8.shape[0]
    topk = topk_block_index.shape[-1]
    if topk_block_index.dtype != torch.int32:
        topk_block_index = topk_block_index.to(torch.int32)
    logits = torch.empty((seq, topk * kv_block_size), device=q_fp8.device, dtype=torch.float32)
    mod = _ensure_built_v2()
    mod.block_sparse_mqa_sm90_v2(
        q_fp8, k_fp8, k_scale, topk_block_index, logits, weights, cu_seqlen_ks, cu_seqlen_ke,
    )
    return logits


_module_v3 = None


def _ensure_built_v3():
    global _module_v3
    if _module_v3 is not None:
        return _module_v3
    (_BUILD_DIR / "v3").mkdir(exist_ok=True)
    _module_v3 = load(
        name="hisa_block_sparse_sm90_v3",
        sources=[str(_DIR / "block_sparse_mqa_sm90_v3.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            f"-I{_DG_INC}",
            "-DCUTE_ARCH_MMA_SM90A_ENABLED",
            "-DCUTE_ARCH_TMA_SM90_ENABLED",
        ],
        extra_cflags=["-O3", f"-I{_DG_INC}"],
        build_directory=str(_BUILD_DIR / "v3"),
        verbose=False,
    )
    return _module_v3


def block_sparse_mqa_cuda_v3(
    q_fp8, k_fp8, k_scale, topk_block_index,
    weights, cu_seqlen_ks, cu_seqlen_ke, kv_block_size=8,
):
    assert kv_block_size == 8
    assert q_fp8.shape[1] == 64 and q_fp8.shape[2] == 128
    seq = q_fp8.shape[0]
    topk = topk_block_index.shape[-1]
    if topk_block_index.dtype != torch.int32:
        topk_block_index = topk_block_index.to(torch.int32)
    logits = torch.empty((seq, topk * kv_block_size), device=q_fp8.device, dtype=torch.float32)
    mod = _ensure_built_v3()
    mod.block_sparse_mqa_sm90_v3(
        q_fp8, k_fp8, k_scale, topk_block_index, logits, weights, cu_seqlen_ks, cu_seqlen_ke,
    )
    return logits


_module_v4 = None


def _ensure_built_v4():
    global _module_v4
    if _module_v4 is not None:
        return _module_v4
    (_BUILD_DIR / "v4").mkdir(exist_ok=True)
    _module_v4 = load(
        name="hisa_block_sparse_sm90_v4",
        sources=[str(_DIR / "block_sparse_mqa_sm90_v4.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            f"-I{_DG_INC}",
            "-DCUTE_ARCH_MMA_SM90A_ENABLED",
            "-DCUTE_ARCH_TMA_SM90_ENABLED",
        ],
        extra_cflags=["-O3", f"-I{_DG_INC}"],
        extra_ldflags=["-lcuda"],
        build_directory=str(_BUILD_DIR / "v4"),
        verbose=False,
    )
    return _module_v4


def block_sparse_mqa_cuda_v4(
    q_fp8, k_fp8, k_scale, topk_block_index,
    weights, cu_seqlen_ks, cu_seqlen_ke, kv_block_size=8,
):
    assert kv_block_size == 8
    assert q_fp8.shape[1] == 64 and q_fp8.shape[2] == 128
    seq = q_fp8.shape[0]
    topk = topk_block_index.shape[-1]
    if topk_block_index.dtype != torch.int32:
        topk_block_index = topk_block_index.to(torch.int32)
    logits = torch.empty((seq, topk * kv_block_size), device=q_fp8.device, dtype=torch.float32)
    mod = _ensure_built_v4()
    mod.block_sparse_mqa_sm90_v4(
        q_fp8, k_fp8, k_scale, topk_block_index, logits, weights, cu_seqlen_ks, cu_seqlen_ke,
    )
    return logits


_module_v5 = None


def _ensure_built_v5():
    global _module_v5
    if _module_v5 is not None:
        return _module_v5
    (_BUILD_DIR / "v5").mkdir(exist_ok=True)
    _module_v5 = load(
        name="hisa_block_sparse_sm90_v5",
        sources=[str(_DIR / "block_sparse_mqa_sm90_v5.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            f"-I{_DG_INC}",
            "-DCUTE_ARCH_MMA_SM90A_ENABLED",
            "-DCUTE_ARCH_TMA_SM90_ENABLED",
        ],
        extra_cflags=["-O3", f"-I{_DG_INC}"],
        extra_ldflags=["-lcuda"],
        build_directory=str(_BUILD_DIR / "v5"),
        verbose=False,
    )
    return _module_v5


def block_sparse_mqa_cuda_v5(
    q_fp8, k_fp8, k_scale, topk_block_index,
    weights, cu_seqlen_ks, cu_seqlen_ke, kv_block_size=8,
):
    assert kv_block_size == 8
    assert q_fp8.shape[1] == 64 and q_fp8.shape[2] == 128
    seq = q_fp8.shape[0]
    topk = topk_block_index.shape[-1]
    if topk_block_index.dtype != torch.int32:
        topk_block_index = topk_block_index.to(torch.int32)
    logits = torch.empty((seq, topk * kv_block_size), device=q_fp8.device, dtype=torch.float32)
    mod = _ensure_built_v5()
    mod.block_sparse_mqa_sm90_v5(
        q_fp8, k_fp8, k_scale, topk_block_index, logits, weights, cu_seqlen_ks, cu_seqlen_ke,
    )
    return logits


_module_v6 = None


def _ensure_built_v6():
    global _module_v6
    if _module_v6 is not None:
        return _module_v6
    (_BUILD_DIR / "v6").mkdir(exist_ok=True)
    _module_v6 = load(
        name="hisa_block_sparse_sm90_v6",
        sources=[str(_DIR / "block_sparse_mqa_sm90_v6.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            f"-I{_DG_INC}",
            "-DCUTE_ARCH_MMA_SM90A_ENABLED",
            "-DCUTE_ARCH_TMA_SM90_ENABLED",
        ],
        extra_cflags=["-O3", f"-I{_DG_INC}"],
        extra_ldflags=["-lcuda"],
        build_directory=str(_BUILD_DIR / "v6"),
        verbose=False,
    )
    return _module_v6


def block_sparse_mqa_cuda_v6(
    q_fp8, k_fp8, k_scale, topk_block_index,
    weights, cu_seqlen_ks, cu_seqlen_ke, kv_block_size=8,
):
    assert kv_block_size == 8
    assert q_fp8.shape[1] == 64 and q_fp8.shape[2] == 128
    seq = q_fp8.shape[0]
    topk = topk_block_index.shape[-1]
    if topk_block_index.dtype != torch.int32:
        topk_block_index = topk_block_index.to(torch.int32)
    logits = torch.empty((seq, topk * kv_block_size), device=q_fp8.device, dtype=torch.float32)
    mod = _ensure_built_v6()
    mod.block_sparse_mqa_sm90_v6(
        q_fp8, k_fp8, k_scale, topk_block_index, logits, weights, cu_seqlen_ks, cu_seqlen_ke,
    )
    return logits


def block_sparse_mqa_cuda_v0(
    q_fp8: torch.Tensor,           # [seq, H, D] fp8 contiguous
    k_fp8: torch.Tensor,           # [seq_kv, D] fp8
    k_scale: torch.Tensor,         # [seq_kv] f32
    topk_block_index: torch.Tensor,  # [seq, topk] i32 or i64
    weights: torch.Tensor,         # [seq, H] f32
    cu_seqlen_ks: torch.Tensor,    # [seq] i32
    cu_seqlen_ke: torch.Tensor,    # [seq] i32
    kv_block_size: int = 8,
) -> torch.Tensor:
    assert kv_block_size == 8, "v0 hardcodes K=8"
    assert q_fp8.shape[1] == 64, "v0 hardcodes H=64"
    assert q_fp8.shape[2] == 128, "v0 hardcodes D=128"
    seq, H, D = q_fp8.shape
    topk = topk_block_index.shape[-1]
    if topk_block_index.dtype != torch.int32:
        topk_block_index = topk_block_index.to(torch.int32)
    logits = torch.empty((seq, topk * kv_block_size), device=q_fp8.device, dtype=torch.float32)
    mod = _ensure_built()
    mod.block_sparse_mqa_sm90_v0(
        q_fp8, k_fp8, k_scale, topk_block_index, logits, weights, cu_seqlen_ks, cu_seqlen_ke,
    )
    return logits

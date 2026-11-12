# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""One-pass FP8 DSA indexer verify logits (K31HT_IDX_FP8_1PASS): a drop-in for deep_gemm.fp8_paged_mqa_logits at 5..8 query rows
per request. The kernel (dg_logits_1p.cuh: sm100_paged_mqa_logits_1p, the vendored DeepGEMM paged MQA logits kernel with a one-pass
entry) keeps all rows of a request in ONE Q-block (UMMA_N = rows x 32), so each 256-token index-K split is streamed through TMA and
MMA'd once per request instead of once per 4-row block; same per-(row, token) arithmetic, so the logits are bitwise equal.
"""

from __future__ import annotations

import ctypes
import hashlib
import re
from functools import cache
from pathlib import Path

import torch

from .jit import check, driver, launch, load

from .dg_logits import LOGITS_ALIGN, MATH_THREADS, SPECIALIZED_THREADS, SPLIT_KV, SPLITS_PER_CHUNK, check_header, compile_unit

HEADER_1P = Path(__file__).with_name("dg_logits_1p.cuh")
HEADS, HEAD_DIM, PAGE = 32, 128, 64
_SYMBOL = re.compile(rb"_ZN9deep_gemm25sm100_paged_mqa_logits_1p[A-Za-z0-9_]+")


def _align(x: int, a: int) -> int:
    return -(-x // a) * a


def _encode(dtype, base: int, dims, strides_bytes, box, swizzle):
    cu = driver()
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    (tensor_map,) = check(
        cu.cuTensorMapEncodeTiled(
            dtype, len(dims), base, [u64(d) for d in dims], [u64(s) for s in strides_bytes], [u32(b) for b in box],
            [u32(1)] * len(dims), cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled(dg_logits_1p)",
    )
    return tensor_map


# ---------------------------------------------------------------------------------------------------------------------------------
# The same one-pass entry for the SERVED FP8 form (FP8 query x FP8 index-K + FP32 per-token scale, no MX scales),
# a drop-in for deep_gemm.fp8_paged_mqa_logits at 5..8 rows per request (no scale TMEM columns, so Q8 fits 2 x 256 accumulator columns).
# Pipeline: 2 query stages; 5 index-K stages (4 at 8 rows, for shared memory). Tensor maps as dg_logits.py (DeepGEMM's FP8 launcher) with
# rows x 32 query rows per box.
ROWS_1P_FP8 = (5, 6, 7, 8)


def stages_fp8(n: int):
    return (2, 4 if n == 8 else 5)


def tmem_stages_fp8(n: int) -> int:
    return 3 if _align(n * HEADS, 8) * 3 <= 512 else 2


def shared_bytes_fp8(n: int) -> int:
    """sizeof(layout::MQALogitsSharedStorage<32, 128, false, n, 256, qs, ks, tmem_stages_fp8(n), e4m3, float>)."""
    qs, ks = stages_fp8(n)
    umma_n = _align(n * HEADS, 8)
    off = qs * umma_n * HEAD_DIM  # smem_q
    off = _align(off, 1024) + ks * SPLIT_KV * HEAD_DIM  # smem_kv
    off = _align(off, 128) + qs * 4  # smem_sf_q (one float per stage without MX)
    off = _align(off, 128) + ks * SPLIT_KV * 4  # smem_sf_kv (per-token FP32 scales)
    off = _align(off, 128) + qs * _align(n * _align(HEADS * 4, 16), 128)  # smem_weights
    off += 8 * (2 * qs + 2 * ks + 2 * tmem_stages_fp8(n)) + 4
    return _align(off, 1024)


def source_fp8(n: int) -> str:
    md5 = hashlib.md5(HEADER_1P.read_bytes()).hexdigest()
    qs, ks = stages_fp8(n)
    return f"""// dg_logits_1p.cuh md5 {md5} (FP8)
#include <dg_logits_1p.cuh>

using namespace deep_gemm;

static_assert(sizeof(layout::MQALogitsSharedStorage<
    {HEADS}, {HEAD_DIM}, false, {n}, {SPLIT_KV}, {qs}, {ks}, {tmem_stages_fp8(n)}, cutlass::float_e4m3_t, float
>) == {shared_bytes_fp8(n)}, "Incorrect MQA logits shared-memory size");

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_paged_mqa_logits_1p<
        {n}, {HEADS},
        {HEAD_DIM}, {PAGE},
        false, true, false,
        {qs}, {ks},
        {SPLIT_KV}, {SPLITS_PER_CHUNK},
        {SPECIALIZED_THREADS}, {MATH_THREADS},
        cutlass::float_e4m3_t, float, float
    >);
}};
"""


def kernel_symbol_fp8(cubin: bytes, n: int) -> str:
    names = sorted({m.decode() for m in _SYMBOL.findall(cubin)}, key=len)
    names = [x for x in names if x.startswith(f"_ZN9deep_gemm25sm100_paged_mqa_logits_1pILj{n}E")]
    if not names or "12float_e4m3_t" not in names[0] or "e2m1" in names[0] or not all(x.startswith(names[0]) for x in names):
        raise RuntimeError(f"dg_logits_1p: expected one FP8 1-pass kernel symbol in the cubin, found {names}")
    return names[0]


@cache
def build_fp8(device_index: int, n: int):
    if n not in ROWS_1P_FP8:
        raise ValueError(f"dg_logits_1p fp8: rows per request {n} not in {ROWS_1P_FP8}")
    check_header(HEADER_1P.read_text())
    with torch.cuda.device(device_index):
        cubin = compile_unit(source_fp8(n))
        return load(cubin, kernel_symbol_fp8(cubin, n), (SPECIALIZED_THREADS + MATH_THREADS, 1, 1), shared_bytes_fp8(n))


_maps_fp8: dict = {}


def _tensor_maps_fp8(q, cache_bytes, weights, n):
    rows = q.shape[0] * q.shape[1]
    blocks = cache_bytes.shape[0]
    page_bytes = cache_bytes.stride(0)
    key = (q.data_ptr(), rows, cache_bytes.data_ptr(), blocks, page_bytes, weights.data_ptr(), weights.stride(0), n)
    maps = _maps_fp8.get(key)
    if maps is None:
        cu = driver()
        u8, f32 = cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8, cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32
        sw128, none = cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
        base = cache_bytes.data_ptr()
        tm_q = _encode(u8, q.data_ptr(), (HEAD_DIM, rows * HEADS), (HEAD_DIM,), (HEAD_DIM, n * HEADS), sw128)
        tm_kv = _encode(u8, base, (HEAD_DIM, PAGE, blocks), (HEAD_DIM, page_bytes), (HEAD_DIM, PAGE, 1), sw128)
        tm_sf_kv = _encode(f32, base + PAGE * HEAD_DIM, (PAGE, blocks), (page_bytes,), (PAGE, 1), none)
        tm_w = _encode(f32, weights.data_ptr(), (HEADS, rows), (weights.stride(0) * 4,), (HEADS, n), none)
        maps = _maps_fp8[key] = (tm_q, tm_sf_kv, tm_kv, tm_w)
    return maps


def paged_mqa_logits_fp8(q, fused_cache, weights, context_lens, block_table, schedule, max_context_len):
    """deep_gemm.fp8_paged_mqa_logits(q, fused_cache, weights, context_lens, block_table, schedule, max_context_len, clean_logits=False)
    for next_n in ROWS_1P_FP8: q FP8 [B, N, 32, 128] contiguous, fused_cache uint8 [blocks, 64, 1, 132], weights FP32 [B * N, 32]."""
    batch, next_n, heads, dim = q.shape
    blocks, page, kv_heads, page_row = fused_cache.shape
    if (heads, dim, page, kv_heads, page_row) != (HEADS, HEAD_DIM, PAGE, 1, HEAD_DIM + 4) or not q.is_contiguous():
        raise ValueError(f"dg_logits_1p fp8: unsupported shapes q {tuple(q.shape)} cache {tuple(fused_cache.shape)}")
    if q.dtype != torch.float8_e4m3fn or fused_cache.dtype != torch.uint8 or weights.dtype != torch.float32:
        raise ValueError("dg_logits_1p fp8: FP8 query, uint8 cache, FP32 weights only")
    if fused_cache.stride(1) != page_row or fused_cache.stride(3) != 1 or weights.stride(1) != 1:
        raise ValueError("dg_logits_1p fp8: cache pages and weight rows must be dense")
    if tuple(context_lens.shape) != (batch, next_n) or not context_lens.is_contiguous() or block_table.stride(1) != 1:
        raise ValueError("dg_logits_1p fp8: context_lens [B, N] contiguous, block_table unit column stride")
    rows = batch * next_n
    device = q.device.index if q.device.index is not None else torch.cuda.current_device()
    kernel = build_fp8(device, next_n)
    stride = -(-(-(-max_context_len // SPLIT_KV) * SPLIT_KV) // LOGITS_ALIGN) * LOGITS_ALIGN
    logits = torch.empty((rows, stride), device=q.device, dtype=torch.float32)[:, :max_context_len]
    tm_q, tm_sf_kv, tm_kv, tm_w = _tensor_maps_fp8(q, fused_cache.view(blocks, page * page_row), weights, next_n)
    args = [
        ctypes.c_uint32(rows), ctypes.c_uint32(stride), ctypes.c_uint32(block_table.stride(0)),
        ctypes.c_void_p(context_lens.data_ptr()), ctypes.c_void_p(logits.data_ptr()),
        ctypes.c_void_p(block_table.data_ptr()), ctypes.c_void_p(0), ctypes.c_void_p(schedule.data_ptr()),
        tm_q, tm_sf_kv, tm_kv, tm_sf_kv, tm_w,
    ]
    launch(kernel, (schedule.shape[0] - 1, 1, 1), args, torch.cuda.current_stream().cuda_stream)
    return logits

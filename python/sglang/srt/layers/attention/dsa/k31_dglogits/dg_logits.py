# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Verify index logits from a vendored copy of DeepGEMM's SM100 paged MQA logits kernel (dg_logits.cuh), launched with
PDL. Drop-in for ``deep_gemm.fp8_paged_mqa_logits`` at the verify shapes (FP8 index query, FP32 head gates and logits,
fused FP8 + FP32-scale index-K pages of 64 tokens, 2D context lengths): the same instantiation unit, nvcc flags, tensor
maps, grid and shared memory as DeepGEMM's own JIT launch, so the code differs only where dg_logits.cuh says and the
logits are bitwise equal. The kernel differs from DeepGEMM's in two places: its TMA warps fence the async proxy after the
grid-dependency wait, and thread 0 triggers its dependents at entry behind a fence.acq_rel.gpu."""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from functools import cache
from pathlib import Path

import torch

from .jit import Kernel, cache_dir, check, driver, launch, load

HEADER = Path(__file__).with_name("dg_logits.cuh")
# Unmodified copies of the sgl-deep-gemm 0.2.3.post2 headers that dg_logits.cuh and dg_logits_1p.cuh include
# (dg_include/README.md). They are searched before the installed package's include tree, so the units compile the same
# on images whose sgl-deep-gemm moved or changed them (0.2.6 moved epilogue/clean_logits.cuh to legacy/); CUTLASS and
# CuTe come from the installed package.
VENDORED_INCLUDE = Path(__file__).with_name("dg_include")
# DeepGEMM's SM100 paged FP8 launch: 4 query rows x 32 heads per block, 256-token splits in chunks of 16, 3 query and
# 5 index-K pipeline stages, 128 specialized + 256 math threads; its shared storage is 220160 bytes.
HEADS, HEAD_DIM, PAGE = 32, 128, 64
BLOCK_Q = 128 // HEADS
SPLIT_KV, SPLITS_PER_CHUNK = 256, 16
Q_STAGES, KV_STAGES = 3, 5
SPECIALIZED_THREADS, MATH_THREADS = 128, 256
SHARED_BYTES = 220160
LOGITS_ALIGN = 1024 // 4  # FP32 logits rows start on 1024-byte boundaries
# DeepGEMM's JIT nvcc options (deep_jit/backend/cuda/options.hpp defaults: -O3, register usage level 10, line info)
NVCC_FLAGS = (
    "-O3", "--compiler-options=-O3", "--ptxas-options=--register-usage-level=10", "--generate-line-info", "-std=c++20",
    "--compiler-options=-fPIC", "--compiler-options=-fconcepts", "--expt-relaxed-constexpr", "--expt-extended-lambda",
    "--diag-suppress=39,161,174,177,186,940", "--compiler-options=-Wno-deprecated-declarations,-Wno-abi",
)


def symbol(tokens_per_request: int) -> str:
    return (f"_ZN9deep_gemm22sm100_paged_mqa_logitsILj{tokens_per_request}ELj32ELj128ELj64ELb0ELb1ELb0ELj3ELj5ELj256ELj16"
            "ELj128ELj256EN7cutlass12float_e4m3_tEffLj2EEEvjjjPKjPT13_S4_S4_S4_14CUtensorMap_stS7_S7_S7_S7_")


def dg_include() -> str:
    import deep_gemm

    return str(Path(deep_gemm.__file__).resolve().parent / "include")


def source(tokens_per_request: int, header: str = "dg_logits.cuh") -> str:
    """DeepGEMM's own instantiation unit for this launch, including ``header`` instead of its sm100_mqa_logits.cuh."""
    return f"""
#include <{header}>

using namespace deep_gemm;

static_assert(sizeof(layout::MQALogitsSharedStorage<
    {HEADS}, {HEAD_DIM}, false, {BLOCK_Q}, {SPLIT_KV}, {Q_STAGES}, {KV_STAGES}, 3, cutlass::float_e4m3_t, float
>) == {SHARED_BYTES}, "Incorrect MQA logits shared-memory size");

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_paged_mqa_logits<
        {tokens_per_request}, {HEADS},
        {HEAD_DIM}, {PAGE},
        false, true, false,
        {Q_STAGES}, {KV_STAGES},
        {SPLIT_KV}, {SPLITS_PER_CHUNK},
        {SPECIALIZED_THREADS}, {MATH_THREADS},
        cutlass::float_e4m3_t, float, float
    >);
}};
"""


def nvcc() -> str:
    root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    return str(Path(root) / "bin" / "nvcc")


def arch() -> str:
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError("dg_logits: SM100-family GPUs only")
    return "sm_100f"  # DeepGEMM's family target on every SM10x GPU


def compile_unit(text: str, kind: str = "cubin", header_dir: str | None = None) -> bytes:
    """nvcc ``text`` to a cubin (or PTX) with DeepGEMM's flags; the vendored header is found in ``header_dir``
    (default: this package), the DeepGEMM headers it includes in VENDORED_INCLUDE. Cached on disk by source, header,
    flags and compiler version."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("compiling dg_logits inside CUDA-graph capture")
    header_dir = header_dir or str(HEADER.parent)
    version = subprocess.run([nvcc(), "--version"], capture_output=True, text=True, check=True).stdout
    flags = (f"--gpu-architecture={arch()}", *NVCC_FLAGS, "--include-path", header_dir, "--include-path", str(VENDORED_INCLUDE),
             "--include-path", dg_include())
    header = Path(header_dir, HEADER.name).read_text()
    key = hashlib.sha256("\0".join([text, header, *flags, version, kind]).encode()).hexdigest()[:24]
    cached = cache_dir() / f"k31_dg_logits-{key}.{kind}"
    if cached.is_file():
        return cached.read_bytes()
    with tempfile.TemporaryDirectory() as work:
        unit, out = Path(work, "kernel.cu"), Path(work, f"kernel.{kind}")
        unit.write_text(text)
        result = subprocess.run([nvcc(), str(unit), f"--{kind}", "--output-file", str(out), *flags],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"dg_logits: nvcc failed: {result.stderr[-4000:]}")
        data = out.read_bytes()
    cached.parent.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(cached)
    return data


# The header's two additions carry its memory-model argument: without the trigger the launch is plain DeepGEMM, without a
# proxy fence a TMA warp may read its predecessors' plain stores through the async proxy unordered, and without the
# fence.acq_rel.gpu in front of the trigger a dependent's reads before its own wait (the top-k's row lengths) are not
# ordered after the stores this grid waited for. A header that lost any of them still compiles and still gives bitwise
# logits in tests, so build() checks the source text and refuses it.
_WAIT = "cudaGridDependencySynchronize();"
_TRIGGER = r'asm volatile("fence.acq_rel.gpu;\n\tgriddepcontrol.launch_dependents;" ::: "memory");'
# Thread 0 alone fences and triggers (only a CTA's first trigger has an effect); __syncwarp reconverges warp 0 after the branch.
_TRIGGER_BLOCK = ("if (threadIdx.x == 0) {", _TRIGGER, "}", "__syncwarp();")
_FENCE = 'asm volatile("fence.proxy.async.global;" ::: "memory");'
_TMA_BRANCHES = ("if (warp_idx == kSpecWarpStart) {", "} else if (warp_idx == kSpecWarpStart + 1) {")
_AFTER_TMA_BRANCHES = "} else if (warp_idx == kSpecWarpStart + 2) {"


def check_header(text: str) -> None:
    """Raise unless ``text`` (dg_logits.cuh) has exactly one grid-dependency wait, directly followed by thread 0's fenced
    dependent-launch trigger (``if (threadIdx.x == 0) {`` one asm block: fence.acq_rel.gpu, then
    griddepcontrol.launch_dependents ``}`` then ``__syncwarp();``) and no other trigger, and after it the two TMA producer
    branches (q / weights, index-K / scales) each starting with fence.proxy.async.global, with every TMA copy of the kernel
    inside those two branches."""
    code = [line for line in (raw.strip() for raw in text.splitlines()) if line and not line.startswith("//")]
    problems = []
    if code.count(_FENCE) != 2:
        problems.append(f"{code.count(_FENCE)} async-proxy fences, expected 2")
    unfenced = sum(1 for line in code
                   if ("cudaTriggerProgrammaticLaunchCompletion" in line or "launch_dependents" in line) and line != _TRIGGER)
    if unfenced:
        problems.append(f"{unfenced} dependent-launch triggers without the fence")
    if code.count(_WAIT) != 1 or code.count(_TRIGGER) != 1:
        problems.append(f"{code.count(_WAIT)} grid-dependency waits and {code.count(_TRIGGER)} fenced triggers, expected 1 each")
    else:
        wait = code.index(_WAIT)
        if tuple(code[wait + 1:wait + 1 + len(_TRIGGER_BLOCK)]) != _TRIGGER_BLOCK:
            problems.append("thread 0's fenced trigger block does not directly follow the wait")
        heads = [next((i for i in range(wait, len(code)) if code[i] == opener), None)
                 for opener in (*_TMA_BRANCHES, _AFTER_TMA_BRANCHES)]
        if None in heads or not heads[0] < heads[1] < heads[2]:
            problems.append("the TMA producer branches were not found after the wait")
        else:
            for name, head in zip(("q / weights", "index-K / scales"), heads[:2]):
                if code[head + 1] != _FENCE:
                    problems.append(f"the {name} TMA branch does not start with the fence")
            stray = sum(1 for i, line in enumerate(code) if "tma::copy" in line and not heads[0] < i < heads[2])
            if stray:
                problems.append(f"{stray} TMA copies outside the fenced branches")
    if problems:
        raise RuntimeError(f"{HEADER.name} lost its PDL synchronization: " + "; ".join(problems))


@cache
def build(device_index: int, tokens_per_request: int) -> Kernel:
    """Check the header's synchronization (check_header), then compile (cached on disk) and load the kernel for
    ``tokens_per_request`` query rows per request. Call outside CUDA-graph capture."""
    check_header(HEADER.read_text())
    with torch.cuda.device(device_index):
        cubin = compile_unit(source(tokens_per_request))
        return load(cubin, symbol(tokens_per_request), (SPECIALIZED_THREADS + MATH_THREADS, 1, 1), SHARED_BYTES)


def _map(dtype, base: int, dims, strides_bytes, box, swizzle_128b: bool):
    cu = driver()
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    (tensor_map,) = check(
        cu.cuTensorMapEncodeTiled(
            dtype, len(dims), base, [u64(d) for d in dims], [u64(s) for s in strides_bytes], [u32(b) for b in box],
            [u32(1)] * len(dims), cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B if swizzle_128b else cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled(dg_logits)",
    )
    return tensor_map


_maps: dict = {}


def _tensor_maps(q, cache_bytes, weights):
    """(q, sf_kv, kv, weights) tensor maps as DeepGEMM builds them; cached by address and shape (graph-captured
    launches keep their buffers)."""
    rows = q.shape[0] * q.shape[1]
    blocks = cache_bytes.shape[0]
    page_bytes = cache_bytes.stride(0)
    key = (q.data_ptr(), rows, cache_bytes.data_ptr(), blocks, page_bytes, weights.data_ptr(), weights.stride(0))
    maps = _maps.get(key)
    if maps is None:
        cu = driver()
        u8, f32 = cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8, cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32
        base = cache_bytes.data_ptr()
        tm_q = _map(u8, q.data_ptr(), (HEAD_DIM, rows * HEADS), (HEAD_DIM,), (HEAD_DIM, BLOCK_Q * HEADS), True)
        tm_kv = _map(u8, base, (HEAD_DIM, PAGE, blocks), (HEAD_DIM, page_bytes), (HEAD_DIM, PAGE, 1), True)
        tm_sf_kv = _map(f32, base + PAGE * HEAD_DIM, (PAGE, blocks), (page_bytes,), (PAGE, 1), False)
        tm_w = _map(f32, weights.data_ptr(), (HEADS, rows), (weights.stride(0) * 4,), (HEADS, BLOCK_Q), False)
        maps = _maps[key] = (tm_q, tm_sf_kv, tm_kv, tm_w)
    return maps


def paged_mqa_logits(q, fused_cache, weights, context_lens, block_table, schedule, max_context_len, clean_logits=False):
    """``deep_gemm.fp8_paged_mqa_logits(q, fused_cache, weights, context_lens, block_table, schedule, max_context_len,
    clean_logits=False)`` for q [B, N, 32, 128] FP8 (contiguous), fused_cache [blocks, 64, 1, 132] uint8, weights
    [B * N, 32] FP32, context_lens [B, N] int32, block_table [B, W] int32, schedule [SMs + 1, 2] int32 from
    deep_gemm.get_paged_mqa_logits_metadata. Returns FP32 logits [B * N, max_context_len] (row stride padded as DG's)."""
    batch, next_n, heads, dim = q.shape
    blocks, page, kv_heads, page_row = fused_cache.shape
    if (heads, dim, page, kv_heads, page_row) != (HEADS, HEAD_DIM, PAGE, 1, HEAD_DIM + 4) or not q.is_contiguous():
        raise ValueError(f"dg_logits: unsupported shapes q {tuple(q.shape)} cache {tuple(fused_cache.shape)}")
    if q.dtype != torch.float8_e4m3fn or fused_cache.dtype != torch.uint8 or weights.dtype != torch.float32:
        raise ValueError("dg_logits: FP8 query, uint8 cache, FP32 weights only")
    if fused_cache.stride(1) != page_row or fused_cache.stride(3) != 1 or weights.stride(1) != 1:
        raise ValueError("dg_logits: cache pages and weight rows must be dense")
    if tuple(context_lens.shape) != (batch, next_n) or not context_lens.is_contiguous() or block_table.stride(1) != 1:
        raise ValueError("dg_logits: context_lens [B, N] contiguous, block_table unit column stride")
    if clean_logits:
        raise ValueError("dg_logits: paged logits are never cleaned (as DeepGEMM)")
    rows = batch * next_n
    stride = -(-(-(-max_context_len // SPLIT_KV) * SPLIT_KV) // LOGITS_ALIGN) * LOGITS_ALIGN
    logits = torch.empty((rows, stride), device=q.device, dtype=torch.float32)[:, :max_context_len]
    kernel = build(q.device.index if q.device.index is not None else torch.cuda.current_device(), next_n)
    tm_q, tm_sf_kv, tm_kv, tm_w = _tensor_maps(q, fused_cache.view(blocks, page * page_row), weights)
    args = [
        ctypes.c_uint32(rows), ctypes.c_uint32(stride), ctypes.c_uint32(block_table.stride(0)),
        ctypes.c_void_p(context_lens.data_ptr()), ctypes.c_void_p(logits.data_ptr()),
        ctypes.c_void_p(block_table.data_ptr()), ctypes.c_void_p(0), ctypes.c_void_p(schedule.data_ptr()),
        tm_q, tm_sf_kv, tm_kv, tm_sf_kv, tm_w,
    ]
    launch(kernel, (schedule.shape[0] - 1, 1, 1), args, torch.cuda.current_stream().cuda_stream)
    return logits

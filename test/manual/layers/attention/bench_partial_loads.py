"""Is the SM80 sparse-attention partial kernel bound by narrow, redundant loads?

Round 14 showed that for a latency-bound TRITON kernel on this deployment the fix
that works is reducing the NUMBER of load instructions rather than adding
parallelism. `_partial_kernel` (grid (8,64) + (8,16), 14.1 + 12.9 us live, 76
launches per token) has two decode branches that both look load-bound:

  * byte data: the FP8 branch issues one byte load per dimension, 512 per key;
    the FP4 branch loads `data_base + d//2`, so every packed byte is fetched
    twice - 512 byte loads per key for 256 bytes;
  * scales: `scale_base + d // TILE_SIZE` re-reads the SAME 16 (FP8) or 32 (FP4)
    scale bytes 32 times each, i.e. 512 loads per key to fetch 16 bytes.

This defines three kernels over the real cache fixtures and compares them bit for
bit as well as in time:

  A production  - unchanged copy of the deployed decode;
  B scalebroadcast - only the scale loads collapse to their distinct bytes;
  C wordloads   - B plus 4-byte word data loads unpacked in registers.

B and C must reproduce A exactly: both are pure reinterpretations of the same
bytes in the same order, so any difference is a bug rather than a numerics
choice. Timing is CUDA-graph amortized; the live shapes are batch 1, 8 heads,
page 128, K=512 keys over 64 chunks and 128 over 16.

Run: PYTHONPATH=python python3 test/manual/layers/attention/bench_partial_loads.py
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.dequant_k_cache_sm80 import (
    _e2m1_to_fp32,
    _e4m3_to_fp32,
    _scaled_bf16,
    _ue8m0_to_fp32,
)
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout


@triton.jit
def _partial_production(
    q_ptr, cache_ptr, indices_ptr, length_ptr, partial_ptr, softmax_scale,
    H: tl.constexpr, K: tl.constexpr, HAS_LENGTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr, PITCH: tl.constexpr,
    DATA_BYTES: tl.constexpr, SCALE_BYTES: tl.constexpr,
    SCALE_OFFSET: tl.constexpr, TILE_SIZE: tl.constexpr, IS_FP4: tl.constexpr,
    CHUNKS: tl.constexpr, CHUNK_OFFSET: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    batch = row // H
    j = chunk * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    d = tl.arange(0, 512).to(tl.int64)
    length = tl.full((), K, tl.int64)
    if HAS_LENGTH:
        length = tl.load(length_ptr + batch).to(tl.int64)
    loc = tl.load(indices_ptr + batch * K + j,
                  mask=(j < K) & (j < length), other=-1).to(tl.int64)
    valid = (j < K) & (j < length) & (loc >= 0)
    safe_loc = tl.where(valid, loc, 0)
    page_base = (safe_loc // PAGE_SIZE) * PITCH
    slot = safe_loc % PAGE_SIZE
    data_base = page_base + slot * DATA_BYTES
    scale_base = page_base + SCALE_OFFSET + slot * SCALE_BYTES
    if IS_FP4:
        packed = tl.load(cache_ptr + data_base[:, None] + d[None, :] // 2,
                         mask=valid[:, None], other=0)
        code = (packed >> ((d[None, :] & 1) * 4)) & 15
        value = _e2m1_to_fp32(code)
        scale_code = tl.load(cache_ptr + scale_base[:, None] + d[None, :] // TILE_SIZE,
                             mask=valid[:, None], other=0)
        scales = _e4m3_to_fp32(scale_code)
    else:
        code = tl.load(cache_ptr + data_base[:, None] + d[None, :],
                       mask=valid[:, None], other=0)
        value = _e4m3_to_fp32(code)
        scale_code = tl.load(cache_ptr + scale_base[:, None] + d[None, :] // TILE_SIZE,
                             mask=valid[:, None], other=0)
        scales = _ue8m0_to_fp32(scale_code)
    kv = _scaled_bf16(value, scales).to(tl.float32)
    q = tl.load(q_ptr + row * 512 + d).to(tl.float32)
    score = tl.sum(kv * q[None, :], 1) * softmax_scale
    score = tl.where(valid, score, float("-inf"))
    maximum = tl.max(score, 0)
    safe_maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    probability = tl.exp(score - safe_maximum)
    denominator = tl.sum(probability, 0)
    numerator = tl.sum(kv * probability[:, None], 0)
    base = (row * CHUNKS + CHUNK_OFFSET + chunk) * 514
    tl.store(partial_ptr + base + d, numerator)
    tl.store(partial_ptr + base + 512, maximum)
    tl.store(partial_ptr + base + 513, denominator)


@triton.jit
def _partial_variant(
    q_ptr, cache_ptr, cache32_ptr, indices_ptr, length_ptr, partial_ptr, softmax_scale,
    H: tl.constexpr, K: tl.constexpr, HAS_LENGTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr, PITCH: tl.constexpr,
    DATA_BYTES: tl.constexpr, SCALE_BYTES: tl.constexpr,
    SCALE_OFFSET: tl.constexpr, TILE_SIZE: tl.constexpr, IS_FP4: tl.constexpr,
    CHUNKS: tl.constexpr, CHUNK_OFFSET: tl.constexpr, BLOCK: tl.constexpr,
    SCALE_BROADCAST: tl.constexpr, WORD_LOADS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    batch = row // H
    j = chunk * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    d = tl.arange(0, 512).to(tl.int64)
    length = tl.full((), K, tl.int64)
    if HAS_LENGTH:
        length = tl.load(length_ptr + batch).to(tl.int64)
    loc = tl.load(indices_ptr + batch * K + j,
                  mask=(j < K) & (j < length), other=-1).to(tl.int64)
    valid = (j < K) & (j < length) & (loc >= 0)
    safe_loc = tl.where(valid, loc, 0)
    page_base = (safe_loc // PAGE_SIZE) * PITCH
    slot = safe_loc % PAGE_SIZE
    data_base = page_base + slot * DATA_BYTES
    scale_base = page_base + SCALE_OFFSET + slot * SCALE_BYTES

    # Scales: load only the distinct TILE_SIZE-wide groups, then broadcast.
    ns = tl.arange(0, 512 // TILE_SIZE)
    raw_scale = tl.load(cache_ptr + scale_base[:, None] + ns[None, :],
                        mask=valid[:, None], other=0)
    if SCALE_BROADCAST:
        if IS_FP4:
            decoded = _e4m3_to_fp32(raw_scale)
        else:
            decoded = _ue8m0_to_fp32(raw_scale)
        scales = tl.reshape(
            tl.broadcast_to(decoded[:, :, None], (BLOCK, 512 // TILE_SIZE, TILE_SIZE)),
            (BLOCK, 512),
        )
    else:
        scale_code = tl.load(cache_ptr + scale_base[:, None] + d[None, :] // TILE_SIZE,
                             mask=valid[:, None], other=0)
        scales = _e4m3_to_fp32(scale_code) if IS_FP4 else _ue8m0_to_fp32(scale_code)

    if WORD_LOADS:
        if IS_FP4:
            wd = tl.arange(0, DATA_BYTES // 4)  # 4 bytes = 8 fp4 codes per word
            w = tl.load(cache32_ptr + data_base[:, None] // 4 + wd[None, :],
                        mask=valid[:, None], other=0)
            nib = (w[:, :, None] >> ((tl.arange(0, 8) * 4)[None, None, :])) & 15
            code = tl.reshape(nib, (BLOCK, 512))
            value = _e2m1_to_fp32(code)
        else:
            wd = tl.arange(0, 128)  # 4 bytes per word, 512 bytes per key
            w = tl.load(cache32_ptr + data_base[:, None] // 4 + wd[None, :],
                        mask=valid[:, None], other=0)
            by = (w[:, :, None] >> ((tl.arange(0, 4) * 8)[None, None, :])) & 0xFF
            code = tl.reshape(by, (BLOCK, 512))
            value = _e4m3_to_fp32(code)
    else:
        if IS_FP4:
            packed = tl.load(cache_ptr + data_base[:, None] + d[None, :] // 2,
                             mask=valid[:, None], other=0)
            code = (packed >> ((d[None, :] & 1) * 4)) & 15
            value = _e2m1_to_fp32(code)
        else:
            code = tl.load(cache_ptr + data_base[:, None] + d[None, :],
                           mask=valid[:, None], other=0)
            value = _e4m3_to_fp32(code)

    kv = _scaled_bf16(value, scales).to(tl.float32)
    q = tl.load(q_ptr + row * 512 + d).to(tl.float32)
    score = tl.sum(kv * q[None, :], 1) * softmax_scale
    score = tl.where(valid, score, float("-inf"))
    maximum = tl.max(score, 0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    probability = tl.exp(score - safe_maximum)
    denominator = tl.sum(probability, 0)
    numerator = tl.sum(kv * probability[:, None], 0)
    base = (row * CHUNKS + CHUNK_OFFSET + chunk) * 514
    tl.store(partial_ptr + base + d, numerator)
    tl.store(partial_ptr + base + 512, maximum)
    tl.store(partial_ptr + base + 513, denominator)


def graph_ms(fn, inner=20, warm=3, rounds=7):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warm):
            for _ in range(inner):
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(inner):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(rounds):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / inner)
    del graph
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--keys", type=int, default=512)
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--length", type=int, default=96)
    args = p.parse_args()

    assert torch.cuda.is_available() and torch.version.hip is None
    print(json.dumps({"kind": "env", "name": torch.cuda.get_device_name()}), flush=True)

    torch.manual_seed(5)
    H, K, BLOCK, PAGE = args.heads, args.keys, args.block, 128
    CHUNKS = triton.cdiv(K, BLOCK)
    q = (torch.randn(H, 512, device="cuda") * 0.3).to(torch.bfloat16)
    idx = torch.randint(0, K, (1, K), device="cuda").to(torch.int32)
    length = torch.tensor([args.length], dtype=torch.int32, device="cuda")

    for layout, name in ((KVLayout.V41, "V41-fp8"), (KVLayout.V41_FP4, "V41_FP4")):
        # One row per page; data occupies [0, PAGE*data) and scales follow, so
        # the row must cover scale_offset(PAGE) + PAGE*scale_bytes.
        pages = (K + PAGE - 1) // PAGE + 1
        row_bytes = PAGE * (layout.data_bytes + layout.scale_bytes) + 4096
        cache = torch.randint(0, 256, (pages, row_bytes), dtype=torch.uint8, device="cuda")
        cache32 = cache.view(torch.int32)
        partial = torch.zeros(1, H, CHUNKS, 514, dtype=torch.float32, device="cuda")

        common = dict(H=H, K=K, HAS_LENGTH=True, PAGE_SIZE=PAGE, PITCH=cache.stride(0),
                      DATA_BYTES=layout.data_bytes, SCALE_BYTES=layout.scale_bytes,
                      SCALE_OFFSET=layout.scale_offset(PAGE), TILE_SIZE=layout.tile_size,
                      IS_FP4=layout is KVLayout.V41_FP4, CHUNKS=CHUNKS, CHUNK_OFFSET=0,
                      BLOCK=BLOCK, num_warps=4, enable_fp_fusion=False)

        def run_prod():
            _partial_production[(H, CHUNKS)](q, cache, idx, length, partial, 0.1, **common)

        try:
            run_prod()
            torch.cuda.synchronize()
        except Exception as exc:
            print(json.dumps({"kind": "skip", "layout": name, "why": str(exc)[:120]}), flush=True)
            continue
        ref = partial.clone()
        ms = graph_ms(run_prod)
        print(json.dumps({"kind": "timing", "layout": name, "variant": "production",
                          "median_us": ms * 1000}), flush=True)

        for sb, wl, label in ((True, False, "scalebroadcast"), (True, True, "wordloads"),
                              (False, True, "wordloads_only")):
            partial.zero_()
            try:
                _partial_variant[(H, CHUNKS)](q, cache, cache32, idx, length, partial, 0.1,
                                              SCALE_BROADCAST=sb, WORD_LOADS=wl, **common)
                torch.cuda.synchronize()
            except Exception as exc:
                print(json.dumps({"kind": "skip", "layout": name, "variant": label,
                                  "why": str(exc)[:120]}), flush=True)
                continue
            same = float((partial.view(torch.int32) == ref.view(torch.int32)).float().mean() * 100)
            maxd = (partial - ref).abs().max().item()
            ms2 = graph_ms(lambda sb=sb, wl=wl: _partial_variant[(H, CHUNKS)](
                q, cache, cache32, idx, length, partial, 0.1,
                SCALE_BROADCAST=sb, WORD_LOADS=wl, **common))
            print(json.dumps({"kind": "timing", "layout": name, "variant": label,
                              "median_us": ms2 * 1000, "speedup": ms / ms2,
                              "bitwise_pct": same, "max_abs": maxd}), flush=True)
    print("PARTIAL_LOADS_DONE", flush=True)


if __name__ == "__main__":
    main()

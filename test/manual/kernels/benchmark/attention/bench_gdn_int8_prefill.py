"""Compare explicit BF16/FP32 state staging with fused INT8 GDN prefill."""

import argparse
import json
import math
import statistics

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from sglang.kernels.ops.attention.fla.chunk_fwd import (
    chunk_gated_delta_rule_fwd_intra,
)
from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_o
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

LAYERS, HEADS, VALUE_HEADS, DIM, CHUNK = 48, 8, 24, 128, 64
ARMS = ("bf16_staged", "fp32_staged", "int8_fused")


@triton.jit
def _stage_state(
    state,
    scales,
    staged,
    stride_scale_slot,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
    cols = tl.arange(0, K)
    offsets = rows[:, None] * K + cols[None, :]
    scale = tl.load(scales + rows // ROWS * stride_scale_slot + rows % ROWS)
    values = tl.load(state + offsets).to(tl.float32) * scale[:, None]
    tl.store(staged + offsets, values)


@triton.jit
def _write_state(
    staged,
    state,
    scales,
    stride_scale_slot,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
    cols = tl.arange(0, K)
    offsets = rows[:, None] * K + cols[None, :]
    values = tl.load(staged + offsets).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), axis=1) / 127.0, 1e-12)
    q = tl.extra.cuda.libdevice.rint(values / scale[:, None])
    q = tl.maximum(tl.minimum(q, 127.0), -127.0).to(tl.int8)
    tl.store(state + offsets, q)
    tl.store(scales + rows // ROWS * stride_scale_slot + rows % ROWS, scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        parser.error("samples must be positive and warmup nonnegative")
    torch.manual_seed(53226)
    device = "cuda"

    def randn(shape, dtype=torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device) * 0.1

    max_batch = 16
    state = torch.randint(
        -127,
        128,
        (LAYERS, 2 * max_batch, VALUE_HEADS, DIM, DIM),
        dtype=torch.int8,
        device=device,
    )
    state_ref = state.clone()
    scale = (
        torch.rand((2 * max_batch, LAYERS, VALUE_HEADS, DIM), device=device) * 0.005
        + 0.001
    ).transpose(0, 1)
    scale_ref = scale.clone()
    rows_per_slot = VALUE_HEADS * DIM
    eviction = torch.empty(64 * 1024 * 1024, device=device)
    results = {}

    for batch, suffix in ((1, 344), (8, 512), (16, 512)):
        total = batch * suffix
        shape = (batch, VALUE_HEADS, DIM, DIM)
        q, k = (randn((1, total, HEADS, DIM)) for _ in range(2))
        v = randn((1, total, VALUE_HEADS, DIM))
        g = -randn((1, total, VALUE_HEADS), torch.float32).abs()
        beta = randn((1, total, VALUE_HEADS)).sigmoid()
        rows = torch.arange(batch, device=device, dtype=torch.int32)
        cu = torch.arange(0, total + 1, suffix, device=device, dtype=torch.int32)
        chunks = prepare_chunk_indices(cu, CHUNK)
        offsets = prepare_chunk_offsets(cu, CHUNK)
        hits = torch.ones(batch, dtype=torch.bool, device=device)
        track_chunk = torch.full_like(rows, 2)
        grid = (batch * rows_per_slot // 16,)

        for tracking in (False, True):
            track_slots = rows + batch if tracking else None

            def reset():
                state.copy_(state_ref)
                scale.copy_(scale_ref)
                eviction.zero_()

            def run(arm):
                for layer in range(LAYERS):
                    active, layer_scale = state[layer], scale[layer]
                    norm_q, norm_k = l2norm_fwd(q), l2norm_fwd(k)
                    gate = chunk_local_cumsum(
                        g, chunk_size=CHUNK, cu_seqlens=cu, chunk_indices=chunks
                    )
                    w, u, _ = chunk_gated_delta_rule_fwd_intra(
                        k=norm_k,
                        v=v,
                        g=gate,
                        beta=beta,
                        cu_seqlens=cu,
                        chunk_indices=chunks,
                    )
                    if arm == "int8_fused":
                        h, v_new = chunk_gated_delta_rule_fwd_h(
                            norm_k,
                            w,
                            u,
                            g=gate,
                            initial_state=active,
                            initial_state_indices=rows,
                            initial_state_scale=layer_scale,
                            has_initial_state=hits,
                            cu_seqlens=cu,
                            chunk_indices=chunks,
                            chunk_offsets=offsets,
                            track_state_indices=track_slots,
                            track_chunk_idx=track_chunk if tracking else None,
                        )
                    else:
                        dtype = (
                            torch.bfloat16 if arm == "bf16_staged" else torch.float32
                        )
                        staged = torch.empty(shape, device=device, dtype=dtype)
                        track = (
                            torch.empty(shape, device=device, dtype=torch.float32)
                            if tracking
                            else None
                        )
                        _stage_state[grid](
                            active,
                            layer_scale,
                            staged,
                            layer_scale.stride(0),
                            rows_per_slot,
                            DIM,
                            16,
                        )
                        h, v_new = chunk_gated_delta_rule_fwd_h(
                            norm_k,
                            w,
                            u,
                            g=gate,
                            initial_state=staged,
                            initial_state_indices=rows,
                            cu_seqlens=cu,
                            chunk_indices=chunks,
                            chunk_offsets=offsets,
                            track_state=track,
                            track_chunk_idx=track_chunk if tracking else None,
                        )
                        _write_state[grid](
                            staged,
                            active,
                            layer_scale,
                            layer_scale.stride(0),
                            rows_per_slot,
                            DIM,
                            16,
                        )
                        if tracking:
                            _write_state[grid](
                                track,
                                active[batch:],
                                layer_scale[batch:],
                                layer_scale.stride(0),
                                rows_per_slot,
                                DIM,
                                16,
                            )
                        del staged, track
                    chunk_fwd_o(
                        q=norm_q,
                        k=norm_k,
                        v=v_new,
                        h=h,
                        g=gate,
                        scale=DIM**-0.5,
                        cu_seqlens=cu,
                        chunk_indices=chunks,
                    )
                    del h, v_new

            graphs = {}
            for arm in ARMS:
                reset()
                run(arm)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(arm)
                graphs[arm] = graph
            del graph

            events = {arm: [] for arm in ARMS}
            for i in range(args.samples + args.warmup):
                for arm in ARMS[i % len(ARMS) :] + ARMS[: i % len(ARMS)]:
                    reset()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[arm].replay()
                    end.record()
                    events[arm].append((start, end))
            torch.cuda.synchronize()
            graphs.clear()

            case = f"B{batch}_suffix{suffix}_track{int(tracking)}"
            results[case] = {}
            for arm, pairs in events.items():
                ms = sorted(
                    start.elapsed_time(end) for start, end in pairs[args.warmup :]
                )
                reset()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                run(arm)
                torch.cuda.synchronize()
                results[case][arm] = {
                    "p50_ms": round(statistics.median(ms), 3),
                    "p90_ms": round(ms[math.ceil(0.9 * len(ms)) - 1], 3),
                    "peak_temp_mib": round(
                        (torch.cuda.max_memory_allocated() - base) / 2**20, 3
                    ),
                }

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "samples_per_arm": args.samples,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

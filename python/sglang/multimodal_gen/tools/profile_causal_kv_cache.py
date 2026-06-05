#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Profile LingBot causal self-attention KV cache update costs.

This script is intentionally a standalone profiling aid. It does not change
runtime behavior and should be run on a CUDA devbox.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

_CACHE_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "runtime"
    / "layers"
    / "kvcache"
    / "causal_attention_cache.py"
)
_CACHE_SPEC = importlib.util.spec_from_file_location(
    "profile_causal_attention_cache", _CACHE_MODULE_PATH
)
if _CACHE_SPEC is None or _CACHE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load {_CACHE_MODULE_PATH}")
_CACHE_MODULE = importlib.util.module_from_spec(_CACHE_SPEC)
_CACHE_SPEC.loader.exec_module(_CACHE_MODULE)
CausalSelfAttentionKVCache = _CACHE_MODULE.CausalSelfAttentionKVCache


@dataclass(frozen=True)
class Scenario:
    name: str
    cache_frames: int
    sink_frames: int
    prev_global_frames: int
    prev_local_frames: int
    current_start_frames: int


def parse_dtype(value: str) -> torch.dtype:
    value = value.lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def make_cache(
    *,
    batch_size: int,
    cache_tokens: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    sink_tokens: int,
) -> CausalSelfAttentionKVCache:
    return CausalSelfAttentionKVCache(
        k=torch.empty(
            (batch_size, cache_tokens, heads, head_dim),
            dtype=dtype,
            device=device,
        ),
        v=torch.empty(
            (batch_size, cache_tokens, heads, head_dim),
            dtype=dtype,
            device=device,
        ),
        global_end_index=torch.zeros(1, dtype=torch.long, device=device),
        local_end_index=torch.zeros(1, dtype=torch.long, device=device),
        global_end_index_int=0,
        local_end_index_int=0,
        cache_size=cache_tokens,
        sink_tokens=sink_tokens,
        attention_window_size=cache_tokens,
    )


def reset_cache(cache: CausalSelfAttentionKVCache, *, global_end: int, local_end: int):
    cache.global_end_index_int = global_end
    cache.local_end_index_int = local_end
    cache.global_end_index.fill_(global_end)
    cache.local_end_index.fill_(local_end)


def expected_move_tokens(
    *,
    cache_tokens: int,
    sink_tokens: int,
    prev_local_end: int,
    appended_tokens: int,
) -> tuple[int, int]:
    if prev_local_end + appended_tokens <= cache_tokens:
        return 0, 0
    evicted = prev_local_end + appended_tokens - cache_tokens
    rolled = max(0, prev_local_end - evicted - sink_tokens)
    return evicted, rolled


def time_cache_update(
    *,
    scenario: Scenario,
    caches: list[CausalSelfAttentionKVCache],
    key: torch.Tensor,
    value: torch.Tensor,
    tokens_per_frame: int,
    warmup: int,
    repeats: int,
) -> dict[str, float | int | str]:
    cache_tokens = scenario.cache_frames * tokens_per_frame
    sink_tokens = scenario.sink_frames * tokens_per_frame
    prev_global = scenario.prev_global_frames * tokens_per_frame
    prev_local = scenario.prev_local_frames * tokens_per_frame
    current_start = scenario.current_start_frames * tokens_per_frame
    appended = max(0, current_start + key.shape[1] - prev_global)
    evicted, rolled = expected_move_tokens(
        cache_tokens=cache_tokens,
        sink_tokens=sink_tokens,
        prev_local_end=prev_local,
        appended_tokens=appended,
    )

    elapsed_ms: list[float] = []
    for idx in range(warmup + repeats):
        for cache in caches:
            reset_cache(cache, global_end=prev_global, local_end=prev_local)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for cache in caches:
            cache.update_and_get_attention_kv(
                key=key,
                value=value,
                current_chunk_start=current_start,
                debug_name=scenario.name,
            )
        end.record()
        torch.cuda.synchronize()
        if idx >= warmup:
            elapsed_ms.append(start.elapsed_time(end))

    return {
        "scenario": scenario.name,
        "cache_frames": scenario.cache_frames,
        "sink_frames": scenario.sink_frames,
        "prev_global_frames": scenario.prev_global_frames,
        "current_start_frames": scenario.current_start_frames,
        "appended_tokens": appended,
        "evicted_tokens": evicted,
        "rolled_tokens": rolled,
        "mean_ms": statistics.mean(elapsed_ms),
        "median_ms": statistics.median(elapsed_ms),
        "p95_ms": percentile(elapsed_ms, 95),
        "min_ms": min(elapsed_ms),
        "max_ms": max(elapsed_ms),
    }


def profile_one_cache_update(
    *,
    scenario: Scenario,
    cache: CausalSelfAttentionKVCache,
    key: torch.Tensor,
    value: torch.Tensor,
    tokens_per_frame: int,
    row_limit: int,
) -> str:
    prev_global = scenario.prev_global_frames * tokens_per_frame
    prev_local = scenario.prev_local_frames * tokens_per_frame
    current_start = scenario.current_start_frames * tokens_per_frame
    reset_cache(cache, global_end=prev_global, local_end=prev_local)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        cache.update_and_get_attention_kv(
            key=key,
            value=value,
            current_chunk_start=current_start,
            debug_name=scenario.name,
        )
    torch.cuda.synchronize()
    return prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=row_limit,
    )


def time_sdpa_reference(
    *,
    q_tokens: int,
    kv_tokens: int,
    batch_size: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float | int | str]:
    q = torch.empty((batch_size, heads, q_tokens, head_dim), dtype=dtype, device=device)
    k = torch.empty((batch_size, heads, kv_tokens, head_dim), dtype=dtype, device=device)
    v = torch.empty((batch_size, heads, kv_tokens, head_dim), dtype=dtype, device=device)
    elapsed_ms: list[float] = []
    for idx in range(warmup + repeats):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        end.record()
        torch.cuda.synchronize()
        if idx >= warmup:
            elapsed_ms.append(start.elapsed_time(end))
    return {
        "scenario": f"sdpa_q{q_tokens}_kv{kv_tokens}",
        "q_tokens": q_tokens,
        "kv_tokens": kv_tokens,
        "mean_ms": statistics.mean(elapsed_ms),
        "median_ms": statistics.median(elapsed_ms),
        "p95_ms": percentile(elapsed_ms, 95),
        "min_ms": min(elapsed_ms),
        "max_ms": max(elapsed_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--latent-height", type=int, default=60)
    parser.add_argument("--latent-width", type=int, default=104)
    parser.add_argument("--patch-height", type=int, default=2)
    parser.add_argument("--patch-width", type=int, default=2)
    parser.add_argument("--chunk-frames", type=int, default=9)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--cache-blocks", type=int, default=1)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--profile-row-limit", type=int, default=12)
    parser.add_argument("--skip-sdpa", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiler")

    dtype = parse_dtype(args.dtype)
    device = torch.device(args.device)
    tokens_per_frame = (
        args.latent_height * args.latent_width // (args.patch_height * args.patch_width)
    )
    chunk_tokens = args.chunk_frames * tokens_per_frame

    print(
        json.dumps(
            {
                "device_name": torch.cuda.get_device_name(device),
                "dtype": str(dtype),
                "tokens_per_frame": tokens_per_frame,
                "chunk_tokens": chunk_tokens,
                "shape": {
                    "batch_size": args.batch_size,
                    "heads": args.heads,
                    "head_dim": args.head_dim,
                    "latent_height": args.latent_height,
                    "latent_width": args.latent_width,
                    "chunk_frames": args.chunk_frames,
                    "layers": args.layers,
                    "cache_blocks_timed": args.cache_blocks,
                },
            },
            indent=2,
        )
    )

    key = torch.empty(
        (args.batch_size, chunk_tokens, args.heads, args.head_dim),
        dtype=dtype,
        device=device,
    )
    value = torch.empty_like(key)

    scenarios = [
        Scenario(
            name="webui_chunk0_cold_append",
            cache_frames=18,
            sink_frames=9,
            prev_global_frames=0,
            prev_local_frames=0,
            current_start_frames=0,
        ),
        Scenario(
            name="webui_same_chunk_overwrite",
            cache_frames=18,
            sink_frames=9,
            prev_global_frames=9,
            prev_local_frames=9,
            current_start_frames=0,
        ),
        Scenario(
            name="webui_chunk1_append_no_roll",
            cache_frames=18,
            sink_frames=9,
            prev_global_frames=9,
            prev_local_frames=9,
            current_start_frames=9,
        ),
        Scenario(
            name="webui_chunk2_evict_no_roll",
            cache_frames=18,
            sink_frames=9,
            prev_global_frames=18,
            prev_local_frames=18,
            current_start_frames=18,
        ),
        Scenario(
            name="long_window_sink9_roll_27f",
            cache_frames=45,
            sink_frames=9,
            prev_global_frames=45,
            prev_local_frames=45,
            current_start_frames=45,
        ),
        Scenario(
            name="arch_default_sink3_roll_33f",
            cache_frames=45,
            sink_frames=3,
            prev_global_frames=45,
            prev_local_frames=45,
            current_start_frames=45,
        ),
    ]

    results = []
    for scenario in scenarios:
        torch.cuda.empty_cache()
        gc.collect()
        cache_tokens = scenario.cache_frames * tokens_per_frame
        sink_tokens = scenario.sink_frames * tokens_per_frame
        caches = [
            make_cache(
                batch_size=args.batch_size,
                cache_tokens=cache_tokens,
                heads=args.heads,
                head_dim=args.head_dim,
                dtype=dtype,
                device=device,
                sink_tokens=sink_tokens,
            )
            for _ in range(args.cache_blocks)
        ]
        result = time_cache_update(
            scenario=scenario,
            caches=caches,
            key=key,
            value=value,
            tokens_per_frame=tokens_per_frame,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        result["estimated_40_layer_mean_ms"] = (
            result["mean_ms"] / args.cache_blocks * args.layers
        )
        result["estimated_40_layer_p95_ms"] = (
            result["p95_ms"] / args.cache_blocks * args.layers
        )
        results.append(result)
        print(json.dumps(result, indent=2))

        if scenario.name in {
            "webui_chunk2_evict_no_roll",
            "arch_default_sink3_roll_33f",
        }:
            print(f"\nProfiler top ops for {scenario.name}:")
            print(
                profile_one_cache_update(
                    scenario=scenario,
                    cache=caches[0],
                    key=key,
                    value=value,
                    tokens_per_frame=tokens_per_frame,
                    row_limit=args.profile_row_limit,
                )
            )
        del caches

    steps = 4
    overwrite = next(r for r in results if r["scenario"] == "webui_same_chunk_overwrite")
    evict = next(r for r in results if r["scenario"] == "webui_chunk2_evict_no_roll")
    per_chunk_ms = (
        float(evict["mean_ms"]) + steps * float(overwrite["mean_ms"])
    ) / args.cache_blocks * args.layers
    print(
        json.dumps(
            {
                "webui_estimated_kv_update_ms_per_steady_chunk_40_layers": per_chunk_ms,
                "assumption": "4 denoise forwards + 1 final context-cache overwrite per chunk",
            },
            indent=2,
        )
    )

    if not args.skip_sdpa:
        for kv_frames in (9, 18):
            result = time_sdpa_reference(
                q_tokens=chunk_tokens,
                kv_tokens=kv_frames * tokens_per_frame,
                batch_size=args.batch_size,
                heads=args.heads,
                head_dim=args.head_dim,
                dtype=dtype,
                device=device,
                warmup=max(3, args.warmup // 2),
                repeats=max(8, args.repeats // 2),
            )
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

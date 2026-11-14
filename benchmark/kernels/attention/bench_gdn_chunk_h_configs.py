"""Offline configuration sweep for the stateful Triton GDN chunk-H kernel.

Every invocation restores a small, non-identity-mapped state pool. CUPTI
records only the chunk-H kernel inside CUDA graph replay, excluding restoration,
allocation and compilation. No tuning runs against a serving engine's state.

Run from an installed SGLang checkout:
    .venv/bin/python benchmark/kernels/attention/bench_gdn_chunk_h_configs.py \
        --output /tmp/gdn-chunk-h.json
"""

import argparse
import gc
import hashlib
import itertools
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.attention.fla import chunk_delta_h
from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)

BASELINE = (32, 4, 2)
CHUNK = 64
# Equal token counts with different request counts, plus both bucket boundaries.
CASES = {
    "short": [512],
    "bucket0_end": [2048],
    "bucket1_start": [2112],
    "long_single": [8192],
    "long_batched": [1024] * 8,
    "long_pair": [4096] * 2,
    "long_quartet": [2048] * 4,
    "long_parallel": [512] * 16,
    "short_pair": [256] * 2,
    "short_quartet": [128] * 4,
    "long_ragged": [127, 513, 1025, 6527],
    "bucket2_start": [8256],
    "longer_single": [16384],
}


def make_inputs(lengths, heads, key_heads, state_dtype, track):
    """Small state pool with unused guard slots and shuffled active slots."""
    n, tokens = len(lengths), sum(lengths)
    counts = [triton.cdiv(x, CHUNK) for x in lengths]
    nt = sum(counts)
    shape = (1, tokens, heads, 128)
    k = torch.nn.functional.normalize(
        torch.randn((1, tokens, key_heads, 128), device="cuda", dtype=torch.float32),
        dim=-1,
    ).to(torch.bfloat16)
    v = (torch.randn(shape, device="cuda") * 0.1).to(torch.bfloat16)
    w = (torch.randn(shape, device="cuda") * 0.05).to(torch.bfloat16)
    # Chunk-local cumulative gates, reset at each request and chunk boundary.
    gate = torch.empty((1, tokens, heads), device="cuda", dtype=torch.float32)
    offset = 0
    for length in lengths:
        for start in range(0, length, CHUNK):
            size = min(CHUNK, length - start)
            raw = -torch.rand((size, heads), device="cuda") * 0.1
            gate[0, offset + start : offset + start + size] = raw.cumsum(0)
        offset += length
    initial = torch.randn((n + 2, heads, 128, 128), device="cuda") * 0.1
    initial = initial.to(state_dtype)
    pool = initial.clone()
    slots = torch.arange(n, 0, -1, dtype=torch.int32, device="cuda")
    cu = torch.tensor(
        [0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32
    )
    chunk_offsets = torch.tensor(
        [0, *itertools.accumulate(counts)], device="cuda", dtype=torch.int32
    )
    tracked = torch.empty((n, heads, 128, 128), device="cuda") if track else None
    track_idx = torch.tensor([c - 1 for c in counts], device="cuda", dtype=torch.int32)
    return dict(
        k=k,
        v=v,
        w=w,
        v_new=torch.empty_like(v),
        g=gate,
        gk=None,
        h=torch.empty((1, nt, heads, 128, 128), device="cuda", dtype=torch.bfloat16),
        initial_state=pool,
        initial_state_indices=slots,
        stride_init_state=pool.stride(0),
        cu_seqlens=cu,
        chunk_offsets=chunk_offsets,
        track_state=tracked,
        track_chunk_idx=track_idx if track else None,
        stride_track_state=tracked.stride(0) if track else 0,
        T=tokens,
        H=heads,
        Hg=key_heads,
        K=128,
        V=128,
        BT=CHUNK,
        USE_G=True,
        USE_GK=False,
        USE_INITIAL_STATE=True,
        INPLACE_UPDATE=True,
        SAVE_NEW_VALUE=True,
        IS_VARLEN=True,
        NT_BUCKET=0 if nt <= 32 else (1 if nt <= 128 else 2),
        USE_EXP2=False,
        TRACK_STATE=track,
    ), initial


def run_kernel(inputs, config):
    bv, warps, stages = config
    # Bypass Autotuner: explicit candidates never benchmark live state in-place.
    kernel = chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    if not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    launch_args = dict(inputs)
    if "NT_BUCKET" not in kernel.arg_names:
        launch_args.pop("NT_BUCKET")
    num_seqs = inputs["cu_seqlens"].numel() - 1
    return kernel[(triton.cdiv(inputs["V"], bv), num_seqs * inputs["H"])](
        **launch_args, BV=bv, num_warps=warps, num_stages=stages
    )


def snapshot(inputs):
    return {
        key: inputs[key].clone()
        for key in ("h", "v_new", "initial_state", "track_state")
        if inputs[key] is not None
    }


def check_candidate(inputs, initial, reference, config):
    inputs["initial_state"].copy_(initial)
    compiled = run_kernel(inputs, config)
    errors = {}
    for key, expected in reference.items():
        actual = inputs[key]
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3)
        errors[key] = float((actual.float() - expected.float()).abs().max())
    # Repeat with exactly the same initial state to exercise restoration itself.
    inputs["initial_state"].copy_(initial)
    run_kernel(inputs, config)
    for key, expected in reference.items():
        torch.testing.assert_close(inputs[key], expected, rtol=1e-2, atol=2e-3)
    for guard in (0, initial.shape[0] - 1):
        torch.testing.assert_close(
            inputs["initial_state"][guard], initial[guard], rtol=0, atol=0
        )
    return errors, compiled


def time_kernel(inputs, initial, config, samples, flush):
    """CUPTI kernel activities exclude the graph's preceding state-copy node."""
    run_kernel(inputs, config)  # All JIT compilation precedes capture/profiling.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        inputs["initial_state"].copy_(initial)
        run_kernel(inputs, config)
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(samples):
            if flush is not None:
                flush.zero_()
            graph.replay()
        torch.cuda.synchronize()
    durations = [
        event.device_time_total
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and event.name.startswith("chunk_gated_delta_rule_fwd_kernel_h_blockdim64")
    ]
    if len(durations) != samples:
        raise RuntimeError(
            f"Expected {samples} chunk-H CUPTI activities, got {len(durations)}"
        )
    # A replay mutates only the restored active slots; guard slots stay intact.
    for guard in (0, initial.shape[0] - 1):
        torch.testing.assert_close(
            inputs["initial_state"][guard], initial[guard], rtol=0, atol=0
        )
    return durations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--key-heads", type=int, default=16)
    parser.add_argument("--state-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--bvs", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--warps", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--stages", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--warm-l2", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip:
        parser.error("This benchmark requires an NVIDIA CUDA GPU")
    if min(args.heads, args.rounds, args.samples) <= 0:
        parser.error("heads, rounds and samples must be positive")
    if args.key_heads <= 0 or args.heads % args.key_heads:
        parser.error("heads must be divisible by positive key-heads")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    configs = sorted(
        set([BASELINE, *itertools.product(args.bvs, args.warps, args.stages)])
    )
    dtype = torch.float32 if args.state_dtype == "fp32" else torch.bfloat16
    device = torch.cuda.get_device_properties(0)
    # Flush before graph replay. State restoration inside the graph necessarily
    # makes that small pool hot; k/v/w and output buffers begin cold.
    flush = (
        None
        if args.warm_l2
        else torch.empty(
            max(64 << 20, 2 * device.L2_cache_size), device="cuda", dtype=torch.int8
        )
    )
    report = dict(
        gpu=device.name,
        torch=torch.__version__,
        triton=triton.__version__,
        cuda=torch.version.cuda,
        state_dtype=args.state_dtype,
        heads=args.heads,
        key_heads=args.key_heads,
        track=args.track,
        timer="CUPTI kernel activities in CUDA graph replay",
        cold_l2=not args.warm_l2,
        baseline=BASELINE,
        seed=args.seed,
        rounds=args.rounds,
        samples=args.samples,
        cases=[],
        source_commit=args.source_commit
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        kernel_source_sha256=hashlib.sha256(
            Path(chunk_delta_h.__file__).read_bytes()
        ).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.cases:
        lengths = CASES[name]
        inputs, initial = make_inputs(
            lengths, args.heads, args.key_heads, dtype, args.track
        )
        run_kernel(inputs, BASELINE)
        reference = snapshot(inputs)
        rows = []
        valid = []
        for config in configs:
            started = time.perf_counter()
            try:
                errors, compiled = check_candidate(inputs, initial, reference, config)
            except triton.OutOfResources as exc:
                rows.append(dict(config=config, status="unsupported", reason=str(exc)))
                continue
            rows.append(
                dict(
                    config=config,
                    status="passed",
                    max_abs_error=errors,
                    warmup_check_seconds=time.perf_counter() - started,
                    registers=compiled.n_regs,
                    shared_bytes=compiled.metadata.shared,
                    round_medians_us=[],
                )
            )
            valid.append((config, rows[-1]))
        for _ in range(0 if args.check_only else args.rounds):
            rng.shuffle(valid)
            for config, row in valid:
                durations = time_kernel(inputs, initial, config, args.samples, flush)
                row["round_medians_us"].append(statistics.median(durations))
        case = dict(
            name=name,
            lengths=lengths,
            total_chunks=sum(triton.cdiv(x, CHUNK) for x in lengths),
            results=rows,
        )
        if not args.check_only:
            for _, row in valid:
                row["median_us"] = statistics.median(row["round_medians_us"])
            baseline = next(row for config, row in valid if config == BASELINE)
            best = min((row for _, row in valid), key=lambda row: row["median_us"])
            case.update(
                baseline_us=baseline["median_us"],
                best_config=best["config"],
                best_us=best["median_us"],
                kernel_speedup=baseline["median_us"] / best["median_us"],
            )
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in case.items() if k != "results"}), flush=True)
        del inputs, initial, reference
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

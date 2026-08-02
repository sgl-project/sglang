"""Kimi-K3 CuTeDSL block-verify self-check and microbenchmark.

Examples:

    uv run python benchmark/bench_linear_attention/bench_kda_mtp_verify.py \
        --mode check
    uv run python benchmark/bench_linear_attention/bench_kda_mtp_verify.py \
        --mode bench --provider cutedsl_split
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from dataclasses import dataclass, fields
from typing import Callable

import torch

from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
    commit_kda_replayssm_spec,
)
from sglang.kernels.ops.kimi_k3.kda_decode_mtp import (
    fused_kda_decode_mtp_dspark as fused_kda_decode_mtp_serial,
)
from sglang.kernels.ops.kimi_k3.kda_decode_mtp_split import (
    fused_kda_decode_mtp_dspark as fused_kda_decode_mtp_split,
)

K = 128
W = 4
LOWER_BOUND = -5.0


@dataclass
class Inputs:
    mixed_qkv: torch.Tensor
    conv_weights: torch.Tensor
    conv_states: torch.Tensor
    gate: torch.Tensor
    beta: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    recurrent_state: torch.Tensor
    cache_indices: torch.Tensor
    query_start_loc: torch.Tensor
    intermediate_state_indices: torch.Tensor
    intermediate_conv: torch.Tensor
    rawv: torch.Tensor
    rawk: torch.Tensor
    ring_g: torch.Tensor
    ring_beta: torch.Tensor
    onorm_gate: torch.Tensor
    onorm_weight: torch.Tensor
    onorm_eps: float
    batch_size: int
    heads: int
    width: int


@dataclass
class CapturedProvider:
    graph: torch.cuda.CUDAGraph
    inputs: list[Inputs]
    outputs: list[torch.Tensor]


def make_inputs(
    *,
    batch_size: int,
    heads: int,
    width: int,
    ring_len: int,
    seed: int,
) -> Inputs:
    if ring_len < width:
        raise ValueError(f"ring_len={ring_len} must be >= width={width}")

    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokens = batch_size * width
    channels = heads * K
    slots = batch_size + 2

    # Match KDA backend layouts: q, k, and v are contiguous channel segments.
    mixed_qkv = torch.randn(batch_size, width, 3 * channels, device=device, dtype=dtype)
    conv_weights = (
        torch.randn(3 * channels, W, device=device, dtype=torch.float32) * 0.1
    )
    conv_states = torch.randn(slots, 3 * channels, W - 1, device=device, dtype=dtype)
    gate = torch.randn(1, tokens, heads, K, device=device, dtype=dtype)
    beta = torch.randn(1, tokens, heads, device=device, dtype=dtype)
    A_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads * K, device=device, dtype=torch.float32)
    recurrent_state = torch.randn(
        slots, heads, K, K, device=device, dtype=torch.float32
    )
    cache_indices = torch.arange(1, batch_size + 1, device=device, dtype=torch.int32)
    query_start_loc = torch.arange(
        0, tokens + 1, width, device=device, dtype=torch.int32
    )
    intermediate_state_indices = torch.arange(
        batch_size, device=device, dtype=torch.int32
    )
    intermediate_conv = torch.zeros(
        batch_size,
        width,
        3 * channels,
        W - 1,
        device=device,
        dtype=dtype,
    )
    rawv = torch.zeros(slots, heads, ring_len, K, device=device, dtype=dtype)
    rawk = torch.zeros_like(rawv)
    ring_g = torch.zeros(slots, heads, ring_len, K, device=device, dtype=torch.float32)
    ring_beta = torch.zeros(slots, heads, ring_len, device=device, dtype=torch.float32)
    onorm_gate = torch.randn(1, tokens, heads, K, device=device, dtype=dtype)
    onorm_weight = torch.randn(K, device=device, dtype=torch.float32)

    return Inputs(
        mixed_qkv=mixed_qkv,
        conv_weights=conv_weights,
        conv_states=conv_states,
        gate=gate,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        intermediate_state_indices=intermediate_state_indices,
        intermediate_conv=intermediate_conv,
        rawv=rawv,
        rawk=rawk,
        ring_g=ring_g,
        ring_beta=ring_beta,
        onorm_gate=onorm_gate,
        onorm_weight=onorm_weight,
        onorm_eps=1e-6,
        batch_size=batch_size,
        heads=heads,
        width=width,
    )


def clone_inputs(inp: Inputs) -> Inputs:
    values = {}
    for field in fields(inp):
        value = getattr(inp, field.name)
        values[field.name] = value.clone() if isinstance(value, torch.Tensor) else value
    return Inputs(**values)


def _run_cutedsl(
    inp: Inputs,
    *,
    split_v: bool,
    cache_ring: bool = True,
    intermediate_ssm: torch.Tensor | None = None,
) -> torch.Tensor:
    if not cache_ring and intermediate_ssm is None:
        raise ValueError("non-ReplaySSM CuTe runs require intermediate_ssm")
    channels = inp.heads * K
    tokens = inp.batch_size * inp.width
    qkv = inp.mixed_qkv.reshape(tokens, 3 * channels)
    x_q, x_k, x_v = qkv.split(channels, dim=-1)
    x_q = x_q.reshape(1, tokens, inp.heads, K)
    x_k = x_k.reshape(1, tokens, inp.heads, K)
    x_v = x_v.reshape(1, tokens, inp.heads, K)
    w_q, w_k, w_v = inp.conv_weights.split(channels, dim=0)
    cs_q, cs_k, cs_v = inp.conv_states.split(channels, dim=1)
    ic_q, ic_k, ic_v = inp.intermediate_conv.split(channels, dim=2)

    kernel = fused_kda_decode_mtp_split if split_v else fused_kda_decode_mtp_serial
    return kernel(
        x_q=x_q,
        x_k=x_k,
        x_v=x_v,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        cs_q=cs_q,
        cs_k=cs_k,
        cs_v=cs_v,
        g=inp.gate,
        beta=inp.beta,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        recurrent_state=inp.recurrent_state,
        intermediate_ssm=intermediate_ssm,
        intermediate_state_indices=inp.intermediate_state_indices,
        intermediate_conv_q=ic_q,
        intermediate_conv_k=ic_k,
        intermediate_conv_v=ic_v,
        ssm_state_indices=inp.cache_indices,
        cu_seqlens=inp.query_start_loc,
        lower_bound=LOWER_BOUND,
        scale=K**-0.5,
        replayssm_rawv=inp.rawv if cache_ring else None,
        replayssm_rawk=inp.rawk if cache_ring else None,
        replayssm_g=inp.ring_g if cache_ring else None,
        replayssm_beta=inp.ring_beta if cache_ring else None,
        onorm_gate=inp.onorm_gate,
        onorm_weight=inp.onorm_weight,
        onorm_eps=inp.onorm_eps,
        **({"split_v": True} if split_v else {}),
    )


def run_cutedsl(inp: Inputs) -> torch.Tensor:
    """Run the production CuTeDSL dispatch policy."""
    return _run_cutedsl(
        inp,
        split_v=inp.batch_size == 1 and inp.width == 16,
    )


def run_cutedsl_serial(inp: Inputs) -> torch.Tensor:
    return _run_cutedsl(inp, split_v=False)


def run_cutedsl_split(inp: Inputs) -> torch.Tensor:
    return _run_cutedsl(inp, split_v=True)


PROVIDERS: dict[str, Callable[[Inputs], torch.Tensor]] = {
    "cutedsl": run_cutedsl,
    "cutedsl_serial": run_cutedsl_serial,
    "cutedsl_split": run_cutedsl_split,
}


def _relative_max(actual: torch.Tensor, expected: torch.Tensor) -> float:
    absolute = (actual.float() - expected.float()).abs().max()
    scale = expected.float().abs().max().clamp_min(1e-6)
    return (absolute / scale).item()


def _committed_state(initial: torch.Tensor, inp: Inputs, accepted: int) -> torch.Tensor:
    state = initial.clone()
    accept_lens = torch.full(
        (inp.batch_size,), accepted, device=inp.rawv.device, dtype=torch.int32
    )
    commit_kda_replayssm_spec(
        state,
        inp.rawv,
        inp.rawk,
        inp.ring_g,
        inp.ring_beta,
        inp.cache_indices,
        accept_lens,
        max_cache_len=inp.rawv.shape[-2],
        num_k_heads=inp.heads,
        use_qk_l2norm_in_kernel=True,
        null_block_id=-1,
    )
    return state


def check_invariants(args: argparse.Namespace) -> None:
    if args.batch_size != 1 or args.width != 16:
        raise ValueError("split-V invariant checks require --batch-size 1 --width 16")

    max_replay_snapshot_rel = 0.0
    for seed in args.check_seeds:
        original = make_inputs(
            batch_size=args.batch_size,
            heads=args.heads,
            width=args.width,
            ring_len=args.ring_len,
            seed=seed,
        )
        cute_inp = clone_inputs(original)
        cute_out = run_cutedsl(cute_inp)
        torch.cuda.synchronize()

        # ReplaySSM commit must reconstruct the same accepted state that
        # the non-ring split-V path snapshots after each token. This is an
        # internal invariant, not a comparison against another backend.
        snapshot_inp = clone_inputs(original)
        snapshots = torch.empty(
            args.batch_size,
            args.width,
            args.heads,
            K,
            K,
            device="cuda",
            dtype=torch.float32,
        )
        snapshot_out = _run_cutedsl(
            snapshot_inp,
            split_v=True,
            cache_ring=False,
            intermediate_ssm=snapshots,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(snapshot_out, cute_out, rtol=0, atol=0)
        slot = int(cute_inp.cache_indices[0].item())
        scratch_row = int(cute_inp.intermediate_state_indices[0].item())
        seed_replay_snapshot_rel = 0.0
        for accepted in range(1, args.width + 1):
            committed = _committed_state(original.recurrent_state, cute_inp, accepted)
            snapshot_rel = _relative_max(
                committed[slot],
                snapshots[scratch_row, accepted - 1],
            )
            seed_replay_snapshot_rel = max(seed_replay_snapshot_rel, snapshot_rel)
            max_replay_snapshot_rel = max(max_replay_snapshot_rel, snapshot_rel)
            torch.testing.assert_close(
                committed[slot],
                snapshots[scratch_row, accepted - 1],
                rtol=1e-5,
                atol=1e-5,
            )

        eager_out = run_cutedsl(cute_inp)
        torch.cuda.synchronize()
        eager_out = eager_out.clone()
        captured = capture_provider("cutedsl", [cute_inp])
        captured.graph.replay()
        torch.cuda.synchronize()
        graph_out = captured.outputs[0].clone()
        for _ in range(100):
            captured.graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured.outputs[0], graph_out, rtol=0, atol=0)
        torch.testing.assert_close(graph_out, eager_out, rtol=0, atol=0)
        print(
            f"seed={seed} replay_snapshot_rel={seed_replay_snapshot_rel:.6g} "
            f"checksum={cute_out.float().sum().item():.6g}"
        )

    print(
        f"checked_seeds={len(args.check_seeds)} "
        f"max_replay_snapshot_rel={max_replay_snapshot_rel:.6g}"
    )

    # Exercise the split kernel's CUDA-graph padding exit. The request uses
    # the null slot; its old physical slot and rollback scratch must remain
    # untouched across repeated graph replays.
    padded = make_inputs(
        batch_size=1,
        heads=args.heads,
        width=args.width,
        ring_len=args.ring_len,
        seed=args.seed,
    )
    padded.cache_indices[0] = -1
    pad_slot = padded.batch_size
    padded.rawv[pad_slot].normal_()
    padded.rawk[pad_slot].normal_()
    padded.ring_g[pad_slot].normal_()
    padded.ring_beta[pad_slot].normal_()
    padded.intermediate_conv[0].normal_()
    sentinels = {
        name: getattr(padded, name).clone()
        for name in ("rawv", "rawk", "ring_g", "ring_beta", "intermediate_conv")
    }
    captured = capture_provider("cutedsl_split", [padded])
    for _ in range(100):
        captured.graph.replay()
    torch.cuda.synchronize()
    for name in ("rawv", "rawk", "ring_g", "ring_beta"):
        if not torch.equal(
            getattr(padded, name)[pad_slot],
            sentinels[name][pad_slot],
        ):
            raise AssertionError(f"padding request modified {name} slot {pad_slot}")
    if not torch.equal(padded.intermediate_conv, sentinels["intermediate_conv"]):
        raise AssertionError("padding request modified rollback convolution scratch")
    pad_out = captured.outputs[0]
    if torch.count_nonzero(pad_out).item() != 0:
        raise AssertionError("padding request output was not zeroed")


def sanitize_once(args: argparse.Namespace) -> None:
    """Run only eager split kernels so compute-sanitizer need not trace graphs."""
    if args.batch_size != 1 or args.width != 16:
        raise ValueError("split-V sanitizer requires --batch-size 1 --width 16")

    valid = make_inputs(
        batch_size=1,
        heads=args.heads,
        width=16,
        ring_len=args.ring_len,
        seed=args.seed,
    )
    valid_out = run_cutedsl_split(valid)
    torch.cuda.synchronize()

    padded = make_inputs(
        batch_size=1,
        heads=args.heads,
        width=16,
        ring_len=args.ring_len,
        seed=args.seed + 1,
    )
    padded.cache_indices[0] = -1
    padded_out = run_cutedsl_split(padded)
    torch.cuda.synchronize()
    if torch.count_nonzero(padded_out).item() != 0:
        raise AssertionError("padding request output was not zeroed")
    print(
        f"sanitized split-V eager launches; checksum={valid_out.float().sum().item():.6g}"
    )


def capture_provider(provider: str, inputs: list[Inputs]) -> CapturedProvider:
    fn = PROVIDERS[provider]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            outputs = [fn(inp) for inp in inputs]
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [fn(inp) for inp in inputs]
    graph.replay()
    stream.synchronize()
    return CapturedProvider(graph=graph, inputs=inputs, outputs=outputs)


def _l2_flush_buffer() -> torch.Tensor:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return torch.empty(5 * props.L2_cache_size, device="cuda", dtype=torch.uint8)


def measure_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    samples: int,
    flush: torch.Tensor | None,
) -> list[float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_us = []
    for _ in range(samples):
        if flush is not None:
            flush.zero_()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)
    return times_us


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(int(q * len(ordered)), len(ordered) - 1)
    return ordered[index]


def _t_critical_95(df: int) -> float:
    # Two-sided Student-t 95% critical values. Linear interpolation is
    # unnecessary for a confidence gate; use the next more conservative row.
    table = (
        (1, 12.706),
        (2, 4.303),
        (3, 3.182),
        (4, 2.776),
        (5, 2.571),
        (6, 2.447),
        (7, 2.365),
        (8, 2.306),
        (9, 2.262),
        (10, 2.228),
        (12, 2.179),
        (15, 2.131),
        (20, 2.086),
        (25, 2.060),
        (30, 2.042),
        (40, 2.021),
        (60, 2.000),
        (120, 1.980),
    )
    for threshold, critical in table:
        if df <= threshold:
            return critical
    return 1.960


def warm_graphs(
    captured: dict[str, CapturedProvider],
    *,
    seconds: float,
) -> None:
    if seconds <= 0:
        return
    deadline = time.perf_counter() + seconds
    provider_names = list(captured)
    iteration = 0
    while time.perf_counter() < deadline:
        captured[provider_names[iteration % len(provider_names)]].graph.replay()
        torch.cuda.synchronize()
        iteration += 1


def _make_layer_inputs(args: argparse.Namespace) -> list[Inputs]:
    return [
        make_inputs(
            batch_size=args.batch_size,
            heads=args.heads,
            width=args.width,
            ring_len=args.ring_len,
            seed=args.seed + layer,
        )
        for layer in range(args.layers)
    ]


def benchmark(args: argparse.Namespace) -> None:
    provider_names = args.provider
    captured = {}
    for provider in provider_names:
        captured[provider] = capture_provider(provider, _make_layer_inputs(args))

    # Keep the device at its normal sustained workload clocks before sampling.
    heater = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(20):
        torch.mm(heater, heater)
    torch.cuda.synchronize()
    warm_graphs(captured, seconds=args.warmup_seconds)

    flush = _l2_flush_buffer() if args.cache == "cold" else None
    rng = random.Random(args.seed)
    trial_medians = {provider: [] for provider in provider_names}
    all_samples = {provider: [] for provider in provider_names}

    for trial in range(args.trials):
        order = provider_names.copy()
        rng.shuffle(order)
        for provider in order:
            times = measure_graph(
                captured[provider].graph, samples=args.samples, flush=flush
            )
            times = [value / args.layers for value in times]
            all_samples[provider].extend(times)
            trial_medians[provider].append(statistics.median(times))
        medians = " ".join(
            f"{provider}={trial_medians[provider][-1]:.3f}us"
            for provider in provider_names
        )
        print(f"trial={trial + 1}/{args.trials} {medians}")

    print(
        f"shape=batch:{args.batch_size},heads:{args.heads},width:{args.width} "
        f"layers/cycle={args.layers} cache={args.cache} trials={args.trials} "
        f"samples/trial={args.samples}"
    )
    for provider in provider_names:
        medians = trial_medians[provider]
        samples = all_samples[provider]
        center = statistics.median(medians)
        spread = statistics.stdev(medians) if len(medians) > 1 else 0.0
        sem95 = 1.96 * spread / math.sqrt(len(medians)) if len(medians) > 1 else 0.0
        print(
            f"{provider}: median_of_trial_medians={center:.3f}us "
            f"trial_min={min(medians):.3f}us trial_max={max(medians):.3f}us "
            f"trial_stdev={spread:.3f}us approx_95ci=±{sem95:.3f}us "
            f"sample_p05={_percentile(samples, 0.05):.3f}us "
            f"sample_p95={_percentile(samples, 0.95):.3f}us"
        )
    if len(provider_names) == 2:
        candidate, baseline = provider_names
        log_ratios = [
            math.log(baseline_time / candidate_time)
            for candidate_time, baseline_time in zip(
                trial_medians[candidate],
                trial_medians[baseline],
            )
        ]
        log_center = statistics.fmean(log_ratios)
        log_stdev = statistics.stdev(log_ratios) if len(log_ratios) > 1 else 0.0
        half_width = (
            _t_critical_95(len(log_ratios) - 1) * log_stdev / math.sqrt(len(log_ratios))
            if len(log_ratios) > 1
            else 0.0
        )
        geometric_speedup = math.exp(log_center)
        ci_low, ci_high = (
            math.exp(log_center - half_width),
            math.exp(log_center + half_width),
        )
        print(
            f"paired_{candidate}_vs_{baseline}={geometric_speedup:.4f}x "
            f"95ci=[{ci_low:.4f}x,{ci_high:.4f}x] "
            f"(>1 means {candidate} is faster)"
        )


def profile(args: argparse.Namespace) -> None:
    if len(args.provider) != 1:
        raise ValueError("--mode profile requires exactly one --provider")
    provider = args.provider[0]
    captured = capture_provider(provider, _make_layer_inputs(args))
    warm_graphs({provider: captured}, seconds=args.warmup_seconds)
    torch.cuda.synchronize()
    marker = f"kda_mtp_profile_{provider}"
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(marker)
    for _ in range(args.profile_iterations):
        captured.graph.replay()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    # Keep output live through the profile range.
    checksum = sum(out.float().sum().item() for out in captured.outputs)
    print(f"profiled={marker} checksum={checksum:.6g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("check", "sanitize", "bench", "profile"),
        default="bench",
    )
    parser.add_argument(
        "--provider",
        choices=tuple(PROVIDERS),
        nargs="+",
        default=["cutedsl"],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--ring-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-seeds", type=int, nargs="+", default=[0, 17, 42])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--layers", type=int, default=69)
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--cache", choices=("warm", "cold"), default="warm")
    parser.add_argument("--profile-iterations", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("KDA CuTeDSL verify benchmark requires an SM10x GPU")
    if args.mode == "check":
        check_invariants(args)
    elif args.mode == "sanitize":
        sanitize_once(args)
    elif args.mode == "bench":
        benchmark(args)
    else:
        profile(args)


if __name__ == "__main__":
    main()

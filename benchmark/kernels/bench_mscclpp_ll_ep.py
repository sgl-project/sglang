#!/usr/bin/env python3
"""Benchmark MSCCL++ LL expert parallelism against SGLang baselines.

Rank-major compares production FusedMoE layers configured as:
  * MSCCL++ LL rank-major + FlashInfer CUTLASS
  * FlashInfer rank-major + FlashInfer CUTLASS

Both rank-major paths are captured and timed through CUDA Graphs.

Expert-major compares:
  * MSCCL++ LL expert-major + Triton
  * A PyTorch variable-split all-to-all dispatcher + Triton FusedMoE

The expert-major baseline is primarily a correctness reference. Its PyTorch
collective orchestration is not equivalent to MSCCL++'s fused all-to-all, so
its latency is reported separately and is not an apples-to-apples comparison.

Pass ``--balanced-routing`` to distribute aggregate top-k routes exactly evenly
across every global expert. This is useful for separating communication and
synchronization costs from router-induced expert load imbalance.

Timing: ``total`` is the median over ``--total-repeats`` CUDA Graph timing
windows of the per-iteration window average, maximized across ranks. Phase
times come from ``--phase-repeats`` passes in which every rank profiles its
CUDA kernels. Each rank's kernel timeline is split into pipeline iterations,
and each kernel counts only the time not overlapped by an earlier-starting
kernel. Per iteration, dispatch and combine take the minimum across ranks (the
rank that waits least for its peers), compute takes the maximum (the slowest
rank) and unclassified takes the median; reported values are medians over
iterations, then over passes.

Teardown closes MSCCL++ dispatchers after releasing captured graphs, then clears
shared communicators before destroying process groups.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable
from contextlib import ExitStack, contextmanager

import msgspec
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.token_dispatcher.mscclpp import MSCCLPPDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
    MSCCLPPEPLayout,
    MSCCLPPMode,
)
from sglang.srt.runtime_context import get_flags, get_parallel
from sglang.test.test_utils import publish_build_topology

DTYPE = torch.bfloat16
MSCCLPP_LL_HIDDEN_SIZES = (1024, 2048, 4096, 4352, 6656, 7168, 8192, 8704, 9216)


class BenchmarkConfig(msgspec.Struct, frozen=True):
    layout: str
    hidden_size: int
    intermediate_size: int
    tokens_per_rank: int
    num_experts: int
    top_k: int
    balanced_routing: bool
    warmup_iters: int
    benchmark_iters: int
    total_repeats: int
    phase_repeats: int
    seed: int
    atol: float
    rtol: float
    world_size: int
    local_world_size: int


class Inputs(msgspec.Struct, frozen=True):
    hidden_states: torch.Tensor
    topk_output: StandardTopKOutput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        choices=("rank-major", "expert-major"),
        required=True,
    )
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--tokens-per-rank", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--balanced-routing",
        action="store_true",
        help=(
            "Generate deterministic top-k assignments with exactly equal aggregate "
            "route counts per global expert. Requires world_size * tokens_per_rank "
            "* top_k to be divisible by num_experts."
        ),
    )
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    parser.add_argument(
        "--total-repeats",
        type=int,
        default=5,
        help="Graph timing windows for total latency; the median is reported.",
    )
    parser.add_argument(
        "--phase-repeats",
        type=int,
        default=3,
        help="All-rank profiled passes for phase latencies; the median is reported.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--atol",
        type=float,
        default=2e-2,
        help="Absolute tolerance for all correctness checks.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=2e-2,
        help="Relative tolerance for all correctness checks.",
    )
    return parser.parse_args()


def initialize_distributed(tokens_per_rank: int) -> tuple[int, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("MSCCL++ LL benchmark requires CUDA GPUs")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        init_distributed_environment(
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            rank=int(os.environ.get("RANK", "0")),
            local_rank=local_rank,
            backend="nccl",
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise RuntimeError(
            "Expert-parallel benchmarking requires at least two ranks; launch "
            "with torchrun --nproc-per-node=<gpu-count>."
        )

    publish_build_topology(
        world_rank=rank,
        tp_size=world_size,
        ep_size=world_size,
        moe_a2a_backend="none",
        moe_runner_backend="triton",
        # Inputs below are already per-rank after attention-TP scatter.
        chunked_prefill_size=tokens_per_rank * world_size,
    )
    initialize_model_parallel(backend="nccl")
    return rank, world_size, int(os.environ.get("LOCAL_WORLD_SIZE", world_size))


def validate_args(args: argparse.Namespace, world_size: int) -> None:
    if args.hidden_size not in MSCCLPP_LL_HIDDEN_SIZES:
        raise ValueError(
            f"--hidden-size must be one of {MSCCLPP_LL_HIDDEN_SIZES} for MSCCL++ LL"
        )
    if args.num_experts <= 0 or args.num_experts % world_size:
        raise ValueError("--num-experts must be positive and divisible by world size")
    if not 0 < args.top_k <= args.num_experts:
        raise ValueError("--top-k must be in [1, num-experts]")
    if args.tokens_per_rank <= 0 or args.intermediate_size <= 0:
        raise ValueError("token and intermediate sizes must be positive")
    if args.warmup_iters < 1 or args.benchmark_iters < 1:
        raise ValueError("warmup and benchmark iteration counts must be positive")
    if args.total_repeats < 1 or args.phase_repeats < 1:
        raise ValueError("--total-repeats and --phase-repeats must be positive")
    if args.layout == "rank-major":
        from flashinfer.comm.mnnvl import is_mnnvl_fabric_supported

        supported = torch.tensor(
            int(is_mnnvl_fabric_supported(torch.cuda.current_device())),
            dtype=torch.int32,
            device="cuda",
        )
        dist.all_reduce(supported, op=dist.ReduceOp.MIN)
        if not supported.item():
            raise RuntimeError(
                "Rank-major comparison requires NVIDIA Fabric/MNNVL support "
                "on every participating GPU for the FlashInfer baseline."
            )


def make_inputs(config: BenchmarkConfig, rank: int, device: torch.device) -> Inputs:
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed + rank * 1_000_003)
    hidden_states = torch.randn(
        config.tokens_per_rank,
        config.hidden_size,
        dtype=DTYPE,
        device=device,
        generator=generator,
    )
    if config.balanced_routing:
        num_local_routes = config.tokens_per_rank * config.top_k
        num_global_routes = config.world_size * num_local_routes
        if num_global_routes % config.num_experts:
            raise ValueError(
                "--balanced-routing requires world_size * tokens_per_rank * top_k "
                "to be divisible by num_experts; "
                f"got {config.world_size} * {config.tokens_per_rank} * "
                f"{config.top_k} routes for {config.num_experts} experts"
            )

        permutation_generator = torch.Generator(device=device)
        permutation_generator.manual_seed(
            config.seed + 20_000_033 + config.tokens_per_rank
        )
        expert_permutation = torch.randperm(
            config.num_experts,
            dtype=torch.int64,
            device=device,
            generator=permutation_generator,
        )
        global_route_offset = rank * num_local_routes
        route_slots = (
            torch.arange(num_local_routes, dtype=torch.int64, device=device)
            + global_route_offset
        ) % config.num_experts
        topk_ids = expert_permutation[route_slots].view(
            config.tokens_per_rank, config.top_k
        )
        selected_logits = torch.randn(
            config.tokens_per_rank,
            config.top_k,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        router_logits = torch.full(
            (config.tokens_per_rank, config.num_experts),
            -torch.inf,
            dtype=torch.float32,
            device=device,
        )
        router_logits.scatter_(1, topk_ids, selected_logits)
    else:
        router_logits = torch.randn(
            config.tokens_per_rank,
            config.num_experts,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        selected_logits, topk_ids = torch.topk(
            router_logits, config.top_k, dim=-1, sorted=True
        )
    topk_weights = torch.softmax(selected_logits, dim=-1, dtype=torch.float32)
    return Inputs(
        hidden_states=hidden_states,
        topk_output=StandardTopKOutput(
            topk_weights.contiguous(),
            topk_ids.to(torch.int32).contiguous(),
            router_logits,
        ),
    )


def verify_balanced_routing(
    inputs: Inputs,
    config: BenchmarkConfig,
    device: torch.device,
) -> int:
    topk_ids = inputs.topk_output.topk_ids.to(torch.int64)
    counts = torch.bincount(topk_ids.flatten(), minlength=config.num_experts)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    expected = topk_ids.numel() * config.world_size // config.num_experts
    if not bool(torch.all(counts == expected)):
        raise AssertionError(
            "Balanced routing produced unequal global expert loads: "
            f"min={counts.min().item()}, max={counts.max().item()}, "
            f"expected={expected}"
        )

    sorted_ids = topk_ids.sort(dim=-1).values
    has_duplicate = torch.tensor(
        int(bool(torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]))),
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(has_duplicate, op=dist.ReduceOp.MAX)
    if has_duplicate.item():
        raise AssertionError("Balanced routing assigned a duplicate expert to a token")
    return expected


class FusedMoEPipeline:
    def __init__(
        self,
        config: BenchmarkConfig,
        rank: int,
        a2a_backend: MoeA2ABackend,
        runner_backend: MoeRunnerBackend,
        device: torch.device,
    ) -> None:
        self._stack = ExitStack()
        self._stack.enter_context(
            get_flags().moe.override(
                a2a_backend=a2a_backend,
                runner_backend=runner_backend,
            )
        )
        self.layer = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=0,
            params_dtype=DTYPE,
            reduce_results=False,
            inplace=False,
        ).to(device)

        generator = torch.Generator(device=device)
        generator.manual_seed(config.seed + 100_003 * rank)
        self.layer.w13_weight.data.normal_(0.0, 0.02, generator=generator)
        self.layer.w2_weight.data.normal_(0.0, 0.02, generator=generator)
        self.layer.quant_method.process_weights_after_loading(self.layer)
        self.layer.eval()

    def __call__(self, inputs: Inputs) -> torch.Tensor:
        return self.layer(inputs.hidden_states, inputs.topk_output)

    def close(self) -> None:
        if isinstance(self.layer.dispatcher, MSCCLPPDispatcher):
            self.layer.dispatcher.close()
        del self.layer
        self._stack.close()


@contextmanager
def flashinfer_cutlass_autotune():
    """Tune CUTLASS tactics consistently across the expert-parallel group."""
    from flashinfer.autotuner import (
        autotune,
        get_autotune_process_group,
        set_autotune_process_group,
    )

    tp_group = get_parallel().tp_group
    process_group = tp_group.cpu_group if tp_group.world_size > 1 else None
    previous_group = get_autotune_process_group()
    set_autotune_process_group(process_group)
    try:
        with autotune(True):
            yield
    finally:
        set_autotune_process_group(previous_group)


def clear_flashinfer_cutlass_autotune() -> None:
    """Force the next CUTLASS backend to tune independently."""
    from flashinfer.autotuner import AutoTuner

    tuner = AutoTuner.get()
    tuner.clear_cache()
    tuner.reset_statistics()


def synchronize() -> None:
    torch.cuda.synchronize()
    dist.barrier()


PHASES = ("dispatch", "compute", "combine", "unclassified")


class PhaseLatencies(msgspec.Struct, frozen=True):
    """Graph latency and per-iteration critical-path CUDA kernel time per phase.

    Ranges are (min, max) across timing windows for ``total`` and across
    profiled passes for the phases. ``median_rank_us`` and ``rank0_us`` hold
    (dispatch, compute, combine) for the median rank and rank 0; their gap to
    the headline dispatch/combine is time spent waiting for slower ranks.
    ``legacy_rank0_us`` is the previous metric (rank-0 median kernel duration
    times invocations per iteration, overlap counted twice) from the same
    profiled passes. ``iteration_split`` is False when some rank's kernel
    timeline could not be split into pipeline iterations and per-rank mean
    times were used instead.
    """

    total_us: float
    dispatch_us: float
    compute_us: float
    combine_us: float
    unclassified_us: float
    total_range_us: tuple[float, float]
    dispatch_range_us: tuple[float, float]
    compute_range_us: tuple[float, float]
    combine_range_us: tuple[float, float]
    median_rank_us: tuple[float, float, float]
    rank0_us: tuple[float, float, float]
    profiled_window_us: float
    legacy_rank0_us: tuple[float, float, float, float]
    iteration_split: bool

    def __str__(self) -> str:
        def span(values: tuple[float, float]) -> str:
            return f"{values[0]:.2f}-{values[1]:.2f} us"

        def phases(values: tuple[float, ...]) -> str:
            return ", ".join(
                f"{phase}={value:.2f} us" for phase, value in zip(PHASES, values)
            )

        phase_sum = (
            self.dispatch_us + self.compute_us + self.combine_us + self.unclassified_us
        )
        split = "per iteration" if self.iteration_split else "per-rank means"
        return (
            f"total={self.total_us:.2f} us "
            f"(critical-path CUDA kernels: dispatch={self.dispatch_us:.2f} us, "
            f"compute={self.compute_us:.2f} us, "
            f"combine={self.combine_us:.2f} us, "
            f"unclassified={self.unclassified_us:.2f} us)\n"
            f"  ranges: total {span(self.total_range_us)} across windows; "
            f"dispatch {span(self.dispatch_range_us)}, "
            f"compute {span(self.compute_range_us)}, "
            f"combine {span(self.combine_range_us)} across profiled passes\n"
            f"  median rank: {phases(self.median_rank_us)}; "
            f"rank 0: {phases(self.rank0_us)}\n"
            f"  profiled window={self.profiled_window_us:.2f} us, "
            f"phase sum={phase_sum:.2f} us ({split}); "
            f"legacy rank-0 median kernels: {phases(self.legacy_rank0_us)}"
        )


def _profiled_kernel_names(prof: profile) -> set[str]:
    return {
        evt.key
        for evt in prof.key_averages()
        if evt.device_type == torch.autograd.DeviceType.CUDA
        and evt.self_device_time_total > 0
    }


def calibrate_phase_kernel_names(
    layer: FusedMoE,
    inputs: Inputs,
    allow_ambiguous: bool = False,
) -> tuple[set[str], set[str], set[str]]:
    """Learn phase kernel names without using eager timings as results."""
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        dispatch_output = layer.dispatcher.dispatch(
            hidden_states=inputs.hidden_states, topk_output=inputs.topk_output
        )
        torch.cuda.synchronize()
    dispatch_names = _profiled_kernel_names(prof)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        combine_input = layer.run_moe_core(dispatch_output=dispatch_output)
        torch.cuda.synchronize()
    compute_names = _profiled_kernel_names(prof)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            final_hidden_states = layer.dispatcher.combine(combine_input=combine_input)
            final_hidden_states[..., : inputs.hidden_states.shape[-1]].contiguous()
        torch.cuda.synchronize()
    combine_names = _profiled_kernel_names(prof)

    overlaps = {
        "dispatch/compute": dispatch_names & compute_names,
        "dispatch/combine": dispatch_names & combine_names,
        "compute/combine": compute_names & combine_names,
    }
    ambiguous = {pair: names for pair, names in overlaps.items() if names}
    if ambiguous:
        if not allow_ambiguous:
            raise RuntimeError(
                "Cannot classify kernels unambiguously because phase kernel "
                f"names overlap: {ambiguous}"
            )
        ambiguous_names = set().union(*ambiguous.values())
        dispatch_names -= ambiguous_names
        compute_names -= ambiguous_names
        combine_names -= ambiguous_names
    synchronize()
    return dispatch_names, compute_names, combine_names


def _profiled_kernel_intervals(prof: profile) -> list[tuple[float, float, str]]:
    return [
        (evt.time_range.start, evt.time_range.end, evt.key)
        for evt in prof.events()
        if evt.device_type == torch.autograd.DeviceType.CUDA
        and evt.self_device_time_total > 0
    ]


def split_phase_timeline_us(
    kernels: list[tuple[float, float, str]],
    phase_kernel_names: tuple[set[str], set[str], set[str]],
) -> list[list[float]]:
    """Return one rank's non-overlapping kernel time per pipeline iteration.

    ``kernels`` holds ``(start_us, end_us, name)`` intervals. Kernels are
    ordered by start time and each counts only its time after the end of every
    earlier-starting kernel, so overlap (e.g. a programmatic dependent launch
    starting before its predecessor ends) is counted once, for the earlier
    kernel. An iteration starts at a dispatch kernel whose preceding classified
    kernel is not a dispatch kernel; kernels before the first dispatch kernel
    (e.g. the barrier that aligns ranks) are ignored. Each iteration is
    ``[dispatch, compute, combine, unclassified]`` in microseconds.
    """
    dispatch_names, compute_names, combine_names = phase_kernel_names
    iterations: list[list[float]] = []
    previous_phase = None
    max_end = float("-inf")
    for start, end, name in sorted(kernels):
        if name in dispatch_names:
            phase = 0
        elif name in compute_names:
            phase = 1
        elif name in combine_names:
            phase = 2
        else:
            phase = 3
        exclusive = max(0.0, end - max(start, max_end))
        max_end = max(max_end, end)
        if phase == 0 and previous_phase != 0:
            iterations.append([0.0] * len(PHASES))
        if phase != 3:
            previous_phase = phase
        if iterations:
            iterations[-1][phase] += exclusive
    return iterations


def reduce_phase_times_us(per_iteration: torch.Tensor) -> list[float]:
    """Reduce ``[ranks, iterations, phases]`` kernel times to medians over
    iterations of: dispatch and combine minimized, compute maximized and
    unclassified median across ranks; then (dispatch, compute, combine) of the
    median rank and of rank 0."""
    per_iteration = per_iteration.double()
    rank_min = per_iteration.amin(dim=0)
    rank_max = per_iteration.amax(dim=0)
    rank_median = per_iteration.quantile(0.5, dim=0)
    critical = (rank_min[:, 0], rank_max[:, 1], rank_min[:, 2], rank_median[:, 3])
    return (
        [values.quantile(0.5).item() for values in critical]
        + [rank_median[:, phase].quantile(0.5).item() for phase in range(3)]
        + [per_iteration[0, :, phase].quantile(0.5).item() for phase in range(3)]
    )


def profile_graph_phase_latencies_us(
    graph: torch.cuda.CUDAGraph,
    phase_kernel_names: tuple[set[str], set[str], set[str]],
    total_us: tuple[float, float, float],
    warmup_iters: int,
    benchmark_iters: int,
    graph_iters: int,
    repeats: int,
) -> PhaseLatencies:
    """Profile CUDA kernels launched by actual graph replays on every rank.

    Each of ``repeats`` passes replays a graph containing ``graph_iters`` full
    pipeline iterations ``benchmark_iters`` times while every rank records its
    CUDA kernels, so profiler overhead is the same on all ranks. Each rank's
    kernel timeline is split into iterations by ``split_phase_timeline_us``
    and the ranks are combined per iteration by ``reduce_phase_times_us``:
    dispatch and combine exclude time spent waiting for slower ranks, and
    compute is the slowest rank's. Values are medians over iterations, then
    over passes. If a rank's timeline cannot be split into exactly
    ``benchmark_iters * graph_iters`` iterations, per-rank mean times are
    reduced the same way instead. These are graph kernel measurements under
    CUPTI, not rescaled eager estimates; their sum approximates the profiled
    window per iteration. ``total_us`` is (median, min, max) from
    ``median_graph_latency_us``.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    iterations = benchmark_iters * graph_iters
    debug = os.environ.get("SGLANG_BENCH_PHASE_DEBUG") == "1"
    passes = []
    iteration_split = True
    for pass_index in range(repeats):
        for _ in range(warmup_iters):
            graph.replay()
        synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            dist.barrier()
            start.record()
            for _ in range(benchmark_iters):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
        window_us = start.elapsed_time(end) * 1000.0 / iterations

        timeline = split_phase_timeline_us(
            _profiled_kernel_intervals(prof), phase_kernel_names
        )
        phase_sums = [sum(phases) for phases in zip(*timeline)] or [0.0] * 4
        if len(timeline) == iterations:
            local_iterations = torch.tensor(timeline, dtype=torch.float64)
        else:
            local_iterations = torch.full((iterations, len(PHASES)), float("nan"))
        legacy = _aggregate_profiled_phase_latencies(
            prof if rank == 0 else None, phase_kernel_names, iterations
        )
        local_summary = torch.tensor(
            [window_us, len(timeline)]
            + [phase_sum / iterations for phase_sum in phase_sums]
            + [legacy[phase] for phase in PHASES],
            dtype=torch.float64,
            device="cuda",
        )
        summary = local_summary.new_empty((world_size, local_summary.numel()))
        dist.all_gather_into_tensor(summary, local_summary)
        local_iterations = local_iterations.to(device="cuda", dtype=torch.float64)
        per_iteration = local_iterations.new_empty(
            (world_size, *local_iterations.shape)
        )
        dist.all_gather_into_tensor(per_iteration, local_iterations)
        summary = summary.cpu()

        split_counts = summary[:, 1]
        if bool((split_counts == iterations).all()):
            values = reduce_phase_times_us(per_iteration.cpu())
        else:
            iteration_split = False
            values = reduce_phase_times_us(summary[:, 2:6].unsqueeze(1))
            if rank == 0:
                bad = [
                    f"rank {r}: {int(count)}"
                    for r, count in enumerate(split_counts.tolist())
                    if count != iterations
                ]
                print(
                    f"WARNING: expected {iterations} pipeline iterations in "
                    f"each kernel timeline, found {', '.join(bad)}; phase "
                    "times use per-rank means",
                    flush=True,
                )
        passes.append(values + [summary[:, 0].max().item()] + summary[0, 6:].tolist())

        if debug and rank == 0:
            for r, row in enumerate(summary.tolist()):
                print(
                    f"PHASE_DEBUG pass={pass_index} rank={r} window={row[0]:.2f} "
                    f"iterations={int(row[1])} "
                    + " ".join(
                        f"{phase}={value:.2f}" for phase, value in zip(PHASES, row[2:6])
                    ),
                    flush=True,
                )

    def median(index: int) -> float:
        return statistics.median(values[index] for values in passes)

    def value_range(index: int) -> tuple[float, float]:
        return (
            min(values[index] for values in passes),
            max(values[index] for values in passes),
        )

    return PhaseLatencies(
        total_us=total_us[0],
        dispatch_us=median(0),
        compute_us=median(1),
        combine_us=median(2),
        unclassified_us=median(3),
        total_range_us=(total_us[1], total_us[2]),
        dispatch_range_us=value_range(0),
        compute_range_us=value_range(1),
        combine_range_us=value_range(2),
        median_rank_us=(median(4), median(5), median(6)),
        rank0_us=(median(7), median(8), median(9)),
        profiled_window_us=median(10),
        legacy_rank0_us=(median(11), median(12), median(13), median(14)),
        iteration_split=iteration_split,
    )


def _aggregate_profiled_phase_latencies(
    prof: profile | None,
    phase_kernel_names: tuple[set[str], set[str], set[str]],
    denominator: int,
) -> dict[str, float]:
    dispatch_names, compute_names, combine_names = phase_kernel_names
    phase_totals = {
        "dispatch": 0.0,
        "compute": 0.0,
        "combine": 0.0,
        "unclassified": 0.0,
    }
    debug_events = []
    if prof is not None:
        durations_by_kernel: dict[str, list[float]] = {}
        for evt in prof.events():
            if (
                evt.device_type != torch.autograd.DeviceType.CUDA
                or evt.self_device_time_total <= 0
            ):
                continue
            durations_by_kernel.setdefault(evt.key, []).append(
                evt.self_device_time_total
            )

        for name, durations in durations_by_kernel.items():
            if name in dispatch_names:
                phase = "dispatch"
            elif name in compute_names:
                phase = "compute"
            elif name in combine_names:
                phase = "combine"
            else:
                phase = "unclassified"
            median_duration_us = statistics.median(durations)
            contribution_us = median_duration_us * len(durations) / denominator
            phase_totals[phase] += contribution_us
            debug_events.append(
                (phase, name, len(durations), median_duration_us, contribution_us)
            )

    if os.environ.get("SGLANG_BENCH_PHASE_DEBUG") == "1" and prof is not None:
        for phase, name, count, median_duration_us, contribution_us in sorted(
            debug_events
        ):
            print(
                f"PHASE_DEBUG phase={phase} count={count} "
                f"median_invocation_us={median_duration_us:.4f} "
                f"per_iteration_us={contribution_us:.4f} kernel={name}",
                flush=True,
            )

    return phase_totals


def profile_eager_phase_latencies_us(
    pipeline: FusedMoEPipeline,
    inputs: Inputs,
    phase_kernel_names: tuple[set[str], set[str], set[str]],
    warmup_iters: int,
    benchmark_iters: int,
) -> tuple[float, float, float, float]:
    """Profile rank-0 CUDA kernels from complete eager pipeline iterations."""
    for _ in range(warmup_iters):
        pipeline(inputs)
    synchronize()

    prof = None
    if dist.get_rank() == 0:
        with profile(activities=[ProfilerActivity.CUDA]) as rank_zero_prof:
            dist.barrier()
            for _ in range(benchmark_iters):
                pipeline(inputs)
            torch.cuda.synchronize()
        prof = rank_zero_prof
    else:
        dist.barrier()
        for _ in range(benchmark_iters):
            pipeline(inputs)
        torch.cuda.synchronize()
    dist.barrier()

    phase_totals = _aggregate_profiled_phase_latencies(
        prof,
        phase_kernel_names,
        benchmark_iters,
    )
    phase_times = torch.tensor(
        [
            phase_totals["dispatch"],
            phase_totals["compute"],
            phase_totals["combine"],
            phase_totals["unclassified"],
        ],
        dtype=torch.float64,
        device="cuda",
    )
    dist.broadcast(phase_times, src=0)
    dispatch_us, compute_us, combine_us, unclassified_us = phase_times.tolist()
    return dispatch_us, compute_us, combine_us, unclassified_us


def assert_close(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    config: BenchmarkConfig,
    label: str,
) -> tuple[float, float]:
    diff = (reference.float() - candidate.float()).abs()
    max_abs = diff.max()
    max_rel = (diff / reference.float().abs().clamp_min(1e-6)).max()
    metrics = torch.stack((max_abs, max_rel))
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)

    close = torch.tensor(
        int(
            torch.allclose(
                reference,
                candidate,
                atol=config.atol,
                rtol=config.rtol,
            )
        ),
        dtype=torch.int32,
        device=reference.device,
    )
    dist.all_reduce(close, op=dist.ReduceOp.MIN)
    max_abs_value, max_rel_value = metrics.tolist()
    if not close.item():
        raise AssertionError(
            f"{label} mismatch: max_abs={max_abs_value:.6g}, "
            f"max_rel={max_rel_value:.6g}, atol={config.atol}, "
            f"rtol={config.rtol}"
        )
    return max_abs_value, max_rel_value


def capture_graph(
    pipeline: FusedMoEPipeline,
    inputs: Inputs,
    warmup_iters: int,
    graph_iters: int = 10,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, int]:
    """Capture ``graph_iters`` back-to-back pipeline calls inside a single
    CUDA Graph. A single ``graph.replay()`` then runs ``graph_iters``
    iterations, amortizing per-replay CPU/event-timing overhead over more
    GPU work, giving a more accurate/efficient per-iteration measurement
    than capturing (and replaying) just one iteration at a time."""
    for _ in range(warmup_iters):
        output = pipeline(inputs)
    synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        for _ in range(graph_iters):
            output = pipeline(inputs)
    synchronize()
    return graph, output, graph_iters


def median_graph_latency_us(
    graph: torch.cuda.CUDAGraph,
    warmup_iters: int,
    benchmark_iters: int,
    graph_iters: int = 1,
    repeats: int = 1,
) -> tuple[float, float, float]:
    """Time ``benchmark_iters`` graph replays as a single elapsed-time
    window (start/end events outside the loop, per ``bench_time`` in
    mscclpp's ``python/test/executor_test.py``), rather than timing each
    replay individually -- this avoids per-replay CPU/event-recording
    overhead from skewing small-workload measurements. Each window gives the
    per-iteration average latency (elapsed time divided by
    ``benchmark_iters * graph_iters``, since each graph replay itself runs
    ``graph_iters`` captured iterations), maximized across ranks. The window
    is repeated ``repeats`` times, each after a barrier; returns the median,
    min and max over windows."""
    for _ in range(warmup_iters):
        graph.replay()

    windows_us = []
    for _ in range(repeats):
        synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(benchmark_iters):
            graph.replay()
        end.record()
        end.synchronize()
        windows_us.append(
            start.elapsed_time(end) * 1000.0 / benchmark_iters / graph_iters
        )

    latency = torch.tensor(windows_us, dtype=torch.float64, device="cuda")
    dist.all_reduce(latency, op=dist.ReduceOp.MAX)
    windows_us = latency.tolist()
    return statistics.median(windows_us), min(windows_us), max(windows_us)


def median_eager_latency_us(
    fn: Callable[[], torch.Tensor], warmup_iters: int, benchmark_iters: int
) -> float:
    for _ in range(warmup_iters):
        fn()
    synchronize()

    samples = []
    for _ in range(benchmark_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    latency = torch.tensor(
        statistics.median(samples), dtype=torch.float64, device="cuda"
    )
    dist.all_reduce(latency, op=dist.ReduceOp.MAX)
    return latency.item()


def run_rank_major(
    config: BenchmarkConfig, rank: int, inputs: Inputs, device: torch.device
) -> None:
    results = {}
    outputs = {}
    paths = (
        (
            "mscclpp-ll-rank-major-cutlass",
            MoeA2ABackend.MSCCLPP,
            MoeRunnerBackend.FLASHINFER_CUTLASS,
        ),
        (
            "flashinfer-rank-major-cutlass",
            MoeA2ABackend.FLASHINFER,
            MoeRunnerBackend.FLASHINFER_CUTLASS,
        ),
    )
    for name, a2a_backend, runner_backend in paths:
        if rank == 0:
            print(f"Preparing {name}...", flush=True)
        benchmark_stream = torch.cuda.Stream()
        graph = None
        eager = None
        graph_output = None
        phase_kernel_names = None
        with torch.cuda.stream(benchmark_stream):
            pipeline = FusedMoEPipeline(
                config, rank, a2a_backend, runner_backend, device
            )
            try:
                if rank == 0:
                    print(f"Autotuning {name} CUTLASS tactics...", flush=True)
                with flashinfer_cutlass_autotune():
                    pipeline(inputs)
                synchronize()
                if rank == 0:
                    print(f"Running {name} eager warmup...", flush=True)
                eager = pipeline(inputs).clone()
                if rank == 0:
                    print(f"Calibrating {name} phase kernel names...", flush=True)
                phase_kernel_names = calibrate_phase_kernel_names(
                    pipeline.layer, inputs
                )
                if rank == 0:
                    print(f"Capturing {name} CUDA Graph...", flush=True)
                graph, graph_output, graph_iters = capture_graph(
                    pipeline, inputs, config.warmup_iters
                )
                graph.replay()
                synchronize()
                assert_close(eager, graph_output, config, f"{name} eager vs graph")
                outputs[name] = graph_output.detach().cpu()
                total_us = median_graph_latency_us(
                    graph,
                    config.warmup_iters,
                    config.benchmark_iters,
                    graph_iters,
                    config.total_repeats,
                )
                if rank == 0:
                    print(f"Profiling {name} CUDA Graph kernels...", flush=True)
                results[name] = profile_graph_phase_latencies_us(
                    graph,
                    phase_kernel_names,
                    total_us,
                    config.warmup_iters,
                    config.benchmark_iters,
                    graph_iters,
                    config.phase_repeats,
                )
                if rank == 0:
                    print(f"Finished {name}.", flush=True)
            finally:
                synchronize()
                if graph is not None:
                    graph.reset()
                del graph, graph_output, eager, phase_kernel_names
                pipeline.close()
                del pipeline
                if a2a_backend == MoeA2ABackend.MSCCLPP:
                    MSCCLPPDispatcher.clear_shared_resources()
                torch.cuda.empty_cache()
                clear_flashinfer_cutlass_autotune()
                synchronize()
        del benchmark_stream

    flashinfer_output = outputs["flashinfer-rank-major-cutlass"].to(device)
    mscclpp_output = outputs["mscclpp-ll-rank-major-cutlass"].to(device)
    max_abs, max_rel = assert_close(
        flashinfer_output,
        mscclpp_output,
        config,
        "MSCCL++ vs FlashInfer rank-major",
    )
    del flashinfer_output, mscclpp_output
    if rank == 0:
        print(f"Correctness: PASS max_abs={max_abs:.6g} max_rel={max_rel:.6g}")
        for name, latencies in results.items():
            print(f"{name}: {latencies}")


def _reference_dispatch(
    inputs: Inputs, config: BenchmarkConfig
) -> tuple[Inputs, list[torch.Tensor], list[int], list[int]]:
    """Pack tokens per-destination and all-to-all them to their target rank.

    Returns the local (post all-to-all) ``Inputs`` for the compute step,
    plus the bookkeeping (``send_token_indices``, ``send_counts``,
    ``recv_counts``) needed by ``_reference_combine`` to route results back.
    """
    world_size = config.world_size
    device = inputs.hidden_states.device
    num_local_experts = config.num_experts // world_size
    topk_ids = inputs.topk_output.topk_ids
    destinations = torch.div(topk_ids, num_local_experts, rounding_mode="floor")

    send_hidden = []
    send_ids = []
    send_weights = []
    send_token_indices = []
    send_counts = []
    for destination in range(world_size):
        route_mask = destinations == destination
        token_mask = route_mask.any(dim=1)
        token_indices = token_mask.nonzero(as_tuple=False).flatten()
        ids = topk_ids[token_indices].clone()
        weights = inputs.topk_output.topk_weights[token_indices].clone()
        local_route_mask = route_mask[token_indices]
        ids.masked_fill_(~local_route_mask, -1)
        weights.masked_fill_(~local_route_mask, 0)
        send_hidden.append(inputs.hidden_states[token_indices])
        send_ids.append(ids)
        send_weights.append(weights)
        send_token_indices.append(token_indices.to(torch.int64))
        send_counts.append(token_indices.numel())

    send_counts_tensor = torch.tensor(
        send_counts, dtype=torch.int64, device=inputs.hidden_states.device
    )
    recv_counts_tensor = torch.empty_like(send_counts_tensor)
    dist.all_to_all_single(recv_counts_tensor, send_counts_tensor)
    recv_counts = recv_counts_tensor.tolist()

    def exchange(chunks, trailing_shape, dtype):
        send = (
            torch.cat(chunks, dim=0)
            if sum(send_counts)
            else torch.empty((0, *trailing_shape), dtype=dtype, device=device)
        )
        recv = torch.empty(
            (sum(recv_counts), *trailing_shape), dtype=dtype, device=device
        )
        dist.all_to_all_single(
            recv,
            send,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
        )
        return recv

    recv_hidden = exchange(send_hidden, (config.hidden_size,), DTYPE)
    recv_ids = exchange(send_ids, (config.top_k,), torch.int32)
    recv_weights = exchange(send_weights, (config.top_k,), torch.float32)
    exchange(send_token_indices, (), torch.int64)
    recv_topk = StandardTopKOutput(
        recv_weights,
        recv_ids,
        torch.empty(0, dtype=torch.float32, device=device),
    )
    return (
        Inputs(recv_hidden, recv_topk),
        send_token_indices,
        send_counts,
        recv_counts,
    )


def _reference_combine(
    recv_output: torch.Tensor,
    inputs: Inputs,
    config: BenchmarkConfig,
    send_token_indices: list[torch.Tensor],
    send_counts: list[int],
    recv_counts: list[int],
) -> torch.Tensor:
    """All-to-all expert outputs back to their originating rank and scatter."""
    device = inputs.hidden_states.device
    returned = torch.empty(
        (sum(send_counts), config.hidden_size), dtype=DTYPE, device=device
    )
    dist.all_to_all_single(
        returned,
        recv_output,
        output_split_sizes=send_counts,
        input_split_sizes=recv_counts,
    )
    returned_indices = torch.cat(send_token_indices)
    output = torch.zeros_like(inputs.hidden_states)
    output.index_add_(0, returned_indices, returned)
    return output


def triton_all_to_all_reference(
    layer: FusedMoEPipeline,
    inputs: Inputs,
    config: BenchmarkConfig,
    rank: int,
) -> torch.Tensor:
    """Variable-split token all-to-all around a rank-local Triton FusedMoE."""
    local_inputs, send_token_indices, send_counts, recv_counts = _reference_dispatch(
        inputs, config
    )
    recv_output = layer(local_inputs)
    return _reference_combine(
        recv_output, inputs, config, send_token_indices, send_counts, recv_counts
    )


def run_expert_major(
    config: BenchmarkConfig, rank: int, inputs: Inputs, device: torch.device
) -> None:
    benchmark_stream = torch.cuda.Stream()
    graph = None
    eager = None
    graph_output = None
    phase_kernel_names = None
    with torch.cuda.stream(benchmark_stream):
        mscclpp = FusedMoEPipeline(
            config,
            rank,
            MoeA2ABackend.MSCCLPP,
            MoeRunnerBackend.TRITON,
            device,
        )
        try:
            if rank == 0:
                print("Running mscclpp-ll-expert-major-triton eager...", flush=True)
            eager = mscclpp(inputs).clone()
            phase_kernel_names = calibrate_phase_kernel_names(
                mscclpp.layer,
                inputs,
                allow_ambiguous=True,
            )
            if rank == 0:
                print(
                    "Capturing mscclpp-ll-expert-major-triton CUDA Graph...",
                    flush=True,
                )
            graph, graph_output, graph_iters = capture_graph(
                mscclpp,
                inputs,
                config.warmup_iters,
            )
            graph.replay()
            synchronize()
            assert_close(
                eager,
                graph_output,
                config,
                "MSCCL++ expert-major eager vs graph",
            )
            mscclpp_output = graph_output.detach().cpu()
            total_us = median_graph_latency_us(
                graph,
                config.warmup_iters,
                config.benchmark_iters,
                graph_iters,
                config.total_repeats,
            )
            mscclpp_latencies = profile_graph_phase_latencies_us(
                graph,
                phase_kernel_names,
                total_us,
                config.warmup_iters,
                config.benchmark_iters,
                graph_iters,
                config.phase_repeats,
            )
        finally:
            synchronize()
            if graph is not None:
                graph.reset()
            del graph, graph_output, eager, phase_kernel_names
            mscclpp.close()
            del mscclpp
            MSCCLPPDispatcher.clear_shared_resources()
            torch.cuda.empty_cache()
            synchronize()
    del benchmark_stream
    synchronize()

    reference = FusedMoEPipeline(
        config,
        rank,
        MoeA2ABackend.NONE,
        MoeRunnerBackend.TRITON,
        device,
    )
    try:
        if rank == 0:
            print("Running triton-all-to-all-reference...", flush=True)
        reference_output = triton_all_to_all_reference(
            reference, inputs, config, rank
        ).clone()
    finally:
        reference.close()

    mscclpp_output = mscclpp_output.to(device)
    max_abs, max_rel = assert_close(
        reference_output,
        mscclpp_output,
        config,
        "MSCCL++ expert-major vs Triton all-to-all reference",
    )
    if rank == 0:
        print(f"Correctness: PASS max_abs={max_abs:.6g} max_rel={max_rel:.6g}")
        print(f"mscclpp-ll-expert-major-triton: {mscclpp_latencies}")
        print(
            "The Triton all-to-all reference is used for correctness only; its "
            "explicit variable-split collectives are not graph benchmarked."
        )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank, world_size, local_world_size = initialize_distributed(args.tokens_per_rank)
    validate_args(args, world_size)
    flags = get_flags().moe
    flags.mscclpp_mode = MSCCLPPMode.LATENCY
    flags.mscclpp_ep_layout = (
        MSCCLPPEPLayout.RANK_MAJOR
        if args.layout == "rank-major"
        else MSCCLPPEPLayout.EXPERT_MAJOR
    )
    config = BenchmarkConfig(
        **vars(args),
        world_size=world_size,
        local_world_size=local_world_size,
    )
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = make_inputs(config, rank, device)
    if config.balanced_routing:
        routes_per_expert = verify_balanced_routing(inputs, config, device)
        if rank == 0:
            print(
                "Balanced routing verified: "
                f"{routes_per_expert} routes/global expert, "
                "no duplicate top-k experts per token.",
                flush=True,
            )

    if rank == 0:
        print("Benchmark configuration:")
        print(json.dumps(msgspec.to_builtins(config), indent=2, sort_keys=True))
        print(f"nodes={world_size // local_world_size}, dtype={DTYPE}")
        print(
            "Timing unit: microseconds per pipeline iteration; total is the "
            "median over --total-repeats graph timing windows of the window "
            "average maximized across ranks; phases come from --phase-repeats "
            "passes profiling CUDA kernels on every rank: non-overlapping "
            "kernel time per phase and iteration, with dispatch and combine "
            "minimized, compute maximized and unclassified the median across "
            "ranks, then medians over iterations and passes"
        )

    try:
        if config.layout == "rank-major":
            run_rank_major(config, rank, inputs, device)
        else:
            run_expert_major(config, rank, inputs, device)
    finally:
        # Destroy cached C++ communication runtimes before the process group.
        MSCCLPPDispatcher.clear_shared_resources()
        torch.cuda.synchronize()
        destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

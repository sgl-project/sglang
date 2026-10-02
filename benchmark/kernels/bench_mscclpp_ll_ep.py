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
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
from contextlib import ExitStack
from typing import Callable, Optional

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
MSCCLPP_LL_HIDDEN_SIZES = (4096, 4352, 5120, 6656, 7168, 8192, 8704, 9216)


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
        del self.layer
        self._stack.close()


def synchronize() -> None:
    torch.cuda.synchronize()
    dist.barrier()


class PhaseLatencies(msgspec.Struct, frozen=True):
    """Wall-clock graph latency and profiled CUDA kernel time per phase."""

    total_us: float
    dispatch_us: float
    compute_us: float
    combine_us: float
    unclassified_us: float

    def __str__(self) -> str:
        return (
            f"total={self.total_us:.2f} us "
            f"(median rank-0 CUDA kernels: dispatch={self.dispatch_us:.2f} us, "
            f"compute={self.compute_us:.2f} us, "
            f"combine={self.combine_us:.2f} us, "
            f"unclassified={self.unclassified_us:.2f} us)"
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


def profile_graph_phase_latencies_us(
    graph: torch.cuda.CUDAGraph,
    phase_kernel_names: tuple[set[str], set[str], set[str]],
    total_us: float,
    warmup_iters: int,
    benchmark_iters: int,
    graph_iters: int,
) -> PhaseLatencies:
    """Profile CUDA kernels launched by actual graph replays.

    Rank 0 profiles ``benchmark_iters`` replays of a graph containing
    ``graph_iters`` full pipeline iterations while every other rank replays
    without CUPTI instrumentation. For each CUDA kernel, its median invocation
    duration is multiplied by its occurrences per pipeline iteration, then
    summed into its phase. These are rank-0 graph kernel measurements, not
    rescaled eager estimates; their sum can differ from wall-clock ``total_us``
    when kernels overlap.
    """
    for _ in range(warmup_iters):
        graph.replay()
    synchronize()

    prof = None
    if dist.get_rank() == 0:
        with profile(activities=[ProfilerActivity.CUDA]) as rank_zero_prof:
            dist.barrier()
            for _ in range(benchmark_iters):
                graph.replay()
            torch.cuda.synchronize()
        prof = rank_zero_prof
    else:
        dist.barrier()
        for _ in range(benchmark_iters):
            graph.replay()
        torch.cuda.synchronize()
    dist.barrier()

    phase_totals = _aggregate_profiled_phase_latencies(
        prof,
        phase_kernel_names,
        benchmark_iters * graph_iters,
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

    return PhaseLatencies(
        total_us=total_us,
        dispatch_us=dispatch_us,
        compute_us=compute_us,
        combine_us=combine_us,
        unclassified_us=unclassified_us,
    )


def _aggregate_profiled_phase_latencies(
    prof: Optional[profile],
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
    with torch.cuda.graph(graph):
        for _ in range(graph_iters):
            output = pipeline(inputs)
    synchronize()
    return graph, output, graph_iters


def median_graph_latency_us(
    graph: torch.cuda.CUDAGraph,
    warmup_iters: int,
    benchmark_iters: int,
    graph_iters: int = 1,
) -> float:
    """Time ``benchmark_iters`` graph replays as a single elapsed-time
    window (start/end events outside the loop, per ``bench_time`` in
    mscclpp's ``python/test/executor_test.py``), rather than timing each
    replay individually -- this avoids per-replay CPU/event-recording
    overhead from skewing small-workload measurements. The result is
    per-iteration average latency (elapsed time divided by
    ``benchmark_iters * graph_iters``, since each graph replay itself runs
    ``graph_iters`` captured iterations)."""
    for _ in range(warmup_iters):
        graph.replay()
    synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(benchmark_iters):
        graph.replay()
    end.record()
    end.synchronize()
    elapsed_us = start.elapsed_time(end) * 1000.0 / benchmark_iters / graph_iters

    latency = torch.tensor(elapsed_us, dtype=torch.float64, device="cuda")
    dist.all_reduce(latency, op=dist.ReduceOp.MAX)
    return latency.item()


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
            "flashinfer-rank-major-cutlass",
            MoeA2ABackend.FLASHINFER,
            MoeRunnerBackend.FLASHINFER_CUTLASS,
        ),
        (
            "mscclpp-ll-rank-major-cutlass",
            MoeA2ABackend.MSCCLPP,
            MoeRunnerBackend.FLASHINFER_CUTLASS,
        ),
    )
    for name, a2a_backend, runner_backend in paths:
        if rank == 0:
            print(f"Preparing {name}...", flush=True)
        pipeline = FusedMoEPipeline(config, rank, a2a_backend, runner_backend, device)
        try:
            if rank == 0:
                print(f"Running {name} eager warmup...", flush=True)
            eager = pipeline(inputs).clone()
            if rank == 0:
                print(f"Calibrating {name} phase kernel names...", flush=True)
            phase_kernel_names = calibrate_phase_kernel_names(pipeline.layer, inputs)
            if rank == 0:
                print(f"Capturing {name} CUDA Graph...", flush=True)
            graph, graph_output, graph_iters = capture_graph(
                pipeline, inputs, config.warmup_iters
            )
            graph.replay()
            synchronize()
            assert_close(eager, graph_output, config, f"{name} eager vs graph")
            outputs[name] = graph_output.clone()
            total_us = median_graph_latency_us(
                graph, config.warmup_iters, config.benchmark_iters, graph_iters
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
            )
            if rank == 0:
                print(f"Finished {name}.", flush=True)
            graph.reset()
        finally:
            pipeline.close()
        synchronize()

    max_abs, max_rel = assert_close(
        outputs["flashinfer-rank-major-cutlass"],
        outputs["mscclpp-ll-rank-major-cutlass"],
        config,
        "MSCCL++ vs FlashInfer rank-major",
    )
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
    recv_token_indices = exchange(send_token_indices, (), torch.int64)
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
    mscclpp = FusedMoEPipeline(
        config,
        rank,
        MoeA2ABackend.MSCCLPP,
        MoeRunnerBackend.TRITON,
        device,
    )
    try:
        if rank == 0:
            print("Running mscclpp-ll-expert-major-triton...", flush=True)
        mscclpp_output = mscclpp(inputs).clone()
        phase_kernel_names = calibrate_phase_kernel_names(
            mscclpp.layer,
            inputs,
            allow_ambiguous=True,
        )
        mscclpp_us = median_eager_latency_us(
            lambda: mscclpp(inputs),
            config.warmup_iters,
            config.benchmark_iters,
        )
        mscclpp_phases = profile_eager_phase_latencies_us(
            mscclpp,
            inputs,
            phase_kernel_names,
            config.warmup_iters,
            config.benchmark_iters,
        )
    finally:
        mscclpp.close()
        MSCCLPPDispatcher.clear_shared_resources()
        gc.collect()
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
        reference_us = median_eager_latency_us(
            lambda: triton_all_to_all_reference(reference, inputs, config, rank),
            config.warmup_iters,
            config.benchmark_iters,
        )
    finally:
        reference.close()

    max_abs, max_rel = assert_close(
        reference_output,
        mscclpp_output,
        config,
        "MSCCL++ expert-major vs Triton all-to-all reference",
    )
    if rank == 0:
        print(f"Correctness: PASS max_abs={max_abs:.6g} max_rel={max_rel:.6g}")
        dispatch_us, compute_us, combine_us, unclassified_us = mscclpp_phases
        print(
            "mscclpp-ll-expert-major-triton: "
            f"eager median latency={mscclpp_us:.2f} us "
            f"(rank-0 median CUDA kernels: dispatch={dispatch_us:.2f} us, "
            f"compute={compute_us:.2f} us, combine={combine_us:.2f} us, "
            f"unclassified={unclassified_us:.2f} us)"
        )
        print(
            f"triton-all-to-all-reference: eager median latency={reference_us:.2f} us"
        )
        print(
            "Reference phase timing is omitted because its explicit PyTorch "
            "variable-split all-to-all calls cannot be classified reliably by "
            "CUDA kernel name. Its performance is not an apples-to-apples "
            "comparison with MSCCL++."
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
            "Timing unit: microseconds; total is the graph-window average "
            "maximized across ranks; phase values sum rank-0 median CUDA "
            "kernel invocation durations"
        )

    try:
        if config.layout == "rank-major":
            run_rank_major(config, rank, inputs, device)
        else:
            run_expert_major(config, rank, inputs, device)
    finally:
        # Destroy cached C++ communication runtimes before the process group.
        MSCCLPPDispatcher.clear_shared_resources()
        gc.collect()
        torch.cuda.synchronize()
        destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

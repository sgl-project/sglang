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
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.token_dispatcher.mscclpp import MSCCLPPDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.runtime_context import get_flags
from sglang.test.test_utils import publish_build_topology

DTYPE = torch.bfloat16
MSCCLPP_LL_HIDDEN_SIZES = (4096, 4352, 5120, 6656, 7168, 8192, 8704, 9216)


@dataclass(frozen=True)
class BenchmarkConfig:
    layout: str
    hidden_size: int
    intermediate_size: int
    tokens_per_rank: int
    num_experts: int
    top_k: int
    warmup_iters: int
    benchmark_iters: int
    seed: int
    atol: float
    rtol: float
    world_size: int
    local_world_size: int


@dataclass(frozen=True)
class Inputs:
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
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    return parser.parse_args()


def initialize_distributed() -> tuple[int, int, int]:
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
        int(torch.allclose(reference, candidate, atol=config.atol, rtol=config.rtol)),
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
    pipeline: FusedMoEPipeline, inputs: Inputs, warmup_iters: int
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    for _ in range(warmup_iters):
        output = pipeline(inputs)
    synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = pipeline(inputs)
    synchronize()
    return graph, output


def median_graph_latency_us(
    graph: torch.cuda.CUDAGraph, warmup_iters: int, benchmark_iters: int
) -> float:
    for _ in range(warmup_iters):
        graph.replay()
    synchronize()

    samples = []
    for _ in range(benchmark_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    latency = torch.tensor(
        statistics.median(samples), dtype=torch.float64, device="cuda"
    )
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
                print(f"Capturing {name} CUDA Graph...", flush=True)
            graph, graph_output = capture_graph(pipeline, inputs, config.warmup_iters)
            graph.replay()
            synchronize()
            assert_close(eager, graph_output, config, f"{name} eager vs graph")
            outputs[name] = graph_output.clone()
            results[name] = median_graph_latency_us(
                graph, config.warmup_iters, config.benchmark_iters
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
        for name, latency in results.items():
            print(f"{name}: median latency={latency:.2f} us")


def triton_all_to_all_reference(
    layer: FusedMoEPipeline,
    inputs: Inputs,
    config: BenchmarkConfig,
    rank: int,
) -> torch.Tensor:
    """Variable-split token all-to-all around a rank-local Triton FusedMoE."""
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
    recv_output = layer(Inputs(recv_hidden, recv_topk))

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
        mscclpp_us = median_eager_latency_us(
            lambda: mscclpp(inputs),
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
        print(f"mscclpp-ll-expert-major-triton: median latency={mscclpp_us:.2f} us")
        print(f"triton-all-to-all-reference: median latency={reference_us:.2f} us")
        print(
            "Performance note: the Triton reference uses explicit PyTorch "
            "variable-split all-to-all calls and is not an apples-to-apples "
            "performance comparison with MSCCL++."
        )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank, world_size, local_world_size = initialize_distributed()
    validate_args(args, world_size)
    config = BenchmarkConfig(
        **vars(args),
        world_size=world_size,
        local_world_size=local_world_size,
    )
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = make_inputs(config, rank, device)

    if rank == 0:
        print("Benchmark configuration:")
        print(json.dumps(asdict(config), indent=2, sort_keys=True))
        print(f"nodes={world_size // local_world_size}, dtype={DTYPE}")
        print(
            "Timing unit: microseconds; aggregation: median on each rank, "
            "then maximum across ranks"
        )

    try:
        if config.layout == "rank-major":
            run_rank_major(config, rank, inputs, device)
        else:
            run_expert_major(config, rank, inputs, device)
    finally:
        destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Compare MoE expert-parallel dispatch/combine backends under CUDA Graphs.

Run the script once per backend so only one MNNVL fabric workspace is resident:

1. ``--backend flashinfer``: FlashInfer ``MoeAlltoAll`` dispatch/combine.
2. ``--backend mscclpp``: MSCCL++ EP low-latency dispatch/combine.

Both paths use BF16 communication and BF16 expert compute with the same input,
routing decisions, routing weights, and rank-local expert weights for a fixed
seed. Both dispatchers produce fixed-capacity token buffers consumed through
SGLang's production FlashInfer CUTLASS fused-runner entry; only
dispatch/combine communication differs.

By default one CUDA Graph captures ``--iters-per-graph`` independently routed
``dispatch -> MoE -> combine`` iterations, and wall-clock time per iteration is
reported over ``--graph-replays`` replays. Capturing many iterations amortizes
launch overhead and averages over routing variation. Pass ``--iters-per-graph 1``
to match SGLang decode CUDA Graph boundaries, and add ``--torch-profiler`` to
report per-stage and per-kernel CUDA time instead of wall-clock time. Profiling
honours ``--iters-per-graph``; keep it high so the spin-waiting dispatch and
combine kernels are not inflated by per-replay launch skew.

Example on one 8-GPU node:

    torchrun --standalone --nproc-per-node=8 \
      benchmark/kernels/bench_moe_ep.py --backend mscclpp

All participating GPUs must belong to the same NVIDIA Fabric cluster.
"""

from __future__ import annotations

import argparse
import gc
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

DTYPE = torch.bfloat16
MSCCLPP_LL_HIDDEN_SIZES = (2048, 4096, 6656, 7168, 8192, 8704, 9216)


@dataclass
class Inputs:
    hidden_states: torch.Tensor
    topk_ids_i64: torch.Tensor
    topk_ids_i32: torch.Tensor
    topk_weights: torch.Tensor


@dataclass
class SglangTopKOutput:
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor


@dataclass
class SglangDispatchOutput:
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor | None
    topk_output: SglangTopKOutput
    moe_output: torch.Tensor


@dataclass
class ExpertWeights:
    w13: torch.Tensor
    w2: torch.Tensor


@dataclass
class Timing:
    dispatch_us: float
    moe_us: float
    combine_us: float
    e2e_us: float


@dataclass
class CapturedGraph:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    iterations: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("flashinfer", "mscclpp"),
        required=True,
        help="Benchmark exactly one backend so MNNVL fabric workspaces do not interfere.",
    )
    parser.add_argument(
        "--tokens-per-rank",
        type=str,
        default="128",
        help="Comma-separated local token counts to benchmark.",
    )
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--warmup-replays",
        type=int,
        default=20,
        help="Unmeasured CUDA Graph replays run after capture, before timing.",
    )
    parser.add_argument(
        "--graph-replays",
        type=int,
        default=20,
        help="Number of measured CUDA Graph replays.",
    )
    parser.add_argument(
        "--iters-per-graph",
        type=int,
        default=100,
        help=(
            "Independently routed dispatch->MoE->combine iterations captured in one "
            "CUDA Graph; reported times are per iteration."
        ),
    )
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Report per-stage CUDA kernel time instead of wall-clock time.",
    )
    parser.add_argument(
        "--skip-moe-autotune",
        action="store_true",
        help="Use FlashInfer CUTLASS fallback tactics instead of exact-shape autotuning.",
    )
    parser.add_argument(
        "--per-rank-moe-autotune",
        action="store_true",
        help="Autotune independently on every rank instead of broadcasting rank 0's "
        "tactics. Ranks that mis-time a candidate become stragglers and, because "
        "dispatch and combine are barriers, stall every peer.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--weight-std",
        type=float,
        default=0.02,
        help="Standard deviation used for random BF16 expert weights.",
    )
    parser.add_argument(
        "--input-std",
        type=float,
        default=0.2,
        help="Standard deviation used for random BF16 input tokens.",
    )
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    return parser.parse_args()


def parse_token_counts(value: str) -> list[int]:
    token_counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not token_counts or any(count <= 0 for count in token_counts):
        raise ValueError("--tokens-per-rank must contain positive integers")
    return token_counts


def initialize_distributed() -> tuple[int, int, dist.ProcessGroup]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device("cuda", local_rank),
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise ValueError("This EP benchmark requires at least two ranks")

    cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    return rank, world_size, cpu_group


def validate_args(args: argparse.Namespace, world_size: int) -> None:
    from flashinfer.comm.mnnvl import is_mnnvl_fabric_supported

    if not is_mnnvl_fabric_supported(torch.cuda.current_device()):
        raise RuntimeError(
            "This benchmark requires NVIDIA Fabric support on every participating GPU"
        )
    if args.num_experts % world_size != 0:
        raise ValueError(
            f"num_experts={args.num_experts} must be divisible by world_size={world_size}"
        )
    if not 0 < args.top_k <= min(9, args.num_experts):
        raise ValueError("--top-k must be in [1, min(9, num_experts)]")
    if args.backend == "mscclpp" and args.hidden_size not in MSCCLPP_LL_HIDDEN_SIZES:
        raise ValueError(
            f"--hidden-size must be one of {MSCCLPP_LL_HIDDEN_SIZES} "
            "for MSCCL++ EP low-latency"
        )
    if args.intermediate_size <= 0:
        raise ValueError("--intermediate-size must be positive")
    if args.graph_replays <= 0:
        raise ValueError("--graph-replays must be positive")
    if args.warmup_replays < 0:
        raise ValueError("--warmup-replays must be non-negative")
    if args.iters_per_graph <= 0:
        raise ValueError("--iters-per-graph must be positive")


def make_local_weights(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> ExpertWeights:
    num_local_experts = args.num_experts // world_size
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 100_003 * rank)

    w13 = torch.empty(
        (num_local_experts, 2 * args.intermediate_size, args.hidden_size),
        dtype=DTYPE,
        device=device,
    )
    w13.normal_(mean=0.0, std=args.weight_std, generator=generator)
    w2 = torch.empty(
        (num_local_experts, args.hidden_size, args.intermediate_size),
        dtype=DTYPE,
        device=device,
    )
    w2.normal_(mean=0.0, std=args.weight_std, generator=generator)
    return ExpertWeights(w13=w13, w2=w2)


def make_inputs(
    args: argparse.Namespace,
    rank: int,
    num_tokens: int,
    device: torch.device,
    sample_index: int = 0,
) -> Inputs:
    generator = torch.Generator(device=device)
    generator.manual_seed(
        args.seed + 1_000_003 * rank + 10_000_019 * sample_index + num_tokens
    )

    hidden_states = torch.empty(
        (num_tokens, args.hidden_size), dtype=DTYPE, device=device
    )
    hidden_states.normal_(mean=0.0, std=args.input_std, generator=generator)

    router_logits = torch.randn(
        (num_tokens, args.num_experts),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    selected_logits, topk_ids_i64 = torch.topk(
        router_logits, k=args.top_k, dim=-1, sorted=True
    )
    topk_weights = torch.softmax(selected_logits, dim=-1, dtype=torch.float32)
    topk_ids_i64 = topk_ids_i64.contiguous()
    return Inputs(
        hidden_states=hidden_states,
        topk_ids_i64=topk_ids_i64,
        topk_ids_i32=topk_ids_i64.to(torch.int32),
        topk_weights=topk_weights.contiguous(),
    )


class SglangCutlassMoe:
    def __init__(
        self,
        weights: ExpertWeights,
        rank: int,
        world_size: int,
        max_dispatched_tokens: int,
        top_k: int,
    ) -> None:
        # Production model loading initializes the quantization registry before the MoE runner.
        import sglang.srt.layers.quantization  # noqa: F401

        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
            FlashInferCutlassMoeQuantInfo,
            _run_flashinfer_cutlass,
        )

        self.run_fused_experts = _run_flashinfer_cutlass
        if weights.w13.dtype != DTYPE or weights.w2.dtype != DTYPE:
            raise TypeError("FlashInfer CUTLASS BF16 MoE requires BF16 expert weights")
        self.quant_info = FlashInferCutlassMoeQuantInfo(
            quant_type="bf16",
            w13_weight=weights.w13,
            w2_weight=weights.w2,
            output_dtype=DTYPE,
            moe_tp_size=1,
            moe_tp_rank=0,
            moe_ep_size=world_size,
            moe_ep_rank=rank,
            apply_routed_scaling_factor=True,
        )
        self.runner_config = MoeRunnerConfig(
            num_experts=weights.w13.shape[0] * world_size,
            num_local_experts=weights.w13.shape[0],
            hidden_size=weights.w13.shape[-1],
            intermediate_size_per_partition=weights.w2.shape[-1],
            top_k=top_k,
            params_dtype=DTYPE,
            activation="silu",
            is_gated=True,
            apply_router_weight_on_input=False,
            inplace=True,
        )
        self.max_dispatched_tokens = max_dispatched_tokens
        self.router_logits = torch.empty(
            0, dtype=torch.float32, device=weights.w13.device
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.dtype != DTYPE or output.dtype != DTYPE:
            raise TypeError("FlashInfer CUTLASS BF16 MoE requires BF16 input/output")
        if hidden_states.shape[0] > self.max_dispatched_tokens:
            raise ValueError(
                "dispatched rows exceed the configured SGLang MoE capacity"
            )
        dispatch_output = SglangDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=SglangTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=self.router_logits,
            ),
            moe_output=output,
        )
        return self.run_fused_experts(
            dispatch_output=dispatch_output,
            quant_info=self.quant_info,
            runner_config=self.runner_config,
            output=output,
            enable_alltoall=True,
        )


def make_torch_distributed_comm_backend(group: dist.ProcessGroup) -> Any:
    from flashinfer.comm.mnnvl import CommBackend

    class TorchDistributedCommBackend(CommBackend):
        def Get_rank(self) -> int:
            return group.rank()

        def Get_size(self) -> int:
            return group.size()

        def allgather(self, data: int) -> list[Any]:
            gathered = [None] * self.Get_size()
            dist.all_gather_object(gathered, data, group=group)
            return gathered

        def bcast(self, data: Any, root: int = 0) -> Any:
            objects = [data]
            dist.broadcast_object_list(objects, src=root, group=group)
            return objects[0]

        def Split(self, color: int, key: int) -> TorchDistributedCommBackend:
            del color, key
            return self

        def barrier(self) -> None:
            dist.barrier(group=group)

    return TorchDistributedCommBackend()


class FlashInferPipeline:
    name = "flashinfer_a2a_mnnvl"

    def __init__(
        self,
        args: argparse.Namespace,
        rank: int,
        world_size: int,
        num_tokens: int,
        weights: ExpertWeights,
        group: dist.ProcessGroup,
    ) -> None:
        from flashinfer.comm import MoeAlltoAll, moe_a2a_get_workspace_size_per_rank
        from flashinfer.comm.mapping import Mapping
        from flashinfer.comm.mnnvl import MnnvlConfig

        dispatch_bytes_per_token = (
            args.hidden_size * DTYPE.itemsize
            + args.top_k * torch.int32.itemsize
            + args.top_k * torch.float32.itemsize
        )
        workspace_size = moe_a2a_get_workspace_size_per_rank(
            ep_size=world_size,
            max_num_tokens=num_tokens,
            total_dispatch_payload_size_per_token=dispatch_bytes_per_token,
            combine_payload_size_per_token=args.hidden_size * DTYPE.itemsize,
        )
        self.a2a = MoeAlltoAll(
            mapping=Mapping(
                rank=rank,
                tp_size=world_size,
                moe_ep_size=world_size,
                world_size=world_size,
                gpus_per_node=torch.cuda.device_count(),
                pp_size=1,
                cp_size=1,
            ),
            max_num_tokens=num_tokens,
            top_k=args.top_k,
            num_experts=args.num_experts,
            workspace_size_per_rank=workspace_size,
            mnnvl_config=MnnvlConfig(
                comm_backend=make_torch_distributed_comm_backend(group)
            ),
        )
        self.num_tokens = num_tokens
        self.hidden_size = args.hidden_size
        self.num_experts = args.num_experts
        self.world_size = world_size
        self.moe = SglangCutlassMoe(
            weights=weights,
            rank=rank,
            world_size=world_size,
            max_dispatched_tokens=world_size * num_tokens,
            top_k=args.top_k,
        )

    def dispatch(self, inputs: Inputs) -> tuple[torch.Tensor, ...]:
        recv_tensors = self.a2a.dispatch(
            inputs.topk_ids_i32,
            [inputs.hidden_states, inputs.topk_ids_i32, inputs.topk_weights],
            self.num_tokens,
            invalid_token_expert_id=self.num_experts,
            expert_id_payload_index=1,
        )
        return tuple(recv_tensors)

    def run_moe(self, state: tuple[torch.Tensor, ...]) -> torch.Tensor:
        recv_hidden, recv_topk_ids, recv_topk_weights = state
        output = self.a2a.get_combine_payload_tensor_in_workspace(
            self.num_tokens, self.hidden_size, DTYPE
        ).view(-1, self.hidden_size)
        return self.moe(
            hidden_states=recv_hidden.view(-1, self.hidden_size),
            topk_ids=recv_topk_ids.view(-1, recv_topk_ids.shape[-1]),
            topk_weights=recv_topk_weights.view(-1, recv_topk_weights.shape[-1]),
            output=output,
        )

    def combine(
        self, state: tuple[torch.Tensor, ...], expert_output: torch.Tensor
    ) -> torch.Tensor:
        del state
        return self.a2a.combine(
            expert_output.view(self.world_size, self.num_tokens, self.hidden_size),
            self.num_tokens,
            payload_in_workspace=True,
        )

    def close(self) -> None:
        self.a2a._reset_workspace()


class MscclppPipeline:
    name = "mscclpp_ep_ll_rank_major"

    def __init__(
        self,
        args: argparse.Namespace,
        rank: int,
        world_size: int,
        num_tokens: int,
        weights: ExpertWeights,
        comm_group: Any,
        device: torch.device,
    ) -> None:
        from mscclpp.ep import (
            CombineMode,
            DispatchLayout,
            MoECommunicator,
            MoECommunicatorConfig,
            MoEMode,
        )

        num_local_experts = args.num_experts // world_size
        self.communicator = MoECommunicator(
            MoECommunicatorConfig(
                comm=comm_group,
                device=device,
                num_experts=args.num_experts,
                num_local_experts=num_local_experts,
                local_expert_start=rank * num_local_experts,
                hidden_size=args.hidden_size,
                topk=args.top_k,
                max_tokens_per_rank=num_tokens,
                mode=MoEMode.LATENCY,
                output_layout=DispatchLayout.RANK_MAJOR,
                invalid_token_expert_id=args.num_experts,
                combine_mode=CombineMode.RANK_LOCAL_REDUCE,
            )
        )
        if not self.communicator.is_available():
            raise RuntimeError("MSCCL++ EP low-latency runtime is unavailable")

        self.combine_output = torch.empty(
            (num_tokens, args.hidden_size), dtype=DTYPE, device=device
        )
        self.moe = SglangCutlassMoe(
            weights=weights,
            rank=rank,
            world_size=world_size,
            max_dispatched_tokens=world_size * num_tokens,
            top_k=args.top_k,
        )

    def dispatch(self, inputs: Inputs) -> tuple[Any, Any]:
        return self.communicator.dispatch(
            inputs.hidden_states,
            inputs.topk_ids_i64,
            inputs.topk_weights,
            output_buffer=None,
        )

    def run_moe(self, state: tuple[Any, Any]) -> torch.Tensor:
        dispatch_output, _ = state
        if dispatch_output.topk_ids is None or dispatch_output.weights is None:
            raise RuntimeError("MSCCL++ RANK_MAJOR dispatch metadata is missing")
        if dispatch_output.combine_input_buffer is None:
            raise RuntimeError("MSCCL++ RANK_MAJOR combine buffer is missing")
        return self.moe(
            hidden_states=dispatch_output.tokens,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.weights,
            output=dispatch_output.combine_input_buffer,
        )

    def combine(
        self, state: tuple[Any, Any], expert_output: torch.Tensor
    ) -> torch.Tensor:
        _, handle = state
        return self.communicator.combine(
            expert_output, handle, out=self.combine_output
        )

    def close(self) -> None:
        return None


def run_once(pipeline: Any, inputs: Inputs) -> torch.Tensor:
    state = pipeline.dispatch(inputs)
    expert_output = pipeline.run_moe(state)
    return pipeline.combine(state, expert_output)


def autotune_moe(
    pipeline: Any,
    inputs: Inputs,
    dispatched_tokens: int,
    sync_group: dist.ProcessGroup,
    per_rank: bool = False,
) -> None:
    from flashinfer.autotuner import AutoTuner, autotune

    torch.cuda.synchronize()
    dist.barrier(group=sync_group)
    # Tuning replays the whole pipeline, including the collective dispatch and
    # combine, so every rank has to run the tuning pass in lockstep.
    with autotune(True, tuning_buckets=(dispatched_tokens,)):
        run_once(pipeline, inputs)
    torch.cuda.synchronize()
    dist.barrier(group=sync_group)
    if per_rank:
        return

    # Every rank tunes the same shapes, so independent picks only add noise: a
    # rank that mis-times one candidate keeps a slower tactic for the whole run
    # and, because dispatch and combine are barriers, stalls every peer. Adopt
    # rank 0's tactics everywhere.
    tuner = AutoTuner.get()
    payload: list[str | None] = [None]
    if dist.get_rank() == 0:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "moe_tactics.json")
            tuner.save_configs(path)
            with open(path, encoding="utf-8") as handle:
                payload[0] = handle.read()
    dist.broadcast_object_list(payload, src=0, group=sync_group)
    if dist.get_rank() != 0:
        configs = payload[0]
        if not configs:
            raise RuntimeError("rank 0 produced no autotuner configs to broadcast")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "moe_tactics.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(configs)
            # Live tuning results take priority over loaded ones, so this rank
            # has to drop its own picks before adopting rank 0's.
            tuner.clear_cache()
            if not tuner.load_configs(path):
                raise RuntimeError(
                    "failed to load autotuner configs broadcast from rank 0; "
                    "ranks may differ in GPU or library versions"
                )
    torch.cuda.synchronize()
    dist.barrier(group=sync_group)


def reduce_max(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.cpu().tolist()


def reduce_mean(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor.cpu().tolist()


def reduce_min_max(
    values: list[float], device: torch.device
) -> tuple[list[float], list[float]]:
    minimum = torch.tensor(values, dtype=torch.float64, device=device)
    maximum = minimum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return minimum.cpu().tolist(), maximum.cpu().tolist()


def gather_per_rank(values: list[float], device: torch.device) -> list[list[float]]:
    world_size = dist.get_world_size()
    local = torch.tensor(values, dtype=torch.float64, device=device)
    gathered = torch.empty(
        (world_size, local.numel()), dtype=torch.float64, device=device
    )
    dist.all_gather_into_tensor(gathered, local)
    return gathered.cpu().tolist()


def synchronize_stream_and_ranks(
    mscclpp_comm_group: Any | None,
    cpu_group: dist.ProcessGroup,
) -> None:
    torch.cuda.current_stream().synchronize()
    if mscclpp_comm_group is not None:
        mscclpp_comm_group.barrier()
    else:
        dist.barrier(group=cpu_group)


def assert_outputs_close(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    rtol: float,
    atol: float,
    device: torch.device,
    label: str,
) -> float:
    diff = (reference.float() - candidate.float()).abs()
    max_abs = diff.max()
    max_rel = (diff / reference.float().abs().clamp_min(1e-6)).max()
    is_close = torch.tensor(
        [int(torch.allclose(reference, candidate, rtol=rtol, atol=atol))],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(is_close, op=dist.ReduceOp.MIN)
    max_abs_value, max_rel_value = reduce_max([max_abs.item(), max_rel.item()], device)
    if not bool(is_close.item()):
        raise AssertionError(
            f"{label} outputs differ: "
            f"max_abs={max_abs_value:.6g}, max_rel={max_rel_value:.6g}, "
            f"rtol={rtol}, atol={atol}"
        )
    return max_abs_value


def capture_graph(
    pipeline: Any,
    input_samples: list[Inputs],
    sync_group: dist.ProcessGroup,
) -> CapturedGraph:
    # The caller already ran every sample eagerly, so lazy workspace and module
    # initialization is complete and capture only records steady-state work.
    torch.cuda.synchronize()
    dist.barrier(group=sync_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for inputs in input_samples:
            output = run_once(pipeline, inputs)

    torch.cuda.synchronize()
    dist.barrier(group=sync_group)
    return CapturedGraph(graph=graph, output=output, iterations=len(input_samples))


def warm_up_graph(
    captured: CapturedGraph,
    warmup_replays: int,
    sync_group: dist.ProcessGroup,
) -> None:
    """Replay the captured graph unmeasured so timing sees steady state."""
    torch.cuda.synchronize()
    dist.barrier(group=sync_group)
    for _ in range(warmup_replays):
        captured.graph.replay()
    torch.cuda.synchronize()
    dist.barrier(group=sync_group)


def time_graph_replays(
    captured: CapturedGraph,
    replays: int,
    device: torch.device,
    sync_group: dist.ProcessGroup,
) -> float:
    torch.cuda.synchronize()
    dist.barrier(group=sync_group)

    start = time.perf_counter()
    for _ in range(replays):
        captured.graph.replay()
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) * 1_000_000
    (per_iteration_us,) = reduce_mean(
        [elapsed_us / (replays * captured.iterations)], device
    )

    dist.barrier(group=sync_group)
    if dist.get_rank() == 0:
        print(
            "Wall-clock CUDA Graph time: "
            f"graph_replays={replays}, iterations_per_graph={captured.iterations}, "
            f"total_iterations={replays * captured.iterations}, "
            f"per_graph={per_iteration_us * captured.iterations:.1f}us, "
            f"per_iteration={per_iteration_us:.1f}us",
            flush=True,
        )
    return per_iteration_us


def classify_profiled_kernel(name: str) -> str | None:
    normalized = name.lower()
    if "moea2a" in normalized:
        if "dispatch" in normalized or "sanitizeexpertids" in normalized:
            return "dispatch"
        if "combine" in normalized:
            return "combine"
    if "mscclpp::ep::" in normalized:
        if "dispatchkernel" in normalized:
            return "dispatch"
        if "combinekernel" in normalized or "combinetmaloadkernel" in normalized:
            return "combine"
    return None


@dataclass
class KernelStat:
    stage: str
    name: str
    launches_per_iteration: float
    total_us_per_iteration: float


def summarize_profiled_replays(
    profiler: torch.profiler.profile,
    expected_replays: int,
) -> tuple[Timing, list[KernelStat]]:
    events = profiler.events()
    launch_ids = [event.id for event in events if event.name == "cudaGraphLaunch"]
    if len(launch_ids) != expected_replays:
        raise RuntimeError(
            f"Torch Profiler observed {len(launch_ids)} CUDA Graph launches; "
            f"expected {expected_replays}"
        )

    launch_id_set = set(launch_ids)
    kernels_by_launch: dict[int, list[Any]] = {
        launch_id: [] for launch_id in launch_ids
    }
    for event in events:
        if (
            event.device_type == torch.autograd.DeviceType.CUDA
            and event.id in launch_id_set
        ):
            kernels_by_launch[event.id].append(event)

    replay_times: list[Timing] = []
    kernel_totals: dict[tuple[str, str], list[float]] = {}
    iterations_seen: list[int] = []
    for launch_id in launch_ids:
        stage_times = {"dispatch": 0.0, "moe": 0.0, "combine": 0.0}
        kernels = sorted(
            kernels_by_launch[launch_id],
            key=lambda event: (event.time_range.start, event.time_range.end),
        )
        # A graph may contain many dispatch -> MoE -> combine iterations, so
        # walk the kernels as a state machine instead of assuming one boundary
        # pair. An unclassified kernel belongs to MoE only when it follows
        # dispatch; anything after combine starts the next iteration.
        stage = None
        iterations = 0
        for event in kernels:
            marker = classify_profiled_kernel(event.name)
            if marker == "dispatch":
                if stage != "dispatch":
                    iterations += 1
                stage = "dispatch"
            elif marker == "combine":
                stage = "combine"
            elif stage == "dispatch":
                stage = "moe"
            elif stage is None:
                kernel_names = "\n".join(f"  {item.name}" for item in kernels)
                raise RuntimeError(
                    "Torch Profiler saw a kernel before any dispatch kernel in "
                    f"a graph replay:\n{kernel_names}"
                )
            stage_times[stage] += float(event.self_device_time_total)
            entry = kernel_totals.setdefault((stage, event.name), [0.0, 0.0])
            entry[0] += 1.0
            entry[1] += float(event.self_device_time_total)
        if iterations == 0 or stage != "combine":
            kernel_names = "\n".join(f"  {item.name}" for item in kernels)
            raise RuntimeError(
                "Torch Profiler could not identify dispatch/combine boundaries "
                f"for a graph replay:\n{kernel_names}"
            )
        iterations_seen.append(iterations)
        replay_times.append(
            Timing(
                dispatch_us=stage_times["dispatch"] / iterations,
                moe_us=stage_times["moe"] / iterations,
                combine_us=stage_times["combine"] / iterations,
                e2e_us=sum(stage_times.values()) / iterations,
            )
        )

    if len(set(iterations_seen)) != 1:
        raise RuntimeError(
            f"graph replays contained differing iteration counts: {sorted(set(iterations_seen))}"
        )

    count = len(replay_times)
    samples = count * iterations_seen[0]
    stage_order = {"dispatch": 0, "moe": 1, "combine": 2}
    kernel_stats = [
        KernelStat(
            stage=stage,
            name=name,
            launches_per_iteration=totals[0] / samples,
            total_us_per_iteration=totals[1] / samples,
        )
        for (stage, name), totals in kernel_totals.items()
    ]
    kernel_stats.sort(
        key=lambda stat: (stage_order[stat.stage], -stat.total_us_per_iteration)
    )
    timing = Timing(
        dispatch_us=sum(item.dispatch_us for item in replay_times) / count,
        moe_us=sum(item.moe_us for item in replay_times) / count,
        combine_us=sum(item.combine_us for item in replay_times) / count,
        e2e_us=sum(item.e2e_us for item in replay_times) / count,
    )
    return timing, kernel_stats


def profile_graph_replays(
    captured: CapturedGraph,
    replays: int,
    device: torch.device,
    sync_group: dist.ProcessGroup,
) -> Timing:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    dist.barrier(group=sync_group)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        acc_events=True,
    ) as profiler:
        # Kineto initialization time varies by process. Align after every rank's
        # profiler is active so the first communication kernel does not measure
        # another rank's profiler startup delay.
        dist.barrier(group=sync_group)
        for _ in range(replays):
            captured.graph.replay()
        torch.cuda.synchronize()

    dist.barrier(group=sync_group)
    local_timing, kernel_stats = summarize_profiled_replays(profiler, replays)
    local_stages = [
        local_timing.dispatch_us,
        local_timing.moe_us,
        local_timing.combine_us,
        local_timing.e2e_us,
    ]
    dispatch_us, moe_us, combine_us, e2e_us = reduce_mean(local_stages, device)
    lo, hi = reduce_min_max(local_stages, device)
    per_rank = gather_per_rank(local_stages, device)
    if dist.get_rank() == 0:
        print(
            "Torch Profiler CUDA kernel time per MoE iteration: "
            f"dispatch={dispatch_us:.1f}us, MoE={moe_us:.1f}us, "
            f"combine={combine_us:.1f}us, sum={e2e_us:.1f}us",
            flush=True,
        )
        print(
            "Per-rank spread (min .. max across ranks): "
            f"dispatch={lo[0]:.1f}..{hi[0]:.1f}us (spread {hi[0] - lo[0]:.1f}), "
            f"MoE={lo[1]:.1f}..{hi[1]:.1f}us (spread {hi[1] - lo[1]:.1f}), "
            f"combine={lo[2]:.1f}..{hi[2]:.1f}us (spread {hi[2] - lo[2]:.1f}), "
            f"sum={lo[3]:.1f}..{hi[3]:.1f}us (spread {hi[3] - lo[3]:.1f})",
            flush=True,
        )
        print_per_rank_stages(per_rank)
        print_kernel_table(kernel_stats, local_timing)
    return Timing(
        dispatch_us=dispatch_us,
        moe_us=moe_us,
        combine_us=combine_us,
        e2e_us=e2e_us,
    )


def print_per_rank_stages(per_rank: list[list[float]]) -> None:
    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE", str(dist.get_world_size()))
    )
    rows = [["rank", "node", "dispatch", "MoE", "combine", "sum"]]
    for rank, (dispatch, moe, combine, total) in enumerate(per_rank):
        rows.append(
            [
                str(rank),
                str(rank // local_world_size),
                f"{dispatch:.1f}",
                f"{moe:.1f}",
                f"{combine:.1f}",
                f"{total:.1f}",
            ]
        )
    print("\nPer-rank stage time per MoE iteration (us):", flush=True)
    print(format_markdown_table(rows), flush=True)


def print_kernel_table(kernel_stats: list[KernelStat], timing: Timing) -> None:
    total = timing.e2e_us
    rows = [["stage", "kernel", "launches/iter", "us/iter", "% of total"]]
    for stat in kernel_stats:
        share = 100.0 * stat.total_us_per_iteration / total if total > 0 else 0.0
        rows.append(
            [
                stat.stage,
                stat.name,
                f"{stat.launches_per_iteration:.1f}",
                f"{stat.total_us_per_iteration:.2f}",
                f"{share:.1f}",
            ]
        )
    print(
        "\nRank-0 per-kernel CUDA time per MoE iteration "
        "(self device time, averaged over all profiled iterations):",
        flush=True,
    )
    print(format_markdown_table(rows), flush=True)


def format_markdown_table(rows: list[list[str]]) -> str:
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]

    def format_row(row: list[str]) -> str:
        return (
            "| "
            + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            + " |"
        )

    header = format_row(rows[0])
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([header, separator, *(format_row(row) for row in rows[1:])])


def build_pipeline(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    num_tokens: int,
    weights: ExpertWeights,
    mscclpp_comm_group: Any | None,
    device: torch.device,
) -> Any:
    if args.backend == "flashinfer":
        return FlashInferPipeline(
            args=args,
            rank=rank,
            world_size=world_size,
            num_tokens=num_tokens,
            weights=weights,
            group=dist.group.WORLD,
        )
    assert mscclpp_comm_group is not None
    return MscclppPipeline(
        args=args,
        rank=rank,
        world_size=world_size,
        num_tokens=num_tokens,
        weights=weights,
        comm_group=mscclpp_comm_group,
        device=device,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    token_counts = parse_token_counts(args.tokens_per_rank)
    rank, world_size, cpu_group = initialize_distributed()
    validate_args(args, world_size)
    device = torch.device("cuda", torch.cuda.current_device())
    mscclpp_comm_group = None

    try:
        import flashinfer

        mscclpp = None
        if args.backend == "mscclpp":
            import mscclpp as imported_mscclpp

            mscclpp = imported_mscclpp
            mscclpp_comm_group = mscclpp.CommGroup(
                torch_group=cpu_group, rank=rank, size=world_size
            )
        weights = make_local_weights(args, rank, world_size, device)
        torch.cuda.synchronize()
        dist.barrier()

        if rank == 0:
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
            weight_gib = (weights.w13.nbytes + weights.w2.nbytes) / (1024**3)
            versions = f"flashinfer={flashinfer.__version__}"
            if mscclpp is not None:
                versions += f", mscclpp={mscclpp.__version__}"
            print(
                "Configuration: "
                f"world_size={world_size}, nodes={world_size // local_world_size}, "
                f"backend={args.backend}, dtype=bf16, "
                f"hidden={args.hidden_size}, intermediate={args.intermediate_size}, "
                f"experts={args.num_experts}, top_k={args.top_k}, "
                f"tokens_per_rank={token_counts}, "
                f"warmup_replays={args.warmup_replays}, "
                f"graph_replays={args.graph_replays}, "
                f"iters_per_graph={args.iters_per_graph}, "
                f"torch_profiler={args.torch_profiler}, "
                f"moe_autotune={not args.skip_moe_autotune}"
                f"{'' if args.skip_moe_autotune else (' (per-rank)' if args.per_rank_moe_autotune else ' (shared from rank 0)')}\n"
                f"Versions: {versions}\n"
                f"Rank-local expert weights: {weight_gib:.2f} GiB\n"
                "Reported times are averaged across ranks.\n",
                flush=True,
            )

        table = (
            [
                [
                    "tokens/rank",
                    "global tokens",
                    "backend",
                    "dispatch kernel (us)",
                    "MoE kernel (us)",
                    "combine kernel (us)",
                    "kernel sum (us)",
                    "global tok/s (kernel)",
                    "graph/eager max abs",
                ]
            ]
            if args.torch_profiler
            else [
                [
                    "tokens/rank",
                    "global tokens",
                    "backend",
                    "E2E (us)",
                    "global tok/s",
                    "graph/eager max abs",
                ]
            ]
        )

        for num_tokens in token_counts:
            input_samples = [
                make_inputs(args, rank, num_tokens, device, sample_index)
                for sample_index in range(args.iters_per_graph)
            ]
            pipeline = build_pipeline(
                args, rank, world_size, num_tokens, weights, mscclpp_comm_group, device
            )

            torch.cuda.synchronize()
            dist.barrier()
            if not args.skip_moe_autotune:
                if rank == 0:
                    print(
                        "Autotuning FlashInfer CUTLASS MoE for "
                        f"{world_size * num_tokens} dispatched rows...",
                        flush=True,
                    )
                autotune_moe(
                    pipeline=pipeline,
                    inputs=input_samples[0],
                    dispatched_tokens=world_size * num_tokens,
                    sync_group=cpu_group,
                    per_rank=args.per_rank_moe_autotune,
                )

            for inputs in input_samples:
                eager_output = run_once(pipeline, inputs)
            eager_output = eager_output.clone()
            torch.cuda.synchronize()
            dist.barrier()
            finite = torch.tensor(
                int(torch.isfinite(eager_output).all()),
                dtype=torch.int32,
                device=device,
            )
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite.item()):
                raise AssertionError(f"{pipeline.name} eager output has NaN or Inf")

            captured = capture_graph(
                pipeline=pipeline,
                input_samples=input_samples,
                sync_group=cpu_group,
            )
            captured.graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            graph_max_abs = assert_outputs_close(
                reference=eager_output,
                candidate=captured.output.clone(),
                rtol=args.rtol,
                atol=args.atol,
                device=device,
                label=f"{pipeline.name} eager and CUDA graph",
            )
            del eager_output

            warm_up_graph(
                captured=captured,
                warmup_replays=args.warmup_replays,
                sync_group=cpu_group,
            )

            global_tokens = num_tokens * world_size
            if args.torch_profiler:
                timing = profile_graph_replays(
                    captured=captured,
                    replays=args.graph_replays,
                    device=device,
                    sync_group=cpu_group,
                )
                if rank == 0:
                    table.append(
                        [
                            str(num_tokens),
                            str(global_tokens),
                            pipeline.name,
                            f"{timing.dispatch_us:.1f}",
                            f"{timing.moe_us:.1f}",
                            f"{timing.combine_us:.1f}",
                            f"{timing.e2e_us:.1f}",
                            f"{global_tokens * 1_000_000 / timing.e2e_us:,.0f}",
                            f"{graph_max_abs:.3g}",
                        ]
                    )
            else:
                e2e_us = time_graph_replays(
                    captured=captured,
                    replays=args.graph_replays,
                    device=device,
                    sync_group=cpu_group,
                )
                if rank == 0:
                    table.append(
                        [
                            str(num_tokens),
                            str(global_tokens),
                            pipeline.name,
                            f"{e2e_us:.1f}",
                            f"{global_tokens * 1_000_000 / e2e_us:,.0f}",
                            f"{graph_max_abs:.3g}",
                        ]
                    )

            torch.cuda.synchronize()
            captured.graph.reset()
            pipeline.close()
            del pipeline, captured, input_samples
            gc.collect()
            torch.cuda.empty_cache()
            synchronize_stream_and_ranks(mscclpp_comm_group, cpu_group)

        if rank == 0:
            print()
            print(format_markdown_table(table))
        synchronize_stream_and_ranks(mscclpp_comm_group, cpu_group)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
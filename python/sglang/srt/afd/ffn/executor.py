"""FFN-side ordered transport and execution for one AFD session."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, NamedTuple

import torch

from sglang.srt.afd.afd_handshake import AFDGraphCapturePlan, AFDHandshakeResult
from sglang.srt.afd.afd_sync import AFDSyncMsg
from sglang.srt.afd.comm import AFDCommBackend
from sglang.srt.afd.dispatcher import AFDFixedBufferPool, afd_work_generation
from sglang.srt.afd.ffn.compute import AFDFFNCompute, _AFDFFNDecodeStepGraph
from sglang.srt.afd.layers import AFDMoELayer
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.layers.dp_attention import set_is_extend_in_batch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.base_cuda_graph_runner import freeze_gc
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
from sglang.srt.runtime_context import get_disagg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class _SourceSegment(NamedTuple):
    peer_global_rank: int
    capacity: int
    rows: slice


def _prepare_source_segments(
    sources: Iterable[tuple[int, int]],
) -> tuple[_SourceSegment, ...]:
    segments = []
    offset = 0
    for peer_global_rank, capacity in sources:
        next_offset = offset + capacity
        segments.append(
            _SourceSegment(peer_global_rank, capacity, slice(offset, next_offset))
        )
        offset = next_offset
    return tuple(segments)


class AFDFFNExecutor:
    """Receive AFD work, run experts, and return results."""

    def __init__(
        self,
        scheduler: Scheduler,
        comm: AFDCommBackend,
        compute: AFDFFNCompute,
        handshake: AFDHandshakeResult,
    ):
        self._comm = comm
        self._compute = compute
        self._device = torch.device(self._comm.config.device)
        self._dtype = scheduler.tp_worker.model_runner.dtype
        self._layers = compute.layers
        self._tp_group = scheduler.tp_group
        self._device_module = scheduler.device_module
        self._rank = self._comm.config.instance_rank
        self._num_ranks = self._comm.config.num_ffn_ranks
        self._scattered = get_disagg().afd_ffn_moe_backend == "deepep"
        self._attn_ranks = self._comm.config.attn_ranks_for_ffn(
            self._rank,
            scattered=self._scattered,
        )
        self._decode_step_graph: _AFDFFNDecodeStepGraph | None = None
        if compute.device != self._device:
            raise RuntimeError(
                f"AFD ffn compute device {compute.device} does not match "
                f"communication device {self._device}"
            )
        if (
            self._tp_group.world_size != self._num_ranks
            or self._tp_group.rank_in_group != self._rank
        ):
            raise RuntimeError(
                "AFD ffn rank topology does not match the local TP group: "
                f"comm={self._rank}/{self._num_ranks}, "
                f"tp_group={self._tp_group.rank_in_group}/"
                f"{self._tp_group.world_size}"
            )

        self._decode_edge_token_buckets = tuple(handshake.peer.decode_token_buckets)
        decode_token_buckets = self._decode_edge_token_buckets
        if self._scattered:
            source_count = max(len(self._attn_ranks), 1)
            decode_token_buckets = [
                capacity * source_count for capacity in decode_token_buckets
            ]
        decode_pool_token_buckets = sorted(
            set(decode_token_buckets) | set(handshake.local.decode_graph_token_buckets)
        )
        self._decode_pool = AFDFixedBufferPool(
            layers=self._layers,
            token_buckets=decode_pool_token_buckets,
            dtype=self._dtype,
            device=self._device,
            role="ffn",
            with_router_logits=True,
            num_slots=1,
        )

    def capture_communication_graphs(self, plan: AFDGraphCapturePlan) -> None:
        if not plan.enabled:
            return
        attn_phase = [
            (
                tuple(variant.child_wire_capacities),
                tuple(
                    capacity
                    for capacity in variant.child_compute_capacities
                    if capacity is not None
                ),
            )
            for variant in plan.attn_graph_variants
        ]
        graph_compute_capacities = [
            capacity
            for _, compute_capacities in attn_phase
            for capacity in compute_capacities
        ]
        if self._rank == 0:
            logger.info(
                "Capture AFD ffn communication graphs begin: wire_to_compute=%s",
                list(plan.ffn_graph_variants),
            )
        self._compute._prepare_graph_capture(max(graph_compute_capacities))
        self._decode_step_graph = _AFDFFNDecodeStepGraph(
            self._compute,
            self._run_integrated_graph_step,
        )
        self._comm.startup_barrier()
        self._capture_variants(attn_phase)
        self._comm.startup_barrier()
        if self._rank == 0:
            logger.info(
                "AFD ffn communication graphs ready: full_steps=%d", len(attn_phase)
            )

    def _decode_source_segments(
        self, edge_wire_capacity: int
    ) -> tuple[_SourceSegment, ...]:
        return _prepare_source_segments(
            (
                self._comm.config.attn_global_rank(attn_rank),
                edge_wire_capacity,
            )
            for attn_rank in self._attn_ranks
        )

    def _recv_source_inputs(
        self,
        source_segments: tuple[_SourceSegment, ...],
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        for peer_global_rank, capacity, rows in source_segments:
            if capacity > 0:
                self._comm.recv_from(
                    [hidden_states[rows], topk_ids[rows], topk_weights[rows]],
                    peer_global_rank,
                )

    def _run_integrated_graph_step(
        self,
        wire_capacities: tuple[int, ...],
        compute_capacities: tuple[int, ...],
    ) -> None:
        edge_wire_capacity = wire_capacities[0]
        compute_capacity = compute_capacities[0]
        source_segments = self._decode_source_segments(edge_wire_capacity)
        source_count = len(source_segments)
        wire_capacity = source_count * edge_wire_capacity
        staging_capacity = max(source_count, 1) * edge_wire_capacity
        direct_nccl_io = source_count == 1 and wire_capacity <= compute_capacity
        work_capacity = compute_capacity if direct_nccl_io else staging_capacity
        for layer_index, layer in enumerate(self._layers):
            work = self._decode_pool.acquire(
                slot_id=0,
                layer_id=layer.layer_id,
                generation=afd_work_generation(0, layer_index),
                capacity=work_capacity,
                input_tokens=wire_capacity,
            )
            self._recv_source_inputs(
                source_segments,
                work.hidden_states,
                work.topk_ids,
                work.topk_weights,
            )
            input_capacity = compute_capacity if direct_nccl_io else wire_capacity
            output = self._compute.run_integrated_layer(
                layer_index,
                work.hidden_states[:input_capacity],
                work.topk_ids[:input_capacity],
                work.topk_weights[:input_capacity],
                wire_capacity,
                compute_capacity,
                stage_inputs=not direct_nccl_io,
            )
            if direct_nccl_io and output.is_contiguous():
                peer_global_rank, capacity, _ = source_segments[0]
                self._comm.send_to(
                    [output[:capacity]],
                    peer_global_rank,
                )
                self._decode_pool.release(work)
                continue
            work.output[:wire_capacity].copy_(output[:wire_capacity])
            for peer_global_rank, capacity, rows in source_segments:
                self._comm.send_to(
                    [work.output[rows]],
                    peer_global_rank,
                )
            self._decode_pool.release(work)

    def _capture_variants(
        self,
        variants: list[tuple[tuple[int, ...], tuple[int, ...]]],
    ) -> None:
        assert self._decode_step_graph is not None
        with (
            torch.inference_mode(),
            model_capture_mode(),
            freeze_gc(self._compute.enable_cudagraph_gc),
            graph_capture() as ctx,
        ):
            with self._decode_step_graph.capture_session(ctx.stream):
                for wire_capacities, compute_capacities in variants:
                    self._decode_step_graph.capture_variant(
                        wire_capacities,
                        compute_capacities,
                    )
        self._device_module.synchronize()
        self._tp_group.barrier()

    @torch.inference_mode()
    def run_step(self, msg: AFDSyncMsg) -> None:
        dp_capacity = self._dp_capacity(msg)
        total_tokens = sum(msg.tokens_per_rank)
        forward_mode = ForwardMode(msg.forward_mode)
        if forward_mode.is_idle():
            if total_tokens != 0 or msg.wire_capacity != 0:
                raise RuntimeError(
                    "AFD idle step must have zero capacity: "
                    f"tokens={total_tokens}, wire={msg.wire_capacity}"
                )
            return
        if total_tokens == 0:
            if msg.wire_capacity != 0:
                raise RuntimeError(
                    f"AFD idle step has nonzero wire capacity {msg.wire_capacity}"
                )
            return
        if forward_mode.is_decode():
            if msg.wire_capacity not in self._decode_edge_token_buckets:
                raise RuntimeError(
                    "AFD wire capacity is not a negotiated decode bucket: "
                    f"{msg.wire_capacity}"
                )
            source_segments = self._decode_source_segments(msg.wire_capacity)
        else:
            source_segments = _prepare_source_segments(
                (
                    self._comm.config.attn_global_rank(attn_rank),
                    (
                        msg.wire_capacity
                        if not self._scattered
                        else msg.tokens_per_rank[attn_rank]
                    ),
                )
                for attn_rank in self._attn_ranks
            )
        wire_capacity = sum(segment.capacity for segment in source_segments)
        set_is_extend_in_batch(forward_mode.is_extend())
        if forward_mode.is_decode():
            if dp_capacity > wire_capacity:
                raise RuntimeError(
                    f"AFD DP capacity {dp_capacity} exceeds wire capacity "
                    f"{wire_capacity}"
                )
        elif wire_capacity != dp_capacity:
            raise RuntimeError(
                "AFD prefill requires exact-size transport: "
                f"dp={dp_capacity}, wire={wire_capacity}"
            )

        if forward_mode.is_decode() and msg.communication_graph:
            if self._decode_step_graph is not None and self._decode_step_graph.replay(
                (msg.wire_capacity,)
            ):
                return
            raise RuntimeError(
                "AFD ffn has no communication graph for wire capacity "
                f"{msg.wire_capacity}"
            )
        self._run_eager_step(msg.step_id, forward_mode, source_segments)

    def _run_eager_step(
        self,
        step_id: int,
        forward_mode: ForwardMode,
        source_segments: tuple[_SourceSegment, ...],
    ) -> None:
        wire_capacity = sum(segment.capacity for segment in source_segments)
        fixed_shape = forward_mode.is_decode()
        allow_compute_graph = (
            fixed_shape and not self._scattered and self._decode_step_graph is None
        )

        for layer_index, layer in enumerate(self._layers):
            work = None
            if fixed_shape and wire_capacity > 0:
                work = self._decode_pool.acquire(
                    slot_id=0,
                    layer_id=layer.layer_id,
                    generation=afd_work_generation(step_id, layer_index),
                    capacity=wire_capacity,
                    input_tokens=wire_capacity,
                )
                hidden_states = work.hidden_states
                topk_ids = work.topk_ids
                topk_weights = work.topk_weights
                if work.router_logits is None:
                    raise RuntimeError("AFD ffn slot has no router logits buffer")
                router_logits = work.router_logits
            else:
                hidden_states, topk_ids, topk_weights = self._make_request_buffers(
                    layer, wire_capacity
                )
                router_logits = torch.empty(
                    (wire_capacity, layer.num_experts),
                    dtype=torch.float32,
                    device=self._device,
                )

            self._recv_source_inputs(
                source_segments, hidden_states, topk_ids, topk_weights
            )

            output = self._compute.run(
                layer_index,
                hidden_states,
                topk_ids,
                topk_weights,
                router_logits,
                allow_graph=allow_compute_graph,
            )
            if work is not None and not self._scattered:
                work.output.copy_(output)
                output = work.output

            for peer_global_rank, capacity, rows in source_segments:
                if capacity > 0:
                    self._comm.send_to(
                        [output[rows].contiguous()],
                        peer_global_rank,
                    )

            if work is not None:
                self._decode_pool.release(work)

    def _dp_capacity(self, msg: AFDSyncMsg) -> int:
        num_attn_ranks = self._comm.config.num_attn_ranks
        if len(msg.tokens_per_rank) != num_attn_ranks:
            raise RuntimeError(
                f"AFD sync tokens_per_rank has {len(msg.tokens_per_rank)} entries, "
                f"but the attn instance has {num_attn_ranks} ranks"
            )
        if self._scattered:
            return sum(msg.tokens_per_rank[attn_rank] for attn_rank in self._attn_ranks)
        return sum(msg.tokens_per_rank)

    def _make_request_buffers(
        self, layer: AFDMoELayer, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = torch.empty(
            (num_tokens, layer.hidden_size), dtype=self._dtype, device=self._device
        )
        topk_ids = torch.empty(
            (num_tokens, layer.top_k), dtype=torch.int32, device=self._device
        )
        topk_weights = torch.empty(
            (num_tokens, layer.top_k), dtype=torch.float32, device=self._device
        )
        return hidden_states, topk_ids, topk_weights

"""FFN-side ordered transport and execution for one AFD session."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, NamedTuple

import torch

from sglang.srt.afd.afd_config import resolve_overlap_sm_partition
from sglang.srt.afd.afd_handshake import AFDGraphCapturePlan, AFDHandshakeResult
from sglang.srt.afd.afd_sync import AFDSyncMsg
from sglang.srt.afd.comm import AFDCommBackend
from sglang.srt.afd.dispatcher import (
    AFDFixedBufferPool,
    AFDWorkHandle,
    afd_work_generation,
    is_afd_fixed_shape_mode,
)
from sglang.srt.afd.ffn.compute import AFDFFNCompute, _AFDFFNDecodeStepGraph
from sglang.srt.afd.layers import AFDMoELayer
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.layers.compute_sm import compute_sm_budget
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


@dataclass(frozen=True)
class _PreparedDecodeChild:
    slot_id: int
    source_segments: tuple[_SourceSegment, ...]
    local_wire_capacity: int
    staging_capacity: int


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
        self._session_id = handshake.session_id
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
        self._decode_num_microbatches = handshake.peer.decode_num_microbatches
        staged_overlap = self._decode_num_microbatches > 1
        self._communication_stream = (
            torch.cuda.Stream(device=self._device) if staged_overlap else None
        )
        self._response_stream = self._communication_stream
        self._request_ready = [
            torch.cuda.Event()
            for _ in range(self._decode_num_microbatches if staged_overlap else 0)
        ]
        self._response_ready = [
            torch.cuda.Event()
            for _ in range(self._decode_num_microbatches if staged_overlap else 0)
        ]
        self._compute_num_sms: int | None = None

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

        if self._decode_num_microbatches > 1:
            overlap_sm_budget = handshake.local.afd_comm_sm_budget
            assert overlap_sm_budget is not None
            total_num_sms = torch.cuda.get_device_properties(
                self._device
            ).multi_processor_count
            (
                self._compute_num_sms,
                communication_num_sms,
            ) = resolve_overlap_sm_partition(
                total_num_sms=total_num_sms,
                overlap_sm_budget=overlap_sm_budget,
            )
            self._compute._graph_compute_num_sms = self._compute_num_sms
            if self._rank == 0:
                logger.info(
                    "AFD ffn overlap SM partition: backend=%s compute=%d "
                    "communication=%d",
                    handshake.local.afd_backend,
                    self._compute_num_sms,
                    communication_num_sms,
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
            num_slots=self._decode_num_microbatches,
        )

        logger.info(
            "AFD ffn executor ready: session=%d rank=%d/%d moe_layers=%d "
            "first_layer=%d last_layer=%d device=%s dtype=%s",
            self._session_id,
            self._rank,
            self._num_ranks,
            len(self._layers),
            self._layers[0].layer_id,
            self._layers[-1].layer_id,
            self._device,
            self._dtype,
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
        if len(wire_capacities) > 1:
            children = self._prepare_decode_children_for_capacities(wire_capacities)
            self._run_overlap_step(0, children, compute_capacities)
            return
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
        fixed_shape = is_afd_fixed_shape_mode(forward_mode)
        if msg.child_wire_capacities and not fixed_shape:
            raise RuntimeError("AFD child plan requires DECODE")

        decode_children = self._prepare_decode_children(msg) if fixed_shape else ()
        source_segments = (
            decode_children[0].source_segments
            if len(decode_children) == 1
            else _prepare_source_segments(
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
        )
        wire_capacity = (
            sum(child.local_wire_capacity for child in decode_children)
            if decode_children
            else sum(segment.capacity for segment in source_segments)
        )
        # Fixed-shape F execution is phase-neutral; capacity alone selects the
        # paired graph for both FULL and SCATTERED layouts.
        set_is_extend_in_batch(forward_mode.is_extend() and not fixed_shape)
        if fixed_shape:
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

        if len(decode_children) > 1:
            child_wire_capacities = msg.child_wire_capacities or (msg.wire_capacity,)
            with torch.inference_mode():
                if msg.communication_graph:
                    if (
                        self._decode_step_graph is not None
                        and self._decode_step_graph.replay(child_wire_capacities)
                    ):
                        return
                    raise RuntimeError(
                        "AFD ffn has no microbatch communication graph for child "
                        f"wire capacities {child_wire_capacities}"
                    )
                self._run_overlap_step(msg.step_id, decode_children)
            return

        if fixed_shape and msg.communication_graph:
            with torch.inference_mode():
                if (
                    self._decode_step_graph is not None
                    and self._decode_step_graph.replay((msg.wire_capacity,))
                ):
                    return
                raise RuntimeError(
                    "AFD ffn has no communication graph for wire capacity "
                    f"{msg.wire_capacity}"
                )

        with torch.inference_mode():
            self._run_eager_step(
                msg.step_id,
                forward_mode,
                source_segments,
            )

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

    def _prepare_decode_children(
        self, msg: AFDSyncMsg
    ) -> tuple[_PreparedDecodeChild, ...]:
        forward_mode = ForwardMode(msg.forward_mode)
        if not is_afd_fixed_shape_mode(forward_mode):
            raise RuntimeError("AFD fixed-shape children require DECODE")
        child_wire_capacities = msg.child_wire_capacities or (msg.wire_capacity,)
        num_children = len(child_wire_capacities)
        if msg.child_wire_capacities and num_children != self._decode_num_microbatches:
            raise RuntimeError(
                "AFD child plan does not match the configured microbatch count: "
                f"children={num_children}, "
                f"configured={self._decode_num_microbatches}"
            )
        children = self._prepare_decode_children_for_capacities(child_wire_capacities)
        if msg.child_wire_capacities and (
            sum(child_wire_capacities) != msg.wire_capacity
        ):
            raise RuntimeError(
                "AFD child capacities do not match the parent wire "
                f"capacity: children={child_wire_capacities}, "
                f"parent={msg.wire_capacity}"
            )
        dp_capacity = self._dp_capacity(msg)
        local_wire_capacity = sum(child.local_wire_capacity for child in children)
        if dp_capacity > local_wire_capacity:
            raise RuntimeError(
                f"AFD DP capacity {dp_capacity} exceeds local wire capacity "
                f"{local_wire_capacity}"
            )
        return children

    def _prepare_decode_children_for_capacities(
        self,
        child_wire_capacities: tuple[int, ...],
    ) -> tuple[_PreparedDecodeChild, ...]:
        num_children = len(child_wire_capacities)
        if num_children not in (1, self._decode_num_microbatches):
            raise RuntimeError(
                "AFD decode child count must be one or the configured "
                f"microbatch count {self._decode_num_microbatches}, got "
                f"{num_children}"
            )
        if any(capacity <= 0 for capacity in child_wire_capacities):
            raise RuntimeError(
                f"AFD decode child capacities must be positive: {child_wire_capacities}"
            )
        return tuple(
            self._prepare_decode_child(wire_capacity, slot_id)
            for slot_id, wire_capacity in enumerate(child_wire_capacities)
        )

    def _prepare_decode_child(
        self, edge_wire_capacity: int, slot_id: int = 0
    ) -> _PreparedDecodeChild:
        if edge_wire_capacity not in self._decode_edge_token_buckets:
            raise RuntimeError(
                "AFD wire capacity is not a negotiated decode bucket: "
                f"{edge_wire_capacity}"
            )
        source_segments = self._decode_source_segments(edge_wire_capacity)
        source_count = len(source_segments)
        return _PreparedDecodeChild(
            slot_id=slot_id,
            source_segments=source_segments,
            local_wire_capacity=sum(segment.capacity for segment in source_segments),
            staging_capacity=max(source_count, 1) * edge_wire_capacity,
        )

    def _enqueue_overlap_request(
        self,
        work: AFDWorkHandle,
        layer_id: int,
        child: _PreparedDecodeChild,
        request_stream: torch.cuda.Stream,
    ) -> None:
        with torch.cuda.stream(request_stream):
            self._recv_source_inputs(
                child.source_segments,
                work.hidden_states,
                work.topk_ids,
                work.topk_weights,
            )
            self._request_ready[child.slot_id].record(request_stream)

    def _run_overlap_step(
        self,
        step_id: int,
        children: tuple[_PreparedDecodeChild, ...],
        child_compute_capacities: tuple[int, ...] | None = None,
    ) -> None:
        with compute_sm_budget(self._compute_num_sms):
            self._run_overlap_step_impl(
                step_id,
                children,
                child_compute_capacities,
            )

    def _run_overlap_step_impl(
        self,
        step_id: int,
        children: tuple[_PreparedDecodeChild, ...],
        child_compute_capacities: tuple[int, ...] | None = None,
    ) -> None:
        assert self._communication_stream is not None
        assert self._response_stream is not None
        compute_stream = torch.cuda.current_stream(self._device)
        request_stream = self._communication_stream
        response_stream = self._response_stream
        request_stream.wait_stream(compute_stream)
        num_children = len(children)
        if num_children <= 1 or num_children != self._decode_num_microbatches:
            raise RuntimeError(
                "AFD overlap child count does not match the configured "
                f"microbatch count: children={num_children}, "
                f"configured={self._decode_num_microbatches}"
            )
        direct_integrated_inputs = tuple(
            len(child.source_segments) == 1
            and child_compute_capacities is not None
            and child.local_wire_capacity <= child_compute_capacities[child_index]
            for child_index, child in enumerate(children)
        )
        slot_work: list[AFDWorkHandle] = []

        first_layer = self._layers[0]
        for child_index, child in enumerate(children):
            work_capacity = (
                child_compute_capacities[child_index]
                if direct_integrated_inputs[child_index]
                else child.staging_capacity
            )
            work = self._decode_pool.acquire(
                slot_id=child.slot_id,
                layer_id=first_layer.layer_id,
                generation=afd_work_generation(step_id, child_index),
                capacity=work_capacity,
                input_tokens=child.local_wire_capacity,
            )
            self._enqueue_overlap_request(
                work,
                first_layer.layer_id,
                child,
                request_stream,
            )
            slot_work.append(work)

        for layer_index, layer in enumerate(self._layers):
            for child_index in range(num_children):
                work_index = layer_index * num_children + child_index
                work = slot_work[child_index]
                child = children[child_index]
                compute_stream.wait_event(self._request_ready[child.slot_id])
                if child_compute_capacities is None:
                    if work.router_logits is None:
                        raise RuntimeError("AFD ffn slot has no router logits buffer")
                    output = self._compute.run(
                        layer_index,
                        work.hidden_states[: child.local_wire_capacity],
                        work.topk_ids[: child.local_wire_capacity],
                        work.topk_weights[: child.local_wire_capacity],
                        work.router_logits[: child.local_wire_capacity],
                        allow_graph=False,
                        defer_output_scaling=True,
                    )
                else:
                    direct_input = direct_integrated_inputs[child_index]
                    compute_capacity = child_compute_capacities[child_index]
                    input_capacity = (
                        compute_capacity if direct_input else child.local_wire_capacity
                    )
                    output = self._compute.run_integrated_layer(
                        layer_index,
                        work.hidden_states[:input_capacity],
                        work.topk_ids[:input_capacity],
                        work.topk_weights[:input_capacity],
                        child.local_wire_capacity,
                        compute_capacity,
                        defer_output_scaling=True,
                        stage_inputs=not direct_input,
                    )
                self._response_ready[child.slot_id].record(compute_stream)
                # Keep allocator-owned compute output alive until response
                # staging on the response stream has consumed it.
                output.record_stream(response_stream)
                with torch.cuda.stream(response_stream):
                    response_stream.wait_event(self._response_ready[child.slot_id])
                    direct_output = (
                        len(child.source_segments) == 1 and output.is_contiguous()
                    )
                    if direct_output:
                        peer_global_rank, capacity, _ = child.source_segments[0]
                        response_output = output[:capacity]
                        self._compute.finalize_deferred_output_in_place(
                            layer_index, response_output
                        )
                        self._comm.send_to(
                            [response_output],
                            peer_global_rank,
                        )
                    else:
                        self._compute.stage_deferred_output(
                            layer_index,
                            work.output[: child.local_wire_capacity],
                            output[: child.local_wire_capacity],
                        )
                        for (
                            peer_global_rank,
                            capacity,
                            rows,
                        ) in child.source_segments:
                            self._comm.send_to(
                                [work.output[rows]],
                                peer_global_rank,
                            )
                self._decode_pool.release(work)

                next_layer_index = layer_index + 1
                if next_layer_index == len(self._layers):
                    continue
                next_layer = self._layers[next_layer_index]
                next_work_index = work_index + num_children
                next_work_capacity = (
                    child_compute_capacities[child_index]
                    if direct_integrated_inputs[child_index]
                    else child.staging_capacity
                )
                next_work = self._decode_pool.acquire(
                    slot_id=child.slot_id,
                    layer_id=next_layer.layer_id,
                    generation=afd_work_generation(step_id, next_work_index),
                    capacity=next_work_capacity,
                    input_tokens=child.local_wire_capacity,
                )
                self._enqueue_overlap_request(
                    next_work,
                    next_layer.layer_id,
                    child,
                    request_stream,
                )
                slot_work[child_index] = next_work

        compute_stream.wait_stream(request_stream)

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

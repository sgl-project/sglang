"""FFN-side execution of merged work from multiple AFD sessions."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Sequence

import torch

from sglang.srt.afd.afd_sync import AFDSyncMsg
from sglang.srt.afd.ffn.executor import AFDFFNExecutor, _PreparedDecodeChild
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.layers.compute_sm import compute_sm_budget
from sglang.srt.layers.dp_attention import set_is_extend_in_batch
from sglang.srt.model_executor.runner.base_cuda_graph_runner import freeze_gc
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

logger = logging.getLogger(__name__)


@dataclass
class _CohortBuffers:
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    router_logits: torch.Tensor


_PreparedChildMembers = tuple[tuple[AFDFFNExecutor, _PreparedDecodeChild], ...]
_PreparedChildren = tuple[_PreparedChildMembers, ...]


class AFDFFNCohortExecutor:
    """Run one merged decode step for a fixed set of AFD sessions."""

    def __init__(self, executors: Sequence[AFDFFNExecutor]):
        self._executors = tuple(executors)
        self._compute = self._executors[0]._compute
        self._step_graph_backend: FullCudaGraphBackend | None = None
        self._step_graph_compute_capacities: dict[tuple[int, ...], tuple[int, ...]] = {}
        max_tokens = sum(
            max(executor._decode_token_buckets) for executor in self._executors
        )
        if self._compute._deepep_dispatch_limit is not None:
            # The selector bounds every SCATTERED cohort by this worst-rank cap.
            max_tokens = min(
                max_tokens,
                self._compute._deepep_dispatch_limit,
            )
        first = self._executors[0]
        self._nccl_overlap_depth = max(
            (
                executor._decode_num_microbatches
                for executor in self._executors
                if executor._decode_transport is None
            ),
            default=1,
        )
        self._uses_nccl_overlap = self._nccl_overlap_depth > 1
        num_buffer_slots = self._nccl_overlap_depth
        self._buffer_slots: list[dict[tuple[int, int, int], _CohortBuffers]] = []
        schemas = {
            (layer.hidden_size, layer.top_k, layer.num_experts)
            for layer in self._compute.layers
        }
        for _ in range(num_buffer_slots):
            slot_buffers = {}
            for hidden_size, top_k, num_experts in schemas:
                slot_buffers[(hidden_size, top_k, num_experts)] = _CohortBuffers(
                    hidden_states=torch.empty(
                        (max_tokens, hidden_size),
                        dtype=first._dtype,
                        device=first._device,
                    ),
                    topk_ids=torch.empty(
                        (max_tokens, top_k),
                        dtype=torch.int32,
                        device=first._device,
                    ),
                    topk_weights=torch.empty(
                        (max_tokens, top_k),
                        dtype=torch.float32,
                        device=first._device,
                    ),
                    router_logits=torch.empty(
                        (max_tokens, num_experts),
                        dtype=torch.float32,
                        device=first._device,
                    ),
                )
            self._buffer_slots.append(slot_buffers)
        self._buffers = self._buffer_slots[0]
        self._communication_stream = (
            torch.cuda.Stream(device=first._device) if self._uses_nccl_overlap else None
        )
        self._request_ready = (
            [torch.cuda.Event() for _ in range(self._nccl_overlap_depth)]
            if self._uses_nccl_overlap
            else []
        )
        self._response_ready = (
            [torch.cuda.Event() for _ in range(self._nccl_overlap_depth)]
            if self._uses_nccl_overlap
            else []
        )

    def capture_communication_graphs(self) -> None:
        if any(executor._decode_transport is None for executor in self._executors):
            return
        if self._compute._scattered and self._compute._deepep_dispatch_limit is None:
            return
        if (
            self._compute._scattered
            and self._compute._disable_padding
            and any(
                executor._comm.config.num_attn_ranks % executor._num_ranks
                for executor in self._executors
            )
        ):
            return

        # A whole-step graph binds every session arena and copy length. Keep
        # variants linear in bucket count by capturing only the full diagonal.
        common_child_plans = set(self._executors[0]._cohort_graph_child_plans)
        for executor in self._executors[1:]:
            common_child_plans.intersection_update(executor._cohort_graph_child_plans)
        compute_capacity_multiplier = (
            sum(executor._max_scattered_source_count for executor in self._executors)
            if self._compute._scattered
            else len(self._executors)
        )
        variants = []
        for child_wire_capacities in sorted(
            common_child_plans,
            reverse=True,
        ):
            required_compute_capacities = tuple(
                wire_capacity * compute_capacity_multiplier
                for wire_capacity in child_wire_capacities
            )
            compute_capacities = tuple(
                self._compute.graph_capacity_for(required_capacity)
                for required_capacity in required_compute_capacities
            )
            if all(capacity is not None for capacity in compute_capacities):
                variants.append(
                    (
                        child_wire_capacities,
                        tuple(
                            capacity
                            for capacity in compute_capacities
                            if capacity is not None
                        ),
                    )
                )
        if not variants:
            return

        self._step_graph_backend = FullCudaGraphBackend(
            self._compute,
            enable_memory_saver=self._compute.enable_memory_saver,
        )
        rank = self._executors[0]._rank
        if rank == 0:
            logger.info(
                "Capture AFD ffn cohort communication graphs begin: "
                "full_membership=true diagonal_children_to_compute=%s",
                variants,
            )
        tic = time.perf_counter()
        with (
            torch.inference_mode(),
            model_capture_mode(),
            freeze_gc(self._compute.enable_cudagraph_gc),
            graph_capture() as ctx,
        ):
            with self._step_graph_backend.capture_session(ctx.stream):
                for child_wire_capacities, compute_capacities in variants:
                    prepared_by_child = tuple(
                        tuple(
                            (
                                executor,
                                executor._prepare_decode_child(
                                    wire_capacity,
                                    child_index,
                                ),
                            )
                            for executor in self._executors
                        )
                        for child_index, wire_capacity in enumerate(
                            child_wire_capacities
                        )
                    )
                    key = (
                        "afd_ffn_cohort_step",
                        child_wire_capacities,
                        compute_capacities,
                    )
                    local_wire_capacities = tuple(
                        sum(child.local_wire_capacity for _, child in prepared)
                        for prepared in prepared_by_child
                    )

                    def reset_inputs() -> None:
                        self._compute.prepare_integrated_capture_inputs(
                            max(compute_capacities)
                        )

                    def capture_or_warmup_step() -> None:
                        # The graph backend reuses this callback for warmup;
                        # transport operations belong only to actual capture.
                        if torch.cuda.is_current_stream_capturing():
                            self._run_prepared_children(
                                prepared_by_child,
                                integrated_compute_capacities=compute_capacities,
                                capture_transport=True,
                            )
                        else:
                            self._warmup_compute_step(
                                local_wire_capacities,
                                compute_capacities,
                            )

                    if self._compute._deepep_adapter is not None:
                        set_is_extend_in_batch(False)
                        self._compute._deepep_adapter.capture(is_extend_in_batch=False)
                    reset_inputs()
                    self._step_graph_backend.capture_one(
                        key,
                        capture_or_warmup_step,
                        post_warmup_hook=reset_inputs,
                    )
                    self._step_graph_compute_capacities[child_wire_capacities] = (
                        compute_capacities
                    )

        self._executors[0]._device_module.synchronize()
        self._executors[0]._tp_group.barrier()
        if rank == 0:
            logger.info(
                "Capture AFD ffn cohort communication graphs end: "
                "elapsed=%.2f s graphs=%d",
                time.perf_counter() - tic,
                len(variants),
            )

    def can_cohort(self, session_id: int, msg: AFDSyncMsg) -> bool:
        executor = self._executors[session_id]
        return executor._can_run_cohort(msg)

    @property
    def cohort_capacity(self) -> int | None:
        return self._compute._deepep_dispatch_limit

    def cohort_cost(self, session_id: int, msg: AFDSyncMsg) -> tuple[int, ...]:
        child_wire_capacities = msg.effective_child_wire_capacities
        if not self._compute._scattered:
            return (0,) * len(child_wire_capacities)
        source_count = self._executors[session_id]._max_scattered_source_count
        return tuple(
            source_count * wire_capacity for wire_capacity in child_wire_capacities
        )

    def run_step(
        self,
        members: Sequence[tuple[int, AFDSyncMsg]],
    ) -> str:
        ordered_members = sorted(members, key=lambda member: member[0])
        merged_dispatch_capacities = None
        if self._compute._scattered:
            # Every F rank must select the same DeepEP graph capacity even when
            # asymmetric fan-in gives them different local token counts.
            costs = tuple(
                self.cohort_cost(session_id, msg) for session_id, msg in ordered_members
            )
            num_children = len(costs[0])
            if any(len(cost) != num_children for cost in costs):
                raise RuntimeError("AFD cohort mixed incompatible child plans")
            merged_dispatch_capacities = tuple(
                sum(cost[child_index] for cost in costs)
                for child_index in range(num_children)
            )
            dispatch_limit = self._compute._deepep_dispatch_limit
            if dispatch_limit is not None and any(
                capacity > dispatch_limit for capacity in merged_dispatch_capacities
            ):
                raise RuntimeError(
                    "AFD cohort selector exceeded the DeepEP dispatch limit: "
                    f"capacities={merged_dispatch_capacities}, "
                    f"limit={dispatch_limit}"
                )

        prepared_by_session = []
        for session_id, msg in ordered_members:
            executor = self._executors[session_id]
            prepared_by_session.append(
                (executor, executor._prepare_decode_children(msg))
            )
        num_children = len(prepared_by_session[0][1])
        if any(len(children) != num_children for _, children in prepared_by_session):
            raise RuntimeError("AFD cohort mixed incompatible child plans")
        prepared_by_child = tuple(
            tuple(
                (executor, children[child_index])
                for executor, children in prepared_by_session
            )
            for child_index in range(num_children)
        )
        child_wire_capacities_by_session = tuple(
            msg.effective_child_wire_capacities for _, msg in ordered_members
        )
        if (
            len(prepared_by_session) == len(self._executors)
            and len(set(child_wire_capacities_by_session)) == 1
            and self._replay_step_graph(child_wire_capacities_by_session[0])
        ):
            return "cohort_step"

        total_tokens = tuple(
            sum(child.local_wire_capacity for _, child in prepared)
            for prepared in prepared_by_child
        )
        required_compute_capacities = (
            merged_dispatch_capacities
            if merged_dispatch_capacities is not None
            else total_tokens
        )
        layer_graph_capacities = tuple(
            self._compute.graph_capacity_for(capacity)
            for capacity in required_compute_capacities
        )
        if (
            self._compute._scattered
            and self._compute._disable_padding
            and any(
                executor._comm.config.num_attn_ranks % executor._num_ranks
                for executor, _ in prepared_by_session
            )
        ):
            layer_graph_capacities = (None,) * num_children
        with torch.inference_mode():
            if num_children > 1 and self._uses_nccl_overlap:
                self._run_nccl_overlap_prepared_children(
                    prepared_by_child,
                    layer_graph_capacities,
                )
            else:
                self._run_prepared_children(
                    prepared_by_child,
                    layer_graph_capacities=layer_graph_capacities,
                )
        return (
            "compute"
            if all(capacity is not None for capacity in layer_graph_capacities)
            else "eager"
        )

    def _replay_step_graph(
        self,
        child_wire_capacities: tuple[int, ...],
    ) -> bool:
        compute_capacities = self._step_graph_compute_capacities.get(
            child_wire_capacities
        )
        if compute_capacities is None:
            return False
        assert self._step_graph_backend is not None
        key = (
            "afd_ffn_cohort_step",
            child_wire_capacities,
            compute_capacities,
        )
        if self._compute._deepep_adapter is not None:
            self._compute._deepep_adapter.replay()
        with self._step_graph_backend.replay_session():
            self._step_graph_backend.replay(key, static_forward_batch=None)
        return True

    def _warmup_compute_step(
        self,
        local_wire_capacities: tuple[int, ...],
        compute_capacities: tuple[int, ...],
    ) -> None:
        set_is_extend_in_batch(False)
        for layer_index, layer in enumerate(self._compute.layers):
            buffers = self._buffers[(layer.hidden_size, layer.top_k, layer.num_experts)]
            for local_wire_capacity, compute_capacity in zip(
                local_wire_capacities,
                compute_capacities,
                strict=True,
            ):
                buffers.hidden_states[:local_wire_capacity].zero_()
                buffers.topk_ids[:local_wire_capacity].zero_()
                buffers.topk_weights[:local_wire_capacity].fill_(1.0 / layer.top_k)
                self._compute.run_integrated_layer(
                    layer_index,
                    buffers.hidden_states[:local_wire_capacity],
                    buffers.topk_ids[:local_wire_capacity],
                    buffers.topk_weights[:local_wire_capacity],
                    local_wire_capacity,
                    compute_capacity,
                )

    @staticmethod
    def _recv_prepared_child(
        layer_id: int,
        prepared: _PreparedChildMembers,
        buffers: _CohortBuffers,
        *,
        capture_transport: bool = False,
    ) -> int:
        offset = 0
        for executor, child in prepared:
            next_offset = offset + child.local_wire_capacity
            executor._recv_decode_child(
                layer_id,
                child,
                buffers.hidden_states[offset:next_offset],
                buffers.topk_ids[offset:next_offset],
                buffers.topk_weights[offset:next_offset],
                capture_transport=capture_transport,
            )
            offset = next_offset
        return offset

    @staticmethod
    def _send_prepared_child(
        layer_id: int,
        prepared: _PreparedChildMembers,
        output: torch.Tensor,
    ) -> None:
        offset = 0
        for executor, child in prepared:
            next_offset = offset + child.local_wire_capacity
            executor._send_decode_child(
                layer_id,
                child,
                output[offset:next_offset],
            )
            offset = next_offset

    def _run_prepared_child_compute(
        self,
        layer_index: int,
        buffers: _CohortBuffers,
        num_tokens: int,
        *,
        integrated_capacity: int | None = None,
        graph_capacity: int | None = None,
    ) -> torch.Tensor:
        hidden_states = buffers.hidden_states[:num_tokens]
        topk_ids = buffers.topk_ids[:num_tokens]
        topk_weights = buffers.topk_weights[:num_tokens]
        if integrated_capacity is not None:
            return self._compute.run_integrated_layer(
                layer_index,
                hidden_states,
                topk_ids,
                topk_weights,
                num_tokens,
                integrated_capacity,
            )
        if graph_capacity is not None:
            return self._compute.run_graph(
                layer_index,
                graph_capacity,
                hidden_states,
                topk_ids,
                topk_weights,
            )
        return self._compute.run(
            layer_index,
            hidden_states,
            topk_ids,
            topk_weights,
            buffers.router_logits[:num_tokens],
            allow_graph=False,
        )

    def _run_nccl_overlap_prepared_children(
        self,
        prepared_by_child: _PreparedChildren,
        layer_graph_capacities: tuple[int | None, ...],
    ) -> None:
        with compute_sm_budget(self._compute._graph_compute_num_sms):
            self._run_nccl_overlap_prepared_children_impl(
                prepared_by_child,
                layer_graph_capacities,
            )

    def _run_nccl_overlap_prepared_children_impl(
        self,
        prepared_by_child: _PreparedChildren,
        layer_graph_capacities: tuple[int | None, ...],
    ) -> None:
        assert self._communication_stream is not None
        set_is_extend_in_batch(False)
        compute_stream = torch.cuda.current_stream(self._compute.device)
        communication_stream = self._communication_stream
        communication_stream.wait_stream(compute_stream)
        total_tokens: list[int] = []

        first_layer = self._compute.layers[0]
        first_schema = (
            first_layer.hidden_size,
            first_layer.top_k,
            first_layer.num_experts,
        )
        with torch.cuda.stream(communication_stream):
            for child_index, prepared in enumerate(prepared_by_child):
                buffers = self._buffer_slots[child_index][first_schema]
                total_tokens.append(
                    self._recv_prepared_child(
                        first_layer.layer_id,
                        prepared,
                        buffers,
                    )
                )
                self._request_ready[child_index].record(communication_stream)

        for layer_index, layer in enumerate(self._compute.layers):
            schema = (layer.hidden_size, layer.top_k, layer.num_experts)
            next_layer_index = layer_index + 1
            for child_index, prepared in enumerate(prepared_by_child):
                buffers = self._buffer_slots[child_index][schema]
                num_tokens = total_tokens[child_index]
                compute_stream.wait_event(self._request_ready[child_index])
                output = self._run_prepared_child_compute(
                    layer_index,
                    buffers,
                    num_tokens,
                    graph_capacity=layer_graph_capacities[child_index],
                )
                # Compute has consumed the request. The communication stream
                # sends this response before reusing the slot for the next recv.
                buffers.hidden_states[:num_tokens].copy_(output)
                self._response_ready[child_index].record(compute_stream)

                with torch.cuda.stream(communication_stream):
                    communication_stream.wait_event(self._response_ready[child_index])
                    self._send_prepared_child(
                        layer.layer_id,
                        prepared,
                        buffers.hidden_states,
                    )

                    if next_layer_index < len(self._compute.layers):
                        next_layer = self._compute.layers[next_layer_index]
                        next_schema = (
                            next_layer.hidden_size,
                            next_layer.top_k,
                            next_layer.num_experts,
                        )
                        next_buffers = self._buffer_slots[child_index][next_schema]
                        self._recv_prepared_child(
                            next_layer.layer_id,
                            prepared,
                            next_buffers,
                        )
                        self._request_ready[child_index].record(communication_stream)

        compute_stream.wait_stream(communication_stream)

    def _run_prepared_children(
        self,
        prepared_by_child: _PreparedChildren,
        *,
        integrated_compute_capacities: tuple[int, ...] | None = None,
        layer_graph_capacities: tuple[int | None, ...] | None = None,
        capture_transport: bool = False,
    ) -> None:
        set_is_extend_in_batch(False)
        for layer_index, layer in enumerate(self._compute.layers):
            for child_index, prepared in enumerate(prepared_by_child):
                buffers = self._buffers[
                    (layer.hidden_size, layer.top_k, layer.num_experts)
                ]
                num_tokens = self._recv_prepared_child(
                    layer.layer_id,
                    prepared,
                    buffers,
                    capture_transport=capture_transport,
                )

                integrated_compute_capacity = (
                    None
                    if integrated_compute_capacities is None
                    else integrated_compute_capacities[child_index]
                )
                layer_graph_capacity = (
                    None
                    if layer_graph_capacities is None
                    else layer_graph_capacities[child_index]
                )
                output = self._run_prepared_child_compute(
                    layer_index,
                    buffers,
                    num_tokens,
                    integrated_capacity=integrated_compute_capacity,
                    graph_capacity=layer_graph_capacity,
                )
                self._send_prepared_child(
                    layer.layer_id,
                    prepared,
                    output,
                )

"""AFD work lifecycle and fixed decode buffers."""

from __future__ import annotations

import bisect
import logging
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

import torch

from sglang.srt.afd.comm.base import AFDCommBackend
from sglang.srt.afd.direct_deepep import AFDDirectDeepEPDispatchOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.afd.afd_handshake import AFDGraphCaptureVariant
    from sglang.srt.afd.comm.nvshmem import NVSHMEMRuntimeOwner
    from sglang.srt.afd.direct_deepep import (
        AFDDirectDeepEPAdapter,
        AFDDirectDeepEPCombineOutput,
    )
    from sglang.srt.afd.layers import AFDMoELayer

logger = logging.getLogger(__name__)


def is_afd_fixed_shape_mode(forward_mode: ForwardMode) -> bool:
    return forward_mode.is_decode()


def build_decode_token_buckets(
    local_batch_sizes: Sequence[int],
    max_local_requests: int,
    wire_token_multiplier: int,
) -> list[int]:
    """Convert A-side per-rank decode buckets to wire capacities."""
    if max_local_requests <= 0 or wire_token_multiplier <= 0:
        raise ValueError(
            "AFD decode bucket bounds must be positive: "
            f"max_local_requests={max_local_requests}, "
            f"wire_token_multiplier={wire_token_multiplier}"
        )
    if any(batch_size <= 0 for batch_size in local_batch_sizes):
        raise ValueError("AFD decode batch sizes must be positive")

    local_buckets = {
        batch_size
        for batch_size in local_batch_sizes
        if batch_size <= max_local_requests
    }
    # max_running_requests can exceed CUDA graph max_bs. Preserve the configured
    # graph buckets, then add progressively coarser transport buckets instead of
    # jumping directly to the largest slot.
    last_bucket = max(local_buckets, default=0)
    while last_bucket < max_local_requests:
        if last_bucket < 16:
            step = 1
        elif last_bucket < 256:
            step = 8
        elif last_bucket < 512:
            step = 16
        else:
            step = 32
        last_bucket = min(
            ((last_bucket // step) + 1) * step,
            max_local_requests,
        )
        local_buckets.add(last_bucket)
    local_buckets.add(max_local_requests)
    return sorted(batch_size * wire_token_multiplier for batch_size in local_buckets)


def select_token_bucket(token_buckets: Sequence[int], num_tokens: int) -> int:
    if num_tokens == 0:
        return 0
    bucket_index = bisect.bisect_left(token_buckets, num_tokens)
    if bucket_index == len(token_buckets):
        raise RuntimeError(
            f"AFD decode token count {num_tokens} exceeds transport capacity "
            f"{token_buckets[-1]}"
        )
    return token_buckets[bucket_index]


def split_afd_batch_size(batch_size: int, num_microbatches: int) -> tuple[int, ...]:
    if num_microbatches < 2:
        raise ValueError("AFD microbatch overlap requires at least two children")
    if batch_size < num_microbatches:
        raise ValueError(
            "AFD parent batch size must be at least the number of microbatches: "
            f"batch_size={batch_size}, microbatches={num_microbatches}"
        )
    child_size, remainder = divmod(batch_size, num_microbatches)
    return tuple(
        child_size + (child_index < remainder)
        for child_index in range(num_microbatches)
    )


def afd_work_generation(step_id: int, work_index: int) -> int:
    # Keep parent steps disjoint when the number of children changes.
    return (step_id << 32) | (work_index + 1)


@dataclass(frozen=True)
class AFDWorkHandle:
    slot_id: Optional[int]
    generation: int
    capacity: int
    input_tokens: int
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    output: torch.Tensor
    router_logits: Optional[torch.Tensor]
    # Shared experts may still read hidden_states on another stream while
    # additional ffn responses arrive.
    response_scratch: Optional[torch.Tensor] = None
    # PyNcclCommunicator does not register external P2P buffers with the
    # caching allocator. Retain direct request inputs until the ordered
    # response has been enqueued.
    request_tensors: Optional[tuple[torch.Tensor, ...]] = None


AFDDispatchWorkHandle = AFDWorkHandle | AFDDirectDeepEPDispatchOutput


@dataclass
class _SlotBuffers:
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    output: torch.Tensor
    router_logits: Optional[torch.Tensor]
    response_scratch: Optional[torch.Tensor]


class AFDFixedBufferPool:
    """Fixed stream-ordered slots shared by layers with the same tensor schema."""

    def __init__(
        self,
        layers: Sequence[AFDMoELayer],
        token_buckets: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        role: str,
        with_router_logits: bool,
        num_slots: int,
        decode_transport: Optional[NVSHMEMRuntimeOwner] = None,
        with_response_scratch: bool = False,
    ):
        if num_slots <= 0:
            raise ValueError("AFD fixed buffer pool requires at least one slot")
        self._num_slots = num_slots
        self._buckets = sorted(set(token_buckets))
        if not self._buckets or self._buckets[0] <= 0:
            raise ValueError("AFD fixed buffer pool requires positive token buckets")
        self._bucket_set = set(self._buckets)

        self._layer_keys = {
            layer.layer_id: (layer.hidden_size, layer.top_k, layer.num_experts)
            for layer in layers
        }
        if len(self._layer_keys) != len(layers):
            raise RuntimeError("AFD found duplicate MoE layer ids")

        max_tokens = self._buckets[-1]
        self._decode_transport = decode_transport
        self._buffers: dict[tuple[int, int, int, int], _SlotBuffers] = {}
        self._router_logits: dict[tuple[int, int, int, int], torch.Tensor] = {}
        schemas = sorted(set(self._layer_keys.values()))
        if decode_transport is None:
            for slot_id in range(num_slots):
                for hidden_size, top_k, num_experts in schemas:
                    key = (slot_id, hidden_size, top_k, num_experts)
                    self._buffers[key] = _SlotBuffers(
                        hidden_states=torch.empty(
                            (max_tokens, hidden_size), dtype=dtype, device=device
                        ),
                        topk_ids=torch.empty(
                            (max_tokens, top_k), dtype=torch.int32, device=device
                        ),
                        topk_weights=torch.empty(
                            (max_tokens, top_k), dtype=torch.float32, device=device
                        ),
                        output=torch.empty(
                            (max_tokens, hidden_size), dtype=dtype, device=device
                        ),
                        router_logits=(
                            torch.empty(
                                (max_tokens, num_experts),
                                dtype=torch.float32,
                                device=device,
                            )
                            if with_router_logits
                            else None
                        ),
                        response_scratch=(
                            torch.empty(
                                (max_tokens, hidden_size),
                                dtype=dtype,
                                device=device,
                            )
                            if with_response_scratch
                            else None
                        ),
                    )
        elif with_router_logits:
            for slot_id in range(num_slots):
                for hidden_size, top_k, num_experts in schemas:
                    key = (slot_id, hidden_size, top_k, num_experts)
                    self._router_logits[key] = torch.empty(
                        (max_tokens, num_experts),
                        dtype=torch.float32,
                        device=device,
                    )

        self._active_generations: list[Optional[int]] = [None] * num_slots
        allocated_bytes = sum(
            tensor.numel() * tensor.element_size()
            for buffers in self._buffers.values()
            for tensor in (
                buffers.hidden_states,
                buffers.topk_ids,
                buffers.topk_weights,
                buffers.output,
                buffers.router_logits,
                buffers.response_scratch,
            )
            if tensor is not None
        ) + sum(
            tensor.numel() * tensor.element_size()
            for tensor in self._router_logits.values()
        )
        logger.info(
            "AFD fixed buffer pool ready: role=%s slots=%d schemas=%d buckets=%d "
            "max_tokens=%d memory=%.2f MiB",
            role,
            num_slots,
            len(schemas),
            len(self._buckets),
            max_tokens,
            allocated_bytes / (1024 * 1024),
        )

    def acquire(
        self,
        slot_id: int,
        layer_id: int,
        generation: int,
        capacity: int,
        input_tokens: int,
    ) -> AFDWorkHandle:
        if not 0 <= slot_id < self._num_slots:
            raise RuntimeError(f"AFD fixed buffer has no slot {slot_id}")
        active_generation = self._active_generations[slot_id]
        if active_generation is not None:
            raise RuntimeError(
                f"AFD fixed buffer slot {slot_id} is still active: "
                f"generation={active_generation}"
            )
        if input_tokens > capacity:
            raise ValueError(
                f"AFD input token count {input_tokens} exceeds capacity {capacity}"
            )
        if layer_id not in self._layer_keys:
            raise RuntimeError(f"AFD fixed buffer has no layer {layer_id}")
        if capacity not in self._bucket_set:
            raise RuntimeError(
                f"AFD capacity {capacity} is not a configured fixed-buffer bucket"
            )
        layer_key = self._layer_keys[layer_id]
        buffer_key = (slot_id, *layer_key)
        if self._decode_transport is None:
            buffers = self._buffers[buffer_key]
            hidden_states = buffers.hidden_states[:capacity]
            topk_ids = buffers.topk_ids[:capacity]
            topk_weights = buffers.topk_weights[:capacity]
            output = buffers.output[:capacity]
            router_logits = (
                buffers.router_logits[:capacity]
                if buffers.router_logits is not None
                else None
            )
            response_scratch = (
                buffers.response_scratch[:capacity]
                if buffers.response_scratch is not None
                else None
            )
        else:
            views = self._decode_transport.decode_views(
                layer_id,
                capacity,
                slot_id=slot_id,
            )
            hidden_states = views.hidden_states
            topk_ids = views.topk_ids
            topk_weights = views.topk_weights
            output = views.output
            router_logits_buffer = self._router_logits.get(buffer_key)
            router_logits = (
                router_logits_buffer[:capacity]
                if router_logits_buffer is not None
                else None
            )
            response_scratch = None
        self._active_generations[slot_id] = generation
        return AFDWorkHandle(
            slot_id=slot_id,
            generation=generation,
            capacity=capacity,
            input_tokens=input_tokens,
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=output,
            router_logits=router_logits,
            response_scratch=response_scratch,
        )

    def release(self, handle: AFDWorkHandle) -> None:
        slot_id = handle.slot_id
        if slot_id is None or not 0 <= slot_id < self._num_slots:
            raise RuntimeError("AFD attempted to release an invalid slot handle")
        active_generation = self._active_generations[slot_id]
        if active_generation != handle.generation:
            raise RuntimeError(
                "AFD fixed buffer generation mismatch: "
                f"slot={slot_id}, active={active_generation}, "
                f"handle={handle.generation}"
            )
        self._active_generations[slot_id] = None


class AFDDispatcher:
    """A-side ordered dispatch/combine with fixed decode staging buffers."""

    def __init__(
        self,
        comm: AFDCommBackend,
        layers: Sequence[AFDMoELayer],
        token_buckets: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        scattered: bool,
        num_slots: int = 1,
        direct_deepep_lanes: Sequence[AFDDirectDeepEPAdapter] = (),
        compute_num_sms: int | None = None,
    ):
        self._comm = comm
        self._decode_transport = comm.decode_transport
        self._layers = list(layers)
        self._dtype = dtype
        self._device = device
        self._scattered = scattered
        self._direct_deepep_lanes = tuple(direct_deepep_lanes)
        self._ffn_global_ranks = tuple(
            comm.config.ffn_global_rank(rank)
            for rank in comm.config.ffn_ranks_for_attn(
                comm.config.instance_rank,
                scattered=scattered,
            )
        )
        self._padding_topk_id = -1 if scattered else 0
        self._num_slots = num_slots
        self._compute_num_sms = compute_num_sms
        self._pool = (
            None
            if self._direct_deepep_lanes
            else AFDFixedBufferPool(
                layers=self._layers,
                token_buckets=token_buckets,
                dtype=dtype,
                device=device,
                role="attn",
                with_router_logits=False,
                num_slots=num_slots,
                decode_transport=self._decode_transport,
                with_response_scratch=len(self._ffn_global_ranks) > 1,
            )
        )
        self._next_step_id = 0
        self._step_id: Optional[int] = None
        self._forward_mode: Optional[ForwardMode] = None
        self._dp_capacity = 0
        self._wire_capacity = 0
        self._child_wire_capacities: tuple[int, ...] = ()
        self._next_dispatch_index = 0
        self._next_combine_index = 0
        self._active_handles: deque[AFDDispatchWorkHandle] = deque()
        self._pending_direct_combines: deque[AFDDirectDeepEPCombineOutput] = deque()
        self._communication_graph_capacities: frozenset[int] = frozenset()
        self._parent_graph_variants_by_batch_size: dict[
            int, AFDGraphCaptureVariant
        ] = {}
        self._parent_graph_variants_by_wire: tuple[AFDGraphCaptureVariant, ...] = ()
        self._paired_parent_graph_variants: frozenset[AFDGraphCaptureVariant] = (
            frozenset()
        )
        self._prepared_parent_graph_variant: Optional[AFDGraphCaptureVariant] = None
        self._current_batch_uses_paired_communication_graph = False
        self._capturing_parent_graph = False
        self._use_eager_parent_graph_breaks = False

    @property
    def needs_empty_decode_dispatch(self) -> bool:
        if (
            (self._direct_deepep_lanes)
            and self._forward_mode is not None
            and is_afd_fixed_shape_mode(self._forward_mode)
        ):
            return True
        return (
            self._scattered
            and self._forward_mode is not None
            and is_afd_fixed_shape_mode(self._forward_mode)
            and self._wire_capacity > 0
        )

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    @property
    def current_num_children(self) -> int:
        return len(self._child_wire_capacities)

    @property
    def compute_num_sms(self) -> int | None:
        return self._compute_num_sms

    def shared_expert_compute_num_sms(self, num_lanes: int) -> int | None:
        if (
            self._direct_deepep_lanes
            and num_lanes > 1
            and self._compute_num_sms is not None
        ):
            lane_sms = self._direct_deepep_lanes[0].overlap_communication_num_sms
            # Attention overlaps the other lanes; shared experts also overlap
            # their own dispatch/combine. Reserve that additional active lane,
            # including during warmup so capture uses the same GEMM kernels.
            num_sms = min(
                self._compute_num_sms,
                self._compute_num_sms + (self._num_slots - 1 - num_lanes) * lane_sms,
            )
            num_sms = num_sms // 2 * 2
            if num_sms <= 0:
                raise RuntimeError("AFD shared experts have no compute SMs left")
            return num_sms
        return None

    @property
    def _prepost_direct_combines(self) -> bool:
        # First-use module loading may synchronize the device. Preserve eager
        # and warmup ordering; the captured graph uses already-warmed kernels.
        return (
            self._capturing_parent_graph
            and bool(self._direct_deepep_lanes)
            and len(self._child_wire_capacities) > 1
            and torch.cuda.is_current_stream_capturing()
        )

    @property
    def use_eager_parent_graph_breaks(self) -> bool:
        return self._use_eager_parent_graph_breaks

    def enable_communication_graph_variants(
        self, variants: Sequence[AFDGraphCaptureVariant]
    ) -> None:
        if self._direct_deepep_lanes:
            self._direct_deepep_lanes[0].arm_capture_start_barrier()
        normalized = self._enable_parent_graph_variants(variants, paired=True)
        self._communication_graph_capacities = frozenset(
            variant.child_wire_capacities[0]
            for variant in normalized
            if len(variant.child_wire_capacities) == 1
        )

    def enable_local_parent_graph_variants(
        self, variants: Sequence[AFDGraphCaptureVariant]
    ) -> None:
        self._enable_parent_graph_variants(variants, paired=False)
        self._use_eager_parent_graph_breaks = True

    def _enable_parent_graph_variants(
        self,
        variants: Sequence[AFDGraphCaptureVariant],
        *,
        paired: bool,
    ) -> tuple[AFDGraphCaptureVariant, ...]:
        normalized = tuple(variants)
        self._parent_graph_variants_by_batch_size = {
            variant.attn_batch_size: variant for variant in normalized
        }
        self._parent_graph_variants_by_wire = tuple(
            sorted(
                normalized,
                key=lambda variant: variant.attn_wire_capacity,
            )
        )
        self._paired_parent_graph_variants = (
            frozenset(normalized) if paired else frozenset()
        )
        return normalized

    def graph_capture_child_plan(self, attn_batch_size: int) -> tuple[int, ...]:
        variant = self._parent_graph_variants_by_batch_size.get(attn_batch_size)
        return () if variant is None else variant.child_wire_capacities

    def select_graph_variant(
        self, minimum_parent_wire_capacity: int
    ) -> Optional[AFDGraphCaptureVariant]:
        for variant in self._parent_graph_variants_by_wire:
            if variant.attn_wire_capacity >= minimum_parent_wire_capacity:
                return variant
        return None

    def can_capture_communication_graph(self, capacity: int) -> bool:
        return (
            not self._capturing_parent_graph
            and capacity in self._communication_graph_capacities
        )

    def select_paired_communication_graph(self, can_run_graph: bool) -> bool:
        variant = self._prepared_parent_graph_variant
        use_parent_graph = variant in self._paired_parent_graph_variants
        use_single_batch_graph = (
            len(self._child_wire_capacities) == 1
            and self._wire_capacity in self._communication_graph_capacities
        )
        self._current_batch_uses_paired_communication_graph = can_run_graph and (
            use_parent_graph or use_single_batch_graph
        )
        return self._current_batch_uses_paired_communication_graph

    def select_parent_graph_variant(self, can_run_graph: bool) -> bool:
        variant = self._prepared_parent_graph_variant
        return (
            can_run_graph
            and variant is not None
            and len(variant.child_wire_capacities) > 1
        )

    def prepared_parent_graph_child_plan(self) -> Optional[tuple[int, ...]]:
        """Child plan prepared for the current parent graph, if any."""
        if self._prepared_parent_graph_variant is None:
            return None
        return self._prepared_parent_graph_variant.child_wire_capacities

    def begin_batch(
        self,
        forward_mode: ForwardMode,
        dp_capacity: int,
        wire_capacity: int,
        child_wire_capacities: Sequence[int] = (),
        graph_variant: Optional[AFDGraphCaptureVariant] = None,
    ) -> int:
        if self._active_handles:
            raise RuntimeError("AFD cannot start a batch with active work")
        previous_work_count = len(self._layers) * len(self._child_wire_capacities)
        previous_batch_completed = (
            self._next_dispatch_index == previous_work_count
            and self._next_combine_index == previous_work_count
        )
        if (
            self._step_id is not None
            and self._dp_capacity > 0
            and not previous_batch_completed
            and not self._current_batch_uses_paired_communication_graph
        ):
            raise RuntimeError(
                "AFD previous batch did not execute every MoE layer: "
                f"dispatched={self._next_dispatch_index}, "
                f"combined={self._next_combine_index}, "
                f"expected={previous_work_count}"
            )
        if dp_capacity < 0 or wire_capacity < 0:
            raise ValueError("AFD token capacities cannot be negative")
        if dp_capacity > wire_capacity:
            raise ValueError(
                f"AFD DP capacity {dp_capacity} exceeds wire capacity {wire_capacity}"
            )
        child_wire_capacities = tuple(child_wire_capacities) or (wire_capacity,)
        if not is_afd_fixed_shape_mode(forward_mode) and (dp_capacity != wire_capacity):
            raise ValueError(
                "AFD non-fixed-shape work requires exact-size transport: "
                f"dp={dp_capacity}, wire={wire_capacity}"
            )
        if len(child_wire_capacities) > self._num_slots:
            raise ValueError(
                "AFD child plan exceeds fixed-buffer slots: "
                f"children={len(child_wire_capacities)}, slots={self._num_slots}"
            )
        if (
            graph_variant is not None
            and child_wire_capacities != graph_variant.child_wire_capacities
        ):
            raise ValueError(
                "AFD graph variant does not match the active child plan: "
                f"variant={graph_variant.child_wire_capacities}, "
                f"children={child_wire_capacities}"
            )

        step_id = self._next_step_id
        self._next_step_id += 1
        self._step_id = step_id
        self._forward_mode = forward_mode
        self._dp_capacity = dp_capacity
        self._wire_capacity = wire_capacity
        self._child_wire_capacities = child_wire_capacities
        self._prepared_parent_graph_variant = graph_variant
        self._current_batch_uses_paired_communication_graph = False
        self._next_dispatch_index = 0
        self._next_combine_index = 0
        return step_id

    def run_inactive_direct_step(
        self,
        forward_mode: ForwardMode,
        wire_capacity: int,
        child_wire_capacities: Sequence[int] = (),
        *,
        do_cpu_sync: bool | None = None,
    ) -> None:
        child_wire_capacities = tuple(child_wire_capacities) or (wire_capacity,)
        self.begin_batch(
            forward_mode=forward_mode,
            dp_capacity=0,
            wire_capacity=wire_capacity,
            child_wire_capacities=child_wire_capacities,
        )
        layer = self._layers[0]
        hidden_states = torch.empty(
            (0, layer.hidden_size), dtype=self._dtype, device=self._device
        )
        topk_ids = torch.empty((0, layer.top_k), dtype=torch.int32, device=self._device)
        topk_weights = torch.empty(
            (0, layer.top_k), dtype=torch.float32, device=self._device
        )
        handles = [
            self.dispatch(
                layer.layer_id,
                hidden_states,
                topk_ids,
                topk_weights,
                direct_do_cpu_sync=do_cpu_sync,
            )
            for _ in child_wire_capacities
        ]
        # Match the active wavefront by refilling each lane after its combine wait.
        for layer_index in range(len(self._layers)):
            next_layer_index = layer_index + 1
            next_handles = []
            for handle in handles:
                self.combine(handle)
                if next_layer_index < len(self._layers):
                    next_handles.append(
                        self.dispatch(
                            self._layers[next_layer_index].layer_id,
                            hidden_states,
                            topk_ids,
                            topk_weights,
                            direct_do_cpu_sync=do_cpu_sync,
                        )
                    )
            handles = next_handles

    @contextmanager
    def capture_parent_graph(self, attn_batch_size: int) -> Iterator[None]:
        variant = self._parent_graph_variants_by_batch_size[attn_batch_size]
        # Keep this field list (and the restore below) in sync with every
        # batch-level attribute mutated by begin_batch() or graph selection.
        saved_state = (
            self._next_step_id,
            self._step_id,
            self._forward_mode,
            self._dp_capacity,
            self._wire_capacity,
            self._child_wire_capacities,
            self._next_dispatch_index,
            self._next_combine_index,
            self._prepared_parent_graph_variant,
            self._current_batch_uses_paired_communication_graph,
            self._capturing_parent_graph,
        )
        self._capturing_parent_graph = True
        self.begin_batch(
            forward_mode=ForwardMode.DECODE,
            dp_capacity=variant.attn_wire_capacity,
            wire_capacity=sum(variant.child_wire_capacities),
            child_wire_capacities=variant.child_wire_capacities,
            graph_variant=variant,
        )
        try:
            yield
        finally:
            (
                self._next_step_id,
                self._step_id,
                self._forward_mode,
                self._dp_capacity,
                self._wire_capacity,
                self._child_wire_capacities,
                self._next_dispatch_index,
                self._next_combine_index,
                self._prepared_parent_graph_variant,
                self._current_batch_uses_paired_communication_graph,
                self._capturing_parent_graph,
            ) = saved_state

    def _can_send_nccl_request_directly(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        capacity: int,
    ) -> bool:
        return (
            self._decode_transport is None
            and len(self._ffn_global_ranks) == 1
            and hidden_states.shape[0] == capacity
            and hidden_states.is_contiguous()
            and topk_ids.is_contiguous()
            and topk_weights.is_contiguous()
        )

    def dispatch_decode_graph_wave(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> AFDDispatchWorkHandle:
        capacity = hidden_states.shape[0]
        layer = next(
            (layer for layer in self._layers if layer.layer_id == layer_id),
            None,
        )
        if layer is None:
            raise RuntimeError(f"AFD has no MoE layer {layer_id}")
        self._validate_request(layer, hidden_states, topk_ids, topk_weights)

        if self._direct_deepep_lanes:
            return self._direct_deepep_lanes[0].dispatch(
                layer_id,
                hidden_states,
                topk_ids,
                topk_weights,
                num_max_dispatch_tokens_per_rank=capacity,
                do_cpu_sync=False,
            )

        work = self._pool.acquire(
            slot_id=0,
            layer_id=layer_id,
            generation=0,
            capacity=capacity,
            input_tokens=capacity,
        )
        peers = self._ffn_global_ranks
        direct_nccl_request = self._can_send_nccl_request_directly(
            hidden_states, topk_ids, topk_weights, capacity
        )
        if peers and not direct_nccl_request:
            work.hidden_states.copy_(hidden_states)
            work.topk_ids.copy_(topk_ids)
            work.topk_weights.copy_(topk_weights)
        if self._decode_transport is not None:
            for peer_global_rank in peers:
                self._decode_transport.put_request(
                    layer_id,
                    capacity,
                    peer_global_rank=peer_global_rank,
                )
        elif peers:
            request = (
                [hidden_states, topk_ids, topk_weights]
                if direct_nccl_request
                else [work.hidden_states, work.topk_ids, work.topk_weights]
            )
            for peer_global_rank in peers:
                self._comm.send_to(request, peer_global_rank)
        return work

    def combine_decode_graph_wave(
        self,
        layer_id: int,
        work: AFDDispatchWorkHandle,
    ) -> torch.Tensor:
        if not isinstance(work, AFDWorkHandle):
            return self._direct_deepep_lanes[work.lane_id].combine_attn(work).wait()

        capacity = work.capacity
        peers = self._ffn_global_ranks
        if self._decode_transport is not None:
            self._decode_transport.launch_wait_response()
            if not peers:
                work.output.zero_()
            else:
                for peer_global_rank in peers[1:]:
                    views = self._decode_transport.decode_views(
                        layer_id,
                        capacity,
                        peer_global_rank=peer_global_rank,
                    )
                    work.output.add_(views.output)
        elif peers:
            self._comm.recv_from([work.output], peers[0])
            if len(peers) > 1:
                assert work.response_scratch is not None
                for peer_global_rank in peers[1:]:
                    self._comm.recv_from(
                        [work.response_scratch],
                        peer_global_rank,
                    )
                    work.output.add_(work.response_scratch)
        else:
            work.output.zero_()
        if self._decode_transport is not None:
            # NVSHMEM arena views are externally allocated. Keep them out of
            # downstream graph-captured collectives that register PyTorch storage.
            output = work.output.clone()
        else:
            output = work.output
        self._pool.release(work)
        return output

    def decode_graph_wave(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        work = self.dispatch_decode_graph_wave(
            layer_id,
            hidden_states,
            topk_ids,
            topk_weights,
        )
        return self.combine_decode_graph_wave(layer_id, work)

    def warmup_decode_transport(self, capacity: int) -> None:
        if self._decode_transport is None:
            raise RuntimeError("AFD decode transport is not initialized")
        for layer in self._layers:
            views = self._decode_transport.decode_views(layer.layer_id, capacity)
            views.hidden_states.zero_()
            views.topk_ids.zero_()
            views.topk_weights.fill_(1.0 / layer.top_k)
            for peer_global_rank in self._ffn_global_ranks:
                self._decode_transport.put_request(
                    layer.layer_id,
                    capacity,
                    peer_global_rank=peer_global_rank,
                )
            self._decode_transport.launch_wait_response()

    def dispatch(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        direct_do_cpu_sync: bool | None = None,
    ) -> AFDDispatchWorkHandle:
        if self._step_id is None or self._forward_mode is None:
            raise RuntimeError("AFD dispatch called before begin_batch")
        num_children = len(self._child_wire_capacities)
        total_work = len(self._layers) * num_children
        if self._next_dispatch_index == total_work:
            raise RuntimeError("AFD batch executed more MoE work than expected")
        if len(self._active_handles) >= num_children:
            raise RuntimeError(
                "AFD dispatch cannot exceed the in-flight child window: "
                f"active={len(self._active_handles)}, children={num_children}"
            )

        layer_index, child_index = divmod(
            self._next_dispatch_index,
            num_children,
        )
        layer = self._layers[layer_index]
        if layer.layer_id != layer_id:
            raise RuntimeError(
                "AFD MoE layer order mismatch: "
                f"expected={layer.layer_id}, got={layer_id}"
            )
        self._validate_request(layer, hidden_states, topk_ids, topk_weights)

        generation = afd_work_generation(
            self._step_id,
            self._next_dispatch_index,
        )
        has_peer = bool(self._ffn_global_ranks)
        direct_nccl_request = False
        input_tokens = hidden_states.shape[0]
        if is_afd_fixed_shape_mode(self._forward_mode):
            capacity = self._child_wire_capacities[child_index]
            if num_children == 1:
                if not self._dp_capacity <= input_tokens <= capacity:
                    raise RuntimeError(
                        "AFD fixed-shape capacity mismatch: "
                        f"dp={self._dp_capacity}, input={input_tokens}, "
                        f"wire={capacity}"
                    )
            elif input_tokens > capacity:
                raise RuntimeError(
                    "AFD fixed-shape child exceeds its wire capacity: "
                    f"child={child_index}, capacity={capacity}, "
                    f"input={input_tokens}"
                )
            if self._direct_deepep_lanes:
                async_with_compute_stream = num_children > 1
                handle = self._direct_deepep_lanes[child_index].dispatch(
                    layer_id,
                    hidden_states,
                    topk_ids,
                    topk_weights,
                    num_max_dispatch_tokens_per_rank=capacity,
                    do_cpu_sync=(
                        not async_with_compute_stream
                        if direct_do_cpu_sync is None
                        else direct_do_cpu_sync
                    ),
                    async_with_compute_stream=async_with_compute_stream,
                )
                self._active_handles.append(handle)
                self._next_dispatch_index += 1
                if self._prepost_direct_combines:
                    self._pending_direct_combines.append(
                        self._direct_deepep_lanes[child_index].combine_attn(
                            handle, async_with_compute_stream=True
                        )
                    )
                return handle
            handle = self._pool.acquire(
                slot_id=child_index,
                layer_id=layer_id,
                generation=generation,
                capacity=capacity,
                input_tokens=input_tokens,
            )
            direct_nccl_request = self._can_send_nccl_request_directly(
                hidden_states, topk_ids, topk_weights, capacity
            )
            if has_peer and not direct_nccl_request:
                handle.hidden_states[:input_tokens].copy_(hidden_states)
                handle.topk_ids[:input_tokens].copy_(topk_ids)
                handle.topk_weights[:input_tokens].copy_(topk_weights)
                if input_tokens < handle.capacity:
                    handle.hidden_states[input_tokens:].zero_()
                    handle.topk_ids[input_tokens:].fill_(self._padding_topk_id)
                    handle.topk_weights[input_tokens:].zero_()
            elif direct_nccl_request:
                handle = replace(
                    handle,
                    request_tensors=(hidden_states, topk_ids, topk_weights),
                )
        else:
            if input_tokens != self._dp_capacity:
                raise RuntimeError(
                    "AFD prefill input size does not match the DP capacity: "
                    f"input={input_tokens}, dp={self._dp_capacity}"
                )
            hidden_states = hidden_states.contiguous()
            topk_ids = topk_ids.contiguous()
            topk_weights = topk_weights.contiguous()
            handle = AFDWorkHandle(
                slot_id=None,
                generation=generation,
                capacity=input_tokens,
                input_tokens=input_tokens,
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                output=torch.empty_like(hidden_states),
                router_logits=None,
            )
        if not has_peer:
            handle.output.zero_()

        if (
            is_afd_fixed_shape_mode(self._forward_mode)
            and self._decode_transport is not None
        ):
            for peer_global_rank in self._ffn_global_ranks:
                self._decode_transport.put_request(
                    layer_id,
                    handle.capacity,
                    slot_id=handle.slot_id,
                    peer_global_rank=peer_global_rank,
                )
        elif has_peer:
            request = (
                handle.request_tensors
                if direct_nccl_request
                else (
                    handle.hidden_states,
                    handle.topk_ids,
                    handle.topk_weights,
                )
            )
            assert request is not None
            for peer_global_rank in self._ffn_global_ranks:
                self._comm.send_to(request, peer_global_rank)
        self._active_handles.append(handle)
        self._next_dispatch_index += 1
        return handle

    def combine(self, handle: AFDDispatchWorkHandle) -> torch.Tensor:
        if self._forward_mode is None:
            raise RuntimeError("AFD combine called without an active forward mode")
        if not self._active_handles or self._active_handles[0] is not handle:
            raise RuntimeError("AFD combine must receive the oldest active work handle")

        num_children = len(self._child_wire_capacities)
        total_work = len(self._layers) * num_children
        required_dispatches = min(
            total_work,
            self._next_combine_index + num_children,
        )
        if self._next_dispatch_index < required_dispatches:
            raise RuntimeError(
                "AFD combine called before filling the in-flight window: "
                f"dispatched={self._next_dispatch_index}, "
                f"required={required_dispatches}"
            )

        if not isinstance(handle, AFDWorkHandle):
            if num_children > 1:
                if not self._pending_direct_combines:
                    # Fork every lane before the next layer queues A compute.
                    for pending_handle in self._active_handles:
                        self._pending_direct_combines.append(
                            self._direct_deepep_lanes[
                                pending_handle.lane_id
                            ].combine_attn(
                                pending_handle,
                                async_with_compute_stream=True,
                            )
                        )
                combine_output = self._pending_direct_combines.popleft()
            else:
                combine_output = self._direct_deepep_lanes[handle.lane_id].combine_attn(
                    handle,
                    async_with_compute_stream=False,
                )
            output = combine_output.wait()
            self._active_handles.popleft()
            self._next_combine_index += 1
            return output

        if (
            is_afd_fixed_shape_mode(self._forward_mode)
            and self._decode_transport is not None
        ):
            if len(self._child_wire_capacities) == 1:
                self._decode_transport.wait_response()
            else:
                self._decode_transport.launch_wait_response(slot_id=handle.slot_id)
            if len(self._ffn_global_ranks) > 1:
                layer_index = self._next_combine_index // num_children
                layer_id = self._layers[layer_index].layer_id
                for peer_global_rank in self._ffn_global_ranks[1:]:
                    views = self._decode_transport.decode_views(
                        layer_id,
                        handle.capacity,
                        slot_id=handle.slot_id,
                        peer_global_rank=peer_global_rank,
                    )
                    handle.output.add_(views.output)
        elif self._ffn_global_ranks:
            self._comm.recv_from(
                [handle.output],
                self._ffn_global_ranks[0],
            )
            if len(self._ffn_global_ranks) > 1:
                scratch = handle.response_scratch
                if scratch is None:
                    scratch = torch.empty_like(handle.output)
                for peer_global_rank in self._ffn_global_ranks[1:]:
                    self._comm.recv_from(
                        [scratch],
                        peer_global_rank,
                    )
                    handle.output.add_(scratch)
        if handle.slot_id is not None:
            output = handle.output[: handle.input_tokens]
            if self._capturing_parent_graph and self._decode_transport is not None:
                # Parent-graph replay must expose PyTorch-owned storage to the
                # model, not the externally allocated NVSHMEM arena view.
                output = output.clone()
            self._pool.release(handle)
        else:
            output = handle.output
        self._active_handles.popleft()
        self._next_combine_index += 1
        return output

    def _validate_request(
        self,
        layer: AFDMoELayer,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        input_tokens = hidden_states.shape[0]
        expected_topk_shape = (input_tokens, layer.top_k)
        if hidden_states.ndim != 2 or hidden_states.shape[1] != layer.hidden_size:
            raise RuntimeError(
                f"AFD layer {layer.layer_id} hidden shape mismatch: "
                f"expected=(*, {layer.hidden_size}), "
                f"got={tuple(hidden_states.shape)}"
            )
        if (
            tuple(topk_ids.shape) != expected_topk_shape
            or tuple(topk_weights.shape) != expected_topk_shape
        ):
            raise RuntimeError(
                f"AFD layer {layer.layer_id} topk shape mismatch: "
                f"expected={expected_topk_shape}, ids={tuple(topk_ids.shape)}, "
                f"weights={tuple(topk_weights.shape)}"
            )
        if hidden_states.dtype != self._dtype:
            raise RuntimeError(
                f"AFD hidden dtype mismatch: expected={self._dtype}, "
                f"got={hidden_states.dtype}"
            )
        if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
            raise RuntimeError(
                "AFD topk tensors must use int32 ids and float32 weights"
            )
        if (
            hidden_states.device != self._device
            or topk_ids.device != self._device
            or topk_weights.device != self._device
        ):
            raise RuntimeError(f"AFD request tensors must be on {self._device}")


_DISPATCHER: Optional[AFDDispatcher] = None


def set_afd_dispatcher(dispatcher: AFDDispatcher) -> None:
    global _DISPATCHER
    if _DISPATCHER is not None:
        raise RuntimeError("AFD dispatcher is already set for this process")
    _DISPATCHER = dispatcher


def get_afd_dispatcher() -> AFDDispatcher:
    if _DISPATCHER is None:
        raise RuntimeError(
            "AFD dispatcher is not initialized; it is created after the "
            "startup handshake"
        )
    return _DISPATCHER


def get_afd_dispatcher_or_none() -> Optional[AFDDispatcher]:
    return _DISPATCHER

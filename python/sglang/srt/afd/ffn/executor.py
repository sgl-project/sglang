"""FFN-side transport execution for eager AFD steps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, NamedTuple

import torch

from sglang.srt.afd.afd_sync import AFDSyncMsg
from sglang.srt.afd.comm import AFDCommBackend
from sglang.srt.afd.ffn.compute import AFDFFNCompute
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_disagg, get_forward

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
    def __init__(
        self,
        scheduler: Scheduler,
        comm: AFDCommBackend,
        compute: AFDFFNCompute,
    ):
        self._comm = comm
        self._compute = compute
        self._device = torch.device(comm.config.device)
        self._dtype = scheduler.tp_worker.model_runner.dtype
        self._layers = compute.layers
        self._scattered = get_disagg().afd_ffn_moe_backend == "deepep"
        self._attn_ranks = comm.config.attn_ranks_for_ffn(
            comm.config.instance_rank, scattered=self._scattered
        )
        if compute.device != self._device:
            raise RuntimeError(
                f"AFD ffn compute device {compute.device} does not match "
                f"communication device {self._device}"
            )
        tp_group = scheduler.tp_group
        if (
            tp_group.world_size != comm.config.num_ffn_ranks
            or tp_group.rank_in_group != comm.config.instance_rank
        ):
            raise RuntimeError(
                "AFD ffn rank topology does not match the local TP group"
            )

    @torch.inference_mode()
    def run_step(self, msg: AFDSyncMsg) -> None:
        if len(msg.tokens_per_rank) != self._comm.config.num_attn_ranks:
            raise RuntimeError(
                f"AFD sync tokens_per_rank has {len(msg.tokens_per_rank)} entries, "
                f"but the attn instance has {self._comm.config.num_attn_ranks} ranks"
            )
        total_tokens = sum(msg.tokens_per_rank)
        expected_wire_capacity = (
            max(msg.tokens_per_rank) if self._scattered else total_tokens
        )
        if msg.wire_capacity != expected_wire_capacity:
            raise RuntimeError(
                "AFD eager execution requires exact-size transport: "
                f"expected={expected_wire_capacity}, wire={msg.wire_capacity}"
            )
        forward_mode = ForwardMode(msg.forward_mode)
        if forward_mode.is_idle() and total_tokens != 0:
            raise RuntimeError("AFD idle step must have zero capacity")
        if total_tokens == 0:
            return

        source_segments = _prepare_source_segments(
            (
                self._comm.config.attn_global_rank(attn_rank),
                msg.tokens_per_rank[attn_rank] if self._scattered else total_tokens,
            )
            for attn_rank in self._attn_ranks
        )
        with get_forward().scoped(is_extend_in_batch=forward_mode.is_extend()):
            self._run_eager_step(source_segments)

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

    def _run_eager_step(self, source_segments: tuple[_SourceSegment, ...]) -> None:
        num_tokens = sum(segment.capacity for segment in source_segments)
        for layer_index, layer in enumerate(self._layers):
            hidden_states = torch.empty(
                (num_tokens, layer.hidden_size), dtype=self._dtype, device=self._device
            )
            topk_ids = torch.empty(
                (num_tokens, layer.top_k), dtype=torch.int32, device=self._device
            )
            topk_weights = torch.empty(
                (num_tokens, layer.top_k), dtype=torch.float32, device=self._device
            )
            # STANDARD routing is already materialized by A; runners use the
            # logits shape for metadata without reading its values.
            router_logits = torch.empty(
                (num_tokens, layer.num_experts),
                dtype=torch.float32,
                device=self._device,
            )
            self._recv_source_inputs(
                source_segments, hidden_states, topk_ids, topk_weights
            )
            output = self._compute.run(
                layer_index, hidden_states, topk_ids, topk_weights, router_logits
            )
            for peer_global_rank, capacity, rows in source_segments:
                if capacity > 0:
                    self._comm.send_to([output[rows].contiguous()], peer_global_rank)

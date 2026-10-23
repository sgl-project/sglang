"""Ordered attention-side dispatch and combine for AFD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.afd.comm import AFDCommBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_resources

if TYPE_CHECKING:
    from sglang.srt.afd.layers import AFDMoELayer


@dataclass
class AFDWorkHandle:
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    output: torch.Tensor


class AFDDispatcher:
    """Keep every rank's remote MoE calls in model-layer order."""

    def __init__(
        self,
        comm: AFDCommBackend,
        layers: Sequence[AFDMoELayer],
        dtype: torch.dtype,
        device: torch.device,
        scattered: bool,
    ):
        self._comm = comm
        self._layers = list(layers)
        self._dtype = dtype
        self._device = device
        self._ffn_global_ranks = tuple(
            comm.config.ffn_global_rank(rank)
            for rank in comm.config.ffn_ranks_for_attn(
                comm.config.instance_rank, scattered=scattered
            )
        )
        self._next_step_id = 0
        self._step_id: Optional[int] = None
        self._wire_capacity = 0
        self._next_layer_index = 0
        self._active_handle: Optional[AFDWorkHandle] = None

    def begin_batch(
        self, forward_mode: ForwardMode, dp_capacity: int, wire_capacity: int
    ) -> int:
        if self._active_handle is not None:
            raise RuntimeError("AFD cannot start a batch with active work")
        if (
            self._step_id is not None
            and self._wire_capacity > 0
            and self._next_layer_index != len(self._layers)
        ):
            raise RuntimeError(
                "AFD previous batch did not execute every MoE layer: "
                f"completed={self._next_layer_index}, expected={len(self._layers)}"
            )
        if wire_capacity < 0:
            raise ValueError("AFD token capacity cannot be negative")
        if wire_capacity != dp_capacity:
            raise ValueError(
                "AFD eager execution requires exact-size transport: "
                f"dp={dp_capacity}, wire={wire_capacity}"
            )
        if forward_mode.is_idle() and wire_capacity != 0:
            raise ValueError("AFD idle step must have zero capacity")

        step_id = self._next_step_id
        self._next_step_id += 1
        self._step_id = step_id
        self._wire_capacity = wire_capacity
        self._next_layer_index = 0
        return step_id

    def dispatch(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> AFDWorkHandle:
        if self._step_id is None:
            raise RuntimeError("AFD dispatch called before begin_batch")
        if self._active_handle is not None:
            raise RuntimeError("AFD dispatch called before combining previous work")
        if self._next_layer_index == len(self._layers):
            raise RuntimeError("AFD batch executed more MoE layers than expected")
        layer = self._layers[self._next_layer_index]
        if layer.layer_id != layer_id:
            raise RuntimeError(
                "AFD MoE layer order mismatch: "
                f"expected={layer.layer_id}, got={layer_id}"
            )
        self._validate_request(layer, hidden_states, topk_ids, topk_weights)
        handle = AFDWorkHandle(
            hidden_states=hidden_states.contiguous(),
            topk_ids=topk_ids.contiguous(),
            topk_weights=topk_weights.contiguous(),
            output=torch.empty_like(
                hidden_states, memory_format=torch.contiguous_format
            ),
        )
        if not self._ffn_global_ranks:
            handle.output.zero_()
        for peer_global_rank in self._ffn_global_ranks:
            self._comm.send_to(
                [handle.hidden_states, handle.topk_ids, handle.topk_weights],
                peer_global_rank,
            )
        self._active_handle = handle
        return handle

    def combine(self, handle: AFDWorkHandle) -> torch.Tensor:
        if self._active_handle is not handle:
            raise RuntimeError("AFD combine requires the active work handle")
        if self._ffn_global_ranks:
            self._comm.recv_from([handle.output], self._ffn_global_ranks[0])
            if len(self._ffn_global_ranks) > 1:
                scratch = torch.empty_like(handle.output)
                for peer_global_rank in self._ffn_global_ranks[1:]:
                    self._comm.recv_from([scratch], peer_global_rank)
                    handle.output.add_(scratch)
        self._active_handle = None
        self._next_layer_index += 1
        return handle.output

    def _validate_request(
        self,
        layer: AFDMoELayer,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        expected_hidden_shape = (self._wire_capacity, layer.hidden_size)
        expected_topk_shape = (self._wire_capacity, layer.top_k)
        if tuple(hidden_states.shape) != expected_hidden_shape:
            raise RuntimeError(
                f"AFD layer {layer.layer_id} hidden shape mismatch: "
                f"expected={expected_hidden_shape}, got={tuple(hidden_states.shape)}"
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


def set_afd_dispatcher(dispatcher: AFDDispatcher) -> None:
    resources = get_resources()
    if resources.afd_dispatcher is not None:
        raise RuntimeError("AFD dispatcher is already set for this process")
    resources.afd_dispatcher = dispatcher


def get_afd_dispatcher() -> AFDDispatcher:
    dispatcher = get_resources().afd_dispatcher
    if dispatcher is None:
        raise RuntimeError(
            "AFD dispatcher is not initialized; it is created after the startup handshake"
        )
    return dispatcher

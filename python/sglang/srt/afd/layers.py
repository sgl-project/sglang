"""AFD model-side layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from sglang.srt.afd.afd_config import AFDRole
from sglang.srt.afd.dispatcher import (
    AFDWorkHandle,
    get_afd_dispatcher,
    get_afd_dispatcher_or_none,
)
from sglang.srt.layers.logits_processor import get_in_autotune_dummy_run
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_utils import capture_mode
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode


@dataclass(frozen=True)
class AFDMoELayer:
    layer_id: int
    experts: nn.Module
    hidden_size: int
    intermediate_size: int
    top_k: int
    num_experts: int
    num_fused_shared_experts: int
    routed_scaling_factor: Optional[float]


@dataclass
class _AFDGraphBreakHandle:
    input_tokens: int
    hidden_size: int
    dtype: torch.dtype
    device: torch.device
    handle: Optional[AFDWorkHandle] = None


@dataclass(frozen=True)
class _AFDCommunicationGraphHandle:
    handle: AFDWorkHandle

    @property
    def input_tokens(self) -> int:
        return self.handle.input_tokens


AFDDispatchHandle = AFDWorkHandle | _AFDGraphBreakHandle | _AFDCommunicationGraphHandle


@eager_on_graph(True)
def _run_remote_experts(
    layer_id: int,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    # Breakable replay is also reported by get_is_capture_mode(); only
    # capture warmup should skip the actual remote-expert call.
    if capture_mode.is_capture_mode:
        return torch.zeros_like(hidden_states)

    dispatcher = get_afd_dispatcher()
    handle = dispatcher.dispatch(layer_id, hidden_states, topk_ids, topk_weights)
    return dispatcher.combine(handle)


@eager_on_graph(True)
def _dispatch_overlap_remote_experts(
    bridge: _AFDGraphBreakHandle,
    layer_id: int,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    if bridge.handle is not None:
        raise RuntimeError("AFD overlap dispatch bridge is still active")
    dispatcher = get_afd_dispatcher()
    if capture_mode.is_capture_mode:
        return

    bridge.handle = dispatcher.dispatch(
        layer_id,
        hidden_states,
        topk_ids,
        topk_weights,
    )


@eager_on_graph(True)
def _combine_overlap_remote_experts(
    bridge: _AFDGraphBreakHandle,
) -> torch.Tensor:
    dispatcher = get_afd_dispatcher()
    if capture_mode.is_capture_mode:
        return torch.zeros(
            (bridge.input_tokens, bridge.hidden_size),
            dtype=bridge.dtype,
            device=bridge.device,
        )
    handle = bridge.handle
    if handle is None:
        raise RuntimeError("AFD overlap combine has no active dispatch handle")
    try:
        return dispatcher.combine(handle)
    finally:
        bridge.handle = None


class AFDAttnMoE(nn.Module):
    """Attention-side routing endpoint with no expert parameters."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_fused_shared_experts = num_fused_shared_experts
        self.num_local_experts = 0
        # Send unscaled routing weights; F returns a scale-final routed output.
        self.should_fuse_routed_scaling_factor_in_topk = False
        self.moe_runner_config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=0,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            layer_id=layer_id,
            top_k=top_k,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            activation=activation,
            inplace=False,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.quant_method = None
        self.supports_deferred_finalize = False

    @staticmethod
    def _check_topk_output(topk_output: TopKOutput) -> None:
        if not TopKOutputChecker.format_is_standard(topk_output):
            # AFD_ATTN selects STANDARD at the shared TopK boundary.
            raise RuntimeError(
                "AFD attention instance requires STANDARD topk output with explicit "
                f"topk_ids/topk_weights; got {type(topk_output).__name__}."
            )

    def dispatch_afd(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> AFDDispatchHandle:
        self._check_topk_output(topk_output)
        dispatcher = get_afd_dispatcher()
        if get_is_capture_mode() and dispatcher.can_capture_communication_graph(
            hidden_states.shape[0]
        ):
            return _AFDCommunicationGraphHandle(
                handle=dispatcher.dispatch_decode_graph_wave(
                    self.layer_id,
                    hidden_states,
                    topk_output.topk_ids,
                    topk_output.topk_weights,
                ),
            )
        if not (
            dispatcher.use_eager_parent_graph_breaks and is_in_breakable_cuda_graph()
        ):
            return dispatcher.dispatch(
                self.layer_id,
                hidden_states,
                topk_output.topk_ids,
                topk_output.topk_weights,
            )
        bridge = _AFDGraphBreakHandle(
            input_tokens=hidden_states.shape[0],
            hidden_size=hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        _dispatch_overlap_remote_experts(
            bridge,
            self.layer_id,
            hidden_states,
            topk_output.topk_ids,
            topk_output.topk_weights,
        )
        return bridge

    def combine_afd(
        self,
        handle: AFDDispatchHandle,
    ) -> torch.Tensor:
        if isinstance(handle, _AFDCommunicationGraphHandle):
            return get_afd_dispatcher().combine_decode_graph_wave(
                self.layer_id,
                handle.handle,
            )
        if isinstance(handle, _AFDGraphBreakHandle):
            return _combine_overlap_remote_experts(handle)
        return get_afd_dispatcher().combine(handle)

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        self._check_topk_output(topk_output)

        # Kernel-autotune dummy forwards run before the A/F handshake creates
        # the data-plane runtime. The ffn side is not stepping either, so skip
        # only remote I/O (routing above already warmed the real attn path).
        if get_in_autotune_dummy_run():
            return hidden_states

        if hidden_states.shape[0] == 0:
            # Prefill has no point-to-point payload for this pair. Decode uses
            # a common bucket, so an empty rank must still send its padding.
            dispatcher = get_afd_dispatcher_or_none()
            if dispatcher is None or not dispatcher.needs_empty_decode_dispatch:
                return torch.empty_like(hidden_states)

        if get_is_capture_mode():
            dispatcher = get_afd_dispatcher_or_none()
            if dispatcher is not None and dispatcher.can_capture_communication_graph(
                hidden_states.shape[0]
            ):
                return dispatcher.decode_graph_wave(
                    self.layer_id,
                    hidden_states,
                    topk_output.topk_ids,
                    topk_output.topk_weights,
                )

        return _run_remote_experts(
            self.layer_id,
            hidden_states,
            topk_output.topk_ids,
            topk_output.topk_weights,
        )


def collect_afd_moe_layers(model: nn.Module, role: AFDRole) -> list[AFDMoELayer]:
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

    experts_type = AFDAttnMoE if role.is_attn() else FusedMoE
    experts_by_layer = sorted(
        (module for module in model.modules() if isinstance(module, experts_type)),
        key=lambda experts: experts.layer_id,
    )
    moe_layers = [
        AFDMoELayer(
            layer_id=experts.layer_id,
            experts=experts,
            hidden_size=experts.hidden_size,
            intermediate_size=(
                experts.moe_runner_config.intermediate_size_per_partition
            ),
            top_k=experts.top_k,
            num_experts=experts.num_experts,
            num_fused_shared_experts=experts.num_fused_shared_experts,
            routed_scaling_factor=experts.moe_runner_config.routed_scaling_factor,
        )
        for experts in experts_by_layer
    ]
    if not moe_layers:
        raise RuntimeError("AFD found no MoE expert layers")
    return moe_layers

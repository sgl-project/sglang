"""FFN-side native expert execution."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch

from sglang.srt.afd.afd_config import AFDRole
from sglang.srt.afd.layers import AFDMoELayer, collect_afd_moe_layers
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.runtime_context import get_disagg, get_exec
from sglang.srt.utils import get_bool_env_var, is_hip

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class _RoutedScalingMode(Enum):
    TOPK_WEIGHTS = "topk_weights"
    RUNNER = "runner"
    OUTPUT = "output"


def _resolve_routed_scaling_mode(layer: AFDMoELayer) -> _RoutedScalingMode:
    if layer.routed_scaling_factor in (None, 1.0):
        return _RoutedScalingMode.RUNNER
    if layer.experts.should_fuse_routed_scaling_factor_in_topk:
        return _RoutedScalingMode.TOPK_WEIGHTS

    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE

    if isinstance(layer.experts, DeepEPMoE):
        return _RoutedScalingMode.RUNNER if _use_aiter else _RoutedScalingMode.OUTPUT

    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        requires_post_expert_routed_scaling,
    )

    if requires_post_expert_routed_scaling(layer.experts):
        return _RoutedScalingMode.OUTPUT
    return _RoutedScalingMode.RUNNER


class AFDFFNCompute:
    """Run the loaded expert layers without an attention forward or KV cache."""

    def __init__(self, scheduler: Scheduler):
        model_runner = scheduler.tp_worker.model_runner
        self._dtype = model_runner.dtype
        self._layers = collect_afd_moe_layers(model_runner.model, AFDRole.FFN)
        self._scaling_modes = [
            _resolve_routed_scaling_mode(layer) for layer in self._layers
        ]
        self._device = next(self._layers[0].experts.parameters()).device
        self._uses_deepep_low_latency = (
            get_disagg().afd_ffn_moe_backend == "deepep"
            and get_exec().moe.deepep_mode != "normal"
        )

    @property
    def layers(self) -> list[AFDMoELayer]:
        return self._layers

    @property
    def device(self) -> torch.device:
        return self._device

    def configure_native_deepep_dispatch_limit(
        self,
        dispatch_limit: int | None,
    ) -> None:
        if not self._uses_deepep_low_latency:
            return
        if dispatch_limit is None:
            raise RuntimeError("AFD DeepEP low-latency capacity was not derived")

        for layer in self._layers:
            for dispatcher in layer.experts.dispatcher._inners:
                if dispatcher.deepep_mode.enable_low_latency():
                    dispatcher._low_latency_dispatcher.num_max_dispatch_tokens_per_rank = dispatch_limit
                if dispatcher.deepep_mode.enable_normal():
                    dispatcher._normal_dispatcher.num_max_dispatch_tokens_per_rank = (
                        dispatch_limit
                    )

    def run(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        layer = self._layers[layer_index]
        scaling_mode = self._scaling_modes[layer_index]
        if scaling_mode is _RoutedScalingMode.TOPK_WEIGHTS:
            topk_weights.mul_(layer.routed_scaling_factor)
        output = layer.experts(
            hidden_states,
            StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
            ),
        )
        if scaling_mode is _RoutedScalingMode.OUTPUT:
            output.mul_(layer.routed_scaling_factor)
        self._validate_output(layer, output, hidden_states.shape[0])
        return output

    def _validate_output(
        self, layer: AFDMoELayer, output: torch.Tensor, num_tokens: int
    ) -> None:
        expected_shape = (num_tokens, layer.hidden_size)
        if (
            tuple(output.shape) != expected_shape
            or output.dtype != self._dtype
            or output.device != self._device
        ):
            raise RuntimeError(
                f"AFD layer {layer.layer_id} output mismatch: "
                f"expected shape={expected_shape}, dtype={self._dtype}, "
                f"device={self._device}; got shape={tuple(output.shape)}, "
                f"dtype={output.dtype}, device={output.device}"
            )

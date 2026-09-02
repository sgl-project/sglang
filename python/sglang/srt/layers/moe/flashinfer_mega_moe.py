"""SGLang adapter for FlashInfer's SM120 MXFP4 x MXFP8 MegaMoE backend."""

from __future__ import annotations

import logging
import weakref
from typing import Any, Optional

import torch
import torch.nn as nn

from sglang.srt.distributed.parallel_state import get_moe_ep_group
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)

_ADAPTERS: dict[int, weakref.ReferenceType["FlashInferMegaMoEAdapter"]] = {}


def _view_byte_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype == dtype:
        return tensor
    if tensor.dtype not in (torch.uint8, torch.int8):
        raise TypeError(f"expected byte storage or {dtype}, got {tensor.dtype}")
    return tensor.view(dtype)


def _lookup_adapter(handle: int) -> FlashInferMegaMoEAdapter:
    ref = _ADAPTERS.get(handle)
    adapter = ref() if ref is not None else None
    if adapter is None:
        raise RuntimeError(f"stale FlashInfer MegaMoE adapter handle {handle}")
    return adapter


class FlashInferMegaMoEAdapter(nn.Module):
    """Own one FlashInfer ``MoEEpLayer`` and its stable output buffer."""

    def __init__(
        self,
        *,
        num_experts: int,
        num_local_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        max_num_tokens: int,
        activation_clamp: Optional[float],
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_num_tokens = max_num_tokens
        self.activation_clamp = activation_clamp
        self.layer: Any = None
        self._tensors_cls: Any = None
        self._fast_tensors: Any = None
        self._output: Optional[torch.Tensor] = None
        self.handle = id(self)
        _ADAPTERS[self.handle] = weakref.ref(self)

    @staticmethod
    def _api() -> dict[str, Any]:
        try:
            from flashinfer.moe_ep import (
                BootstrapConfig,
                FleetParams,
                MegaConfig,
                MoEEpLayer,
                MoEEpTensors,
                PrequantizedMoEWeights,
                Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "flashinfer_megamoe requires the FlashInfer "
                "sm120-w4a8-megamoe branch with flashinfer.moe_ep support"
            ) from exc
        return {
            "BootstrapConfig": BootstrapConfig,
            "FleetParams": FleetParams,
            "MegaConfig": MegaConfig,
            "MoEEpLayer": MoEEpLayer,
            "MoEEpTensors": MoEEpTensors,
            "PrequantizedMoEWeights": PrequantizedMoEWeights,
            "KernelConfig": Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
        }

    def _validate_runtime(self, device: torch.device) -> None:
        capability = torch.cuda.get_device_capability(device)
        if capability[0] != 12:
            raise NotImplementedError(
                "FlashInfer MXFP4 x MXFP8 MegaMoE requires SM120; got "
                f"SM{capability[0]}{capability[1]}."
            )
        parallel = get_parallel()
        if parallel.attn_tp_size != 1 or parallel.moe_tp_size != 1:
            raise ValueError(
                "FlashInfer MegaMoE requires attention TP1 and MoE TP1. "
                f"Derived widths are attn_tp={parallel.attn_tp_size}, "
                f"moe_tp={parallel.moe_tp_size}."
            )
        if self.hidden_size % 128 or self.intermediate_size % 128:
            raise ValueError(
                "FlashInfer MegaMoE requires hidden and intermediate sizes "
                "to be multiples of 128."
            )

    def finalize_weights(
        self,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        if self.layer is not None:
            return
        self._validate_runtime(w13.device)
        api = self._api()
        ep_group = get_moe_ep_group()
        if ep_group.world_size * self.num_local_experts != self.num_experts:
            raise ValueError(
                "FlashInfer MegaMoE requires an even, non-replicated EP "
                "expert partition."
            )

        weights = api["PrequantizedMoEWeights"](
            w13=_view_byte_dtype(w13, torch.float4_e2m1fn_x2),
            w2=_view_byte_dtype(w2, torch.float4_e2m1fn_x2),
            w13_scale=_view_byte_dtype(w13_scale, torch.float8_e8m0fnu),
            w2_scale=_view_byte_dtype(w2_scale, torch.float8_e8m0fnu),
        )
        bootstrap = api["BootstrapConfig"](
            world_size=ep_group.world_size,
            rank=ep_group.rank_in_group,
            device=torch.cuda.current_device(),
            process_group=ep_group.device_group,
        )
        fleet = api["FleetParams"](
            num_experts=self.num_experts,
            max_tokens_per_rank=self.max_num_tokens,
            token_hidden_size=self.hidden_size,
        )
        kernel_config = api["KernelConfig"](
            intermediate_size=self.intermediate_size,
            top_k=self.top_k,
            gate_up_clamp=self.activation_clamp,
        )
        self.layer = api["MoEEpLayer"](
            bootstrap=bootstrap,
            fleet_params=fleet,
            weights=weights,
            backend=api["MegaConfig"](
                megakernel=kernel_config,
                quantize_input=True,
                preprocess_weights=True,
            ),
        )
        self._tensors_cls = api["MoEEpTensors"]
        self._output = self.layer.output_buffer
        logger.info(
            "Initialized FlashInfer MegaMoE: experts=%d local_experts=%d "
            "top_k=%d hidden=%d intermediate=%d capacity=%d",
            self.num_experts,
            self.num_local_experts,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.max_num_tokens,
        )

    @property
    def output_buffer(self) -> torch.Tensor:
        if self._output is None:
            raise RuntimeError("FlashInfer MegaMoE weights are not finalized")
        return self._output

    def stage_into(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        output: torch.Tensor,
        *,
        compile_tokens_per_rank: int,
        activation_clamp: Optional[float],
    ) -> None:
        if self.layer is None or self._tensors_cls is None:
            raise RuntimeError("FlashInfer MegaMoE weights are not finalized")
        if hidden_states.shape[0] > self.max_num_tokens:
            raise ValueError(
                f"local token count {hidden_states.shape[0]} exceeds FlashInfer "
                f"MegaMoE capacity {self.max_num_tokens}"
            )
        if compile_tokens_per_rank < hidden_states.shape[0]:
            raise ValueError(
                "compile_tokens_per_rank cannot be smaller than the local rows"
            )
        if activation_clamp != self.activation_clamp:
            raise ValueError(
                "FlashInfer MegaMoE activation clamp changed after construction: "
                f"{self.activation_clamp} -> {activation_clamp}"
            )
        if topk_ids.dtype != torch.int64 or topk_weights.dtype != torch.float32:
            raise TypeError(
                "FlashInfer MegaMoE requires int64 topk_ids and fp32 topk_weights; "
                f"got {topk_ids.dtype} and {topk_weights.dtype}."
            )

        if self._fast_tensors is None:
            self._fast_tensors = self._tensors_cls(
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                output=output,
            )
        else:
            self._fast_tensors.hidden_states = hidden_states
            self._fast_tensors.topk_ids = topk_ids
            self._fast_tensors.topk_weights = topk_weights
            self._fast_tensors.output = output
        self.layer.stage_inputs(
            self._fast_tensors,
            compile_tokens_per_rank=compile_tokens_per_rank,
        )

    def compute_staged_into(self, output: torch.Tensor) -> None:
        if self.layer is None:
            raise RuntimeError("FlashInfer MegaMoE weights are not finalized")
        self.layer.compute_staged(output=output)


@register_custom_op(mutates_args=["output"])
def flashinfer_megamoe_stage(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    handle: int,
    compile_tokens_per_rank: int,
    activation_clamp: Optional[float],
) -> None:
    _lookup_adapter(handle).stage_into(
        hidden_states,
        topk_weights,
        topk_ids,
        output,
        compile_tokens_per_rank=compile_tokens_per_rank,
        activation_clamp=activation_clamp,
    )


@register_custom_op(mutates_args=["output"])
def flashinfer_megamoe_compute(output: torch.Tensor, handle: int) -> None:
    _lookup_adapter(handle).compute_staged_into(output)


def finalize_flashinfer_megamoe_weights(layer, *, max_num_tokens: int) -> None:
    if getattr(layer, "flashinfer_megamoe_adapter", None) is not None:
        return
    if layer.num_fused_shared_experts != 0:
        raise ValueError(
            "FlashInfer MegaMoE requires --disable-shared-experts-fusion; "
            "the shared expert is computed separately."
        )
    adapter = FlashInferMegaMoEAdapter(
        num_experts=layer.num_experts,
        num_local_experts=layer.num_local_experts,
        top_k=layer.moe_runner_config.top_k,
        hidden_size=layer.moe_runner_config.hidden_size,
        intermediate_size=layer.intermediate_size_per_partition,
        max_num_tokens=max_num_tokens,
        activation_clamp=layer.moe_runner_config.swiglu_limit,
    )
    w13_scale_name = (
        "w13_weight_scale_inv"
        if hasattr(layer, "w13_weight_scale_inv")
        else "w13_weight_scale"
    )
    w2_scale_name = (
        "w2_weight_scale_inv"
        if hasattr(layer, "w2_weight_scale_inv")
        else "w2_weight_scale"
    )
    adapter.finalize_weights(
        layer.w13_weight.data,
        layer.w2_weight.data,
        getattr(layer, w13_scale_name).data,
        getattr(layer, w2_scale_name).data,
    )
    layer.flashinfer_megamoe_adapter = adapter
    # FlashInfer owns the preprocessed weight pack after construction. Drop the
    # loader-side Parameters to avoid retaining a second full expert copy.
    layer.w13_weight = None
    layer.w2_weight = None
    setattr(layer, w13_scale_name, None)
    setattr(layer, w2_scale_name, None)
    layer._mega_moe_weights_built = True


def run_flashinfer_megamoe(
    layer,
    hidden_states: torch.Tensor,
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    compile_tokens_per_rank: int,
    activation_clamp: Optional[float],
) -> torch.Tensor:
    adapter = getattr(layer, "flashinfer_megamoe_adapter", None)
    if adapter is None:
        raise RuntimeError("FlashInfer MegaMoE adapter is not initialized")
    output = adapter.output_buffer
    flashinfer_megamoe_stage(
        hidden_states,
        topk_weights,
        topk_ids,
        output,
        adapter.handle,
        compile_tokens_per_rank,
        activation_clamp,
    )
    flashinfer_megamoe_compute(output, adapter.handle)
    return output[: hidden_states.shape[0]]

"""Native SGLang integration for GLM Quark MXFP4 Gluon MoE kernels."""

from __future__ import annotations

import importlib
import logging
import math
from typing import Optional

import torch

from sglang.srt.layers.moe.gluon_backend import GluonMoeBackend

_WEIGHT_NAMES = (
    "w13_weight",
    "w13_weight_scale",
    "w2_weight",
    "w2_weight_scale",
)
_SUPPORTED_TOPOLOGIES = {
    # (total TP, EP): (rank-local experts, rank-local intermediate size)
    (4, 1): (256, 512),
    (4, 4): (64, 2048),
    (8, 1): (256, 256),
    (8, 2): (128, 512),
    (8, 4): (64, 1024),
    (8, 8): (32, 2048),
}
_NEXTN_LOCAL_EXPERTS = {
    (1, 4): 256,
    (4, 1): 64,
    (1, 8): 256,
    (2, 4): 128,
    (4, 2): 64,
    (8, 1): 32,
}
_KERNEL_PACKAGE = "sglang.srt.layers.moe.gluon_kernels.glm_mxfp4"
_LOGGED_SHARED_MODES = set()
logger = logging.getLogger(__name__)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _pack_weight(value: torch.Tensor) -> torch.Tensor:
    experts, output_size, input_size = value.shape
    return (
        value.reshape(experts, output_size // 16, 16, input_size // 32, 2, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .reshape(experts, output_size, input_size)
    )


def _pack_scale(value: torch.Tensor) -> torch.Tensor:
    experts, output_size, groups = value.shape
    padded_groups = (groups + 7) // 8 * 8
    _require(output_size % 256 == 0, "Unsupported GLM MXFP4 scale layout")
    value = torch.nn.functional.pad(value, (0, padded_groups - groups), value=127)
    return (
        value.reshape(experts, output_size // 32, 2, 16, padded_groups // 8, 2, 4)
        .permute(0, 1, 4, 6, 3, 5, 2)
        .contiguous()
        .reshape(experts, output_size, padded_groups)
    )


def _kernel_name(total_tp: int, ep_size: int, is_nextn: bool, m: int) -> str:
    if total_tp == 4 or ep_size > 1:
        if 17 <= m <= 64:
            return "fused_moe_tp4_m32_64"
        if 1 <= m <= 16:
            return "fused_moe_tp4_m1_16"
        if m <= 4192:
            return "fused_moe_tp4_m128_4192"
        if m <= 32768 and ep_size > 1:
            # The rank-local shared kernel accepts arbitrary expert offsets and
            # local expert banks, so it also covers 32K EP prefill.  Keep EP1
            # on the dedicated full-expert adapter below.
            return "fused_moe_tp4_m4193_16768_shared"
        if total_tp == 4 and ep_size == 1 and m <= 32768:
            return "fused_moe_tp4_m16769_32768_shared"
    elif is_nextn:
        if 1 <= m <= 8 or m in (12, 16):
            return "fused_moe_tp8_m1_16"
        if m <= 17:
            return "fused_moe_tp8_m1_16_mtp"
        if m <= 1023:
            return "fused_moe_tp8_m32_256_mtp"
        if m <= 4192:
            return "fused_moe_tp8_m1024_4192_mtp"
        if m <= 32768:
            return "fused_moe_tp8_m4193_16768"
    else:
        if 1 <= m <= 16:
            return "fused_moe_tp8_m1_16"
        if m <= 255:
            return "fused_moe_tp8_m32_128"
        if m <= 1023:
            return "fused_moe_tp8_m256"
        if m <= 4192:
            return "fused_moe_tp8_m1024_4192"
        if m <= 32768:
            return "fused_moe_tp8_m4193_16768"
    raise RuntimeError(
        f"GLM Gluon MoE has no specialization for TP={total_tp}, EP={ep_size}, "
        f"NextN={is_nextn}, M={m}"
    )


def validate_gluon_quant_method(layer, quant_method) -> None:
    """Validate the only target and draft expert ABIs consumed by this backend."""

    from sglang.srt.layers.quantization.quark.quark import QuarkFusedMoEMethod
    from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
        QuarkW4A4MXFp4MoE,
    )
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    serialized_quark_mxfp4 = (
        isinstance(quant_method, QuarkFusedMoEMethod)
        and isinstance(getattr(layer, "scheme", None), QuarkW4A4MXFp4MoE)
        and layer.scheme.is_checkpoint_mxfp4_serialized
    )
    moe_ep_size = getattr(layer, "moe_ep_size", 1)
    moe_tp_size = getattr(layer, "moe_tp_size", None)
    local_routed = getattr(
        layer, "_num_local_routed", getattr(layer, "num_experts", None)
    )
    glm_nextn_topology = (
        _NEXTN_LOCAL_EXPERTS.get((moe_ep_size, moe_tp_size)) == local_routed
    )
    glm_nextn_bf16 = (
        isinstance(quant_method, UnquantizedFusedMoEMethod)
        and str(getattr(layer, "layer_name", "")).endswith("decoder.mlp.experts")
        and getattr(layer, "num_experts", None) == 256
        and getattr(layer, "hidden_size", None) == 6144
        and getattr(layer, "top_k", None) == 8
        and glm_nextn_topology
        and getattr(layer, "intermediate_size_per_partition", None) * moe_tp_size
        == 2048
        and getattr(layer, "w13_weight", None) is not None
        and getattr(layer, "w2_weight", None) is not None
        and layer.w13_weight.dtype == torch.bfloat16
        and layer.w2_weight.dtype == torch.bfloat16
        and tuple(layer.w13_weight.shape)
        == (local_routed, 2 * layer.intermediate_size_per_partition, 6144)
        and tuple(layer.w2_weight.shape)
        == (local_routed, 6144, layer.intermediate_size_per_partition)
    )
    if not (serialized_quark_mxfp4 or glm_nextn_bf16):
        raise ValueError(
            "--moe-runner-backend gluon supports only serialized Quark W4A4 "
            "MXFP4 target experts or the GLM NextN BF16 draft-expert ABI; "
            "other formats and topologies are not supported."
        )


class GlmMxfp4GluonMoeBackend(GluonMoeBackend):
    """Strict GLM-5.2/5.3 W4A4 backend with TP/EP-aware dispatch."""

    def __init__(self) -> None:
        self.layer = None
        self.experts = None
        self.parameters = None

    def bind(self, layer: torch.nn.Module, experts) -> None:
        from sglang.srt import utils
        from sglang.srt.layers.quantization.quark.quark import QuarkFusedMoEMethod
        from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
            QuarkW4A4MXFp4MoE,
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
        from sglang.srt.runtime_context import get_exec, get_parallel

        config = layer.config
        topology = (layer.tp_size, layer.moe_ep_size)
        local_shape = _SUPPORTED_TOPOLOGIES.get(topology)
        quant_method = getattr(experts, "quant_method", None)
        quant_scheme = getattr(experts, "scheme", None)
        serialized_mxfp4 = (
            isinstance(quant_method, QuarkFusedMoEMethod)
            and isinstance(quant_scheme, QuarkW4A4MXFp4MoE)
            and quant_scheme.is_checkpoint_mxfp4_serialized
        )
        nextn_bf16 = layer.is_nextn and isinstance(
            quant_method, UnquantizedFusedMoEMethod
        )
        checks = {
            "GLM MoE DSA model": getattr(config, "model_type", None) == "glm_moe_dsa",
            "hidden size 6144": getattr(config, "hidden_size", None) == 6144,
            "256 routed experts": getattr(config, "n_routed_experts", None) == 256,
            "top-8 routing": getattr(config, "num_experts_per_tok", None) == 8,
            "MoE intermediate size 2048": getattr(config, "moe_intermediate_size", None)
            == 2048,
            "one shared expert": getattr(config, "n_shared_experts", None) == 1,
            "normalized sigmoid routing": getattr(config, "scoring_func", None)
            == "sigmoid"
            and getattr(config, "norm_topk_prob", None) is True,
            "serialized MXFP4 target or BF16 NextN experts": serialized_mxfp4
            or nextn_bf16,
            "TP4/EP1/4 or TP8/EP1/2/4/8 topology": local_shape
            == (experts._num_local_routed, experts.intermediate_size_per_partition),
            "gfx950 GPU": utils.is_gfx95_supported(),
            "single stream": layer.alt_stream is None,
            "no A2A MoE": not layer._enable_a2a_moe,
            "no EPLB": not get_exec().moe.enable_eplb,
            "MoE DP size 1": get_parallel().moe_dp_size == 1,
            "supported shared expert layout": layer.num_fused_shared_experts in (0, 1)
            and not (
                layer.num_fused_shared_experts == 1
                and layer.is_nextn
            ),
            "no replicated shared expert": not layer._shared_expert_tp1,
            "no SBO shared-expert fusion": not layer._fuse_shared_experts_inside_sbo,
            "positive routed scaling": isinstance(
                layer.routed_scaling_factor, (int, float)
            )
            and not isinstance(layer.routed_scaling_factor, bool)
            and math.isfinite(layer.routed_scaling_factor)
            and layer.routed_scaling_factor > 0,
        }
        unsupported = [name for name, valid in checks.items() if not valid]
        _require(
            not unsupported,
            f"GLM Gluon MoE layer {layer.layer_id} has unsupported contract: "
            + ", ".join(unsupported),
        )
        self.layer = layer
        self.experts = experts
        self.local_experts, self.local_intermediate = local_shape
        self.expert_start = experts.moe_ep_rank * self.local_experts
        self.total_tp, self.ep_size = topology
        self.is_nextn = layer.is_nextn
        # The generic model/loader owns the policy. When enabled, consume the
        # same appended shared-expert slot as AITER instead of repacking a
        # second copy of the dense shared weights here.
        self.fuse_shared_expert = layer.num_fused_shared_experts == 1
        mode_key = (self.total_tp, self.ep_size, self.is_nextn, self.fuse_shared_expert)
        if mode_key not in _LOGGED_SHARED_MODES:
            _LOGGED_SHARED_MODES.add(mode_key)
            logger.info(
                "GLM Gluon MoE TP%d/EP%d%s uses %s shared expert",
                self.total_tp,
                self.ep_size,
                " NextN" if self.is_nextn else "",
                "fused" if self.fuse_shared_expert else "native",
            )

    def _routed_weights(self):
        values = tuple(getattr(self.experts, name, None) for name in _WEIGHT_NAMES)
        stored_experts = self.local_experts + int(self.fuse_shared_expert)
        packed_shapes = (
            (stored_experts, 2 * self.local_intermediate, 3072),
            (stored_experts, 2 * self.local_intermediate, 192),
            (stored_experts, 6144, self.local_intermediate // 2),
            (stored_experts, 6144, self.local_intermediate // 32),
        )
        if all(
            isinstance(value, torch.nn.Parameter)
            and value.dtype == torch.uint8
            and value.is_contiguous()
            and tuple(value.shape) == shape
            for value, shape in zip(values, packed_shapes)
        ):
            return values

        _require(self.is_nextn, "GLM target experts must use serialized MXFP4")
        w13, w2 = values[0], values[2]
        _require(
            values[1] is None
            and values[3] is None
            and isinstance(w13, torch.nn.Parameter)
            and isinstance(w2, torch.nn.Parameter)
            and w13.dtype == torch.bfloat16
            and w2.dtype == torch.bfloat16
            and tuple(w13.shape)
            == (self.local_experts, 2 * self.local_intermediate, 6144)
            and tuple(w2.shape) == (self.local_experts, 6144, self.local_intermediate),
            "Unsupported GLM NextN routed-expert storage",
        )
        from sglang.srt.layers.quantization.quark.utils import b_dynamic_mxfp4_quant

        w13_quant, w13_scale = b_dynamic_mxfp4_quant(w13)
        w2_quant, w2_scale = b_dynamic_mxfp4_quant(w2)
        return (
            _pack_weight(w13_quant),
            _pack_scale(w13_scale),
            _pack_weight(w2_quant),
            _pack_scale(w2_scale),
        )

    def prepare_weights(self) -> None:
        _require(
            not torch.cuda.is_current_stream_capturing(),
            "GLM Gluon weights must be prepared before CUDA graph capture",
        )
        routed = self._routed_weights()
        if self.fuse_shared_expert:
            packed = list(routed)
        else:
            packed = []
            for index, (name, routed_value) in enumerate(zip(_WEIGHT_NAMES, routed)):
                shared_value = (
                    torch.full_like(routed_value[:1], 127)
                    if index % 2
                    else torch.zeros_like(routed_value[:1])
                )
                backing = torch.cat((routed_value, shared_value), dim=0)
                current = getattr(self.experts, name, None)
                prefix = backing[: self.local_experts]
                if isinstance(current, torch.nn.Parameter):
                    current.data = prefix
                else:
                    setattr(
                        self.experts,
                        name,
                        torch.nn.Parameter(prefix, requires_grad=False),
                    )
                packed.append(backing)

        self.parameters = (
            self.layer.gate.weight,
            self.layer.gate.e_score_correction_bias,
            *packed,
        )
        self.native_shared = not self.fuse_shared_expert

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        gemm_output_zero_allocator=None,
        input_ids: Optional[torch.Tensor] = None,
        input_ids_global: Optional[torch.Tensor] = None,
        skip_shared_experts: bool = False,
        num_token_non_padded: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _require(
            hidden_states.ndim == 2,
            "GLM Gluon MoE requires a two-dimensional input",
        )
        _require(
            hidden_states.shape[1] == 6144
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous(),
            "GLM Gluon MoE requires contiguous BF16 [M, 6144] input",
        )
        _require(not skip_shared_experts, "GLM Gluon MoE requires shared experts")
        _require(input_ids_global is None, "GLM Gluon MoE rejects global input IDs")
        _require(
            self.parameters is not None,
            "GLM Gluon weights were not prepared after checkpoint loading",
        )

        name = _kernel_name(
            self.total_tp, self.ep_size, self.is_nextn, hidden_states.shape[0]
        )
        fused_moe = importlib.import_module(f"{_KERNEL_PACKAGE}.{name}").fused_moe
        kwargs = {"routed_scaling_factor": float(self.layer.routed_scaling_factor)}
        if self.total_tp == 4 or self.ep_size > 1:
            kwargs["expert_start"] = self.expert_start
            kwargs["fuse_shared_expert"] = self.fuse_shared_expert
        output = fused_moe(hidden_states, *self.parameters, **kwargs)

        if self.native_shared:
            shared = self.layer._forward_shared_experts(
                hidden_states,
                gemm_output_zero_allocator=gemm_output_zero_allocator,
            )
            _require(shared is not None, "GLM native shared expert is unavailable")
            output = output + shared
        return output

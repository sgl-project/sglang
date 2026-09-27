"""Strict DeepSeek-V4 Pro TP8 Gluon MoE integration."""

from __future__ import annotations

import math
from typing import Optional

import torch

from sglang.srt.layers.moe.gluon_backend import GluonMoeBackend


def _require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


class DeepseekV4ProGluonMoeBackend(GluonMoeBackend):
    """Model-specific backend for the public DeepSeek-V4 Pro FP4 contract.

    The first three hash-routed layers are deliberately handled by the native
    AMD runner. This backend owns every later routed layer and never falls back
    when a call is outside its supported decode shapes.
    """

    _SUPPORTED_M = frozenset((1, 4, 6))

    def __init__(self) -> None:
        self.layer = None
        self.experts = None
        self.parameters = None

    def bind(self, layer: torch.nn.Module, experts) -> None:
        from sglang.srt import utils
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

        quant_method = getattr(experts, "quant_method", None)
        config = layer.config
        checks = {
            "DeepSeek-V4 Pro model type": getattr(config, "model_type", None)
            == "deepseek_v4",
            "hidden size 7168": getattr(config, "hidden_size", None) == 7168,
            "384 routed experts": getattr(config, "n_routed_experts", None) == 384,
            "top-6 routing": getattr(config, "num_experts_per_tok", None) == 6,
            "MoE intermediate size 3072": getattr(config, "moe_intermediate_size", None)
            == 3072,
            "61 transformer layers": getattr(config, "num_hidden_layers", None) == 61,
            "three hash layers": getattr(config, "num_hash_layers", None) == 3,
            "one shared expert": getattr(config, "n_shared_experts", None) == 1,
            "sqrtsoftplus routing": getattr(config, "scoring_func", None)
            == "sqrtsoftplus",
            "normalized routing": getattr(config, "norm_topk_prob", None) is True,
            "SwiGLU clamp 10": getattr(config, "swiglu_limit", None) == 10.0,
            "TP8": getattr(layer, "tp_size", None) == 8,
            "EP1": getattr(layer, "moe_ep_size", None) == 1,
            "non-hash layer": not getattr(layer, "is_hash", False),
            "router correction bias": getattr(
                layer.gate, "e_score_correction_bias", None
            )
            is not None,
            "native packed FP4 routed experts": isinstance(quant_method, Fp8MoEMethod)
            and quant_method.is_fp4_expert
            and getattr(quant_method.quant_config, "is_dsv4_fp4_experts", False)
            and getattr(
                quant_method.quant_config, "is_checkpoint_fp8_serialized", False
            ),
            "UE8M0 128x128 scales": isinstance(quant_method, Fp8MoEMethod)
            and getattr(quant_method.quant_config, "scale_fmt", None) == "ue8m0"
            and getattr(quant_method.quant_config, "weight_block_size", None)
            == [128, 128],
            "gfx950 GPU": utils.is_gfx95_supported(),
            "single stream": getattr(layer, "alt_stream", None) is None,
            "no A2A MoE": not getattr(layer, "_enable_a2a_moe", False),
            "unfused shared expert": getattr(layer, "num_fused_shared_experts", 0) == 0,
            "no SBO shared-expert fusion": not getattr(
                layer, "_fuse_shared_experts_inside_sbo", False
            ),
            "tensor-parallel shared expert": not getattr(
                layer, "_shared_expert_tp1", False
            ),
            "shared expert": getattr(layer, "shared_experts", None) is not None,
            "positive routed scaling": isinstance(
                getattr(layer, "routed_scaling_factor", None), (int, float)
            )
            and not isinstance(layer.routed_scaling_factor, bool)
            and math.isfinite(layer.routed_scaling_factor)
            and layer.routed_scaling_factor > 0,
        }
        unsupported = [name for name, valid in checks.items() if not valid]
        _require(
            not unsupported,
            f"DeepSeek-V4 Pro Gluon MoE layer {layer.layer_id} has unsupported "
            "contract: " + ", ".join(unsupported),
        )
        self.layer = layer
        self.experts = experts

    def prepare_weights(self) -> None:
        _require(self.layer is not None, "DeepSeek-V4 Pro Gluon backend is not bound")
        _require(
            not torch.cuda.is_current_stream_capturing(),
            "DeepSeek-V4 Pro Gluon weights must be prepared before graph capture",
        )
        w13 = self.experts.w13_weight
        w2 = self.experts.w2_weight
        s13 = self.experts.w13_weight_scale_inv
        s2 = self.experts.w2_weight_scale_inv
        _require(
            tuple(w13.shape) == (384, 768, 3584)
            and tuple(w2.shape) == (384, 7168, 192),
            "DeepSeek-V4 Pro Gluon MoE received incompatible FP4 weights",
        )
        _require(
            getattr(w13, "is_shuffled", False) and getattr(w2, "is_shuffled", False),
            "DeepSeek-V4 Pro FP4 expert weights were not shuffled",
        )
        _require(
            s13.numel() == 384 * 768 * 224 and s2.numel() == 384 * 7168 * 16,
            "DeepSeek-V4 Pro Gluon MoE received incompatible UE8M0 scales",
        )
        self.parameters = (
            self.layer.gate.weight,
            self.layer.gate.e_score_correction_bias,
            w13,
            s13,
            w2,
            s2,
        )

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
        _require(self.layer is not None, "DeepSeek-V4 Pro Gluon backend is not bound")
        _require(
            hidden_states.ndim == 2 and hidden_states.shape[0] in self._SUPPORTED_M,
            "DeepSeek-V4 Pro Gluon MoE supports only c=1 decode shapes "
            "M=1 (target) and M=4/6 (MTP)",
        )
        _require(
            hidden_states.shape[1] == 7168,
            f"DeepSeek-V4 Pro Gluon MoE requires H=7168, got {hidden_states.shape[1]}",
        )
        _require(
            hidden_states.dtype == torch.bfloat16,
            f"DeepSeek-V4 Pro Gluon MoE requires BF16 activations, got {hidden_states.dtype}",
        )
        _require(hidden_states.is_contiguous(), "Gluon MoE requires contiguous input")
        _require(
            input_ids_global is None,
            "Gluon MoE does not accept global input IDs",
        )
        _require(
            self.parameters is not None,
            "DeepSeek-V4 Pro Gluon weights were not prepared after checkpoint loading",
        )
        from sglang.srt.layers.moe.gluon_kernels.deepseek_v4_pro_tp8 import (
            fused_moe,
        )

        routed = fused_moe(
            hidden_states,
            *self.parameters,
            routed_scaling_factor=float(self.layer.routed_scaling_factor),
            swiglu_limit=float(self.layer.config.swiglu_limit),
        )
        if skip_shared_experts:
            return routed
        shared = self.layer._forward_shared_experts(
            hidden_states,
            gemm_output_zero_allocator=gemm_output_zero_allocator,
        )
        _require(shared is not None, "DeepSeek-V4 Pro shared expert is unavailable")
        return routed + shared

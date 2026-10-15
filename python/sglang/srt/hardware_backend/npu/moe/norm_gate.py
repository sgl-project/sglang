"""Opt-in native MXFP8 input and router for the NPU W4A8-MXFP4 TP MoE path."""

from functools import lru_cache

import torch

from sglang.srt.hardware_backend.npu.moe.tp_fusion import (
    mxfp8_input,
    record_mxfp8_operand,
    require_npu_op,
    supports_mxfp8_linear,
)
from sglang.srt.hardware_backend.npu.quantization.moe_methods import _require_e8m0_dtype
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.base_config import QuantizeMethodBase


@lru_cache(maxsize=None)
def native_norm_op():
    op = require_npu_op(
        "npu_add_rms_norm_dynamic_mx_quant",
        ("beta", "epsilon", "scale_alg", "round_mode", "dst_type"),
    )
    if len(op.default._schema.returns) != 4:
        raise RuntimeError(f"Unsupported native MX norm ABI: {op.default._schema}")
    return op


@lru_cache(maxsize=None)
def mxfp8_gate_op():
    _require_e8m0_dtype()
    return require_npu_op(
        "npu_quant_matmul",
        (
            "pertoken_scale",
            "output_dtype",
            "scale_dtype",
            "pertoken_scale_dtype",
            "group_sizes",
        ),
    )


def mx_scale_layout(scale, rows, hidden_size):
    if scale.dtype not in (torch.uint8, getattr(torch, "float8_e8m0fnu", None)):
        raise ValueError(f"Expected E8M0 scale bytes, got {scale.dtype}")
    # The existing W4A8 TP consumers require adjacent pairs of 32-value scales.
    return scale.view(rows, hidden_size // 64, 2)


def native_add_rmsnorm_mxfp8(x, residual, gamma, eps):
    op = native_norm_op()
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != gamma.shape[0]:
        raise ValueError(
            "Native MoE norm requires BF16 [T, hidden] matching the norm weight"
        )
    if residual.shape != x.shape or residual.dtype != x.dtype:
        raise ValueError("Native MoE norm requires a matching BF16 residual")
    q, residual_out, scale, rstd = op(
        x,
        residual,
        gamma,
        beta=None,
        epsilon=eps,
        scale_alg=0,
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
    )
    if (
        q.dtype != torch.float8_e4m3fn
        or q.shape != x.shape
        or residual_out.dtype != torch.bfloat16
        or residual_out.shape != x.shape
        or rstd.dtype != torch.float32
    ):
        raise RuntimeError("Unsupported native MX norm output order/dtypes/shapes")
    return q, residual_out, mx_scale_layout(scale, x.shape[0], x.shape[1]), rstd


class NativeMXFP8MoENorm(RMSNorm):
    def __init__(self, hidden_size, eps):
        native_norm_op()
        super().__init__(hidden_size, eps=eps)

    def forward_npu(
        self, x, residual=None, post_residual_addition=None, quant_linear=None
    ):
        if post_residual_addition is not None or quant_linear is not None:
            raise ValueError("Native MoE norm does not support extra fused branches")
        residual_input = torch.zeros_like(x) if residual is None else residual
        q, residual_out, scale, _ = native_add_rmsnorm_mxfp8(
            x, residual_input, self.weight.data, self.variance_epsilon
        )
        # Keep the scale on the FP8 tensor without a tensor -> tuple -> tensor cycle.
        q._npu_mxfp8_scale = scale
        return q if residual is None else (q, residual_out)

    # The downstream consumers require MXFP8 with its scales.
    forward = forward_npu


class MXFP8GateMethod(QuantizeMethodBase):
    def __init__(self):
        self.matmul = mxfp8_gate_op()
        self.quantize = require_npu_op(
            "npu_dynamic_mx_quant",
            ("dst_type", "block_size", "scale_alg", "round_mode"),
        )

    @torch.no_grad()
    def process_weights_after_loading(self, layer):
        if (
            layer.weight.dtype
            not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            )
            or layer.weight.ndim != 2
        ):
            raise ValueError(
                "MXFP8 gate requires a 2-D FP16/BF16/FP32 "
                "[n_routed_experts, hidden_size] weight; "
                f"got {layer.weight.dtype} {tuple(layer.weight.shape)}"
            )
        num_experts, hidden_size = layer.weight.shape
        layer.mxfp8_hidden_size = hidden_size
        # Retain the loaded gate dtype/values for the production reference.
        q, scale = self.quantize(
            layer.weight.data,
            dst_type=torch.float8_e4m3fn,
            block_size=32,
            scale_alg=0,
            round_mode="rint",
        )
        if q.dtype != torch.float8_e4m3fn or q.shape != layer.weight.shape:
            raise RuntimeError("Unsupported native gate MXFP8 quantization output")
        weight = q.transpose(0, 1)
        weight_scale = mx_scale_layout(scale, num_experts, hidden_size).transpose(0, 1)
        layer.register_buffer("mxfp8_weight", weight, persistent=False)
        layer.register_buffer("mxfp8_weight_scale", weight_scale, persistent=False)

    def apply(self, layer, hidden_states):
        if not hasattr(layer, "mxfp8_weight"):
            raise RuntimeError("MXFP8 gate weights were not prepared after loading")
        operand = (
            hidden_states
            if isinstance(hidden_states, tuple)
            else mxfp8_input(hidden_states)
        )
        q, scale = operand
        if (
            q.dtype != torch.float8_e4m3fn
            or q.ndim != 2
            or q.shape[1] != layer.mxfp8_hidden_size
        ):
            raise ValueError("MXFP8 gate requires E4M3 [T, hidden] activations")
        record_mxfp8_operand(operand)
        logits = self.matmul(
            q,
            layer.mxfp8_weight,
            layer.mxfp8_weight_scale,
            pertoken_scale=mx_scale_layout(scale, q.shape[0], q.shape[1]),
            scale_dtype=_require_e8m0_dtype(),
            pertoken_scale_dtype=_require_e8m0_dtype(),
            output_dtype=torch.float32,
            group_sizes=[1, 1, 32],
        )
        if logits.dtype != torch.float32:
            raise RuntimeError("MXFP8 gate did not return FP32 logits")
        return logits


def configure_native_norm_gate(moe):
    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUW4A8MXFP4MoEMethod,
    )
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher
    from sglang.srt.layers.moe.utils import is_sbo_enabled, is_tbo_enabled
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if not (
        moe.moe_ep_size == 1
        and parallel.moe_dp_size == 1
        and parallel.attn_dp_size == 1
        and parallel.attn_cp_size == 1
        and not parallel.enable_prefill_cp
        and not (is_sbo_enabled() or is_tbo_enabled())
        and not moe._enable_a2a_moe
        and not moe._shared_expert_tp1
        and not moe._fuse_shared_experts_inside_sbo
        and moe.num_fused_shared_experts == 0
        and isinstance(moe.experts.dispatcher, AscendTPDispatcher)
        and isinstance(getattr(moe.experts, "w2_kernel", None), NPUW4A8MXFP4MoEMethod)
        and envs.SGLANG_NPU_TP_MOE_PREQUANT_INPUT.get()
        and envs.SGLANG_NPU_TP_MOE_REUSE_MXFP8.get()
        and hasattr(moe, "shared_experts")
        and supports_mxfp8_linear(moe.shared_experts.gate_up_proj)
        and supports_mxfp8_linear(moe.shared_experts.down_proj)
    ):
        raise ValueError(
            "Native norm/gate requires the W4A8 MXFP TP/EP1 prequant/reuse setup"
        )
    if envs.SGLANG_NPU_TP_MOE_NATIVE_NORM_MXFP8.get():
        if not envs.SGLANG_NPU_TP_MOE_MXFP8_GATE.get():
            raise ValueError("Native MXFP8 norm requires MXFP8_GATE=1")
        native_norm_op()
    moe.gate.quant_method = MXFP8GateMethod()
    moe.gate._npu_mxfp8_gate = True

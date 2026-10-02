"""gfx950 dense route of the DeepSeek-V4 attention for 32-wide-block (V4.1) fp8 checkpoints: the
fused RMSNorm + fake-quant producers hand wqkv_a / wq_b their operand on the fp8 grid
(Fp8GridActivation) or as native MXFP8 (Mxfp8Activation); deepseek_v4 binds it under _is_hip."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import nn

from sglang.kernels.ops.layernorm.mhc_boundary_hip import rmsnorm_with_sinkhorn
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
    Fp8GridActivation,
    Mxfp8Activation,
)
from sglang.kernels.ops.quantization.rmsnorm_fake_quant_amd_gfx95 import (
    rmsnorm_fake_quant_fp8,
)
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import resolve_block_fp8_mxfp8_backend
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.srt.utils.common import is_gfx1250_supported

logger = logging.getLogger(__name__)

_is_hip = is_hip()
_is_gfx95_supported = is_gfx95_supported()
_is_gfx1250_supported = is_gfx1250_supported()
_use_aiter = envs.SGLANG_USE_AITER.get() and _is_hip

# V4.1's wo_a route on gfx950: an aiter batched GEMM fork with wo_b's fp8-grid rounding in its
# epilogue; None keeps the aiter kernel
_wo_a_fp8_grid_gemm = None
if _use_aiter and _is_gfx95_supported:
    from sglang.kernels.ops.gemm.gfx95_batched_gemm_bf16_fp8_grid import (
        batched_gemm_bf16_fp8_grid as _wo_a_fp8_grid_gemm,
    )


def fused_rmsnorm_fp8_quant_eligible(
    quant_config: Optional[QuantizationConfig],
) -> bool:
    """Whether the aiter fused RMSNorm + fp8 quant applies: its (fp8, 128-group scale)
    output is consumed only by the 128x128-block dense GEMM."""
    return bool(
        _use_aiter
        and (_is_gfx95_supported or _is_gfx1250_supported)
        and isinstance(quant_config, Fp8Config)
        and quant_config.weight_block_size == [128, 128]
    )


def fused_rmsnorm_fake_quant_eligible(
    quant_config: Optional[QuantizationConfig],
) -> bool:
    """Whether rmsnorm_fake_quant_fp8 applies: gfx950 with a 32-wide-block checkpoint
    (V4.1), whose native or aiter dense route takes the norm output already on the fp8
    grid (Fp8GridActivation) or as fp8 + ue8m0 (Mxfp8Activation)."""
    if not (_is_hip and _is_gfx95_supported and isinstance(quant_config, Fp8Config)):
        return False
    block = quant_config.weight_block_size
    backend = resolve_block_fp8_mxfp8_backend()
    return (
        block is not None
        and block[1] == 32
        and quant_config.scale_fmt == "ue8m0"
        and (backend.is_gfx95_mxfp8_native() or backend.is_gfx95_mxfp8_aiter())
    )


def _mxfp8_consumer(linear: Optional[nn.Module]) -> bool:
    """Whether linear runs a gfx950 MXFP8 route that consumes fp8 + ue8m0 scales directly
    at every M: aiter's MXFP8 GEMM, or the native kernels on a tiled weight."""
    if linear is None:
        return False
    quant_method = linear.quant_method
    return (
        isinstance(quant_method, Fp8LinearMethod)
        and quant_method.block_fp8_as_mxfp8
        and linear.block_fp8_mxfp8_ready
        and (
            quant_method.mxfp8_dense_backend.is_gfx95_mxfp8_aiter()
            or (
                quant_method.mxfp8_dense_backend.is_gfx95_mxfp8_native()
                and linear.mxfp8_native_ready
            )
        )
    )


def _fake_quant_applies(norm: nn.Module, x: torch.Tensor) -> bool:
    return x.dim() == 2 and x.dtype == norm.weight.dtype


def q_norm_fake_quant(attn, q_lora: torch.Tensor) -> Tuple[torch.Tensor, object]:
    """attn.q_norm(q_lora) as (the bf16 norm the indexer reads, the operand wq_b
    consumes, already on the fp8 grid); the plain norm, twice, when the rows are not
    a 2-D bf16 batch."""
    if not _fake_quant_applies(attn.q_norm, q_lora):
        q_lora = attn.q_norm(q_lora)
        return q_lora, q_lora
    if not attn._wq_b_native_consumer_checked:
        attn._wq_b_native_consumer = _mxfp8_consumer(attn.wq_b)
        attn._wq_b_native_consumer_checked = True
    q_for_wq_b, q_lora = rmsnorm_fake_quant_fp8(
        q_lora,
        attn.q_norm.weight.data,
        attn.q_norm.variance_epsilon,
        emit_fp8=attn._wq_b_native_consumer,
    )
    return q_lora, q_for_wq_b


def input_norm_fake_quant(
    layer, hidden_states: torch.Tensor, coefficients=None
) -> Tuple[torch.Tensor, Optional[object]]:
    """layer.input_layernorm(hidden_states) as (the bf16 norm attention reads, the fp8-grid operand
    of its dense projections, or None for non-2-D / non-bf16 rows). coefficients rides in the norm
    launch when the fused kernel runs, else it is materialized here."""
    norm = layer.input_layernorm
    if not _fake_quant_applies(norm, hidden_states):
        if coefficients is not None:
            coefficients.materialize()
        return norm(hidden_states), None
    if not layer._wqkv_a_native_consumer_checked:
        # wqkv_a exists only when the q / kv projections are fused
        layer._wqkv_a_native_consumer = _mxfp8_consumer(
            layer.self_attn.wqkv_a if layer.self_attn.fuse_wqa_wkv else None
        )
        layer._wqkv_a_native_consumer_checked = True
    emit_fp8 = layer._wqkv_a_native_consumer
    if coefficients is not None and not coefficients.materialized:
        x_quant, hidden_states = rmsnorm_with_sinkhorn(
            hidden_states,
            norm.weight.data,
            norm.variance_epsilon,
            coefficients,
            emit_fp8=emit_fp8,
        )
        return hidden_states, x_quant
    x_quant, hidden_states = rmsnorm_fake_quant_fp8(
        hidden_states,
        norm.weight.data,
        norm.variance_epsilon,
        emit_fp8=emit_fp8,
    )
    return hidden_states, x_quant


def post_attention_norm(layer, x: torch.Tensor, coefficients=None) -> torch.Tensor:
    """layer.post_attention_layernorm(x); with coefficients (HcCoefficients of the
    boundary that produced x) still pending, the reduce + sinkhorn rides in the norm
    launch, which then is the Triton row norm rather than the aiter one."""
    norm = layer.post_attention_layernorm
    if (
        coefficients is None
        or coefficients.materialized
        or not _fake_quant_applies(norm, x)
    ):
        if coefficients is not None:
            coefficients.materialize()
        return norm(x)
    if _ffn_norm_emits_mxfp8(layer, x.shape[0]):
        x_quant, out = rmsnorm_with_sinkhorn(
            x, norm.weight.data, norm.variance_epsilon, coefficients, emit_fp8=True
        )
        # the shared expert's gate_up takes it in place of its own quant (_forward_shared_experts)
        out._hip_mxfp8_operand = x_quant
        return out
    _, out = rmsnorm_with_sinkhorn(
        x, norm.weight.data, norm.variance_epsilon, coefficients, fake_quant=False
    )
    return out


_FFN_NORM_EMIT_MXFP8 = envs.SGLANG_HIP_FFN_NORM_MXFP8.get()


def _ffn_norm_emits_mxfp8(layer, num_tokens: int) -> bool:
    """SGLANG_HIP_FFN_NORM_MXFP8: whether the FFN norm launch also emits fp8 + ue8m0 for the
    shared expert's gate_up (its route consumes them at this token count)."""
    if not _FFN_NORM_EMIT_MXFP8:
        return False
    consumer = getattr(layer, "_ffn_shared_mxfp8_consumer", None)
    if consumer is None:
        shared = getattr(getattr(layer, "mlp", None), "shared_experts", None)
        consumer = layer._ffn_shared_mxfp8_consumer = _mxfp8_consumer(
            getattr(shared, "gate_up_proj", None)
        )
    return consumer


def live_rows(activation, num_tokens: int):
    """The first num_tokens rows of a break input; the gfx950 fused q_norm
    hands the indexer its Fp8GridActivation wrapper, whose rows live in .x."""
    if isinstance(activation, Fp8GridActivation):
        return Fp8GridActivation(activation.x[:num_tokens])
    if isinstance(activation, Mxfp8Activation):
        return Mxfp8Activation(activation.q[:num_tokens], activation.scale[:num_tokens])
    return activation[:num_tokens]


def wo_b_takes_fp8_grid(attn) -> bool:
    """Whether attn.wo_b consumes an Fp8GridActivation, which the wo_a GEMM
    then emits from its epilogue; resolved on first use, once the weights are loaded."""
    if attn._wo_b_fp8_grid_operand is None:
        quant_method = attn.wo_b.quant_method
        attn._wo_b_fp8_grid_operand = bool(
            _wo_a_fp8_grid_gemm is not None
            and isinstance(quant_method, Fp8LinearMethod)
            and quant_method.block_fp8_as_mxfp8
            and attn.wo_b.block_fp8_mxfp8_ready
            and quant_method.mxfp8_dense_backend.is_gfx95_mxfp8_native()
        )
    return attn._wo_b_fp8_grid_operand


_WO_A_EMIT_MXFP8 = envs.SGLANG_HIP_WO_A_MXFP8.get()


def wo_b_emits_mxfp8(attn, num_tokens: int) -> bool:
    """SGLANG_HIP_WO_A_MXFP8: whether the wo_a GEMM hands wo_b fp8 + ue8m0 from its epilogue
    (wo_b's route consumes them at this token count) instead of bf16 plus a separate quant."""
    if not _WO_A_EMIT_MXFP8 or _wo_a_fp8_grid_gemm is None:
        return False
    consumer = getattr(attn, "_wo_b_mxfp8_consumer", None)
    if consumer is None:
        consumer = attn._wo_b_mxfp8_consumer = _mxfp8_consumer(attn.wo_b)
    return consumer


def wo_a_fp8_grid_matmul(
    o: torch.Tensor, wo_a: torch.Tensor, fp8_grid: bool, emit_fp8: bool = False
):
    """o [T, G, D] @ wo_a [G, R, D]^T through the gfx950 fork of the aiter batched
    GEMM: bf16 [T, G, R], or with fp8_grid rounded onto wo_b's fp8 grid as
    an Fp8GridActivation [T, G * R], or with emit_fp8 an Mxfp8Activation [T, G * R];
    None when the fork did not import."""
    if _wo_a_fp8_grid_gemm is None:
        return None
    # the split-K regime ends at 64 rows, so a request's verify and decode rows would
    # take different reduction orders; deterministic inference keeps the single chain
    split_k = False if get_exec().deterministic.enable_deterministic_inference else None
    if emit_fp8:
        q, scale = _wo_a_fp8_grid_gemm(
            o, wo_a, fp8_grid=False, split_k=split_k, emit_fp8=True
        )
        return Mxfp8Activation(q, scale)
    y = _wo_a_fp8_grid_gemm(o, wo_a, fp8_grid=fp8_grid, split_k=split_k)
    if fp8_grid:
        return Fp8GridActivation(y)
    return y.view(o.shape[0], wo_a.shape[0], wo_a.shape[1])

"""gfx950 dense route of the DeepSeek-V4 attention for 32-wide-block (V4.1) fp8 checkpoints.

The fused RMSNorm + fake-quant producers hand ``wqkv_a`` / ``wq_b`` their operand
already on the fp8 grid (``Fp8GridActivation``), or as native MXFP8 when the consumer
runs the native kernels (``Mxfp8Activation``), and the ``wo_a`` batched GEMM fork rounds
its epilogue onto ``wo_b``'s grid. Every helper takes the attention / decoder layer,
caching what it resolves from the loaded weights on that layer behind a ``*_checked``
flag; ``deepseek_v4`` binds this module only under ``_is_hip``.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_utils import resolve_block_fp8_mxfp8_backend
from sglang.srt.utils import get_bool_env_var, is_gfx95_supported, is_hip
from sglang.srt.utils.common import is_gfx1250_supported

logger = logging.getLogger(__name__)

_is_hip = is_hip()
_is_gfx95_supported = is_gfx95_supported()
_is_gfx1250_supported = is_gfx1250_supported()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

Fp8GridActivation = None
Mxfp8Activation = None
rmsnorm_fake_quant_fp8 = None
if _is_hip and _is_gfx95_supported:
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
        Fp8GridActivation,
        Mxfp8Activation,
    )
    from sglang.kernels.ops.quantization.rmsnorm_fake_quant_amd_gfx95 import (
        rmsnorm_fake_quant_fp8,
    )

# gfx950 fork of the aiter batched GEMM with wo_b's fp8-grid rounding in its epilogue; None keeps the aiter kernel
_wo_a_fp8_grid_gemm = None
if _use_aiter and _is_gfx95_supported and envs.SGLANG_OPT_USE_AITER_BATCHED_GEMM.get():
    try:
        from sglang.kernels.ops.gemm.gfx95_batched_gemm_bf16_fp8_grid import (
            batched_gemm_bf16_fp8_grid as _wo_a_fp8_grid_gemm,
        )
    except Exception as err:  # pragma: no cover - env-dependent
        logger.warning(
            "wo_a fp8-grid batched GEMM import failed; the aiter kernel and a "
            "separate fake-quant serve wo_a -> wo_b for this process: %s",
            err,
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
    """Whether `rmsnorm_fake_quant_fp8` applies: gfx950 with a 32-wide-block checkpoint
    (V4.1), whose dense route takes the norm output already on the fp8 grid as an
    `Fp8GridActivation`."""
    if not (_is_hip and _is_gfx95_supported and isinstance(quant_config, Fp8Config)):
        return False
    block = quant_config.weight_block_size
    if (
        block is None
        or len(block) != 2
        or block[1] != 32
        or quant_config.scale_fmt != "ue8m0"
    ):
        return False
    return resolve_block_fp8_mxfp8_backend().takes_fp8_grid_activation()


def _native_mxfp8_consumer(linear: Optional[nn.Module]) -> Optional[Tuple[int, int]]:
    """``(N, K)`` of ``linear`` when it runs the gfx950 native MXFP8 route with a weight the
    native kernels tile (it then consumes fp8 + ue8m0 scales directly), else None."""
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    qm = getattr(linear, "quant_method", None)
    if not (
        isinstance(qm, Fp8LinearMethod)
        and qm.block_fp8_as_mxfp8
        and linear.block_fp8_mxfp8_ready
        and qm.mxfp8_dense_backend.is_gfx95_mxfp8_native()
        and linear.mxfp8_native_ready
    ):
        return None
    tiles, steps, _ = linear.weight.shape  # the lane-order layout [N/16, K/128, 2048]
    return tiles * 16, steps * 128


def _emit_native_fp8(consumer: Optional[Tuple[int, int]], num_tokens: int) -> bool:
    """Whether the fused producer hands the native route fp8 + ue8m0 for this token count:
    the skinny kernel's range, or an M bucket served by the dot_scaled tile (hipBLASLt
    buckets keep the fp8-grid bf16 operand)."""
    if consumer is None:
        return False
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        native_consumer_wants_fp8,
    )

    return native_consumer_wants_fp8(num_tokens, consumer[0], consumer[1])


def _fake_quant_applies(norm: nn.Module, x: torch.Tensor) -> bool:
    return x.dim() == 2 and x.dtype == norm.weight.dtype


def q_norm_fake_quant(attn, q_lora: torch.Tensor) -> Tuple[torch.Tensor, object]:
    """`attn.q_norm(q_lora)` as (the bf16 norm the indexer reads, the operand `wq_b`
    consumes, already on the fp8 grid); the plain norm, twice, when the rows are not
    a 2-D bf16 batch."""
    if not _fake_quant_applies(attn.q_norm, q_lora):
        q_lora = attn.q_norm(q_lora)
        return q_lora, q_lora
    if not attn._wq_b_native_consumer_checked:
        attn._wq_b_native_consumer = _native_mxfp8_consumer(attn.wq_b)
        attn._wq_b_native_consumer_checked = True
    q_for_wq_b, q_lora = rmsnorm_fake_quant_fp8(
        q_lora,
        attn.q_norm.weight.data,
        attn.q_norm.variance_epsilon,
        emit_fp8=_emit_native_fp8(attn._wq_b_native_consumer, q_lora.shape[0]),
    )
    return q_lora, q_for_wq_b


def input_norm_fake_quant(
    layer, hidden_states: torch.Tensor
) -> Tuple[torch.Tensor, Optional[object]]:
    """`layer.input_layernorm(hidden_states)` as (the bf16 norm attention reads, the
    operand of its dense projections already on the fp8 grid, or None when the rows are
    not a 2-D bf16 batch)."""
    norm = layer.input_layernorm
    if not _fake_quant_applies(norm, hidden_states):
        return norm(hidden_states), None
    if not layer._wqkv_a_native_consumer_checked:
        # wqkv_a exists only when the q / kv projections are fused
        layer._wqkv_a_native_consumer = _native_mxfp8_consumer(
            getattr(layer.self_attn, "wqkv_a", None)
        )
        layer._wqkv_a_native_consumer_checked = True
    x_quant, hidden_states = rmsnorm_fake_quant_fp8(
        hidden_states,
        norm.weight.data,
        norm.variance_epsilon,
        emit_fp8=_emit_native_fp8(
            layer._wqkv_a_native_consumer, hidden_states.shape[0]
        ),
    )
    return hidden_states, x_quant


def live_rows(activation, num_tokens: int):
    """The first ``num_tokens`` rows of a break input; the gfx950 fused q_norm
    hands the indexer its ``Fp8GridActivation`` wrapper, whose rows live in ``.x``."""
    if _is_hip and _is_gfx95_supported:
        if isinstance(activation, Fp8GridActivation):
            return Fp8GridActivation(activation.x[:num_tokens])
        if isinstance(activation, Mxfp8Activation):
            return Mxfp8Activation(
                activation.q[:num_tokens], activation.scale[:num_tokens]
            )
    return activation[:num_tokens]


def wo_b_takes_fp8_grid(attn) -> bool:
    """Whether ``attn.wo_b`` consumes an ``Fp8GridActivation``, which the ``wo_a`` GEMM
    then emits from its epilogue; resolved on first use, once the weights are loaded."""
    if not attn._wo_b_fp8_grid_checked:
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        qm = getattr(attn.wo_b, "quant_method", None)
        attn._wo_b_fp8_grid_operand = bool(
            _wo_a_fp8_grid_gemm is not None
            and isinstance(qm, Fp8LinearMethod)
            and qm.block_fp8_as_mxfp8
            and attn.wo_b.block_fp8_mxfp8_ready
            and qm.mxfp8_dense_backend.takes_fp8_grid_activation()
        )
        attn._wo_b_fp8_grid_checked = True
    return attn._wo_b_fp8_grid_operand


def wo_a_fp8_grid_matmul(o: torch.Tensor, wo_a: torch.Tensor, fp8_grid: bool):
    """``o [T, G, D] @ wo_a [G, R, D]^T`` through the gfx950 fork of the aiter batched
    GEMM: bf16 ``[T, G, R]``, or with ``fp8_grid`` rounded onto ``wo_b``'s fp8 grid as
    an ``Fp8GridActivation`` ``[T, G * R]``; None when the fork did not import."""
    if _wo_a_fp8_grid_gemm is None:
        return None
    y = _wo_a_fp8_grid_gemm(o, wo_a, fp8_grid=fp8_grid)
    if fp8_grid:
        return Fp8GridActivation(y)
    return y.view(o.shape[0], wo_a.shape[0], wo_a.shape[1])

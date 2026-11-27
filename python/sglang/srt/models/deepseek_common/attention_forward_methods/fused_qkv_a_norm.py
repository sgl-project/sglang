"""MLA's q_a and kv_a RMSNorms as one launch (SGLANG_OPT_FUSED_MLA_QKV_A_NORM).

At decode, verify and draft extend the two norms are two small launches (on two
streams in CUDA graphs), after fused_qkv_a's split-K partial sum when that
projection splits K. ``fused_qkv_a_rmsnorm`` replaces them with one Triton launch
that also does the sum, returning what the unfused path would.
"""

import functools
import logging
from typing import Optional, Tuple

import torch

from sglang.kernels.fused_op import get_fused_op_backend
from sglang.kernels.ops.layernorm.mla_qkv_a_rmsnorm import mla_qkv_a_rmsnorm
from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.unquant import bf16_linear_splitk_partials
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_forward

logger = logging.getLogger(__name__)


def fused_qkv_a_rmsnorm(
    attn: torch.nn.Module, forward_batch: ForwardBatch
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``(q, k_nope, latent_cache)`` as the unfused q_a/kv_a norms leave them
    (``q`` and ``k_nope`` normalized and contiguous, ``latent_cache`` the
    projection's ``[k_nope | k_pe]`` columns), or None where the unfused path
    must run."""
    mode = forward_batch.forward_mode
    attn_inputs = get_forward().attn_inputs
    if not (
        (mode.is_decode() or mode.is_target_verify() or mode.is_draft_extend_v2())
        and attn_inputs is not None
        and _is_plain_rmsnorm(attn.q_a_layernorm)
        and _is_plain_rmsnorm(attn.kv_a_layernorm)
    ):
        return None
    partials = _qkv_a_splitk_partials(attn, attn_inputs)
    src = attn_inputs.fetch_qkv_latent() if partials is None else partials
    # The sum of partials is written in the weights' dtype, as the BF16 route's.
    if partials is None and src.dtype != torch.bfloat16:
        return None
    _log_once(partials is not None)
    latent, q, k_nope = mla_qkv_a_rmsnorm(
        src,
        q_weight=attn.q_a_layernorm.weight,
        q_eps=attn.q_a_layernorm.variance_epsilon,
        kv_weight=attn.kv_a_layernorm.weight,
        kv_eps=attn.kv_a_layernorm.variance_epsilon,
    )
    return q, k_nope, latent[..., attn.q_lora_rank :]


def _qkv_a_splitk_partials(attn, attn_inputs) -> Optional[torch.Tensor]:
    """fused_qkv_a's FP32 split-K partials, where prepare_qkv_latent would run
    that projection unfetched, unscattered, through the split-K route; else None."""
    if (
        attn_inputs.qkv_latent_ is not None
        or not isinstance(attn_inputs.hidden_states_local, torch.Tensor)
        or attn_inputs.qkv_latent_func != attn.prepare_qkv_latent
        or (get_attn_tp_context().input_scattered and not attn_inputs.is_pre_gathered)
        # prepare_qkv_latent's own test for separate q_a / kv_a projections.
        or hasattr(attn, "q_a_proj")
        or not isinstance(attn.fused_qkv_a_proj_with_mqa, ReplicatedLinear)
    ):
        return None
    return bf16_linear_splitk_partials(
        attn.fused_qkv_a_proj_with_mqa, attn_inputs.hidden_states_local
    )


def _is_plain_rmsnorm(norm: torch.nn.Module) -> bool:
    return (
        type(norm) is RMSNorm
        and norm.has_weight
        and not norm.cast_x_before_out_mul
        and norm.variance_size_override is None
        and norm.override_orig_dtype is None
        and norm._quant_linear is None
        and norm.weight.dtype == torch.bfloat16
        and not norm.is_torch_compile
        and get_fused_op_backend() is None
        and not is_batch_invariant_mode_enabled()
    )


@functools.cache
def _log_once(sums_partials: bool) -> None:
    logger.info(
        "Fused MLA q_a/kv_a RMSNorm: one launch%s",
        " with the fused_qkv_a split-K sum" if sums_partials else "",
    )

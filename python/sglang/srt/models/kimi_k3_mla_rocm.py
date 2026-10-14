"""Kimi-K3 (NoPE MLA) ROCm decode / target-verify fusions on the aiter backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.lora.deepseek_mla_correction import is_kv_b_lora_active
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()


def mla_cat_cache(
    attn: DeepseekV2AttentionMLA,
    q_nope_out: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    forward_batch: ForwardBatch,
) -> Optional[torch.Tensor]:
    """NoPE MLA (rotary_emb None, e.g. Kimi-K3)
    decode / target verify on the aiter backend. One Triton launch builds
    q = [q_nope_out | q_pe] and writes the latent row into the MLA KV cache,
    replacing two torch.cat + set_kv_buffer's cast + index_put (bit-identical).
    Returns q, or None when not covered (the caller keeps the unfused path)."""
    if not (
        attn.rotary_emb is None
        and attn.current_attention_backend == "aiter"
        and not get_parallel().dcp_enabled
        and (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        )
    ):
        return None
    from sglang.kernels.ops.attention import k3_mla_cat_cache_hip as cc
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    pool = get_token_to_kv_pool()
    inner = getattr(pool, "full_kv_pool", pool)
    if type(inner) is not MLATokenToKVPool or getattr(
        inner, "dsa_kv_cache_store_fp8", False
    ):
        return None
    if getattr(inner, "requires_physical_write_loc", False) and not getattr(
        forward_batch, "out_cache_loc_is_physical", False
    ):
        return None  # the write door would refuse this loc; keep its error path
    loc = forward_batch.out_cache_loc
    kv = pool.get_key_buffer(attn.attn_mqa.layer_id)
    if not cc.covered(q_nope_out, q_pe, k_nope, k_pe, kv, loc):
        return None
    return cc.k3_mla_cat_cache(q_nope_out, q_pe, k_nope, k_pe, kv, loc)


def _absorb_plain_bf16(w, w_scale) -> bool:
    return (
        isinstance(w, torch.Tensor)
        and w.dtype == torch.bfloat16
        and isinstance(w_scale, (int, float))
        and w_scale == 1.0
    )


def absorb_q_cache_target(
    attn: DeepseekV2AttentionMLA,
    forward_batch: ForwardBatch,
    num_rows: int,
):
    """Fused-absorb gate for the W_UK half: (kv_buffer,
    loc) when the absorb BMM + q cat + latent KV write can run as one launch,
    else None. Same eligibility as mla_cat_cache plus a bf16 W_UK and
    <= 64 rows; prepare and core both call it, so they agree."""
    if not (
        attn.rotary_emb is None
        and attn.current_attention_backend == "aiter"
        and not get_parallel().dcp_enabled
        and not attn.use_deep_gemm_bmm
        and not is_kv_b_lora_active(attn)
        and not _SGLANG_EXPERIMENTAL_LORA_OPTI
        and 0 < num_rows <= 64
        and _absorb_plain_bf16(getattr(attn, "w_kc", None), attn.w_scale)
        and (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        )
    ):
        return None
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    pool = get_token_to_kv_pool()
    inner = getattr(pool, "full_kv_pool", pool)
    if type(inner) is not MLATokenToKVPool or getattr(
        inner, "dsa_kv_cache_store_fp8", False
    ):
        return None
    if getattr(inner, "requires_physical_write_loc", False) and not getattr(
        forward_batch, "out_cache_loc_is_physical", False
    ):
        return None
    return pool.get_key_buffer(attn.attn_mqa.layer_id), forward_batch.out_cache_loc


def absorb_q_cat_cache(
    attn: DeepseekV2AttentionMLA,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    forward_batch: ForwardBatch,
) -> Optional[torch.Tensor]:
    """One launch: q = [q_nope @ W_UK | q_pe] and the latent KV-cache write.
    Returns q, or None when not covered (caller runs the unfused chain)."""
    target = absorb_q_cache_target(attn, forward_batch, q_nope.shape[0])
    if target is None:
        return None
    from sglang.kernels.ops.attention import k3_mla_absorb as ka

    kv, loc = target
    if not ka.covered_q(q_nope, q_pe, attn.w_kc, k_nope, k_pe, kv, loc):
        return None
    return ka.k3_mla_absorb_q_cat_cache(q_nope, q_pe, attn.w_kc, k_nope, k_pe, kv, loc)


def absorb_v_gate(
    attn: DeepseekV2AttentionMLA,
    attn_output: torch.Tensor,
    forward_batch: ForwardBatch,
) -> Optional[torch.Tensor]:
    """Fused-absorb W_UV half: one launch for
    attn_output @ W_UV and, when the K3 output gate is pending, the gate
    multiply the o_proj wrapper would otherwise apply. Returns the o_proj
    input [M, H * v_head_dim], or None (caller keeps rocm_absorb_v_bmm)."""
    if not (
        attn.rotary_emb is None
        and not attn.use_deep_gemm_bmm
        and not is_kv_b_lora_active(attn)
        and not _SGLANG_EXPERIMENTAL_LORA_OPTI
        and not get_parallel().dcp_enabled
        and not is_in_tc_piecewise_cuda_graph()
        and getattr(attn.o_proj, "weight", None) is not None
        and attn.o_proj.weight.dtype == torch.bfloat16
        and _absorb_plain_bf16(getattr(attn, "w_vc", None), attn.w_scale)
        and (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        )
    ):
        return None
    from sglang.kernels.ops.attention import k3_mla_absorb as ka

    if not ka.covered_v(attn_output, attn.w_vc, None):
        return None
    gate_input = getattr(attn, "_gate_hidden_states", None)
    if gate_input is None:
        return ka.k3_mla_absorb_v_gate(attn_output, attn.w_vc, None)
    # Take the gate over from the o_proj wrapper (KimiK3MLAAttention): with
    # _gate_hidden_states cleared and the side-stream join done here, the
    # wrapper passes straight through to o_proj.
    attn._gate_hidden_states = None
    gate = attn._compute_output_gate(gate_input)
    if ka.covered_v(attn_output, attn.w_vc, gate):
        return ka.k3_mla_absorb_v_gate(attn_output, attn.w_vc, gate)
    x = ka.k3_mla_absorb_v_gate(attn_output, attn.w_vc, None)
    return x * torch.sigmoid(gate)

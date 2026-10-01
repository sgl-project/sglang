"""ROCm glue of deepseek_v4: the branches the model takes under _is_hip, bound there as
_hip. The gfx950 dense fp8 routes live in deepseek_v4_gfx95_dense, the fused mHC
boundary in deepseek_v4_fused_mhc."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h
from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.environ import envs
from sglang.srt.layers.attention.hip_flash_mla import (
    hip_attention_needs_head_pad,
    resolve_hip_flashmla_backend,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.models.deepseek_common.amd import deepseek_v4_gfx95_dense as gfx95_dense
from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (  # noqa: F401  deepseek_v4 reaches apply_attention_mhc through this module
    apply_attention_mhc,
    forward_hc_pre_from_prev_fused_boundary,
)
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import is_gfx95_supported

live_rows = gfx95_dense.live_rows
wo_a_fp8_grid_matmul = gfx95_dense.wo_a_fp8_grid_matmul

_is_gfx95_supported = is_gfx95_supported()


# ---- MqaAttentionBase / MQALayer ----


def init_mqa_layer(attn, quant_config) -> None:
    """The ROCm state of MQALayer.__init__. On the gfx950 32-block route q_norm also emits
    the fp8-grid operand of wq_b; whether wq_b consumes native MXFP8, and wo_b the fp8-grid
    operand wo_a emits, resolve on first use, once the weights are loaded."""
    attn.fused_rmsnorm_fake_quant = gfx95_dense.fused_rmsnorm_fake_quant_eligible(
        quant_config
    )
    attn._wq_b_native_consumer_checked = False
    attn._wq_b_native_consumer = False
    attn._wo_b_fp8_grid_operand = None


def wo_a_emits_fp8_grid(attn) -> bool:
    """Whether V4.1's gfx950 wo_a GEMM rounds its output onto wo_b's fp8 grid."""
    return attn.is_dsv41 and gfx95_dense.wo_b_takes_fp8_grid(attn)


def wo_a_split_k_allowed() -> bool:
    """Whether the V4.1 gfx950 wo_a decode / verify kernels may run: their split-K
    reduction order depends on the row count, which batch-invariant and deterministic
    inference rule out."""
    return not (
        is_batch_invariant_mode_enabled()
        or get_exec().deterministic.enable_deterministic_inference
    )


def use_fused_qk_norm_rope(attn) -> bool:
    return bool(envs.SGLANG_OPT_USE_FUSED_QK_NORM_ROPE.get() and attn.q_head_norm)


def q_norm_for_wq_b(attn, q_lora: torch.Tensor) -> Tuple[torch.Tensor, object]:
    """attn.q_norm(q_lora) as (the bf16 norm the indexer reads, the operand wq_b
    consumes); on the gfx950 32-block route the second is already on the fp8 grid."""
    if attn.fused_rmsnorm_fake_quant:
        return gfx95_dense.q_norm_fake_quant(attn, q_lora)
    q_lora = attn.q_norm(q_lora)
    return q_lora, q_lora


def fuses_q_rope_into_k_store(
    attn, q_out: Optional[torch.Tensor], *, unified: bool, use_cp: bool
) -> bool:
    """Whether the K norm-rope-store launch ropes the query heads too: only the plain store
    path, where it is the cache writer and the query is consumed in place (no padded copy)."""
    return (
        not unified
        and not use_cp
        and q_out is None
        and not attn.q_head_norm
        and not envs.SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE.get()
    )


def wq_b_unroped(attn, q) -> torch.Tensor:
    """attn._compute_q_b without the RoPE, which the K store launch applies."""
    q, _ = attn.wq_b(q)
    return q.view(-1, attn.n_local_heads, attn.head_dim)


def compute_q_b(
    attn,
    q_lora: torch.Tensor,
    q_for_wqb,
    positions: torch.Tensor,
    q_out: Optional[torch.Tensor],
    unified: bool,
    use_cp: bool,
):
    """attn._compute_q_b(q_for_wqb, positions, q_out) as (q, the q_lora the indexer reads,
    whether the K norm-rope-store launch ropes q). The indexer takes the fused operand:
    its wq_b would re-round onto the same grid."""
    fuse_q_rope = _is_gfx95_supported and fuses_q_rope_into_k_store(
        attn, q_out, unified=unified, use_cp=use_cp
    )
    if fuse_q_rope:
        q = wq_b_unroped(attn, q_for_wqb)
    else:
        q = attn._compute_q_b(q_for_wqb, positions, q_out)
    return q, q_for_wqb, fuse_q_rope


def skip_head_pad(attn) -> bool:
    # only tilelang is built for padded head widths; aiter and Triton take the real head count
    return attn.attn_tp_size > 1 and not hip_attention_needs_head_pad()


def attention_inv_rope(
    attn,
    positions: torch.Tensor,
    forward_batch,
    unified: bool,
    *,
    wo_a_applies_inv_rope: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """(freqs_real, positions) for aiter's gfx950 sparse kernel to apply the inverse RoPE
    itself; None for the other kernels, and where the fp8 wo_a front end or the prefill
    graph op keeps it."""
    if (
        unified
        or not _is_gfx95_supported
        or resolve_hip_flashmla_backend() != "aiter_sparse"
        or wo_a_applies_inv_rope
    ):
        return None
    if forward_batch.forward_mode.is_extend() and is_in_breakable_cuda_graph():
        return None
    return torch.view_as_real(attn.freqs_cis).flatten(-2), positions


# ---- DeepseekV4DecoderLayer ----


def init_decoder_layer(layer, quant_config) -> None:
    """The ROCm state of DeepseekV4DecoderLayer.__init__. input_layernorm also emits the
    pre-quantized operand of the dense projections (whether wqkv_a consumes native MXFP8
    resolves on first use), and on gfx950 one launch runs hc_post, the pre-collapse and
    the mixing stats; the kernel only supports hc_mult 4."""
    layer.fused_rmsnorm_fp8_quant = gfx95_dense.fused_rmsnorm_fp8_quant_eligible(
        quant_config
    )
    layer.fused_rmsnorm_fake_quant = gfx95_dense.fused_rmsnorm_fake_quant_eligible(
        quant_config
    )
    layer._wqkv_a_native_consumer_checked = False
    layer._wqkv_a_native_consumer = False
    layer.hc_boundary_fused = (
        _is_gfx95_supported and layer.hc_pre_from_prev_sublayer and layer.hc_mult == 4
    )


def hc_post(layer, x, residual, post, comb) -> Optional[torch.Tensor]:
    """V4.1's gfx950 hc_post at 768-4096 rows: the split-H kernel with 2048-wide blocks,
    whose ordinary stores preserve locality for the following mHC reader; None elsewhere."""
    if not (
        _is_gfx95_supported
        and layer.config.model_type == "deepseek_v41"
        and 768 <= x.shape[0] <= 4096
        and x.shape[1] == 5120
        and residual.shape == (x.shape[0], 4, 5120)
        and post.shape == (x.shape[0], 4)
        and comb.shape == (x.shape[0], 4, 4)
        and x.dtype == residual.dtype == torch.bfloat16
        and post.dtype == comb.dtype == torch.float32
        and all(t.is_contiguous() for t in (x, residual, post, comb))
    ):
        return None
    return mhc_post_split_h(x, residual, post, comb, block_size=2048)


def input_norm(
    layer,
    hidden_states: torch.Tensor,
    allow_aiter_quant: bool,
    coefficients,
    fused_rmsnorm_fp8_quant,
) -> Tuple[torch.Tensor, Optional[object]]:
    """layer.input_layernorm(hidden_states) as (the bf16 norm attention reads, the
    pre-quantized operand of its dense projections or None). coefficients is the fused
    boundary's pending reduce + sinkhorn, hosted by the gfx950 norm launch when that one runs."""
    if layer.fused_rmsnorm_fp8_quant and allow_aiter_quant:
        if coefficients is not None:
            coefficients.materialize()
        x_quant, hidden_states = fused_rmsnorm_fp8_quant(
            hidden_states, layer.input_layernorm.weight, layer.rms_norm_eps
        )
        return hidden_states, x_quant
    if layer.fused_rmsnorm_fake_quant:
        return gfx95_dense.input_norm_fake_quant(layer, hidden_states, coefficients)
    if coefficients is not None:
        coefficients.materialize()
    return layer.input_layernorm(hidden_states), None


# ---- DeepseekV4Model layer loop ----


def forward_layer_fused_boundary(
    model,
    i: int,
    *,
    positions: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    input_ids: torch.Tensor,
    forward_batch,
    input_ids_global: torch.Tensor,
    prev_pre: Optional[torch.Tensor],
    pending_post: Optional[Tuple[torch.Tensor, ...]],
    capture_dspark: bool,
):
    """Layer i through the fused mHC boundary. Its FFN hc_post stays pending for the next
    layer's boundary launch unless that layer reads the residual stream first (Engram gate,
    DSpark capture) or there is none; hidden_states is None while a post is pending."""
    nxt = i + 1
    defer_post = (
        nxt < model.end_layer
        and model.layers[nxt].engram is None
        and not (capture_dspark and nxt in model.dspark_layers_to_capture)
    )
    return forward_hc_pre_from_prev_fused_boundary(
        model.layers[i],
        positions=positions,
        hidden_states=hidden_states,
        input_ids=input_ids,
        forward_batch=forward_batch,
        input_ids_global=input_ids_global,
        prev_pre=prev_pre,
        pending_post=pending_post,
        defer_post=defer_post,
    )

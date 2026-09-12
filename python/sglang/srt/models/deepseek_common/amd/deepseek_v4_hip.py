"""ROCm glue of `deepseek_v4`: the branches the model takes under `_is_hip`, bound there as
`_hip`. The gfx950 dense fp8 routes live in `deepseek_v4_gfx95_dense`, the fused mHC
boundary in `deepseek_v4_fused_mhc`."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.hip_flash_mla import (
    hip_attention_needs_head_pad,
    hip_fused_decode_glue,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.models.deepseek_common.amd import deepseek_v4_gfx95_dense as gfx95_dense
from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
    forward_hc_pre_from_prev_fused_boundary,
)

live_rows = gfx95_dense.live_rows
wo_a_fp8_grid_matmul = gfx95_dense.wo_a_fp8_grid_matmul
wo_b_takes_fp8_grid = gfx95_dense.wo_b_takes_fp8_grid


# ---- MqaAttentionBase / MQALayer ----


def init_mqa_attention_base(attn) -> None:
    # resolved on first use by `wo_b_takes_fp8_grid`, once the weights are loaded
    attn._wo_b_fp8_grid_checked = False
    attn._wo_b_fp8_grid_operand = False


def use_fused_qk_norm_rope(attn, quant_config) -> bool:
    # the fused qk-norm-rope store quantizes for a 128x128-block wq_b; 32-block keeps the plain path
    return bool(
        envs.SGLANG_OPT_USE_FUSED_QK_NORM_ROPE.get()
        and attn.q_head_norm
        and isinstance(quant_config, Fp8Config)
        and quant_config.weight_block_size == [128, 128]
    )


def init_mqa_layer(attn, quant_config) -> None:
    # gfx950 32-block route: `q_norm` also emits the fp8-grid operand of `wq_b`
    attn.fused_rmsnorm_fake_quant = gfx95_dense.fused_rmsnorm_fake_quant_eligible(
        quant_config
    )
    attn._wq_b_native_consumer_checked = False
    attn._wq_b_native_consumer = None


def q_norm_for_wq_b(attn, q_lora: torch.Tensor) -> Tuple[torch.Tensor, object]:
    """`attn.q_norm(q_lora)` as (the bf16 norm the indexer reads, the operand `wq_b`
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
    """`attn._compute_q_b` without the RoPE, which the K store launch applies."""
    q, _ = attn.wq_b(q)
    return q.view(-1, attn.n_local_heads, attn.head_dim)


def skip_head_pad(attn) -> bool:
    # only tilelang is built for padded head widths; aiter and Triton take the real head count
    return attn.attn_tp_size > 1 and not hip_attention_needs_head_pad()


def attention_inv_rope(
    attn, positions: torch.Tensor, forward_batch, *, wo_a_applies_inv_rope: bool
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """`(freqs_real, positions)` for the attention kernel to apply the inverse RoPE itself;
    None where the fp8 wo_a front end or the prefill graph op keeps it."""
    if wo_a_applies_inv_rope:
        return None
    if forward_batch.forward_mode.is_extend() and is_in_breakable_cuda_graph():
        return None
    return torch.view_as_real(attn.freqs_cis).flatten(-2), positions


# ---- DeepseekV4DecoderLayer ----


def init_decoder_layer(layer, quant_config) -> None:
    layer.fused_rmsnorm_fp8_quant = gfx95_dense.fused_rmsnorm_fp8_quant_eligible(
        quant_config
    )
    layer.fused_rmsnorm_fake_quant = gfx95_dense.fused_rmsnorm_fake_quant_eligible(
        quant_config
    )
    layer._wqkv_a_native_consumer_checked = False
    layer._wqkv_a_native_consumer = None
    # hc_post, pre-collapse and mixing stats in one launch; the kernel only supports hc_mult 4
    layer.hc_boundary_fused = layer.hc_pre_from_prev_sublayer and layer.hc_mult == 4


def input_norm(
    layer, hidden_states: torch.Tensor, allow_aiter_quant: bool, coefficients
) -> Tuple[torch.Tensor, Optional[object]]:
    """`layer.input_layernorm(hidden_states)` as (the bf16 norm attention reads, the
    pre-quantized operand of its dense projections or None). ``coefficients`` is the fused
    boundary's pending reduce + sinkhorn, hosted by the gfx950 norm launch when that one runs."""
    if layer.fused_rmsnorm_fp8_quant and allow_aiter_quant:
        # deepseek_v4 binds this module at import, so the helper is read at call time
        from sglang.srt.models.deepseek_v4 import _fused_rmsnorm_fp8_quant

        if coefficients is not None:
            coefficients.materialize()
        x_quant, hidden_states = _fused_rmsnorm_fp8_quant(
            hidden_states, layer.input_layernorm.weight, layer.rms_norm_eps
        )
        return hidden_states, x_quant
    if layer.fused_rmsnorm_fake_quant:
        return gfx95_dense.input_norm_fake_quant(layer, hidden_states, coefficients)
    if coefficients is not None:
        coefficients.materialize()
    return layer.input_layernorm(hidden_states), None


# ---- DeepseekV4Model layer loop ----


def engram_image_select(config, input_ids: torch.Tensor):
    """`(input_ids, image_token_id)` when the fused Engram gate keeps the image-token rows
    itself (the model's `torch.where` after the gate), else None."""
    if not (config.model_type == "deepseek_v41" and config.vision_n_layers > 0):
        return None
    if not (input_ids.is_cuda and hip_fused_decode_glue()):
        return None
    return input_ids.contiguous(), config.image_token_id


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
    """Layer `i` through the fused mHC boundary. Its FFN hc_post stays pending for the next
    layer's boundary launch unless that layer reads the residual stream first (Engram gate,
    DSpark capture) or there is none; `hidden_states` is None while a post is pending."""
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

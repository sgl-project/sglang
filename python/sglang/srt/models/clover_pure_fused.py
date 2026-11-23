# SPDX-License-Identifier: Apache-2.0
"""Opt-in grouped BF16 preview for Clover Pure Block Diffusion, TP=1."""

from copy import copy

import torch
import torch.nn.functional as F
import triton.language as tl

from sglang.kernels.ops.attention.clover_gdn_triton import clover_local_memory_triton
from sglang.kernels.ops.mamba.clover_conv_triton import clover_local_memory_conv
from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_req_to_token_pool,
)
from sglang.srt.models.clover_pure_eager import (
    full_attention,
    norm,
    route,
    shared_expert,
)


def grouped_experts(layer, x):
    x = x[0]
    ids, weights = route(layer, x, True)
    ids = ids.int().contiguous()
    top_k = ids.shape[1]
    align = moe_align_block_size(ids, 16, layer.gate.weight.shape[0])
    config = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=1)

    def project(rows, matrix, divisor, multiply_weight):
        out = rows.new_empty((ids.numel(), matrix.shape[1]))
        invoke_fused_moe_kernel(
            rows,
            matrix,
            None,
            out,
            None,
            None,
            None,
            weights,
            ids,
            *align,
            mul_routed_weight=multiply_weight,
            top_k=divisor,
            config=config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
        )
        return out

    gate, up = project(x, layer.experts.w13_weight, top_k, False).chunk(2, -1)
    hidden = (F.silu(gate.float()) * up.float()).to(x.dtype)
    weighted = project(hidden, layer.experts.w2_weight, 1, True).view(len(x), top_k, -1)
    contributions = weighted.gather(
        1, ids.argsort(dim=-1)[..., None].expand_as(weighted)
    )
    out = contributions[:, 0].float()
    for slot in range(1, top_k):
        out = out + contributions[:, slot].float()
    return (out.to(x.dtype) + shared_expert(layer, x))[None]


def _linear_preview(layer, x, slots, valid):
    t = x.shape[1]
    raw, z = F.linear(x, layer.in_proj_qkvz.weight).split(
        (layer.conv_dim, layer.value_dim), -1
    )
    cache = get_req_to_token_pool().mamba2_layer_cache(layer.layer_id)
    mixed = clover_local_memory_conv(
        raw[0],
        clean_conv_states=cache.conv[0],
        cache_indices=slots,
        conv_weights=layer.conv1d.weight,
        bias=None,
        activation="silu",
        block_size=t,
        active_valid_mask=valid,
        conv_mode="independent_symmetric",
        conv_right_weight=layer.conv_right_weight,
    )
    q, k, v = mixed.split((layer.key_dim, layer.key_dim, layer.value_dim), -1)
    q = q.view(1, t, layer.num_k_heads, layer.head_k_dim)
    k = k.view_as(q)
    v = v.view(1, t, layer.num_v_heads, layer.head_v_dim)
    _, aw = layer.in_proj_ba.weight.chunk(2)
    g = -layer.A_log.float().exp() * F.softplus(
        F.linear(x, aw).float() + layer.dt_bias.float()
    )
    output = clover_local_memory_triton(
        q,
        k,
        v,
        g,
        clean_ssm_states=cache.temporal,
        cache_indices=slots,
        active_valid_mask=valid,
        scale=layer.head_k_dim**-0.5,
        softmax_logit_scale=layer.bd_softmax_logit_scale,
        softmax_history_bias=layer.bd_softmax_history_bias,
        softmax_history_c=layer.bd_softmax_history_c,
        pure_decay=True,
    )
    y = norm(output, layer.norm, layer.config.rms_norm_eps, z.reshape_as(output))
    return F.linear(y.reshape(1, t, -1), layer.out_proj.weight)


def _preview(model, input_ids, batch, slots, pages, valid):
    x = F.embedding(input_ids[None], model.model.embed_tokens.weight)
    eps = model.config.rms_norm_eps
    block_valid = torch.ones_like(input_ids, dtype=torch.bool)
    for layer in model.model.layers:
        h = norm(x, layer.input_layernorm, eps)
        if hasattr(layer, "linear_attn"):
            x = x + _linear_preview(layer.linear_attn, h, slots, block_valid)
        else:
            x = x + full_attention(layer, h, batch, True, (pages, valid))
        h = norm(x, layer.post_attention_layernorm, eps)
        x = x + grouped_experts(layer.mlp, h)
    return norm(x, model.model.norm, eps)[0]


def preview_forward(model, input_ids, batch):
    prefix = batch.extend_prefix_lens_cpu[0]
    capacity = 1 << max(7, (prefix - 1).bit_length())
    if envs.SGLANG_CLOVER_PURE_DISABLE_GRAPH.get():
        slots = (
            get_attn_backend().linear_attn_backend.forward_metadata.mamba_cache_indices
        )
        prefix_pages = (
            get_req_to_token_pool()
            .req_to_token[batch.req_pool_indices[0], :prefix]
            .long()
        )
        pages = torch.cat((prefix_pages, prefix_pages[:1].expand(capacity - prefix)))
        valid = torch.arange(capacity, device=pages.device) < prefix
        return _preview(model, input_ids, batch, slots.long(), pages, valid)
    cached = getattr(model, "_pure_fused_graph", None)
    if cached is None or cached[0] != capacity:
        # Keep only one bucket resident. Positions, page IDs and the recurrent
        # slot are refreshed on every replay, including after request-slot reuse.
        model._pure_fused_graph = None
        cached = None
        static_batch = copy(batch)
        positions = torch.empty_like(batch.positions)
        static_batch.positions = positions
        tokens = torch.empty_like(input_ids)
        slots = batch.req_pool_indices.new_empty(1, dtype=torch.long)
        pages = slots.new_zeros(capacity)
        valid = pages.new_zeros(capacity, dtype=torch.bool)
        graph = torch.cuda.CUDAGraph()
    else:
        _, graph, tokens, positions, slots, pages, valid, output = cached

    tokens.copy_(input_ids)
    positions.copy_(batch.positions)
    slots.copy_(
        get_attn_backend().linear_attn_backend.forward_metadata.mamba_cache_indices
    )
    pages[:prefix].copy_(
        get_req_to_token_pool().req_to_token[batch.req_pool_indices[0], :prefix]
    )
    if prefix < capacity:
        pages[prefix:].copy_(pages[:1].expand(capacity - prefix))
    valid[:prefix].fill_(True)
    valid[prefix:].fill_(False)
    if cached is None:
        _preview(model, tokens, static_batch, slots, pages, valid)
        with torch.cuda.graph(graph):
            output = _preview(model, tokens, static_batch, slots, pages, valid)
        model._pure_fused_graph = (
            capacity,
            graph,
            tokens,
            positions,
            slots,
            pages,
            valid,
            output,
        )
    graph.replay()
    return output

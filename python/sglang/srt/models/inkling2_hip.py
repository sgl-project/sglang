"""ROCm: Inkling2 decoder layers apply their input norms inside the block attention
residual kernel instead of after it."""

import torch

from sglang.kernels.ops.inkling2.norm_hip import attn_res
from sglang.srt.models.inkling_common.inkling2_attn import AttentionSources


def _norm(norm):
    return (norm.weight, norm.pre_weight, norm.eps, norm.output_dtype)


def _normed_attn_res(state, blocks, partial, queries, scaled, norms, out=None):
    """block_attn_res with one RMSNorm per query applied in the same kernel; out
    receives the normalized (queries, tokens, hidden)."""
    if queries.ndim == 1:
        queries = queries.unsqueeze(0)
    addend = None
    if isinstance(partial, tuple):
        partial, addend = partial
    sources = [*blocks, partial]
    return attn_res(
        sources,
        addend,
        scaled.view(queries.shape)
        if scaled is not None
        else queries.float() * (sources[0].shape[-1] ** -0.5),
        normalize_values=state.attn_res.normalize_values,
        norms=norms,
        out=out,
    )


def _attention_inputs(layer, state, norms, out=None):
    if state.blocks:
        queries = state.attn_res.attn_query(layer.layer_id)
        scaled = state.attn_res.scaled_query("attn", layer.layer_id)
    else:
        # A single source takes all the weight, whatever the query.
        queries = scaled = state.partial.new_zeros(
            (len(norms), state.partial.shape[-1]), dtype=torch.float32
        )
    return _normed_attn_res(state, state.blocks, state.partial, queries, scaled, norms, out)


def attention_input(layer, state):
    """The layer's attention input: the block attention residual with attn_norm (local
    layers) or the q / k / v input norms (global layers) applied in the same kernel."""
    if layer.is_local:
        return _attention_inputs(layer, state, [_norm(layer.attn_norm)])[0]
    partial = state.partial
    stacked = partial.new_empty(
        (3, *partial.shape), dtype=layer.q_input_norm.output_dtype
    )
    q, k, v = _attention_inputs(
        layer,
        state,
        [
            _norm(norm)
            for norm in (layer.q_input_norm, layer.k_input_norm, layer.v_input_norm)
        ],
        stacked,
    )
    return AttentionSources(q, k, v)


def mlp_input(layer, state, attention):
    """The layer's MLP input, with mlp_norm applied in the residual kernel."""
    res, idx = state.attn_res, layer.layer_id
    if res.is_block_boundary(idx):
        blocks, partial = [*state.blocks, state.partial], attention
    else:
        blocks, partial = state.blocks, (state.partial, attention)
    return _normed_attn_res(
        state,
        blocks,
        partial,
        res.mlp_queries[idx],
        res.scaled_query("mlp", idx),
        [_norm(layer.mlp_norm)],
    )[0]

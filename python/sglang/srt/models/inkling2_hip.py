"""ROCm: Inkling2 decoder layers apply their input norms inside the block attention
residual kernel instead of after it."""

from sglang.srt.models.inkling_common.inkling2_attn import AttentionSources


def _norm(norm):
    return (norm.weight, norm.pre_weight, norm.eps, norm.output_dtype)


def attention_input(layer, state):
    """The layer's attention input: the block attention residual with attn_norm (local
    layers) or the q / k / v input norms (global layers) applied in the same kernel."""
    if layer.is_local:
        return state.attention_input(layer.layer_id, norms=[_norm(layer.attn_norm)])[0]
    partial = state.partial
    stacked = partial.new_empty(
        (3, *partial.shape), dtype=layer.q_input_norm.output_dtype
    )
    q, k, v = state.attention_input(
        layer.layer_id,
        norms=[
            _norm(norm)
            for norm in (layer.q_input_norm, layer.k_input_norm, layer.v_input_norm)
        ],
        out=stacked,
    )
    return AttentionSources(q, k, v)


def mlp_input(layer, state, attention):
    """The layer's MLP input, with mlp_norm applied in the residual kernel."""
    return state.mlp_input(layer.layer_id, attention, norms=[_norm(layer.mlp_norm)])

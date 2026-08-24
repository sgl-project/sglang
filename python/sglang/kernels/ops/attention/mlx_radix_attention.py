"""Backend-neutral MLX attention primitives."""

from sglang.kernels.ops.attention._deferred_radix_attention_mlx import (
    DeferredAttentionSpec,
    deferred_attention_reject_reason,
    radix_decode_deferred,
    radix_prefill_deferred,
)


def causal_gqa(
    query,
    key,
    value,
    *,
    spec: DeferredAttentionSpec,
):
    """Run one prefix-free causal GQA prefill sequence in MLX."""
    import mlx.core as mx

    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("causal GQA requires [tokens, heads, head_dim] inputs")
    if (
        query.shape[1:] != (spec.num_q_heads, spec.head_dim)
        or key.shape[1:] != (spec.num_kv_heads, spec.head_dim)
        or value.shape != key.shape
    ):
        raise ValueError("causal GQA tensors do not match the attention spec")
    output = mx.fast.scaled_dot_product_attention(
        query.transpose(1, 0, 2)[None, ...],
        key.transpose(1, 0, 2)[None, ...],
        value.transpose(1, 0, 2)[None, ...],
        scale=spec.attention_scale,
        mask="causal",
    )
    return output[0].transpose(1, 0, 2)


__all__ = [
    "DeferredAttentionSpec",
    "causal_gqa",
    "deferred_attention_reject_reason",
    "radix_decode_deferred",
    "radix_prefill_deferred",
]

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
    batch_size: int = 1,
):
    """Run independent, equally sized prefix-free causal GQA segments."""
    import mlx.core as mx

    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("causal GQA requires [tokens, heads, head_dim] inputs")
    if (
        query.shape[1:] != (spec.num_q_heads, spec.head_dim)
        or key.shape[1:] != (spec.num_kv_heads, spec.head_dim)
        or value.shape != key.shape
        or query.shape[0] != key.shape[0]
    ):
        raise ValueError("causal GQA tensors do not match the attention spec")
    if batch_size <= 0 or query.shape[0] % batch_size:
        raise ValueError("causal GQA token count must divide into positive batch_size")
    seq_len = query.shape[0] // batch_size
    output = mx.fast.scaled_dot_product_attention(
        query.reshape(batch_size, seq_len, spec.num_q_heads, spec.head_dim).transpose(
            0, 2, 1, 3
        ),
        key.reshape(batch_size, seq_len, spec.num_kv_heads, spec.head_dim).transpose(
            0, 2, 1, 3
        ),
        value.reshape(batch_size, seq_len, spec.num_kv_heads, spec.head_dim).transpose(
            0, 2, 1, 3
        ),
        scale=spec.attention_scale,
        mask="causal",
    )
    return output.transpose(0, 2, 1, 3).reshape(query.shape)


__all__ = [
    "DeferredAttentionSpec",
    "causal_gqa",
    "deferred_attention_reject_reason",
    "radix_decode_deferred",
    "radix_prefill_deferred",
]

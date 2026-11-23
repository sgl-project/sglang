"""Shared TOP head normalization; the stock fused kernel owns the arithmetic."""

import torch

from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
from sglang.srt.utils.custom_op import register_custom_op


def supports_fused_qk_norm(head_dim: int, dtype: torch.dtype) -> bool:
    """Only the BF16 head128 path is checked to be batch- and head-partition invariant."""
    return head_dim == 128 and dtype == torch.bfloat16


@register_custom_op(op_name="top_fused_qk_norm", mutates_args=["q", "k"])
def fused_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> None:
    if not supports_fused_qk_norm(q.shape[-1], q.dtype):
        raise NotImplementedError("TOP fused QK norm requires BF16 head dimension 128")
    # The stock kernel validates tensor shapes, strides, devices and weight dtypes.
    fused_inplace_qknorm(q, k, q_weight, k_weight, eps=eps, head_dim=128)


def maybe_fused_qk_norm(q, k, q_norm, k_norm, *, head_dim, batch_size, allow_inplace):
    """TOP's fused Q/K RMSNorm for apply_qk_norm; None when its conditions do not hold."""
    from sglang.srt.true_on_policy import is_true_on_policy_enabled
    from sglang.srt.utils import is_cuda

    q_eps = q_norm.variance_epsilon
    if not (
        is_cuda()
        and is_true_on_policy_enabled()
        and supports_fused_qk_norm(head_dim, q.dtype)
        and q_eps == k_norm.variance_epsilon
        and all(
            norm.weight.dtype == q.dtype
            and not norm.cast_x_before_out_mul
            and norm.override_orig_dtype is None
            and norm.variance_size_override is None
            for norm in (q_norm, k_norm)
        )
    ):
        return None
    if not allow_inplace:
        q, k = q.clone(), k.clone()
    fused_qk_norm(
        q.view(batch_size, -1, head_dim),
        k.view(batch_size, -1, head_dim),
        q_norm.weight,
        k_norm.weight,
        q_eps,
    )
    return q, k

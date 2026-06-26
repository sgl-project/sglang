# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/batch_invariant_ops.py

import batch_invariant_ops  # noqa: F401
import torch
import torch_npu


def npu_mm_batch_invariant(a, b):
    return torch.ops.batch_invariant_ops.npu_mm_batch_invariant(a, b)


def npu_matmul_batch_invariant(a, b):
    return torch.ops.batch_invariant_ops.npu_matmul_batch_invariant(a, b)


def npu_mean_batch_invariant(
    input, dim, keepdim=False, dtype: torch.dtype | None = None
):
    assert dtype is None or dtype == torch.float32, f"unsupported dtype: {dtype}"
    if len(dim) == 1:
        return torch.ops.batch_invariant_ops.npu_reduce_mean_batch_invariant(
            input, dim[0], keepdim=keepdim
        )
    else:
        assert input.dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }, "only float types supported for now"
        n_elems = 1
        for d in dim:
            n_elems *= input.shape[d]
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=torch.float32) / n_elems


def npu_log_softmax_batch_invariant(input, dim, _half_to_float):
    assert not _half_to_float, "not implemented"
    return torch.ops.batch_invariant_ops.npu_log_softmax_batch_invariant(input, dim=dim)


def npu_fused_infer_attention_score_batch_invariant(*args, **kwargs):
    return (
        torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant(
            *args, **kwargs
        )
    )


def npu_add_rms_norm_batch_invariant(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
):
    """
    AclnnAddRmsNorm can't ensure batch invariant,
    so we need to split it into add and rms_norm.
    """
    x_ = x + residual
    residual_ = x_
    x_, _ = torch_npu.npu_rms_norm(x_, weight, eps)
    return x_, None, residual_

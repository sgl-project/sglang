# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/batch_invariant_ops.py

import batch_invariant_ops  # noqa: F401
import torch
import torch_npu
from torch_npu.npu import NpuGraphOpHandler, register_npu_graph_handler


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


# Preserve the OpOverloadPacket, including its graph-compatible .out overload.
npu_fused_infer_attention_score_batch_invariant = (
    torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant
)
npu_fia_batch_invariant_get_max_workspace = torch.ops.batch_invariant_ops._npu_fused_infer_attention_score_batch_invariant_get_max_workspace


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


@register_npu_graph_handler(
    [
        "npu_fused_infer_attention_score_batch_invariant.default",
        "npu_fused_infer_attention_score_batch_invariant.out",
    ]
)
class NativeFIAGraphHandler(NpuGraphOpHandler):
    """Rebind the native FIA's keyword-only CPU lengths on graph updates."""

    @classmethod
    def prepare_capture(cls, func, args, kwargs):
        ops = torch.ops.batch_invariant_ops
        out_op = npu_fused_infer_attention_score_batch_invariant.out
        # graph.update replaces matching kwargs; include optional lengths too.
        kwargs = dict(kwargs)
        kwargs.setdefault("actual_seq_lengths", None)
        kwargs.setdefault("actual_seq_lengths_kv", None)
        if func is out_op:
            return func, args, kwargs
        workspace = npu_fia_batch_invariant_get_max_workspace(*args, **kwargs)
        # Match infer_output's keyword-only schema; omit absent kwargs to keep
        # its defaults. Sequence lengths are not accepted by this helper;
        # they remain in kwargs for FIA execution and graph updates.
        keys = [
            "input_layout",
            "quant_scale2",
            "block_table",
            "num_heads",
            "num_key_value_heads",
            "softmax_lse_flag",
            "query_rope",
        ]
        # FIA positional args are (query, key, value): args[0] is Q, args[2] is V.
        # This helper derives output shapes/dtypes from Q, V and the options
        # above, then allocates (attention_output, softmax_lse) for .out.
        # It does not compute attention; .out fills these buffers.
        output = ops._npu_fused_infer_attention_score_batch_invariant_infer_output(
            args[0], args[2], **{k: kwargs[k] for k in keys if k in kwargs}
        )
        kwargs.update(workspace=workspace, out=list(output))
        return out_op, args, kwargs

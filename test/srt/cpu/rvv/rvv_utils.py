import importlib.util

import torch


def has_sgl_kernel_op(op_name: str) -> bool:
    """Check if a specific operator is available in sgl_kernel."""
    if not importlib.util.find_spec("sgl_kernel"):
        return False
    try:
        getattr(torch.ops.sgl_kernel, op_name)
        return True
    except (AttributeError, RuntimeError):
        return False


# Numeric tolerances for shared RVV test paths: precision[key][dtype].
precision = {
    "pointwise_default": {
        torch.bfloat16: 3e-2,
        torch.float16: 1e-3,
        torch.float32: 1e-5,
    },
    "norm_layer_bias": {
        torch.bfloat16: 3e-2,
        torch.float16: 1e-2,
        torch.float32: 1e-5,
    },
    "linear_gemm": {torch.bfloat16: 1.5e-1, torch.float16: 1e-1},
    "rotary_embedding": {torch.bfloat16: 3e-2, torch.float16: 1e-2},
    "attention_decode_logit_cap": {torch.float16: 1e-2, torch.bfloat16: 1e-1},
    "attention_decode": {
        torch.bfloat16: 1e-1,
        torch.float16: 7e-2,
        torch.float32: 1e-5,
    },
    "attention_extend": {
        torch.bfloat16: 1e-1,
        torch.float16: 3e-3,
        torch.float32: 1e-5,
    },
}


def helper_non_contiguous(t: torch.Tensor) -> torch.Tensor:
    """Return a non-contiguous view of t by striding the batch dimension 2x."""
    buf = torch.empty(t.shape[0] * 2, *t.shape[1:], dtype=t.dtype)
    buf[::2] = t
    return buf[::2]

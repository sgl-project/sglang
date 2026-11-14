"""Speculative verify sampling ops, resolved once per platform.

A name is ``None`` where the platform has no kernel for it.
"""

import torch

from sglang.srt.utils import is_cuda, is_musa, is_npu


def _with_zero_draft_probs(kernel):
    """Adapt a kernel that still takes ``draft_probs`` to the CUDA/MUSA signature.

    The CUDA and MUSA kernels mask the tokens rejected at the last accepted node
    themselves. The NPU kernel still takes a zeroed ``draft_probs`` scratch
    buffer, so it is allocated here for that kernel only.
    """

    def tree_speculative_sampling_target_only(*, target_probs, **kwargs):
        kernel(
            target_probs=target_probs,
            draft_probs=torch.zeros_like(target_probs),
            **kwargs,
        )

    return tree_speculative_sampling_target_only


if is_npu():
    try:
        from sgl_kernel_npu.sample import (
            chain_speculative_sampling_triton,
        )
        from sgl_kernel_npu.sample import (
            tree_speculative_sampling_target_only as _npu_tree_sampling,
        )
    except ImportError:
        chain_speculative_sampling_triton = None
        tree_speculative_sampling_target_only = None
    else:
        tree_speculative_sampling_target_only = _with_zero_draft_probs(
            _npu_tree_sampling
        )
else:
    from sglang.kernels.ops.speculative.reject_sampling import (
        chain_speculative_sampling_triton,
    )

    if is_cuda():
        from sglang.kernels.ops.speculative.sampling import (
            tree_speculative_sampling_target_only,
        )
    elif is_musa():
        try:
            from sgl_kernel import tree_speculative_sampling_target_only
        except Exception:
            tree_speculative_sampling_target_only = None
    else:
        tree_speculative_sampling_target_only = None

__all__ = ["chain_speculative_sampling_triton", "tree_speculative_sampling_target_only"]

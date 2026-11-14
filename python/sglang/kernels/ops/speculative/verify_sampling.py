"""Speculative verify sampling ops, resolved once per platform.

A name is ``None`` where the platform has no kernel for it.
"""

from sglang.srt.utils import is_cuda, is_musa, is_npu, is_xpu

if is_npu():
    try:
        from sgl_kernel_npu.sample import (
            chain_speculative_sampling_triton,
            tree_speculative_sampling_target_only,
        )
    except ImportError:
        chain_speculative_sampling_triton = None
        tree_speculative_sampling_target_only = None
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
    elif is_xpu():
        # Top-1 chains only; EAGLE verify keeps XPU tree drafts on greedy.
        from sglang.kernels.ops.speculative.reject_sampling import (
            chain_speculative_sampling_target_only_triton as tree_speculative_sampling_target_only,
        )
    else:
        tree_speculative_sampling_target_only = None

__all__ = ["chain_speculative_sampling_triton", "tree_speculative_sampling_target_only"]

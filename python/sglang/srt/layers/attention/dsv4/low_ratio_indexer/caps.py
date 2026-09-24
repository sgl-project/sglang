"""Which low-ratio indexer kernels this build can run."""

from __future__ import annotations

import functools

import torch


@functools.cache
def is_sm100_or_newer() -> bool:
    # DeepGEMM's fp8_fp4 mqa-logits kernels need SM100+; Hopper takes the torch indexer.
    return torch.cuda.get_device_capability()[0] >= 10


@functools.cache
def has_dense_fp4_indexer() -> bool:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    return hasattr(deep_gemm, "fp8_fp4_mqa_logits")


@functools.cache
def use_deep_gemm_prefill() -> bool:
    """Whether an eager prefill scores on DeepGEMM rather than torch. A CP rank
    scores rank-local rows, which only DeepGEMM serves, so CP ignores the env."""
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_parallel

    if not (is_sm100_or_newer() and has_dense_fp4_indexer()):
        return False
    return (
        get_parallel().attn_cp_size > 1
        or not envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.get()
    )

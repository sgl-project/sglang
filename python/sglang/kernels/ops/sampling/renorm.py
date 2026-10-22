"""Probability renormalization and DSpark sampling primitives.

Renormalization names are ``None`` where the platform has no kernel for them.
DSpark primitives resolve their platform implementation on first use.
"""

from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu
from sglang.utils import LazyImport

if is_cuda():
    # Resolved on first call: CUDA builds without FlashInfer must still import
    # this module.
    top_k_renorm_prob = LazyImport("flashinfer.sampling", "top_k_renorm_probs")
    top_p_renorm_prob = LazyImport("flashinfer.sampling", "top_p_renorm_probs")
elif is_hip():
    from sglang.kernels.ops.sampling.renorm_triton import (
        top_k_renorm_probs_triton as top_k_renorm_prob,
    )
    from sglang.kernels.ops.sampling.renorm_triton import (
        top_p_renorm_probs_triton as top_p_renorm_prob,
    )
elif is_npu():
    try:
        from sgl_kernel_npu.sample import top_k_renorm_prob, top_p_renorm_prob
    except ImportError:
        top_k_renorm_prob = None
        top_p_renorm_prob = None
elif is_musa():
    try:
        from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob
    except Exception:
        top_k_renorm_prob = None
        top_p_renorm_prob = None
else:
    top_k_renorm_prob = None
    top_p_renorm_prob = None

__all__ = ["top_k_renorm_prob", "top_p_renorm_prob"]


# DSpark sampling primitives share the GPU signatures and tensor contracts.
if is_npu():
    markov_sample_step = LazyImport("sgl_kernel_npu.sample", "markov_sample_step")
    markov_greedy_step_triton = LazyImport(
        "sgl_kernel_npu.sample", "markov_greedy_step_triton"
    )
    row_argmax = LazyImport("sgl_kernel_npu.sample", "row_argmax")
    compute_dflash_correct_drafts_and_bonus = LazyImport(
        "sgl_kernel_npu.sample", "compute_dflash_correct_drafts_and_bonus"
    )
    softmax_temp_triton = LazyImport("sgl_kernel_npu.sample", "softmax_temp_triton")
else:
    markov_sample_step = LazyImport(
        "sglang.kernels.ops.speculative.dspark.markov_sample", "markov_sample_step"
    )
    markov_greedy_step_triton = LazyImport(
        "sglang.kernels.ops.speculative.dspark.dspark_draft_model",
        "markov_greedy_step_triton",
    )
    row_argmax = LazyImport("sglang.kernels.ops.speculative.row_argmax", "row_argmax")
    compute_dflash_correct_drafts_and_bonus = LazyImport(
        "sglang.srt.speculative.dflash_utils",
        "compute_dflash_correct_drafts_and_bonus",
    )
    softmax_temp_triton = LazyImport(
        "sglang.kernels.ops.speculative.dspark.dspark_accept", "softmax_temp_triton"
    )

__all__ += [
    "markov_sample_step",
    "markov_greedy_step_triton",
    "row_argmax",
    "compute_dflash_correct_drafts_and_bonus",
    "softmax_temp_triton",
]

"""Top-k / top-p probability renormalization, resolved once per platform.

A name is ``None`` where the platform has no kernel for it.
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

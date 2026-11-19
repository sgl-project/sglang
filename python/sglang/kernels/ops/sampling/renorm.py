"""Top-k / top-p probability renormalization, resolved once per platform.

A name is ``None`` where the platform has no kernel for it.
"""

import torch

from sglang._lazy_import import LazyImport
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, is_xpu

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

    def _batch_scalar(value, probs, dtype):
        if not isinstance(value, torch.Tensor):
            return torch.full(
                (probs.shape[0],), value, device=probs.device, dtype=dtype
            )
        if value.ndim == 0:
            return value.to(device=probs.device, dtype=dtype).expand(probs.shape[0])
        return value.to(device=probs.device, dtype=dtype)

    def top_k_top_p_renorm_prob(probs, top_ks, top_ps, max_keep):
        """Apply top-p and top-k together and write back only the survivors.

        Both filters keep a prefix of the sorted probabilities, so the joint
        result equals running the two kernels in sequence. Writing back only
        max_keep columns avoids the full-vocabulary un-sort that the top-p
        kernel performs, a scalar-bound ScatterElementsV2 costing ~21 ms per
        call for a [20, 248320] verify batch on 910B.
        """
        vocab_size = probs.shape[-1]
        width = min(int(max_keep), vocab_size)
        top_ks = _batch_scalar(top_ks, probs, torch.long).clamp(min=1, max=vocab_size)
        top_ps = _batch_scalar(top_ps, probs, probs.dtype).clamp(min=0.0, max=1.0)
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        sorted_probs = sorted_probs[:, :width]
        positions = torch.arange(width, device=probs.device).view(1, -1)
        keep = positions < top_ks.view(-1, 1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        keep = keep & (cumulative_probs - sorted_probs <= top_ps.view(-1, 1))
        sorted_probs = sorted_probs.masked_fill(~keep, 0.0)
        total = sorted_probs.sum(dim=-1, keepdim=True).clamp_min_(1e-20)
        kept = sorted_probs / total
        out = torch.zeros_like(probs)
        return out.scatter_(dim=-1, index=sorted_indices[:, :width], src=kept)
elif is_musa() or is_xpu():
    try:
        from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob
    except Exception:
        top_k_renorm_prob = None
        top_p_renorm_prob = None
else:
    top_k_renorm_prob = None
    top_p_renorm_prob = None

__all__ = ["top_k_renorm_prob", "top_k_top_p_renorm_prob", "top_p_renorm_prob"]

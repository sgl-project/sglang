"""Top-k / top-p probability renormalization, resolved once per platform.

A name is ``None`` where the platform has no kernel for it.
"""

import torch

from sglang._lazy_import import LazyImport
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, is_xpu

# Whether the platform top-k kernel accepts an optional bound on how many
# leading (sorted) columns can survive its mask. Only the NPU kernel implements
# it today; callers probe this before passing max_keep, because every other
# platform rejects the argument.
supports_max_keep_bound = False


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
        from sgl_kernel_npu.sample import top_k_renorm_prob as _ref_top_k_renorm_prob
        from sgl_kernel_npu.sample import top_p_renorm_prob as _ref_top_p_renorm_prob
    except ImportError:
        _ref_top_k_renorm_prob = None
        _ref_top_p_renorm_prob = None

    if _ref_top_k_renorm_prob is not None:

        def _as_batch(value, probs, dtype, name):
            if not isinstance(value, torch.Tensor):
                return torch.full(
                    (probs.shape[0],), value, device=probs.device, dtype=dtype
                )
            if value.ndim == 0:
                return value.to(device=probs.device, dtype=dtype).expand(probs.shape[0])
            if value.ndim != 1 or value.shape[0] != probs.shape[0]:
                raise ValueError(f"{name} must be a scalar or have shape [batch]")
            return value.to(device=probs.device, dtype=dtype)

        def _renorm_prefix(probs, sorted_probs, sorted_indices):
            kept = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min_(
                1e-20
            )
            out = torch.zeros_like(probs)
            return out.scatter_(dim=-1, index=sorted_indices, src=kept)

        def top_k_renorm_prob(probs, top_ks, max_keep=None):
            """Keep each row top-k probabilities and renormalize the row.

            With max_keep (an upper bound on every row top-k) the un-sort only
            touches the leading columns: everything past max_keep was zeroed by
            the mask, and zeros_like already holds zeros there. The reference
            kernel instead scatters all [rows, vocab] entries, which aclnn
            lowers to a scalar-bound ScatterElementsV2 (aiv_scalar_ratio 0.993),
            measured at ~21 ms per call for 20 x 248320 on 910B.
            """
            if max_keep is None:
                return _ref_top_k_renorm_prob(probs, top_ks)
            vocab_size = probs.shape[-1]
            width = min(int(max_keep), vocab_size)
            sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
            top_ks = _as_batch(top_ks, probs, torch.long, "top_ks").clamp(
                min=1, max=vocab_size
            )
            positions = torch.arange(vocab_size, device=probs.device).view(1, -1)
            sorted_probs.masked_fill_(positions >= top_ks.view(-1, 1), 0.0)
            return _renorm_prefix(
                probs, sorted_probs[:, :width], sorted_indices[:, :width]
            )

        top_p_renorm_prob = _ref_top_p_renorm_prob

        supports_max_keep_bound = True
    else:
        top_k_renorm_prob = None
        top_p_renorm_prob = None

elif is_musa() or is_xpu():
    try:
        from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob
    except Exception:
        top_k_renorm_prob = None
        top_p_renorm_prob = None
else:
    top_k_renorm_prob = None
    top_p_renorm_prob = None

__all__ = ["top_k_renorm_prob", "top_p_renorm_prob", "supports_max_keep_bound"]

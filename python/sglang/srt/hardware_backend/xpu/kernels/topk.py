"""Intel XPU MoE top-k routing kernels.

Mirrors the MUSA backend precedent (``srt/hardware_backend/musa/kernels/topk.py``):
each non-CUDA backend exposes ``topk_sigmoid`` with the CUDA-compatible call
signature so the shared routing code in ``srt/layers/moe/topk.py`` can dispatch
to it behind a platform flag.

The heavy lifting is a native SYCL kernel shipped in the ``sgl_kernel`` (XPU)
wheel and registered as ``torch.ops.sgl_kernel.topk_sigmoid``. That kernel
computes the ungrouped biased-sigmoid top-k (selection uses ``sigmoid(logits) +
correction_bias``; the returned weights are the raw ``sigmoid`` values, optionally
renormalized to sum to 1). It does NOT handle ``routed_scaling_factor`` or
``num_fused_shared_experts`` — those are applied here in Python to match
``biased_grouped_topk_impl``.
"""

from typing import Optional, Tuple

import torch
from sgl_kernel import topk_sigmoid as _sgl_kernel_topk_sigmoid


def biased_grouped_topk_sigmoid_xpu(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ungrouped biased-sigmoid top-k on XPU via the native SYCL kernel.

    Returns ``(topk_weights [num_tokens, topk] float32, topk_ids
    [num_tokens, topk] int32)`` matching ``biased_grouped_topk_impl`` for the
    ungrouped (``num_expert_group == 1``, ``num_fused_shared_experts == 0``)
    case.
    """
    num_tokens = gating_output.shape[0]
    device = gating_output.device

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=device)
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)

    if num_tokens == 0:
        return topk_weights, topk_ids

    # The kernel selects experts with sigmoid(logits)+bias and writes the raw
    # sigmoid weights (renormalized when renormalize=True).
    _sgl_kernel_topk_sigmoid(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        correction_bias,
    )

    # biased_grouped_topk_impl applies routed_scaling_factor to the output only
    # when it renormalizes; mirror that here.
    if (
        renormalize
        and apply_routed_scaling_factor_on_output
        and routed_scaling_factor is not None
    ):
        topk_weights = topk_weights * routed_scaling_factor

    return topk_weights, topk_ids

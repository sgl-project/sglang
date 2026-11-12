"""Reference oracle for ``speculative.verify_target_probs``.

Composes the exact unfused calls ``eagle_sample`` makes today, in the same
order, so the fused kernel is checked against production behavior rather than a
re-derivation. Keep this in sync with ``srt/speculative/eagle_utils.py``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def reference_verify_target_probs(
    logits: torch.Tensor,
    *,
    rows_per_request: int,
    temperatures: torch.Tensor,
    additive_penalties: Optional[torch.Tensor] = None,
    scaling_penalties: Optional[torch.Tensor] = None,
    logit_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(target_probs, penalized_logits)``; ``logits`` is not modified."""
    from sglang.kernels.ops.sampling import softmax as sampling_softmax
    from sglang.srt.sampling.penaltylib.repetition_penalty import (
        apply_scaling_penalties,
    )

    logits = logits.clone()
    k = rows_per_request
    if additive_penalties is not None:
        logits.add_(torch.repeat_interleave(additive_penalties, k, dim=0))
    if scaling_penalties is not None:
        apply_scaling_penalties(
            logits, torch.repeat_interleave(scaling_penalties, k, dim=0)
        )
    if logit_bias is not None:
        logits.add_(torch.repeat_interleave(logit_bias, k, dim=0))

    probs = sampling_softmax(logits, temperatures=temperatures, rows_per_request=k)
    return probs, logits

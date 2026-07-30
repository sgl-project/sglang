from __future__ import annotations

from typing import Optional

import msgspec
import torch


class VerifyTreeMask(msgspec.Struct):
    """Target-verify tree-mask scratch + whether verify reads it.

    Keep together: a wrapper forwarding only one silently falls back to a fresh
    mask per step. Temporary home -- a phase-level buffer with no owner today
    (``spec_info`` is a per-phase union, so the graph registry cannot slot it).
    """

    buffer: torch.Tensor
    is_read: bool = True


def maybe_create_verify_tree_mask(
    *,
    is_draft_runner: bool,
    skip_prefill: bool,
    max_num_tokens: int,
    max_context_len: int,
    num_draft_tokens: Optional[int],
    device: torch.device | str,
    is_read: bool,
    dtype: torch.dtype = torch.bool,
) -> Optional[VerifyTreeMask]:
    """Worst-case scratch, sized to the tree kernel's write bound -- which holds
    even when nothing reads it. 100s of MB at long context, hence the gate."""
    if is_draft_runner or skip_prefill or not num_draft_tokens:
        return None
    return VerifyTreeMask(
        buffer=torch.zeros(
            max_num_tokens * (max_context_len + num_draft_tokens),
            dtype=dtype,
            device=device,
        ),
        is_read=is_read,
    )

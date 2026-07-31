from __future__ import annotations

from typing import Optional

import msgspec
import torch

from sglang.srt.speculative.eagle_utils import TreeMaskMode, default_tree_mask_mode


def tree_mask_numel(
    mode: TreeMaskMode, bs: int, num_draft_tokens: int, max_context_len: int
) -> int:
    """Cells the tree kernel writes for ``bs`` requests under ``mode``.

    FULL_MASK reaches 100s of MB at long context; QLEN_ONLY stays in the KBs.
    Bit-packed layouts are not sized here -- falling through to FULL_MASK would
    over-allocate them by orders of magnitude.
    """
    if mode == TreeMaskMode.QLEN_ONLY:
        per_req = num_draft_tokens * num_draft_tokens
    elif mode == TreeMaskMode.FULL_MASK:
        per_req = num_draft_tokens * (max_context_len + num_draft_tokens)
    else:
        raise NotImplementedError(f"Invalid tree mask: {mode=}")
    return bs * per_req


class VerifyMask(msgspec.Struct):
    """The target-verify mask.

    ``build_tree_kernel_efficient`` writes the buffer in place after draft, which
    is what lets the worker skip the ``seq_lens_sum`` D2H sync. Keep the three
    together: taking the buffer without its layout has the kernel write a shape
    the reader does not expect. The kernel writes every cell even when unread, so
    the buffer is always allocated.

    Temporary home -- a phase-level buffer with no owner today (``spec_info`` is a
    per-phase union the graph registry cannot slot).
    """

    buffer: torch.Tensor
    mode: TreeMaskMode
    is_read: bool = True

    def fits(self, bs: int, num_draft_tokens: int) -> bool:
        """Whether this batch's writes stay inside the buffer.

        Only the compact layout is checked. FULL_MASK keeps its pre-existing
        unconditional reuse -- its bound needs a max_context_len that composite
        backends do not carry -- so a batch past max_bs can still overflow it
        when draft * sum(seq_len) exceeds the buffer, as it could before.
        """
        if self.mode != TreeMaskMode.QLEN_ONLY:
            return True
        return self.buffer.numel() >= bs * num_draft_tokens * num_draft_tokens


def maybe_create_verify_mask(
    *,
    is_draft_runner: bool,
    skip_prefill: bool,
    max_bs: int,
    max_context_len: int,
    num_draft_tokens: Optional[int],
    device: torch.device | str,
    is_read: bool,
    dtype: torch.dtype = torch.bool,
) -> Optional[VerifyMask]:
    """Allocate for the captured max batch; None when nothing verifies."""
    if is_draft_runner or skip_prefill or not num_draft_tokens:
        return None
    mode = default_tree_mask_mode() if is_read else TreeMaskMode.QLEN_ONLY
    return VerifyMask(
        buffer=torch.zeros(
            tree_mask_numel(mode, max_bs, num_draft_tokens, max_context_len),
            dtype=dtype,
            device=device,
        ),
        mode=mode,
        is_read=is_read,
    )

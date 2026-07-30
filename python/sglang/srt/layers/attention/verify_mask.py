from __future__ import annotations

from typing import Optional

import msgspec
import torch

from sglang.srt.speculative.eagle_utils import TreeMaskMode, default_tree_mask_mode


def tree_mask_numel(
    mode: TreeMaskMode, bs: int, num_draft_tokens: int, max_context_len: int
) -> int:
    """Cells the tree kernel writes for ``bs`` requests under ``mode``.

    FULL_MASK spans the context dimension (100s of MB at long context);
    QLEN_ONLY is the qlen x qlen block alone and stays in the KBs.
    """
    per_req = (
        num_draft_tokens * num_draft_tokens
        if mode == TreeMaskMode.QLEN_ONLY
        else num_draft_tokens * (max_context_len + num_draft_tokens)
    )
    return bs * per_req


class VerifyMask(msgspec.Struct):
    """The target-verify mask: its buffer, its layout, and whether verify reads it.

    ``build_tree_kernel_efficient`` writes the buffer in place after draft, which
    is what lets the worker skip the ``seq_lens_sum`` D2H sync. The three travel
    together because the backend decides them at once -- a caller that took the
    buffer without the layout would have the kernel write a different shape than
    the reader expects.

    ``is_read=False`` (a chain tree has no branching to mask, some backends never
    consult one) skips the per-step fill and takes the compact layout, since
    nothing interprets the content. The kernel writes every cell either way, so
    the buffer is always allocated.

    Temporary home: a phase-level buffer with no owner today (``spec_info`` is a
    per-phase union, so the graph registry cannot slot it).
    """

    buffer: torch.Tensor
    mode: TreeMaskMode
    is_read: bool = True

    def fits(self, bs: int, num_draft_tokens: int, max_context_len: int) -> bool:
        """Whether this batch's writes stay inside the buffer.

        Sized for the captured max batch, so an eager batch past it must fall
        back to a fresh allocation -- the compact layout has no context-dimension
        slack to absorb the overflow.
        """
        return self.buffer.numel() >= tree_mask_numel(
            self.mode, bs, num_draft_tokens, max_context_len
        )


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
    """Allocate for the captured max batch, or nothing at all.

    Skipped for runners that never verify: draft runners, decode-only
    (``skip_prefill``) targets, and non-spec servers.
    """
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

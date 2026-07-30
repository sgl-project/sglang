from __future__ import annotations

from typing import Optional

import torch


class VerifyTreeMask:
    """The target-verify tree-mask scratch, owned in one place.

    ``build_tree_kernel_efficient`` writes this buffer in place once the draft
    stage finishes, which is what lets the worker size the mask without a
    ``seq_lens_sum`` D2H sync. Verify attention then reads it -- but only when
    ``is_read``: a chain (topk<=1) tree carries no branching to mask off, and
    some backends never consult the mask at all, so the per-step prefix fill
    has no reader and is skipped.

    The kernel writes every tree cell regardless, so the buffer must always be
    allocated at worst-case size even when nothing reads it.
    """

    def __init__(self, buffer: torch.Tensor, is_read: bool):
        self.buffer = buffer
        self.is_read = is_read

    @classmethod
    def create_full_mask(
        cls,
        *,
        max_num_tokens: int,
        max_context_len: int,
        num_draft_tokens: int,
        device: torch.device | str,
        is_read: bool,
        dtype: torch.dtype = torch.bool,
    ) -> VerifyTreeMask:
        """Allocate a worst-case FULL_MASK scratch.

        Sized for ``num_draft_tokens * (seq_len + num_draft_tokens)`` cells per
        request at the longest supported context -- the bound the tree kernel
        writes up to. Costs ``max_num_tokens * max_context_len`` bytes, which
        reaches 100s of MB at long context.
        """
        return cls(
            buffer=torch.zeros(
                max_num_tokens * (max_context_len + num_draft_tokens),
                dtype=dtype,
                device=device,
            ),
            is_read=is_read,
        )


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
    """Allocate the scratch only for a target runner that runs verify.

    Draft runners never verify, and a decode-only (``skip_prefill``) target
    never reaches the verify path, so neither should pay for the buffer.
    """
    if is_draft_runner or skip_prefill or not num_draft_tokens:
        return None
    return VerifyTreeMask.create_full_mask(
        max_num_tokens=max_num_tokens,
        max_context_len=max_context_len,
        num_draft_tokens=num_draft_tokens,
        device=device,
        is_read=is_read,
        dtype=dtype,
    )

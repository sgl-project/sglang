from __future__ import annotations

from typing import Optional

import msgspec
import torch


class VerifyTreeMask(msgspec.Struct):
    """The target-verify tree-mask scratch and whether verify reads it.

    ``build_tree_kernel_efficient`` writes this buffer in place once the draft
    stage finishes, which is what lets the worker size the mask without a
    ``seq_lens_sum`` D2H sync. Verify attention then reads it -- but only when
    ``is_read``: a chain (topk<=1) tree has no branching to mask off, and some
    backends never consult the mask at all, so the per-step prefix fill has no
    reader and is skipped. The two travel together because the backend decides
    both at once, and a wrapper forwarding only one of them would silently
    fall back to allocating a fresh mask every step.

    Temporary home: this is a phase-level buffer with no owner today (the graph
    buffer registry mirrors ForwardBatch fields, and ``spec_info`` is a
    per-phase union it cannot model), so it lives next to the backends that
    allocate it.
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
    """Allocate a worst-case FULL_MASK scratch for a target runner that verifies.

    Sized for ``num_draft_tokens * (seq_len + num_draft_tokens)`` cells per
    request at the longest supported context -- the bound the tree kernel writes
    up to, which holds even when nothing reads the mask. Costs
    ``max_num_tokens * max_context_len`` bytes, reaching 100s of MB at long
    context, so it is skipped entirely for runners that never verify: draft
    runners, decode-only (``skip_prefill``) targets, and non-spec servers.
    """
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

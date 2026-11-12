from typing import Optional

import torch


def track_indices(buffer: Optional[torch.Tensor], bs: int):
    return None if buffer is None else buffer[:bs]


def refresh_track_indices(buffer, src, *, raw_bs, bs, copy_dsts=None, copy_srcs=None):
    """Refresh virtual destinations without clearing rows that will be copied.

    Ordinary EAGLE joins its existing grouped copy. The multi-layer runner
    uses a fused preparation kernel for other inputs and copies here instead.
    """
    if buffer is None:
        return
    if src is None:
        buffer[:bs].zero_()
        return
    if raw_bs < bs:
        buffer[raw_bs:bs].zero_()
    if copy_dsts is None:
        buffer[:raw_bs].copy_(src)
    else:
        copy_dsts.append(buffer[:raw_bs])
        copy_srcs.append(src)

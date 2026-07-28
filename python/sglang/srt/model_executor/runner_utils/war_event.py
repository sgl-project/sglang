"""WAR-barrier read-done event factory for the CUDA graph runners."""

from typing import Optional

import torch

from sglang.srt.utils import is_cuda


def make_war_read_done_event(device_module) -> Optional[torch.cuda.Event]:
    """External event whose in-capture record() becomes a graph node that
    re-arms on every replay; None when unsupported (fallback paths)."""
    if not is_cuda():
        return None
    try:
        return device_module.Event(external=True)
    except TypeError:
        return None

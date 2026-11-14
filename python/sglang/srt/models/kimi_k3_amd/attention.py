"""Kimi-K3's attention on ROCm. ``kimi_k3`` calls these behind ``_is_hip``."""

from typing import Optional

import torch


def mono_output_gate_hip(pre: dict, alt: Optional[torch.cuda.Stream]) -> torch.Tensor:
    """g_proj(x) from a monokernel plugin's projection front (``_mono_pre``),
    once the gate stream the layer left pending has finished."""
    if alt is not None:
        torch.cuda.current_stream().wait_stream(alt)
    return pre["gate"]

from __future__ import annotations

import torch


def use_torch_reference(device: torch.device) -> bool:
    """Whether a canary launcher must fall back to its byte-equal torch reference.

    The write / verify / plan-entries kernels are CUDA-JIT only; HIP keeps them
    since torch reports it as ``"cuda"``. XPU / CPU / anything else falls back.
    """
    return device.type != "cuda"

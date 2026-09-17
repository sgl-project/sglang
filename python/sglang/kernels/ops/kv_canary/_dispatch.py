from __future__ import annotations

import torch


def use_torch_reference(device: torch.device) -> bool:
    """Whether a canary launcher must fall back to its byte-equal torch reference.

    The write / verify / plan-entries kernels are CUDA-JIT only; HIP keeps them
    since torch reports it as ``"cuda"``. XPU / CPU / anything else falls back.
    """
    if device.type == "cuda":
        return False

    # The references do host work and D2H, so a captured graph would record none of
    # their launches and replay a permanently clean canary. install_canary refuses
    # this pairing at startup; this catches any other capture context.
    # Imported here, not at module scope: the kernels layer must not depend on srt.
    from sglang.srt.utils import is_device_stream_capturing

    assert not is_device_stream_capturing(device), (
        f"kv-canary: the torch reference path ({device.type}) cannot run under graph "
        "capture; launch canary outside capture or run with --disable-cuda-graph"
    )
    return True

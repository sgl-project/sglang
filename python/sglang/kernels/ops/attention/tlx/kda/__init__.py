"""Kimi Delta Attention for gfx950 from the TLX op library.

``gfx950_prefill`` (``attention.kda_paged_prefill``) is a TLX kernel.
``gfx950_decode`` (``attention.kda_recurrent_decode``) uses no TLX primitive and
compiles on any Triton. Both are upstream's code reformatted for this
repository: re-vendor them rather than patching them here.
"""

from __future__ import annotations

import os

import torch

from sglang.kernels.ops.attention.tlx import is_available as tlx_available


def _is_gfx950(device: torch.device | None) -> bool:
    if not torch.version.hip or not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(device).gcnArchName
    return arch.split(":", 1)[0] == "gfx950"


def decode_available(device: torch.device | None = None) -> bool:
    return _is_gfx950(device)


def _tlx_is_native(tlx) -> bool:
    """Whether ``tlx`` ships inside the Triton package (fbtriton), not as a plugin."""
    import triton

    path = getattr(tlx, "__file__", None)
    triton_dir = os.path.dirname(os.path.abspath(triton.__file__))
    return path is not None and os.path.abspath(path).startswith(triton_dir + os.sep)


def prefill_available(device: torch.device | None = None) -> bool:
    """gfx950 and a TLX-capable Triton that will compile the prefill.

    A plugin TLX (uTLX) needs async copy for the prefill:
    ``TRITON_HIP_USE_ASYNC_COPY=0`` drops the pass that lays out its
    direct-to-LDS copies for their padded LDS layouts, and compilation fails.
    """
    if not (_is_gfx950(device) and tlx_available()):
        return False
    import triton
    from triton.language.extra import tlx

    return _tlx_is_native(tlx) or triton.knobs.amd.use_async_copy is not False

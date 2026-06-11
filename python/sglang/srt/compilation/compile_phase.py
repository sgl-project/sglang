"""torch.compile-internal phase markers used by the tc_piecewise backend.

Two pieces of state, both private to the torch.compile path (the
``cuda_piecewise_backend`` FX backend and the runner that drives it):

* ``_in_torch_compile_warmup`` — true during the warmup-compile loop
  where we run the compiled callable to trigger inductor compilation
  but explicitly do **not** capture into a CUDA graph yet.
  ``cuda_piecewise_backend`` reads this to short-circuit the capture
  branch.
* ``_pcg_capture_stream`` — the CUDA stream on which the runner is
  performing capture, surfaced so the FX backend can use the same
  stream for its own ``torch.cuda.graph(...)`` calls.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

_in_torch_compile_warmup = False
_pcg_capture_stream: "torch.cuda.Stream | None" = None


def is_in_torch_compile_warmup() -> bool:
    """True while inside the tc_piecewise warmup-compile pass. Strict subset of
    ``torch.compiler.is_compiling()``.
    """
    return _in_torch_compile_warmup


@contextmanager
def enable_torch_compile_warmup():
    """Mark the enclosed scope as the tc_piecewise warmup-compile pass. The FX
    piecewise backend uses this to skip CUDA graph capture during warmup.
    """
    global _in_torch_compile_warmup
    _in_torch_compile_warmup = True
    try:
        yield
    finally:
        _in_torch_compile_warmup = False


def get_pcg_capture_stream() -> "torch.cuda.Stream | None":
    return _pcg_capture_stream


@contextmanager
def set_pcg_capture_stream(stream: torch.cuda.Stream):
    global _pcg_capture_stream
    _pcg_capture_stream = stream
    try:
        yield
    finally:
        _pcg_capture_stream = None

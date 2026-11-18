"""DVPP JPEGD/VPC decoding; unsupported images retain PIL semantics."""

from __future__ import annotations

import atexit
import os
import threading
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from sglang.srt.hardware_backend.npu.dvpp import ACLFailure, check, supports_device
from sglang.srt.hardware_backend.npu.jpeg_header import prepare_jpeg

_local = threading.local()
_live = {}
_guard = threading.Lock()
_pid = os.getpid()
_closed = False


@dataclass
class _Entry:
    owner: object
    decoder: object
    retiring: bool = False


def _check_process():
    if os.getpid() != _pid:
        raise RuntimeError("NPU JPEG decoders must be initialized after worker spawn")


def _ensure_cleanup_context(device):
    """Initialize the current thread on its selected device only if needed."""
    import acl
    import torch

    context, ret = acl.rt.get_context()
    if ret == 107002:  # ACL_ERROR_RT_CONTEXT_NULL
        torch.npu.set_device(device)
        context, ret = acl.rt.get_context()
    check(ret, "get caller context for decoder cleanup")
    if not context:
        raise ACLFailure("decoder cleanup requires an active caller context")


def _reap_locked(device):
    context_ready = False
    for token, entry in list(_live.items()):
        owner = entry.owner()
        if entry.retiring or owner is None or not owner.is_alive():
            entry.retiring = True
            if not context_ready:
                _ensure_cleanup_context(device)
                context_ready = True
            entry.decoder.close()
            del _live[token]


def _get_dvpp(device):
    _check_process()
    key = (os.getpid(), threading.get_ident(), device)
    with _guard:
        if _closed:
            raise RuntimeError("NPU JPEG decoder pool has been closed")
        _reap_locked(device)
        previous = getattr(_local, "slot", None)
        if previous and previous[0] == key and not previous[1].closed:
            return previous[1]
        if previous:
            _local.slot = None
            _live[id(previous[1])].retiring = True
            _ensure_cleanup_context(device)
            previous[1].close()
            del _live[id(previous[1])]
        from sglang.srt.hardware_backend.npu.dvpp import Dvpp

        decoder = Dvpp()
        _live[id(decoder)] = _Entry(weakref.ref(threading.current_thread()), decoder)
        _local.slot = (key, decoder)
        return decoder


def _close_cached_dvpp():
    """Terminal cleanup after workers join; initialize a missing thread context.

    An existing caller context is restored by each decoder; if the caller has
    no current context, cleanup initializes the current NPU's default context.
    The NPU device is never reset.
    """
    global _closed
    if os.getpid() != _pid:
        return
    first_error = None
    with _guard:
        _closed = True
        if _live:
            import torch

            _ensure_cleanup_context(torch.npu.current_device())
        for token, entry in list(_live.items()):
            try:
                entry.retiring = True
                entry.decoder.close()
                del _live[token]
            except Exception as error:
                first_error = first_error or error
    _local.slot = None
    if first_error:
        raise first_error


atexit.register(_close_cached_dvpp)


def decode_jpeg(
    data: bytes | torch.Tensor, device: str | torch.device = "npu"
) -> torch.Tensor | None:
    """Decode encoded CPU uint8 data to contiguous CHW RGB on an NPU.

    Accept a one-dimensional CPU uint8 tensor, or bytes for direct callers.
    Contiguous tensors share their encoded storage with the decoder. Strided
    inputs are made contiguous; SOF changes use a private input buffer.
    An unindexed device uses the calling thread's current NPU.
    Unsupported images/devices return None for PIL fallback before creating
    resources. Execution errors propagate; the caller's device is restored.
    """
    import torch

    prepared = prepare_jpeg(data)
    if prepared is None:
        return None
    data, _ = prepared
    import torch_npu  # noqa: F401 - registers the NPU device backend

    target = torch.device(device)
    if target.type != "npu":
        raise ValueError("NPU JPEG decoding requires an NPU device")
    device_index = target.index
    if device_index is None:
        device_index = torch.npu.current_device()
    if not supports_device(device_index):
        return None
    with torch.npu.device(device_index):
        return _get_dvpp(device_index).process_jpeg_dec(data)

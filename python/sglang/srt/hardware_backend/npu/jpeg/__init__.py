"""Opt-in DVPP/Triton JPEG decoding; unsupported images retain PIL semantics."""

import atexit
import os
import threading
import weakref
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from sglang.srt.hardware_backend.npu.jpeg.dvpp import UnsupportedJPEG, inspect_jpeg

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


def _reap_locked():
    for token, entry in list(_live.items()):
        owner = entry.owner()
        if entry.retiring or owner is None or not owner.is_alive():
            entry.retiring = True
            entry.decoder.close()
            del _live[token]


def _decoder(device):
    _check_process()
    key = (os.getpid(), threading.get_ident(), device)
    with _guard:
        if _closed:
            raise RuntimeError("NPU JPEG decoder pool has been closed")
        _reap_locked()
        previous = getattr(_local, "slot", None)
        if previous and previous[0] == key:
            return previous[1]
        if previous:
            _local.slot = None
            _live[id(previous[1])].retiring = True
            previous[1].close()
            del _live[id(previous[1])]
        from sglang.srt.hardware_backend.npu.jpeg.dvpp import TritonJPEGDecoder

        decoder = TritonJPEGDecoder(device)
        _live[id(decoder)] = _Entry(weakref.ref(threading.current_thread()), decoder)
        _local.slot = (key, decoder)
        return decoder


def close_decoders():
    """Terminal cleanup, after request workers have joined; never reset NPU."""
    global _closed
    if os.getpid() != _pid:
        return
    first_error = None
    with _guard:
        _closed = True
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


atexit.register(close_decoders)


def decode_jpeg(data: bytes):
    """Return uint8 CHW on the current NPU, or None for supported PIL fallback.

    No device work is performed for ineligible headers. Runtime failures and
    malformed/over-budget inputs remain errors rather than hidden CPU success.
    """
    if not isinstance(data, bytes):
        raise TypeError("NPU JPEG input must be bytes")
    if not (data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")):
        return None
    try:
        inspect_jpeg(data, width_alignment=64)
    except UnsupportedJPEG:
        return None
    with Image.open(BytesIO(data)) as image:
        if image.getexif() or image.info.get("icc_profile") or image.info.get("exif"):
            return None

    import torch
    import torch_npu  # noqa: F401 - registers the NPU device backend

    return _decoder(torch.npu.current_device()).decode(data)

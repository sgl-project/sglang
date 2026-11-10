# SPDX-License-Identifier: Apache-2.0
"""Device failures that make a diffusion worker or client unsafe to reuse."""

import torch


class DiffusionRuntimeError(RuntimeError):
    """The inference runtime failed and the generator must be recreated."""


# Non-OOM device errors after which the device context or the caching allocator
# cannot be trusted, matched in the lowercased message: driver checks and
# allocator asserts surface as plain RuntimeErrors, and worker errors reach the
# client as strings.
_UNRECOVERABLE_DEVICE_ERROR_MARKERS = (
    "cuda error:",
    "cuda driver error:",
    "hip error:",
    "cudacachingallocator.cpp",
    "device-side assert triggered",
    "an illegal memory access was encountered",
    "misaligned address",
    "unspecified launch failure",
)


def is_unrecoverable_device_error(error: BaseException | str) -> bool:
    """Recognize context/allocator failures, including wrapped RPC errors.

    An out-of-memory error stays recoverable: the allocator releases what it
    mapped, and the cache is emptied after the request.
    """
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, torch.OutOfMemoryError):
            return False
        message = str(error).lower()
        if "out of memory" not in message and (
            isinstance(error, torch.AcceleratorError)
            or any(marker in message for marker in _UNRECOVERABLE_DEVICE_ERROR_MARKERS)
        ):
            return True
        if not isinstance(error, BaseException):
            break
        error = error.__cause__ or error.__context__
    return False

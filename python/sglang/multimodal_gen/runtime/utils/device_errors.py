# SPDX-License-Identifier: Apache-2.0
"""Device failures that make a diffusion client unsafe to reuse."""


class DiffusionRuntimeError(RuntimeError):
    """The inference runtime failed and the generator must be recreated."""


def is_unrecoverable_device_error(error: BaseException | str) -> bool:
    """Recognize context/allocator failures, including wrapped RPC errors.

    Worker errors arrive as strings, so exception types alone are insufficient.
    Ordinary request errors and recoverable out-of-memory errors keep their
    existing behavior.
    """
    markers = (
        "device-side assert triggered",
        "an illegal memory access was encountered",
        "misaligned address",
        "unspecified launch failure",
        "cuda driver error:",
        "cudacachingallocator.cpp",
    )
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        message = str(error).lower()
        if "out of memory" not in message and any(
            marker in message for marker in markers
        ):
            return True
        if not isinstance(error, BaseException):
            break
        error = error.__cause__ or error.__context__
    return False

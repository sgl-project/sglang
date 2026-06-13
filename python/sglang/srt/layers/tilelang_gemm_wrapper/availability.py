"""TileLang GEMM availability checks.

TileLang is intentionally optional. Keep imports lazy so normal SGLang startup
does not depend on TileLang or its cache directory being usable.
"""

from __future__ import annotations

import importlib
from typing import Optional

from packaging.version import Version

from sglang.srt.utils import get_device_sm, is_cuda

TILELANG_MIN_VERSION = "0.1.11"
SUPPORTED_SMS = (89, 90)


def get_availability_error() -> Optional[str]:
    if not is_cuda():
        return "TileLang FP8 GEMM requires CUDA."

    sm = get_device_sm()
    if sm not in SUPPORTED_SMS:
        return (
            "TileLang FP8 GEMM currently supports only SM89 and SM90 GPUs; "
            f"detected SM{sm}."
        )

    try:
        tilelang = importlib.import_module("tilelang")
    except Exception as err:
        return (
            f"TileLang FP8 GEMM requires tilelang>={TILELANG_MIN_VERSION}. "
            f"Install it with `pip install 'tilelang>={TILELANG_MIN_VERSION}'` "
            "or `pip install sglang[tilelang]`. "
            "If TileLang cannot create its cache under your home directory, set "
            "`TILELANG_CACHE_DIR` to a writable path."
            f" Original error: {err}"
        )

    version = getattr(tilelang, "__version__", None)
    if version is None:
        return (
            f"TileLang FP8 GEMM requires tilelang>={TILELANG_MIN_VERSION}, "
            "but the installed version is unknown."
        )

    if Version(version) < Version(TILELANG_MIN_VERSION):
        return (
            f"TileLang FP8 GEMM requires tilelang>={TILELANG_MIN_VERSION}; "
            f"detected tilelang=={version}."
        )

    return None


def is_available() -> bool:
    return get_availability_error() is None


def assert_available() -> None:
    error = get_availability_error()
    if error is not None:
        raise RuntimeError(error)

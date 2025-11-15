"""Tiny stub mimicking :mod:`PIL.Image` APIs used by transformers.

The shim only supports ``fromarray`` inputs and represents images as simple
Python objects wrapping numpy arrays. The heavy lifting is still performed by
NumPy/PyTorch downstream, so these stubs are sufficient for environments where
installing Pillow is not possible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class _ArrayImage:
    data: np.ndarray

    def resize(self, size: tuple[int, int], resample: int | None = None) -> "_ArrayImage":
        # Basic nearest-neighbor resize to satisfy potential calls. We rely on
        # NumPy to avoid bringing in other heavy dependencies.
        height, width = size
        y_scale = height / self.data.shape[0]
        x_scale = width / self.data.shape[1]
        y_idx = (np.arange(height) / y_scale).astype(int)
        x_idx = (np.arange(width) / x_scale).astype(int)
        resized = self.data[y_idx][:, x_idx]
        return _ArrayImage(resized)

    def convert(self, mode: str) -> "_ArrayImage":
        if mode.upper() not in {"RGB", "RGBA"}:
            return self
        arr = self.data
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if mode.upper() == "RGBA" and arr.shape[-1] == 3:
            alpha = np.full(arr.shape[:2] + (1,), 255, dtype=arr.dtype)
            arr = np.concatenate([arr, alpha], axis=2)
        if mode.upper() == "RGB" and arr.shape[-1] == 4:
            arr = arr[..., :3]
        return _ArrayImage(arr)

    def getexif(self) -> dict[str, Any]:
        return {}

    def tobytes(self) -> bytes:
        return self.data.tobytes()


class Image:  # pragma: no cover - compatibility shim
    Image = _ArrayImage


def fromarray(array: Any) -> _ArrayImage:
    return _ArrayImage(np.asarray(array))


def open(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("Pillow stub cannot open image files")

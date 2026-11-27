"""Tensors stored flat under caller-chosen names; the caller keeps any structure."""

from __future__ import annotations

from typing import Any, Mapping

import msgspec
import numpy as np

from radix_transfer_plugins.schema import RawField

_TORCH_DTYPE_NAMES = (
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
)


class TensorStash(msgspec.Struct, frozen=True):
    """``name -> CPU tensor``; the stored dtype is the torch dtype name, such as ``bfloat16``."""

    tensors: dict[str, Any]

    def is_empty(self) -> bool:
        return not self.tensors

    def to_fields(self) -> dict[str, RawField]:
        return {name: _to_raw_field(tensor) for name, tensor in self.tensors.items()}

    @classmethod
    def from_fields(cls, fields: Mapping[str, RawField]) -> TensorStash:
        return cls({name: _from_raw_field(field) for name, field in fields.items()})


def _to_raw_field(tensor: Any) -> RawField:
    import torch

    flat = tensor.detach().cpu().contiguous().reshape(-1)
    return RawField(
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(tensor.shape),
        data=memoryview(flat.view(torch.uint8).numpy()),
    )


def _from_raw_field(field: RawField) -> Any:
    import torch

    if field.dtype not in _TORCH_DTYPE_NAMES:
        raise ValueError(f"unknown tensor dtype {field.dtype!r}")
    dtype = getattr(torch, field.dtype)
    if field.data.nbytes == 0:
        return torch.empty(field.shape, dtype=dtype)
    return torch.frombuffer(
        np.frombuffer(field.data, dtype=np.uint8), dtype=dtype
    ).reshape(field.shape)

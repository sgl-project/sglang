# SPDX-License-Identifier: Apache-2.0
"""Value-only, versioned descriptions of finalized inference state.

Storage groups describe allocations; tensor groups describe object identity.
Neither group identifier is a pointer or a reusable CUDA IPC send reference.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch

CACHE_ABI = 1


def canonical_digest(value: dict) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype) or name != str(dtype).removeprefix("torch."):
        raise ValueError(f"Unsupported canonical tensor dtype: {name!r}")
    # Quantized/shell dtypes that cannot form ordinary strided views are not
    # admitted by traversal. Their support requires a separate adapter/ABI.
    return dtype


def _nonnegative_int(value: int, field: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")


def _name(value: str, *, root: bool = False) -> None:
    if not isinstance(value, str) or (
        not (root and value == "") and any(not part for part in value.split("."))
    ):
        raise ValueError(f"Invalid module/tensor path: {value!r}")


@dataclass(frozen=True)
class StorageDescriptor:
    group: str
    nbytes: int


@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    kind: str  # parameter | buffer
    persistent: bool
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int  # in dtype elements, not bytes
    storage_group: str
    tensor_group: str
    requires_grad: bool

    def object_signature(self) -> tuple:
        # Persistence belongs to a registration path: the same buffer object
        # can be persistent at one path and non-persistent at another.
        return (
            self.kind,
            self.dtype,
            self.shape,
            self.stride,
            self.storage_offset,
            self.storage_group,
            self.requires_grad,
        )


@dataclass(frozen=True)
class StateManifest:
    storages: tuple[StorageDescriptor, ...]
    tensors: tuple[TensorDescriptor, ...]
    training: tuple[tuple[str, bool], ...]
    cache_abi: int = CACHE_ABI

    def validate(self) -> None:
        if type(self.cache_abi) is not int or self.cache_abi != CACHE_ABI:
            raise ValueError(f"Unsupported weight-cache ABI: {self.cache_abi}")
        storages = {}
        for storage in self.storages:
            if not isinstance(storage.group, str) or not storage.group:
                raise ValueError("Storage group must be a nonempty string")
            if storage.group in storages:
                raise ValueError(f"Duplicate storage group: {storage.group}")
            _nonnegative_int(storage.nbytes, "storage nbytes")
            storages[storage.group] = storage
        modules = {}
        for name, training in self.training:
            _name(name, root=True)
            if name in modules or type(training) is not bool:
                raise ValueError(f"Invalid/duplicate training state: {name!r}")
            modules[name] = training
        if "" not in modules:
            raise ValueError("Missing root module training state")
        for name in modules:
            if name and name.rpartition(".")[0] not in modules:
                raise ValueError(f"Missing parent module: {name!r}")
        names, referenced, objects = set(), set(), {}
        for tensor in self.tensors:
            _name(tensor.name)
            if tensor.name in names:
                raise ValueError(f"Duplicate tensor name: {tensor.name}")
            names.add(tensor.name)
            if tensor.name.rpartition(".")[0] not in modules:
                raise ValueError(f"Missing tensor parent: {tensor.name}")
            if tensor.kind not in ("parameter", "buffer"):
                raise ValueError(f"Invalid tensor kind: {tensor.kind}")
            if type(tensor.persistent) is not bool or (
                tensor.kind == "parameter" and not tensor.persistent
            ):
                raise ValueError(f"Invalid persistence: {tensor.name}")
            if type(tensor.requires_grad) is not bool:
                raise ValueError(f"Invalid requires_grad: {tensor.name}")
            dtype = resolve_dtype(tensor.dtype)
            if len(tensor.shape) != len(tensor.stride):
                raise ValueError(f"Shape/stride rank mismatch: {tensor.name}")
            for value in (*tensor.shape, *tensor.stride, tensor.storage_offset):
                _nonnegative_int(value, f"layout of {tensor.name}")
            storage = storages.get(tensor.storage_group)
            if storage is None:
                raise ValueError(f"Unknown storage group: {tensor.storage_group}")
            referenced.add(storage.group)
            itemsize = dtype.itemsize
            end = tensor.storage_offset
            if math.prod(tensor.shape):
                end += 1 + sum(
                    (size - 1) * stride
                    for size, stride in zip(tensor.shape, tensor.stride)
                )
            if end * itemsize > storage.nbytes:
                raise ValueError(f"Tensor exceeds storage bounds: {tensor.name}")
            if not isinstance(tensor.tensor_group, str) or not tensor.tensor_group:
                raise ValueError(f"Invalid tensor group: {tensor.name}")
            signature = tensor.object_signature()
            if objects.setdefault(tensor.tensor_group, signature) != signature:
                raise ValueError(f"Conflicting exact tensor tie: {tensor.name}")
        if referenced != set(storages):
            raise ValueError("Manifest contains unreferenced storage groups")

    @property
    def unique_storage_bytes(self) -> int:
        return sum(storage.nbytes for storage in self.storages)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> StateManifest:
        if set(value) != {"cache_abi", "storages", "tensors", "training"}:
            raise ValueError("Unexpected state manifest fields")
        result = cls(
            cache_abi=value["cache_abi"],
            storages=tuple(StorageDescriptor(**item) for item in value["storages"]),
            tensors=tuple(
                TensorDescriptor(
                    **{
                        **item,
                        "shape": tuple(item["shape"]),
                        "stride": tuple(item["stride"]),
                    }
                )
                for item in value["tensors"]
            ),
            training=tuple((name, mode) for name, mode in value["training"]),
        )
        result.validate()
        return result

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self.to_dict())

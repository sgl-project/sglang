"""Full, typed identities for processor outputs and encoder inputs.

Tensor bytes use the same BLAKE3-256 algorithm on CPU and GPU. The outer
digest also commits to types, shapes and container boundaries. Device and
strides are not semantic: transporting or making a tensor contiguous must not
change its identity. Callers own the inputs until hashing finishes.
"""

import dataclasses
import struct
from collections import defaultdict
from collections.abc import Mapping
from enum import Enum
from operator import attrgetter

import blake3
import numpy as np
import torch
from PIL import Image

from sglang.kernels.selector import get_kernel

_DOMAIN = b"sglang-mm-structured-identity-v1"


def _frame(tag: bytes, *parts: bytes) -> bytes:
    return tag + b"".join(len(part).to_bytes(8, "big") + part for part in parts)


def dataclass_items(value):
    """Read declared fields without deep-copying tensor storage, including slots."""
    return (
        (field.name, attrgetter(field.name)(value))
        for field in dataclasses.fields(value)
    )


def _children(value):
    if isinstance(value, Mapping):
        return value.values()
    if isinstance(value, (list, tuple)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (child for _, child in dataclass_items(value))
    return ()


def _gpu_leaves(value, batches):
    if isinstance(value, torch.Tensor) and value.is_cuda:
        batches[value.device][id(value)] = value
    else:
        for child in _children(value):
            _gpu_leaves(child, batches)


def _encode(value, gpu_digests) -> bytes:
    if value is None:
        return b"none"
    if isinstance(value, Enum):
        return _frame(
            b"enum", str(type(value)).encode(), _encode(value.value, gpu_digests)
        )
    if isinstance(value, bool):
        return b"true" if value else b"false"
    if isinstance(value, int):
        return _frame(b"int", str(value).encode())
    if isinstance(value, float):
        return _frame(b"float", struct.pack(">d", value))
    if isinstance(value, str):
        return _frame(b"str", value.encode())
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _frame(b"bytes", blake3.blake3(value).digest())
    if isinstance(value, Image.Image):
        palette = value.palette
        return _frame(
            b"pil",
            _encode(
                (
                    value.mode,
                    value.size,
                    None if palette is None else (palette.mode, palette.tobytes()),
                    value.info.get("transparency"),
                ),
                gpu_digests,
            ),
            blake3.blake3(value.tobytes()).digest(),
        )
    if isinstance(value, (torch.dtype, torch.device, np.dtype)):
        return _frame(b"type", str(value).encode())
    if isinstance(value, np.generic):
        value = np.asarray(value)
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError("Multimodal identities require dense, unquantized tensors")
        if value.is_cuda:
            digest = gpu_digests[id(value)]
        else:
            raw = (
                value.detach()
                .resolve_conj()
                .resolve_neg()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .cpu()
                .numpy()
            )
            digest = blake3.blake3(memoryview(raw)).digest()
        return _frame(
            b"tensor",
            str(value.dtype).encode(),
            _encode(tuple(value.shape), gpu_digests),
            digest,
        )
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot identify multimodal features")
        raw = np.ascontiguousarray(value)
        return _frame(
            b"numpy",
            value.dtype.str.encode(),
            _encode(tuple(value.shape), gpu_digests),
            blake3.blake3(memoryview(raw.reshape(-1)).cast("B")).digest(),
        )
    if isinstance(value, (list, tuple)):
        return _frame(
            b"list" if isinstance(value, list) else b"tuple",
            *(_encode(child, gpu_digests) for child in value),
        )
    if isinstance(value, Mapping):
        pairs = sorted(
            (_encode(key, gpu_digests), _encode(child, gpu_digests))
            for key, child in value.items()
        )
        return _frame(b"dict", *(_frame(b"pair", key, child) for key, child in pairs))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _frame(
            b"dataclass",
            str(type(value)).encode(),
            _encode(dict(dataclass_items(value)), gpu_digests),
        )
    raise TypeError(f"Unsupported multimodal identity input: {type(value).__name__}")


def structured_digest(value) -> bytes:
    """Return all 32 bytes; never use a truncated integer as a cache key.

    GPU leaves are hashed independently and only their digests leave the GPU.
    One host synchronization per device also covers nested multi-image inputs.
    """
    batches = defaultdict(dict)
    _gpu_leaves(value, batches)
    gpu_digests = {}
    for tensors in batches.values():
        digests = get_kernel("memory.gpu_tensors_blake3")(list(tensors.values()))
        gpu_digests.update(zip(tensors, digests))
    return blake3.blake3(_DOMAIN + _encode(value, gpu_digests)).digest()


def media_identity(content_config_digest: bytes, *, modality: str, metadata) -> bytes:
    """Bind an existing full source/config digest to model-specific geometry."""
    if len(content_config_digest) != 32:
        raise ValueError("Media identity requires a full 32-byte content/config digest")
    return structured_digest(
        ("media-source-v1", modality, content_config_digest, metadata)
    )

# SPDX-License-Identifier: Apache-2.0
"""Zero-copy tensor codec for ZMQ multipart messages.

Frame 0: JSON metadata (tensor descriptors + scalar fields)
Frame 1-N: Raw tensor data buffers (one per tensor)
"""

import ctypes
import json
import logging
from dataclasses import dataclass

import torch
import zmq

logger = logging.getLogger(__name__)

_DTYPE_TO_STR = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


def dtype_to_str(dtype: torch.dtype) -> str:
    s = _DTYPE_TO_STR.get(dtype)
    if s is None:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return s


def str_to_dtype(s: str) -> torch.dtype:
    d = _STR_TO_DTYPE.get(s)
    if d is None:
        raise ValueError(f"Unknown dtype string: {s}")
    return d


class TensorWrapper:
    """Expose a CPU-contiguous tensor's data buffer for zero-copy ZMQ send."""

    def __init__(self, tensor: torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        self.tensor = tensor
        data_ptr = tensor.data_ptr()
        total_bytes = tensor.numel() * tensor.element_size()
        self._c_buf = (ctypes.c_char * total_bytes).from_address(data_ptr)
        self._view = memoryview(self._c_buf)


@dataclass
class TensorDescriptor:
    field_name: str
    shape: list[int]
    dtype: str
    list_index: int = -1  # -1 means not part of a list

    def to_dict(self) -> dict:
        return {
            "field_name": self.field_name,
            "shape": self.shape,
            "dtype": self.dtype,
            "list_index": self.list_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TensorDescriptor":
        return cls(
            field_name=d["field_name"],
            shape=d["shape"],
            dtype=d["dtype"],
            list_index=d.get("list_index", -1),
        )


def pack_tensors(
    tensor_fields: dict[str, torch.Tensor | list[torch.Tensor] | None],
    scalar_fields: dict | None = None,
) -> tuple[bytes, list[TensorWrapper]]:
    """Pack tensor fields into metadata + buffer list for send_multipart."""
    descriptors = []
    buffers = []

    for field_name, value in tensor_fields.items():
        if value is None:
            continue

        if isinstance(value, torch.Tensor):
            wrapper = TensorWrapper(value)
            descriptors.append(
                TensorDescriptor(
                    field_name=field_name,
                    shape=list(value.shape),
                    dtype=dtype_to_str(value.dtype),
                )
            )
            buffers.append(wrapper)

        elif isinstance(value, list):
            for i, t in enumerate(value):
                if t is None:
                    continue
                if not isinstance(t, torch.Tensor):
                    raise TypeError(
                        f"Expected Tensor in list for field '{field_name}', "
                        f"got {type(t)}"
                    )
                wrapper = TensorWrapper(t)
                descriptors.append(
                    TensorDescriptor(
                        field_name=field_name,
                        shape=list(t.shape),
                        dtype=dtype_to_str(t.dtype),
                        list_index=i,
                    )
                )
                buffers.append(wrapper)

    metadata = {
        "tensor_descriptors": [d.to_dict() for d in descriptors],
        "scalar_fields": scalar_fields or {},
    }
    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return metadata_bytes, buffers


def send_tensors(
    socket: zmq.Socket,
    tensor_fields: dict[str, torch.Tensor | list[torch.Tensor] | None],
    scalar_fields: dict | None = None,
    flags: int = 0,
) -> None:
    """Send tensors over ZMQ using multipart with zero-copy."""
    metadata_bytes, buffers = pack_tensors(tensor_fields, scalar_fields)
    parts: list = [metadata_bytes]
    parts.extend(w._view if isinstance(w, TensorWrapper) else w for w in buffers)
    socket.send_multipart(parts, flags=flags, copy=True)


def unpack_tensors(
    parts: list,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor | list[torch.Tensor]], dict]:
    """Unpack multipart message frames into tensor fields and scalar fields."""
    metadata_frame = parts[0]
    metadata_bytes = (
        bytes(metadata_frame.buffer)
        if hasattr(metadata_frame, "buffer")
        else bytes(metadata_frame)
    )
    metadata = json.loads(metadata_bytes)

    descriptors = [
        TensorDescriptor.from_dict(d) for d in metadata["tensor_descriptors"]
    ]
    scalar_fields = metadata.get("scalar_fields", {})

    if len(parts) - 1 != len(descriptors):
        raise ValueError(
            f"Expected {len(descriptors)} tensor frames, got {len(parts) - 1}"
        )

    tensor_fields: dict[str, torch.Tensor | list[torch.Tensor]] = {}
    list_sizes: dict[str, int] = {}
    for desc in descriptors:
        if desc.list_index >= 0:
            current_max = list_sizes.get(desc.field_name, 0)
            list_sizes[desc.field_name] = max(current_max, desc.list_index + 1)

    for field_name, size in list_sizes.items():
        tensor_fields[field_name] = [None] * size

    for i, desc in enumerate(descriptors):
        frame = parts[i + 1]
        buf = frame.buffer if hasattr(frame, "buffer") else bytes(frame)
        dtype = str_to_dtype(desc.dtype)
        # clone() to own the memory (decouple from ZMQ buffer lifetime)
        tensor = torch.frombuffer(buf, dtype=dtype).reshape(desc.shape).clone()
        if device != "cpu" and device != torch.device("cpu"):
            tensor = tensor.to(device)

        if desc.list_index >= 0:
            tensor_fields[desc.field_name][desc.list_index] = tensor
        else:
            tensor_fields[desc.field_name] = tensor

    return tensor_fields, scalar_fields

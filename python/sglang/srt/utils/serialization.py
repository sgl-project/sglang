# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tensor and pickle serialization helpers, including the restricted unpickler."""

from __future__ import annotations

import binascii
import io
import pickle
from multiprocessing.reduction import ForkingPickler
from typing import List, Union

import pybase64
import torch

__all__ = [
    "MultiprocessingSerializer",
    "SerializedTensorPayload",
    "normalize_serialized_named_tensor_payload",
    "normalize_serialized_named_tensor_payloads",
    "SafeUnpickler",
    "safe_pickle_load",
    "safe_pickle_loads",
]


class MultiprocessingSerializer:
    @staticmethod
    def serialize(obj, output_str: bool = False):
        """
        Serialize a Python object using ForkingPickler.

        Args:
            obj: The object to serialize.
            output_str (bool): If True, return a base64-encoded string instead of raw bytes.

        Returns:
            bytes or str: The serialized object.
        """
        buf = io.BytesIO()
        ForkingPickler(buf).dump(obj)
        buf.seek(0)
        output = buf.read()

        if output_str:
            # Convert bytes to base64-encoded string
            output = pybase64.b64encode(output).decode("utf-8")

        return output

    @staticmethod
    def deserialize(data):
        """
        Deserialize a previously serialized object.

        Args:
            data (bytes or str): The serialized data, optionally base64-encoded.

        Returns:
            The deserialized Python object.
        """
        if isinstance(data, str):
            # Decode base64 string to bytes
            data = pybase64.b64decode(data, validate=True)

        return SafeUnpickler(io.BytesIO(data)).load()


SerializedTensorPayload = Union[str, bytes, bytearray, memoryview]


def _looks_like_pickle_payload(data: bytes) -> bool:
    return len(data) >= 2 and data[0] == 0x80 and data[1] <= pickle.HIGHEST_PROTOCOL


def normalize_serialized_named_tensor_payload(data: SerializedTensorPayload) -> bytes:
    """Normalize a serialized tensor payload to raw MultiprocessingSerializer bytes."""
    if isinstance(data, str):
        return pybase64.b64decode(data, validate=True)

    if isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
        if _looks_like_pickle_payload(data):
            return data
        try:
            return pybase64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            return data

    raise TypeError(
        "serialized_named_tensors entries must be base64 strings or bytes-like "
        f"payloads, got {type(data).__name__}"
    )


def normalize_serialized_named_tensor_payloads(
    payloads: List[SerializedTensorPayload],
) -> List[bytes]:
    return [normalize_serialized_named_tensor_payload(data) for data in payloads]


def _safe_load_torch_storage(data: bytes):
    storage = torch.load(io.BytesIO(data), weights_only=True)
    if not isinstance(storage, (torch.storage.TypedStorage, torch.UntypedStorage)):
        raise pickle.UnpicklingError(
            f"Expected a Torch storage, got {type(storage).__name__}"
        )
    return storage


class SafeUnpickler(pickle.Unpickler):
    # Standard-library modules expose powerful callables alongside harmless data
    # types. Keep these globals exact so a newly added callable is denied by
    # default instead of silently expanding the unpickling attack surface.
    ALLOWED_GLOBALS = {
        # --- Python types ---
        ("builtins", "bool"),
        ("builtins", "bytearray"),
        ("builtins", "bytes"),
        ("builtins", "complex"),
        ("builtins", "dict"),
        ("builtins", "float"),
        ("builtins", "frozenset"),
        ("builtins", "int"),
        ("builtins", "list"),
        ("builtins", "range"),
        ("builtins", "set"),
        ("builtins", "slice"),
        ("builtins", "str"),
        ("builtins", "tuple"),
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
        ("collections", "deque"),
        ("collections", "Counter"),
        ("copyreg", "__newobj__"),
        ("copyreg", "__newobj_ex__"),
        ("functools", "partial"),
        ("itertools", "chain"),
        ("itertools", "repeat"),
        ("multiprocessing.reduction", "_rebuild_partial"),
        ("multiprocessing.reduction", "_rebuild_socket"),
        ("multiprocessing.resource_sharer", "DupFd"),
        ("types", "SimpleNamespace"),
        ("_codecs", "encode"),
        # --- PyTorch data containers & rebuild functions ---
        # Code-module prefixes (torch.*, sglang.srt.*) are NOT allowed: they
        # contain gadgets like sglang.srt.utils.common.dynamic_import
        ("torch", "Tensor"),
        ("torch", "BFloat16Tensor"),
        ("torch", "BoolTensor"),
        ("torch", "ByteTensor"),
        ("torch", "CharTensor"),
        ("torch", "DoubleTensor"),
        ("torch", "FloatTensor"),
        ("torch", "HalfTensor"),
        ("torch", "IntTensor"),
        ("torch", "LongTensor"),
        ("torch", "ShortTensor"),
        ("torch.cuda", "BFloat16Tensor"),
        ("torch.cuda", "BoolTensor"),
        ("torch.cuda", "ByteTensor"),
        ("torch.cuda", "CharTensor"),
        ("torch.cuda", "DoubleTensor"),
        ("torch.cuda", "FloatTensor"),
        ("torch.cuda", "HalfTensor"),
        ("torch.cuda", "IntTensor"),
        ("torch.cuda", "LongTensor"),
        ("torch.cuda", "ShortTensor"),
        ("torch.cuda.sparse", "BFloat16Tensor"),
        ("torch.cuda.sparse", "ByteTensor"),
        ("torch.cuda.sparse", "CharTensor"),
        ("torch.cuda.sparse", "DoubleTensor"),
        ("torch.cuda.sparse", "FloatTensor"),
        ("torch.cuda.sparse", "HalfTensor"),
        ("torch.cuda.sparse", "IntTensor"),
        ("torch.cuda.sparse", "LongTensor"),
        ("torch.cuda.sparse", "ShortTensor"),
        ("torch.sparse", "BFloat16Tensor"),
        ("torch.sparse", "ByteTensor"),
        ("torch.sparse", "CharTensor"),
        ("torch.sparse", "DoubleTensor"),
        ("torch.sparse", "FloatTensor"),
        ("torch.sparse", "HalfTensor"),
        ("torch.sparse", "IntTensor"),
        ("torch.sparse", "LongTensor"),
        ("torch.sparse", "ShortTensor"),
        ("torch", "Size"),
        ("torch", "device"),
        ("torch", "dtype"),
        ("torch", "bcomplex32"),
        ("torch", "bfloat16"),
        ("torch", "bit"),
        ("torch", "bits16"),
        ("torch", "bits1x8"),
        ("torch", "bits2x4"),
        ("torch", "bits4x2"),
        ("torch", "bits8"),
        ("torch", "bool"),
        ("torch", "cdouble"),
        ("torch", "cfloat"),
        ("torch", "chalf"),
        ("torch", "complex128"),
        ("torch", "complex32"),
        ("torch", "complex64"),
        ("torch", "double"),
        ("torch", "float"),
        ("torch", "float16"),
        ("torch", "float32"),
        ("torch", "float4_e2m1fn_x2"),
        ("torch", "float64"),
        ("torch", "float8_e4m3fn"),
        ("torch", "float8_e4m3fnuz"),
        ("torch", "float8_e5m2"),
        ("torch", "float8_e5m2fnuz"),
        ("torch", "float8_e8m0fnu"),
        ("torch", "half"),
        ("torch", "int"),
        ("torch", "int1"),
        ("torch", "int16"),
        ("torch", "int2"),
        ("torch", "int3"),
        ("torch", "int32"),
        ("torch", "int4"),
        ("torch", "int5"),
        ("torch", "int6"),
        ("torch", "int64"),
        ("torch", "int7"),
        ("torch", "int8"),
        ("torch", "long"),
        ("torch", "qint32"),
        ("torch", "qint8"),
        ("torch", "quint2x4"),
        ("torch", "quint4x2"),
        ("torch", "quint8"),
        ("torch", "short"),
        ("torch", "uint1"),
        ("torch", "uint16"),
        ("torch", "uint2"),
        ("torch", "uint3"),
        ("torch", "uint32"),
        ("torch", "uint4"),
        ("torch", "uint5"),
        ("torch", "uint6"),
        ("torch", "uint64"),
        ("torch", "uint7"),
        ("torch", "uint8"),
        ("torch.nn.parameter", "Parameter"),
        ("torch.serialization", "_get_layout"),
        ("torch._utils", "_rebuild_tensor"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch._utils", "_rebuild_tensor_v3"),
        ("torch._utils", "_rebuild_parameter"),
        ("torch._utils", "_rebuild_parameter_with_state"),
        ("torch._utils", "_rebuild_qtensor"),
        ("torch._utils", "_rebuild_sparse_tensor"),
        ("torch._utils", "_rebuild_meta_tensor_no_storage"),
        ("torch._utils", "_rebuild_wrapper_subclass"),
        ("torch._utils", "_rebuild_device_tensor_from_numpy"),
        ("torch._utils", "_rebuild_device_tensor_from_cpu_tensor"),
        ("torch._tensor", "_rebuild_from_type_v2"),
        ("torch.storage", "UntypedStorage"),
        ("torch.storage", "_UntypedStorage"),
        ("torch.storage", "TypedStorage"),
        ("torch", "UntypedStorage"),
        ("torch", "BFloat16Storage"),
        ("torch", "BoolStorage"),
        ("torch", "ByteStorage"),
        ("torch", "CharStorage"),
        ("torch", "ComplexDoubleStorage"),
        ("torch", "ComplexFloatStorage"),
        ("torch", "DoubleStorage"),
        ("torch", "FloatStorage"),
        ("torch", "HalfStorage"),
        ("torch", "IntStorage"),
        ("torch", "LongStorage"),
        ("torch", "QInt32Storage"),
        ("torch", "QInt8Storage"),
        ("torch", "QUInt2x4Storage"),
        ("torch", "QUInt4x2Storage"),
        ("torch", "QUInt8Storage"),
        ("torch", "ShortStorage"),
        ("torch.cuda", "BFloat16Storage"),
        ("torch.cuda", "BoolStorage"),
        ("torch.cuda", "ByteStorage"),
        ("torch.cuda", "CharStorage"),
        ("torch.cuda", "ComplexDoubleStorage"),
        ("torch.cuda", "ComplexFloatStorage"),
        ("torch.cuda", "DoubleStorage"),
        ("torch.cuda", "FloatStorage"),
        ("torch.cuda", "HalfStorage"),
        ("torch.cuda", "IntStorage"),
        ("torch.cuda", "LongStorage"),
        ("torch.cuda", "ShortStorage"),
        ("torch.multiprocessing.reductions", "rebuild_tensor"),
        ("torch.multiprocessing.reductions", "rebuild_meta_tensor"),
        ("torch.multiprocessing.reductions", "rebuild_cuda_tensor"),
        ("sglang.srt.utils.patch_torch", "_rebuild_cuda_tensor_modified"),
        ("torch_npu.multiprocessing.reductions", "rebuild_npu_tensor"),
        ("sglang.srt.utils.patch_torch", "_rebuild_npu_tensor_modified"),
        ("torch.multiprocessing.reductions", "rebuild_nested_tensor"),
        ("torch.multiprocessing.reductions", "rebuild_sparse_coo_tensor"),
        ("torch.multiprocessing.reductions", "rebuild_sparse_compressed_tensor"),
        ("torch.multiprocessing.reductions", "rebuild_storage_fd"),
        ("torch.multiprocessing.reductions", "rebuild_storage_filename"),
        ("torch.multiprocessing.reductions", "rebuild_storage_empty"),
        ("torch.multiprocessing.reductions", "rebuild_typed_storage"),
        ("torch.multiprocessing.reductions", "rebuild_typed_storage_child"),
        ("torch", "per_tensor_affine"),
        ("torch", "per_tensor_symmetric"),
        ("torch", "per_channel_affine"),
        ("torch", "per_channel_symmetric"),
        ("torch", "per_channel_affine_float_qparams"),
        # --- SGLang data containers only (no code modules) ---
        ("sglang.srt.managers.io_struct", "GenerateReqInput"),
        ("sglang.srt.managers.io_struct", "EmbeddingReqInput"),
        ("sglang.srt.disaggregation.encoder.receiver", "EmbeddingData"),
        ("sglang.srt.managers.schedule_batch", "Modality"),
        ("sglang.srt.multimodal.modality", "Modality"),
        ("sglang.srt.weight_sync.tensor_bucket", "FlattenedTensorMetadata"),
        ("sglang.srt.weight_sync.tensor_bucket", "FlattenedTensorBucket"),
        (
            "sglang.srt.model_executor.model_runner_components.weight_updater",
            "LocalSerializedTensor",
        ),
        ("sglang.srt.model_executor.model_runner", "LocalSerializedTensor"),
    }

    def find_class(self, module, name):
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            # Torch's helper calls an unrestricted nested torch.load.
            return _safe_load_torch_storage
        if (module, name) in self.ALLOWED_GLOBALS:
            return super().find_class(module, name)

        raise RuntimeError(
            f"Blocked unsafe global ({module}.{name}) during pickle deserialization"
        )


def safe_pickle_load(fp):
    """Drop-in replacement for pickle.load() that blocks unsafe class loading."""
    return SafeUnpickler(fp).load()


def safe_pickle_loads(data):
    """Drop-in replacement for pickle.loads() that blocks unsafe class loading."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        buf = bytes(data)
    else:
        # zmq.Frame and other buffer-protocol objects
        buf = bytes(memoryview(data))
    return SafeUnpickler(io.BytesIO(buf)).load()

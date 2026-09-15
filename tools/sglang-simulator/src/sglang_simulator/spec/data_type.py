from enum import Enum, unique
from typing import Dict, Optional

_BYTES_MAP: dict["DataType", float] = {}
_ALIAS_MAP: Dict[str, str] = {}
_TORCH_DTYPE_TO_DATA_TYPE: Dict[str, "DataType"] = {}


@unique
class DataType(Enum):
    INT4 = "INT4"
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    FP4 = "FP4"
    FP8 = "FP8"
    FP16 = "FP16"
    BF16 = "BF16"
    TF32 = "TF32"
    FP32 = "FP32"
    FP64 = "FP64"
    # tensor
    INT4_TENSOR = "INT4_TENSOR"
    INT8_TENSOR = "INT8_TENSOR"
    INT16_TENSOR = "INT16_TENSOR"
    INT32_TENSOR = "INT32_TENSOR"
    INT64_TENSOR = "INT64_TENSOR"
    FP4_TENSOR = "FP4_TENSOR"
    FP8_TENSOR = "FP8_TENSOR"
    FP16_TENSOR = "FP16_TENSOR"
    BF16_TENSOR = "BF16_TENSOR"
    TF32_TENSOR = "TF32_TENSOR"
    FP32_TENSOR = "FP32_TENSOR"
    FP64_TENSOR = "FP64_TENSOR"

    # FIXME: This map will be added as a enum member.

    @property
    def bytes(self) -> float:
        return _BYTES_MAP.get(self, 1)

    @classmethod
    def tensor_suffix(cls) -> str:
        return "_TENSOR"

    @classmethod
    def alias(cls):
        return _ALIAS_MAP

    @classmethod
    def from_torch_dtype(cls, dtype: str) -> Optional["DataType"]:
        return _TORCH_DTYPE_TO_DATA_TYPE.get(dtype.lower())


_BYTES_MAP.update(
    {
        DataType.INT4: 0.5,
        DataType.INT8: 1,
        DataType.INT16: 2,
        DataType.INT32: 4,
        DataType.INT64: 8,
        DataType.FP4: 0.5,
        DataType.FP8: 1,
        DataType.FP16: 2,
        DataType.BF16: 2,
        DataType.TF32: 4,
        DataType.FP32: 4,
        DataType.FP64: 8,
        DataType.INT4_TENSOR: 0.5,
        DataType.INT8_TENSOR: 1,
        DataType.INT16_TENSOR: 2,
        DataType.INT32_TENSOR: 4,
        DataType.INT64_TENSOR: 8,
        DataType.FP4_TENSOR: 0.5,
        DataType.FP8_TENSOR: 1,
        DataType.FP16_TENSOR: 2,
        DataType.BF16_TENSOR: 2,
        DataType.TF32_TENSOR: 4,
        DataType.FP32_TENSOR: 4,
        DataType.FP64_TENSOR: 8,
    }
)

_ALIAS_MAP.update(
    {
        "int8": "INT8",
        "float8": "FP8",
        "float16": "FP16",
        "float32": "FP32",
        "bfloat16": "BF16",
    }
)

_TORCH_DTYPE_TO_DATA_TYPE.update(
    {
        "fp8": DataType.FP8,
        "int8": DataType.INT8,
        "float8": DataType.FP8,
        "float16": DataType.FP16,
        "float32": DataType.FP32,
        "bfloat16": DataType.BF16,
    }
)

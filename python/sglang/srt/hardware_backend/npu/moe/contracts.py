"""Activation contracts at native MoE compute and routing boundaries."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ActivationContract:
    kind: str
    block_size: int = 0
    scale_layout: str = "per_token_fp32"
    packing: int = 1
    round_mode: str = "default"
    scale_alg: int = 0


INT8 = ActivationContract("int8")
INT4 = ActivationContract("int4", packing=8)
MXFP8 = ActivationContract("mxfp8", 32, "e8m0_pairs")
MXFP4 = ActivationContract("mxfp4", 32, "e8m0_pairs", 2, "rint")


@dataclass(frozen=True)
class ProjectionContract:
    method: str
    input: Optional[ActivationContract]
    weight_dtype: torch.dtype
    weight_shape: tuple
    weight_format: Optional[int]
    side_tensors: tuple


def projection_contract(layer, prefix):
    weight = getattr(layer, f"{prefix}_weight")
    weight_format = None
    if weight.device.type == "npu":
        import torch_npu

        weight_format = torch_npu.get_npu_format(weight)
    metadata = tuple(
        (name, value.dtype, tuple(value.shape))
        for name, value in layer.named_parameters(recurse=False)
        if name.startswith(prefix + "_") and name != f"{prefix}_weight"
    )
    kernel = getattr(layer, f"{prefix}_kernel")
    return ProjectionContract(
        type(kernel).__name__,
        moe_input_contract(kernel),
        weight.dtype,
        tuple(weight.shape),
        weight_format,
        metadata,
    )


@dataclass(frozen=True)
class QuantizedOperand:
    data: torch.Tensor
    scale: torch.Tensor
    contract: ActivationContract
    source_shape: tuple
    source_dtype: torch.dtype = torch.bfloat16

    def __iter__(self):
        return iter((self.data, self.scale))

    def __getitem__(self, index):
        return (self.data, self.scale)[index]

    def validate(self, expected: ActivationContract):
        if self.contract != expected:
            raise ValueError(f"Incompatible NPU operand: {self.contract} -> {expected}")
        if self.source_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("NPU operand requires FP16/BF16 source activations")
        if self.data.device != self.scale.device:
            raise ValueError("NPU operand data and scale must share a device")
        if expected == INT8 and self.data.dtype != torch.int8:
            raise ValueError("INT8 operand requires integer INT8 data")
        if expected == MXFP8 and self.data.dtype != torch.float8_e4m3fn:
            raise ValueError("MXFP8 operand requires E4M3 data")
        if len(self.source_shape) != 2 or self.data.shape[0] != self.source_shape[0]:
            raise ValueError("NPU operand must preserve its logical token rows")
        if expected.packing == 1 and tuple(self.data.shape) != self.source_shape:
            raise ValueError("NPU operand shape differs from its source")
        if expected.packing > 1:
            if self.data.dtype not in (torch.int32, torch.uint8, torch.int8):
                raise ValueError("Packed operand requires integer storage")
            logical_elements = self.source_shape[0] * self.source_shape[1]
            if self.data.numel() * self.data.element_size() * 2 != logical_elements:
                raise ValueError(
                    "Packed operand must store exactly two 4-bit values per byte"
                )
        if expected.block_size:
            expected_scales = (
                self.source_shape[0] * self.source_shape[1] // expected.block_size
            )
            if self.source_shape[-1] % 64 or self.scale.numel() != expected_scales:
                raise ValueError("MX operand requires paired block32 scales")
            if self.scale.element_size() != 1:
                raise ValueError("MX operand requires byte-sized E8M0 scales")
        elif (
            self.scale.numel() != self.source_shape[0]
            or self.scale.dtype != torch.float32
        ):
            raise ValueError("Integer operand requires FP32 per-token scales")

    def record_stream(self):
        stream = torch.get_device_module(self.data.device).current_stream()
        self.data.record_stream(stream)
        self.scale.record_stream(stream)


def linear_kernel(layer):
    return getattr(getattr(layer, "scheme", None), "kernel", None) or getattr(
        layer, "quant_method", None
    )


def linear_input_contract(layer) -> Optional[ActivationContract]:
    from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
        NPU_W4A4DynamicLinearMethod,
        NPUMXFP4W4A8OfflineLinearMethod,
        NPUMXFP8LinearMethod,
        NPUSingleLevelMXFP4OfflineLinearMethod,
        NPUW8A8Int8DynamicLinearMethod,
    )

    kernel = linear_kernel(layer)
    if isinstance(kernel, (NPUMXFP8LinearMethod, NPUMXFP4W4A8OfflineLinearMethod)):
        return MXFP8
    if isinstance(kernel, NPUW8A8Int8DynamicLinearMethod):
        return INT8
    if isinstance(kernel, NPU_W4A4DynamicLinearMethod):
        return INT4
    if isinstance(kernel, NPUSingleLevelMXFP4OfflineLinearMethod):
        return MXFP4
    return None


def moe_input_contract(kernel) -> Optional[ActivationContract]:
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        _DEFAULT_DYNAMIC_QUANT,
        NPUMXFP8MoEMethod,
        NPUW4A4Int4MoEMethod,
        NPUW4A4MXFP4MoEMethod,
        NPUW4A8Int8MoEMethod,
        NPUW4A8MXFP4MoEMethod,
        NPUW8A8Int8MoEMethod,
    )

    if isinstance(kernel, NPUMXFP8MoEMethod):
        return MXFP8
    if isinstance(kernel, NPUW4A8MXFP4MoEMethod):
        return MXFP8 if kernel.dynamic_quant_kwargs is _DEFAULT_DYNAMIC_QUANT else None
    if isinstance(kernel, (NPUW4A8Int8MoEMethod, NPUW8A8Int8MoEMethod)):
        return INT8
    if isinstance(kernel, NPUW4A4Int4MoEMethod):
        return INT4
    if isinstance(kernel, NPUW4A4MXFP4MoEMethod):
        return MXFP4
    return None


def unpack_operand(value, expected):
    if isinstance(value, QuantizedOperand):
        value.validate(expected)
        value.record_stream()
        return value.data, value.scale, value.source_dtype, value.source_shape
    data, scale = value
    stream = torch.get_device_module(data.device).current_stream()
    data.record_stream(stream)
    scale.record_stream(stream)
    return data, scale, torch.bfloat16, tuple(data.shape)

# SPDX-License-Identifier: Apache-2.0
"""Native BMG ConvRot INT8: SYCL rotation/quantization and SYCL-TLA GEMM.

No comfy_kitchen dispatch or eager arithmetic fallback is used. The FP32
butterfly is not bit-exact with dense-matmul rotation at rounding boundaries.
"""

from __future__ import annotations

import functools

import torch
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.convrot_int8_config import (
    XPU_GROUP_SIZE,
    XPU_MAX_INPUT_SIZE,
    XPU_OUTPUT_ALIGNMENT,
    ConvRotInt8Config,
    xpu_supports_input_size,
)
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs

_OPS = ("convrot_rotate_quantize", "convrot_int8_linear_prequant")
# sgl_kernel.query_device reports BMG (Xe2) as (2, 0).
_BMG_DEVICE = (2, 0)


@functools.cache
def _load_xpu_kernels() -> tuple[torch._ops.OpOverload, torch._ops.OpOverload]:
    """Checks the ops and device once per process; returns the two op overloads."""
    try:
        import sgl_kernel
    except ImportError as exc:
        raise ImportError(
            "convrot_int8 backend xpu requires sgl-kernel-xpu built with "
            "convrot_rotate_quantize and convrot_int8_linear_prequant"
        ) from exc
    for name in _OPS:
        if not hasattr(sgl_kernel, name) or not hasattr(torch.ops.sgl_kernel, name):
            raise RuntimeError(
                f"convrot_int8 backend xpu requires {name}; rebuild and install "
                "sgl-kernel-xpu with ConvRot INT8 support"
            )
    if not torch.xpu.is_available():
        raise RuntimeError("convrot_int8 backend xpu requires an Intel BMG XPU")
    device = torch.xpu.current_device()
    if tuple(torch.ops.sgl_kernel.query_device(device)) != _BMG_DEVICE:
        raise RuntimeError("convrot_int8 backend xpu requires an Intel BMG/Xe2 XPU")
    return tuple(getattr(torch.ops.sgl_kernel, name).default for name in _OPS)


class ConvRotInt8XpuLinearMethod(LinearMethodBase):
    """Load or create compatible INT8 weights, then call the two native ops."""

    def __init__(
        self,
        quant_config: ConvRotInt8Config,
        *,
        group_size: int,
        is_checkpoint_serialized: bool,
    ) -> None:
        if group_size != XPU_GROUP_SIZE:
            raise ValueError(
                f"convrot_int8 backend xpu requires group_size {XPU_GROUP_SIZE}"
            )
        self.quant_config = quant_config
        self.group_size = group_size
        self.is_checkpoint_serialized = is_checkpoint_serialized
        self._rotate_quantize, self._linear_prequant = _load_xpu_kernels()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if params_dtype != torch.bfloat16:
            raise ValueError(
                "convrot_int8 backend xpu requires BF16 parameters and activations; "
                "use --dit-precision bf16"
            )
        # The config screened these shapes already; this guards direct use.
        if not xpu_supports_input_size(input_size_per_partition):
            raise ValueError(
                "convrot_int8 backend xpu requires input_size_per_partition in "
                f"[{XPU_GROUP_SIZE}, {XPU_MAX_INPUT_SIZE}], divisible by "
                f"{XPU_GROUP_SIZE}; got {input_size_per_partition}"
            )
        rows = sum(output_partition_sizes)
        if rows <= 0 or rows % XPU_OUTPUT_ALIGNMENT:
            raise ValueError(
                "convrot_int8 backend xpu requires a positive output size per "
                f"partition divisible by {XPU_OUTPUT_ALIGNMENT}; got {rows}"
            )
        UnquantizedLinearMethod.create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            torch.int8 if self.is_checkpoint_serialized else params_dtype,
            **extra_weight_attrs,
        )
        if self.is_checkpoint_serialized:
            # Comfy stores the per-row scale as [N, 1]; the GEMM accepts it as is.
            scale = Parameter(
                torch.empty(rows, 1, dtype=torch.float32), requires_grad=False
            )
            set_weight_attrs(scale, {"output_dim": 0})
            set_weight_attrs(scale, extra_weight_attrs)
            layer.register_parameter("weight_scale", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.is_checkpoint_serialized:
            self.quant_config.note_loaded()
            return
        weight = layer.weight.data
        if weight.dtype == torch.int8:
            return
        home = weight.device
        # Offloaded layers are quantized one at a time on this worker's XPU.
        device = (
            home
            if home.type == "xpu"
            else torch.device("xpu", torch.xpu.current_device())
        )
        qdata, scale = self._rotate_quantize(
            weight.to(device=device, non_blocking=True).contiguous(), self.group_size
        )
        layer.weight = Parameter(qdata.to(home), requires_grad=False)
        layer.register_parameter(
            "weight_scale", Parameter(scale.to(home), requires_grad=False)
        )
        self.quant_config.note_quantized(weight.numel() * weight.element_size())

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or not x.is_xpu:
            raise ValueError(
                "convrot_int8 backend xpu requires BF16 activations on XPU, "
                f"got {x.dtype} on {x.device}"
            )
        # The ops validate K against the weight and the supported range.
        rows = x.reshape(-1, x.shape[-1]).contiguous()
        qdata, scale = self._rotate_quantize(rows, self.group_size)
        out = self._linear_prequant(
            qdata, layer.weight, scale, layer.weight_scale, bias
        )
        return out.view(*x.shape[:-1], out.shape[-1])

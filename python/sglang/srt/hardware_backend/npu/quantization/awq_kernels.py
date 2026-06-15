from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe.moe_runner.torch_npu import (
    TorchNpuQuantInfo,
)
from sglang.srt.layers.quantization.utils import replace_parameter

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A16Int4MoEMethod,
)
import torch_npu


class AWQAscendLinearKernel:
    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Original scales are (K // group_size, N) → transpose to (N, K // group_size)
        layer.scales = torch.nn.Parameter(
            layer.scales.data.transpose(0, 1).contiguous(), requires_grad=False
        )

        # Unpack qweight and qzeros
        qweight_tmp = torch.zeros_like(layer.qweight.data)   # (K, N // pack_factor) int32
        qzeros_tmp = layer.qzeros.data
        qzeros_list = []
        shifts = [0, 4, 1, 5, 2, 6, 3, 7]

        for i in range(self.quant_config.pack_factor):
            shift_num = shifts[i] * 4
            qzeros_list.append((qzeros_tmp.reshape(-1, 1) >> shift_num) & 0xF)
            qweight_tmp.bitwise_or_(
                ((layer.qweight.data >> shift_num) & 0xF) << (4 * i)
            )

        qweight_tmp.bitwise_xor_(0x88888888)

        # qzeros unpacking
        qzeros_tmp = torch.cat(qzeros_list, dim=-1)  # (num_groups * pack_factor, N // pack_factor) ?
        # After unpacking, each row expands by pack_factor along columns → shape (num_groups, N)
        qzeros_tmp = qzeros_tmp.reshape(layer.qzeros.shape[0], -1)   # (num_groups, N)
        qzeros_tmp = -(qzeros_tmp - 8)
        qzeros_tmp = qzeros_tmp.to(layer.scales.data.dtype)

        # Transpose zeros to (N, num_groups) to match scales
        qzeros_tmp = qzeros_tmp.transpose(0, 1).contiguous()

        layer.zeros = torch.nn.Parameter(qzeros_tmp, requires_grad=False)

        # Transpose weight from (K, N // pack_factor) to (N, K // pack_factor)
        qweight_tmp = qweight_tmp.transpose(0, 1).contiguous()
        layer.weight = torch.nn.Parameter(qweight_tmp, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.weight          # (N, K // pack_factor)
        scales = layer.scales           # (N, 1) or (N, num_groups)
        qzeros = layer.zeros            # (N, 1) or (N, num_groups)
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[0],)
        reshaped_x = x.reshape(-1, x.shape[-1])

        K = qweight.shape[1] * pack_factor   # input features

        # Derive group_size from scales
        if scales.ndim == 2 and scales.shape[1] == 1:
            group_size = 0                     # per‑channel
        elif scales.ndim == 2:
            num_groups = scales.shape[1]       # K // group_size
            if K % num_groups != 0:
                raise RuntimeError(f"K={K} not divisible by scale groups {num_groups}")
            group_size = K // num_groups
            # NPU constraint: must be multiple of 32, between 32 and K-1
            if group_size % 32 != 0 or not (32 <= group_size <= K - 1):
                raise RuntimeError(
                    f"Derived group_size={group_size} is invalid for NPU"
                )
        else:
            raise RuntimeError(f"Unexpected scale shape: {scales.shape}")

        if bias is not None and bias.dtype == torch.bfloat16:
            bias = bias.float()

        out = torch_npu.npu_weight_quant_batchmatmul(
            reshaped_x,
            qweight,
            antiquant_scale=scales,
            antiquant_offset=qzeros,
            antiquant_group_size=group_size,
            bias=bias,
        )
        return out.reshape(out_shape)


class AWQAscendMoEKernel:
    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config
        self.w13_kernel = NPUW4A16Int4MoEMethod()
        self.w2_kernel = NPUW4A16Int4MoEMethod()

    @staticmethod
    def _register_or_replace_parameter(
        layer: torch.nn.Module, name: str, tensor: torch.Tensor
    ) -> None:
        if hasattr(layer, name):
            replace_parameter(layer, name, tensor)
        else:
            layer.register_parameter(
                name, torch.nn.Parameter(tensor, requires_grad=False)
            )

    def _convert_awq_weight_to_npu_layout(self, qweight: torch.Tensor) -> torch.Tensor:
        num_experts, input_size, _ = qweight.shape
        unpacked_weight = (
            self.w13_kernel._unpack_from_int32(qweight.flatten(0, 1), 4)
            .view(num_experts, input_size, -1)
            .transpose(1, 2)
            .contiguous()
            .int()
        )
        return self.w13_kernel._pack_to_int32(unpacked_weight)

    def _convert_awq_qzeros_to_npu_offset(
        self, qzeros: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        num_experts, num_groups, _ = qzeros.shape
        offset = (
            -self.w13_kernel._unpack_from_int32(qzeros.flatten(0, 1), 4)
            .view(num_experts, num_groups, -1)
            .transpose(1, 2)
            .contiguous()
        )
        return offset.to(dtype)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._register_or_replace_parameter(
            layer,
            "w13_weight",
            self._convert_awq_weight_to_npu_layout(layer.w13_qweight.data),
        )
        self._register_or_replace_parameter(
            layer,
            "w2_weight",
            self._convert_awq_weight_to_npu_layout(layer.w2_qweight.data),
        )
        self._register_or_replace_parameter(
            layer,
            "w13_weight_scale",
            layer.w13_scales.data.transpose(1, 2).contiguous(),
        )
        self._register_or_replace_parameter(
            layer,
            "w2_weight_scale",
            layer.w2_scales.data.transpose(1, 2).contiguous(),
        )
        self._register_or_replace_parameter(
            layer,
            "w13_weight_offset",
            self._convert_awq_qzeros_to_npu_offset(
                layer.w13_qzeros.data, layer.w13_scales.data.dtype
            ),
        )
        self._register_or_replace_parameter(
            layer,
            "w2_weight_offset",
            self._convert_awq_qzeros_to_npu_offset(
                layer.w2_qzeros.data, layer.w2_scales.data.dtype
            ),
        )

        self.w13_kernel.process_weights_after_loading(layer, "w13")
        self.w2_kernel.process_weights_after_loading(layer, "w2")

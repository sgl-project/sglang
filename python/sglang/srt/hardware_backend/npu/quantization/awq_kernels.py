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
import torch.nn.functional as F


class AWQAscendLinearKernel:
    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Keep scales as a parameter (groups, N) – needed for NPU path
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

        # Unpack weight
        qweight_tmp = torch.zeros_like(layer.qweight.data)
        qzeros_tmp = layer.qzeros.data
        qzeros_list = []
        shifts = [0, 4, 1, 5, 2, 6, 3, 7]

        for i in range(self.quant_config.pack_factor):
            shift_num = shifts[i] * 4
            qzeros_list.append((qzeros_tmp.reshape(-1, 1) >> shift_num) & 0xF)
            qweight_tmp.bitwise_or_(
                ((layer.qweight.data >> shift_num) & 0xF) << (4 * i)
            )

        qweight_tmp.bitwise_xor_(0x88888888)   # signed int4 now

        # Unpack zeros – not needed beyond shape calculations, but compute anyway
        qzeros_tmp = torch.cat(qzeros_list, dim=-1).reshape(layer.qzeros.shape[0], -1)

        pack_factor = self.quant_config.pack_factor
        K = qweight_tmp.shape[0]
        N = qweight_tmp.shape[1] * pack_factor
        num_groups = layer.scales.shape[0]

        if K % num_groups != 0:
            raise RuntimeError(f"K={K} not divisible by scale groups {num_groups}")
        group_size = K // num_groups

        # NPU constraint
        npu_ok = (group_size == 0) or (group_size % 32 == 0 and 32 <= group_size < K)

        if npu_ok:
            # Use NPU kernel – keep weight, scales, and precomputed group_size
            layer.register_parameter("weight", torch.nn.Parameter(qweight_tmp, requires_grad=False))
            layer.use_npu_matmul = True
            layer.npu_group_size = group_size
            # scales already a parameter, do not delete
        else:
            # Fallback: dequantize to bfloat16
            weight_int8 = torch.zeros((K, N), dtype=torch.int8, device=qweight_tmp.device)
            for i in range(pack_factor):
                weight_int8[:, i::pack_factor] = ((qweight_tmp >> (4 * i)) & 0xF).to(torch.int8)

            if group_size > 0:
                scales_exp = layer.scales.data.repeat_interleave(group_size, dim=0)   # (K, N)
            else:
                scales_exp = layer.scales.data

            # Weight already signed, no zero‑point subtraction
            weight_float = weight_int8.float() * scales_exp.float()
            weight_float = weight_float.t().contiguous().to(torch.bfloat16)

            w = weight_float
            print(w.min(), w.max(), torch.isinf(w).any())

            layer.register_parameter("weight", torch.nn.Parameter(weight_float, requires_grad=False))
            # No need for scales anymore – we can delete them
            delattr(layer, "scales")
            layer.use_npu_matmul = False

        # Always delete original packed tensors – no longer needed
        for attr in ("qweight", "qzeros"):
            if hasattr(layer, attr):
                delattr(layer, attr)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])

        if layer.use_npu_matmul:
            qweight = layer.weight          # (K, N//pack) int32, XORed signed
            scales = layer.scales           # (groups, N) – must exist
            pack_factor = self.quant_config.pack_factor
            out_shape = x.shape[:-1] + (qweight.shape[1] * pack_factor,)

            if bias is not None and bias.dtype == torch.bfloat16:
                bias = bias.float()

            out = torch_npu.npu_weight_quant_batchmatmul(
                reshaped_x,
                qweight,
                antiquant_scale=scales,
                antiquant_offset=None,                 # no offset
                antiquant_group_size=layer.npu_group_size,
                bias=bias,
            )
            print('quant', out)
            return out.reshape(out_shape)
        else:
            # fallback: weight is (N, K) bfloat16
            print('unquant', out)
            out = F.linear(x, layer.weight, bias)
            return out


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

        # --- Free memory: delete original AWQ parameters ---
        for attr in ("w13_qweight", "w13_qzeros", "w13_scales",
                     "w2_qweight", "w2_qzeros", "w2_scales"):
            if hasattr(layer, attr):
                delattr(layer, attr)

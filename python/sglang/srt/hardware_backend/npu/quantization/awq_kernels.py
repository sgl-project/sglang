from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
import torch_npu

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUWNA16Int4MoEMethod,
)
from sglang.srt.layers.quantization.utils import replace_parameter

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


class AWQAscendLinearKernel:
    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Keep scales as (groups, N) – NPU kernel expects this layout
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

        raw_qweight = layer.qweight.data  # (K, N // pack_factor)
        raw_qzeros = layer.qzeros.data  # (groups, N // pack_factor)

        pack_factor = self.quant_config.pack_factor
        # shifts control which 4-bit nibble we extract from each packed byte:
        #   byte = [nibble_7 | nibble_6 | ... | nibble_0]
        #   shift = 4*i gives the i-th nibble's bit offset.
        shifts = [0, 4, 1, 5, 2, 6, 3, 7]

        K = raw_qweight.shape[0]
        N = raw_qweight.shape[1] * pack_factor
        num_groups = layer.scales.shape[0]

        if K % num_groups != 0:
            raise RuntimeError(f"K={K} not divisible by scale groups {num_groups}")
        group_size = K // num_groups

        # NPU fast-path constraint:
        #   The NPU's `npu_weight_quant_batchmatmul` kernel requires group_size
        #   to be a multiple of 32 and at least 32, but less than K (otherwise
        #   per-tensor scaling would apply, which is a different code path).
        #   This aligns with the NPU's SIMD vectorization width (32 elements)
        #   and ensures efficient memory access patterns.
        is_support_npu_quant_mm = (group_size == 0) or (
            group_size % 32 == 0 and 32 <= group_size < K
        )

        if is_support_npu_quant_mm:
            # ----- NPU fast path: unsigned weight + raw zero point -----
            # The NPU kernel expects:
            #   1. qweight: packed unsigned 4-bit values (no XOR)
            #   2. zeros: raw zero-point values (not dequantized)
            #
            # Step 1: Pack weight as unsigned nibbles (NO XOR).
            #   We extract each 4-bit nibble from the original packed tensor
            #   and repack them into a new tensor where each byte contains
            #   two 4-bit values in the order expected by the NPU kernel.
            qweight_tmp = torch.zeros_like(raw_qweight)
            qzeros_list = []
            for i in range(pack_factor):
                shift_num = shifts[i] * 4
                qzeros_list.append((raw_qzeros.reshape(-1, 1) >> shift_num) & 0xF)
                qweight_tmp.bitwise_or_(
                    ((layer.qweight.data >> shift_num) & 0xF) << (4 * i)
                )

            # Step 2: XOR with 0x88888888 to convert from signed to unsigned
            #   representation. The original weights are stored as signed int4
            #   (values -8..7). XOR with 0x8 flips the sign bit, mapping
            #   -8 → 0, -7 → 1, ..., 7 → 15. This yields the unsigned
            #   representation the NPU kernel expects.
            #
            #   Mathematical formula:
            #     unsigned_val = signed_val ^ 0x8   (for each 4-bit nibble)
            #   Since we pack two nibbles per byte, we XOR the whole byte
            #   with 0x88 to flip both sign bits simultaneously.
            qweight_tmp.bitwise_xor_(0x88888888)  # 0x88 per byte = flip sign bit of both nibbles

            # Step 3: Convert zero points from signed to unsigned.
            #   The zero points are stored as signed int4 (-8..7).
            #   We convert them to unsigned (0..15) by subtracting 8,
            #   then negate to get the raw zero-point value expected by the NPU.
            #     unsigned_zero = signed_zero + 8
            #     raw_zero = -unsigned_zero
            qzeros_tmp = torch.cat(qzeros_list, dim=-1).reshape(raw_qzeros.shape[0], -1)
            qzeros_tmp = -(qzeros_tmp - 8)  # convert signed → unsigned → negated
            qzeros_tmp = qzeros_tmp.to(layer.scales.data.dtype)

            layer.zeros = torch.nn.Parameter(qzeros_tmp, requires_grad=False)
            layer.weight = torch.nn.Parameter(qweight_tmp, requires_grad=False)

            layer.use_npu_matmul = True
            layer.npu_group_size = group_size
        else:
            # ----- Fallback: asymmetric dequantisation on CPU/NPU via standard linear -----
            # When group_size doesn't meet the NPU constraint, we fall back to
            # a standard dequantisation + FP16 linear. This is gives memory overhead but correct
            # for all group_size values.
            weight_u8 = torch.zeros((K, N), dtype=torch.int8, device=raw_qweight.device)
            zeros_u8 = torch.zeros(
                (num_groups, N), dtype=torch.int8, device=raw_qzeros.device
            )

            for i in range(pack_factor):
                shift = shifts[i] * 4
                nib_w = (raw_qweight >> shift) & 0xF
                weight_u8[:, i::pack_factor] = nib_w.to(torch.int8)
                nib_z = (raw_qzeros >> shift) & 0xF
                zeros_u8[:, i::pack_factor] = nib_z.to(torch.int8)

            # Dequantize: weight_fp = (weight_u8 - zeros) * scales
            if group_size > 0:
                zeros_exp = zeros_u8.repeat_interleave(group_size, dim=0)
                scales_exp = layer.scales.data.repeat_interleave(group_size, dim=0)
            else:
                zeros_exp = zeros_u8
                scales_exp = layer.scales.data

            weight_float = (weight_u8.float() - zeros_exp.float()) * scales_exp.float()
            weight_float = weight_float.t().contiguous().to(torch.bfloat16)

            layer.register_parameter(
                "weight", torch.nn.Parameter(weight_float, requires_grad=False)
            )
            delattr(layer, "scales")
            layer.use_npu_matmul = False

        # Clean original packed tensors to free memory
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
        pack_factor = self.quant_config.pack_factor

        if layer.use_npu_matmul:
            qweight = layer.weight  # (K, N//pack) int32, unsigned
            scales = layer.scales  # (groups, N)
            offset = layer.zeros  # (groups, N) raw zero point

            out_shape = x.shape[:-1] + (qweight.shape[1] * pack_factor,)
            if bias is not None and bias.dtype == torch.bfloat16:
                bias = bias.float()

            # NPU-accelerated quantized matmul.
            # The kernel internally does:
            #   out = (x @ qweight_dequantized) + bias
            # where qweight_dequantized = (qweight_unsigned - offset) * scales
            # with group-wise scaling applied.
            out = torch_npu.npu_weight_quant_batchmatmul(
                reshaped_x,
                qweight,
                antiquant_scale=scales,
                antiquant_offset=offset,  # raw zero point
                antiquant_group_size=layer.npu_group_size,
                bias=bias,
            )
            return out.reshape(out_shape)
        else:
            return F.linear(x, layer.weight, bias)


class AWQAscendMoEKernel:
    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config
        self.w13_kernel = NPUWNA16Int4MoEMethod()
        self.w2_kernel = NPUWNA16Int4MoEMethod()

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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13_qweight_tmp = torch.zeros_like(layer.w13_qweight.data)
        w2_qweight_tmp = torch.zeros_like(layer.w2_qweight.data)
        w13_qzeros_list = []
        w2_qzeros_list = []

        # shifts control which 4-bit nibble we extract from each packed byte.
        # For AWQ with pack_factor=8, each byte contains 8 nibbles (4-bit values).
        # shifts = [0,4,1,5,2,6,3,7] extracts nibbles in the order:
        #   nibble_0, nibble_1, nibble_2, ..., nibble_7
        # but interleaved to match the NPU kernel's expected layout.
        shifts = [0, 4, 1, 5, 2, 6, 3, 7]

        for i in range(self.quant_config.pack_factor):
            shift_num = shifts[i] * 4
            w13_qzeros_list.append(
                (layer.w13_qzeros.data.reshape(-1, 1) >> shift_num) & 0xF
            )
            w2_qzeros_list.append(
                (layer.w2_qzeros.data.reshape(-1, 1) >> shift_num) & 0xF
            )
            w13_qweight_tmp.bitwise_or_(
                ((layer.w13_qweight.data >> shift_num) * (2 ** (4 * i)))
                & (0xF << (4 * i))
            )
            w2_qweight_tmp.bitwise_or_(
                ((layer.w2_qweight.data >> shift_num) * (2 ** (4 * i)))
                & (0xF << (4 * i))
            )

        # XOR with 0x88888888 converts signed int4 to unsigned int4.
        # Each byte contains two 4-bit values, so 0x88 flips the sign bit
        # of both nibbles simultaneously.
        #
        #   signed_val:  -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7
        #   unsigned_val: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        #   signed_val ^ 0x8 maps -8→0, -7→1, ..., 7→15.
        w13_qweight_tmp.bitwise_xor_(0x88888888)
        w2_qweight_tmp.bitwise_xor_(0x88888888)

        # Convert zero points: signed int4 → unsigned → negated.
        # The NPU kernel expects raw zero-point values (not dequantized).
        w13_qzeros_tmp = torch.cat(w13_qzeros_list, dim=-1).reshape(
            layer.w13_qzeros.shape[0], layer.w13_qzeros.shape[1], -1
        )
        w13_qzeros_tmp = -(w13_qzeros_tmp - 8)  # signed → unsigned → negated
        w13_qzeros_tmp = w13_qzeros_tmp.to(layer.w13_scales.data.dtype)

        w2_qzeros_tmp = torch.cat(w2_qzeros_list, dim=-1).reshape(
            layer.w2_qzeros.shape[0], layer.w2_qzeros.shape[1], -1
        )
        w2_qzeros_tmp = -(w2_qzeros_tmp - 8)
        w2_qzeros_tmp = w2_qzeros_tmp.to(layer.w2_scales.data.dtype)

        layer.register_parameter(
            "w13_qzeros", torch.nn.Parameter(w13_qzeros_tmp, requires_grad=False)
        )
        layer.register_parameter(
            "w13_qweight", torch.nn.Parameter(w13_qweight_tmp, requires_grad=False)
        )
        layer.register_parameter(
            "w2_qzeros", torch.nn.Parameter(w2_qzeros_tmp, requires_grad=False)
        )
        layer.register_parameter(
            "w2_qweight", torch.nn.Parameter(w2_qweight_tmp, requires_grad=False)
        )

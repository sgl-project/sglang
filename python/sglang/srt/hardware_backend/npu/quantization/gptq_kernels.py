from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

# NPU utility for NZ format conversion
from sglang.srt.hardware_backend.npu.utils import npu_format_cast


def unpack_from_int32(
    weight: torch.Tensor,
    num_bits: int,
    packed_dim: int = 1,
) -> torch.Tensor:
    """
    Unpacks quantized weights from int32 format back to original bits.

    :param weight: The packed int32 tensor containing quantized weights
    :param num_bits: The number of bits used for quantization (<= 8)
    :param packed_dim: Dimension along which weights are packed (0 or 1), defaults to 1
    :return: Unpacked tensor with int8 dtype after applying offset correction
    """
    assert (
        weight.dtype == torch.int32
    ), f"Expecting `weight.dtype` is torch.int32 but got {weight.dtype}."
    assert (
        num_bits <= 8
    ), f"Expecting `num_bits` should not be larger than 8 but got {num_bits}."

    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1

    if packed_dim == 1:
        unpacked_weight = torch.zeros(
            (weight.shape[0], weight.shape[1] * pack_factor),
            device=weight.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked_weight[:, i::pack_factor] = (weight >> (num_bits * i)) & mask
    else:
        unpacked_weight = torch.zeros(
            (weight.shape[0] * pack_factor, weight.shape[1]),
            device=weight.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked_weight[i::pack_factor, :] = (weight >> (num_bits * i)) & mask
    offset = pow(2, num_bits) // 2
    unpacked_weight = (unpacked_weight - offset).to(torch.int8)
    return unpacked_weight

class GPTQLinearAscendKernel:
    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config
        self.use_v2_format = quant_config.checkpoint_format == "gptq_v2"

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        layer.qzeros = torch.nn.Parameter(
            unpack_from_int32(
                layer.qzeros.data.contiguous(),
                self.quant_config.weight_bits,
                packed_dim=1,
            ).to(layer.scales.dtype),
            requires_grad=False,
        )
        if not self.use_v2_format:
            layer.qzeros += 1

        qweight_tmp = unpack_from_int32(
            layer.qweight.data.contiguous(), self.quant_config.weight_bits, packed_dim=0
        )
        # use int8 to store weight by default
        if self.quant_config.weight_bits != 4:
            layer.qweight = torch.nn.Parameter(
                qweight_tmp,
                requires_grad=False,
            )
            return

        # for 4bit case we need to pack 4bit weight to int32 to save memory
        layer.qweight = torch.nn.Parameter(
            torch.ops.npu.npu_convert_weight_to_int4pack(qweight_tmp.to(torch.int32)),
            requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros

        reshaped_x = x.reshape(-1, x.shape[-1])

        if bias is not None and bias.dtype == torch.bfloat16:
            bias = bias.float()

        # 4bit weight is packed to int32(8 x int4)
        if self.quant_config.weight_bits == 4:
            out_shape = x.shape[:-1] + (qweight.shape[-1] * 8,)
        else:
            out_shape = x.shape[:-1] + (qweight.shape[-1],)

        out = torch.ops.npu.npu_weight_quant_batchmatmul(
            reshaped_x,
            qweight,
            antiquant_scale=scales,
            antiquant_offset=qzeros,
            antiquant_group_size=self.quant_config.group_size,
            bias=bias,
        )

        return out.reshape(out_shape)


class GPTQMoEAscendKernel:
    def __init__(self, quant_config: Optional["QuantizationConfig"] = None):
        self.quant_config = quant_config
        self.use_v2_format = quant_config.checkpoint_format == "gptq_v2"

    def _pack_to_int32(self, weight: torch.Tensor) -> torch.Tensor:
        # pack 4 int8 (representing 8 int4) into int32
        assert weight.shape[-1] % 4 == 0, (
            f"Last dimension of weight must be divisible by 4 for int8→int32 packing, "
            f"got shape {weight.shape}"
        )
        return weight.contiguous().view(torch.int32)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pack_factor = 32 // self.quant_config.weight_bits
    
        # ---------- w13 ----------
        # zero points (unchanged)
        w13_qzeros_2d = layer.w13_qzeros.data.contiguous().reshape(-1, layer.w13_qzeros.shape[-1])
        layer.w13_qzeros = torch.nn.Parameter(
            unpack_from_int32(w13_qzeros_2d, self.quant_config.weight_bits, packed_dim=1)
            .reshape(layer.w13_qzeros.shape[0], layer.w13_qzeros.shape[1], -1)
            .to(layer.w13_scales.dtype),
            requires_grad=False,
        )
        if not self.use_v2_format:
            layer.w13_qzeros += 1
    
        # Unpack weight
        w13_qweight_2d = (
            layer.w13_qweight.data.transpose(-1, -2)
            .contiguous()
            .reshape(-1, layer.w13_qweight.shape[-2])
        )
        w13_qweight_tmp = unpack_from_int32(w13_qweight_2d, self.quant_config.weight_bits, packed_dim=1)
        # w13_qweight_tmp shape: (E * K, N)
    
        E_w13 = layer.w13_qweight.shape[0]          # num experts
        N_w13 = w13_qweight_tmp.shape[1]            # output features (N)
        K_w13 = w13_qweight_tmp.shape[0] // E_w13   # input features (K)
    
        # 1) Reshape to [E, K, N] – used for sign flipping
        w13_weight = w13_qweight_tmp.reshape(E_w13, K_w13, N_w13)   # [E, K, N]
    
        # 2) Handle negative scales (4‑bit only) – mask matches [E, K, N]
        if self.quant_config.weight_bits == 4:
            group_size = self.quant_config.group_size
            # Scale shape: [E, groups, N] -> expand to [E, K, N]
            scale_expanded = layer.w13_scales.data.repeat_interleave(group_size, dim=1)  # [E, K, N]
            neg_mask = scale_expanded < 0
            if neg_mask.any():
                w13_weight[neg_mask] = -w13_weight[neg_mask]
                w13_weight.clamp_(max=7)
            layer.w13_scales.data.abs_()   # scales stay [E, groups, N]
    
        # 3) Permute to [E, N, K] required by NPU matmul
        w13_weight = w13_weight.permute(0, 2, 1).contiguous()   # [E, N, K]
    
        # 4) Convert to NPU NZ format
        w13_weight = npu_format_cast(w13_weight)
    
        # 5) Pack 4 int8 values into int32 along K dimension
        if self.quant_config.weight_bits == 4:
            assert w13_weight.shape[-1] % 4 == 0
            w13_weight = self._pack_to_int32(w13_weight)   # [E, N, K//4] int32
    
        layer.w13_qweight = torch.nn.Parameter(w13_weight, requires_grad=False)
    
        # ---------- w2 (identical pattern) ----------
        w2_qzeros_2d = layer.w2_qzeros.data.contiguous().reshape(-1, layer.w2_qzeros.shape[-1])
        layer.w2_qzeros = torch.nn.Parameter(
            unpack_from_int32(w2_qzeros_2d, self.quant_config.weight_bits, packed_dim=1)
            .reshape(layer.w2_qzeros.shape[0], layer.w2_qzeros.shape[1], -1)
            .to(layer.w2_scales.dtype),
            requires_grad=False,
        )
        if not self.use_v2_format:
            layer.w2_qzeros += 1
    
        w2_qweight_2d = (
            layer.w2_qweight.data.transpose(-1, -2)
            .contiguous()
            .reshape(-1, layer.w2_qweight.shape[-2])
        )
        w2_qweight_tmp = unpack_from_int32(w2_qweight_2d, self.quant_config.weight_bits, packed_dim=1)
    
        E_w2 = layer.w2_qweight.shape[0]
        N_w2 = w2_qweight_tmp.shape[1]
        K_w2 = w2_qweight_tmp.shape[0] // E_w2
    
        # 1) Reshape to [E, K, N]
        w2_weight = w2_qweight_tmp.reshape(E_w2, K_w2, N_w2)   # [E, K, N]
    
        # 2) Sign flip with mask in [E, K, N]
        if self.quant_config.weight_bits == 4:
            scale_expanded = layer.w2_scales.data.repeat_interleave(group_size, dim=1)  # [E, K, N]
            neg_mask = scale_expanded < 0
            if neg_mask.any():
                w2_weight[neg_mask] = -w2_weight[neg_mask]
                w2_weight.clamp_(max=7)
            layer.w2_scales.data.abs_()
    
        # 3) Permute to [E, N, K]
        w2_weight = w2_weight.permute(0, 2, 1).contiguous()   # [E, N, K]
    
        # 4) NZ format conversion
        w2_weight = npu_format_cast(w2_weight)
    
        # 5) Pack to int32 (for 4-bit)
        if self.quant_config.weight_bits == 4:
            assert w2_weight.shape[-1] % 4 == 0
            w2_weight = self._pack_to_int32(w2_weight)
    
        layer.w2_qweight = torch.nn.Parameter(w2_weight, requires_grad=False)

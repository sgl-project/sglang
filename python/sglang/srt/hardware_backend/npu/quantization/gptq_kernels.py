from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

import logging

logger = logging.getLogger(__name__)


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
    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config
        self.use_v2_format = quant_config.checkpoint_format == "gptq_v2"

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # ----- zero‑points (unchanged) -----
        w13_qzeros_2d = layer.w13_qzeros.data.contiguous().reshape(
            -1, layer.w13_qzeros.shape[-1]
        )
        layer.w13_qzeros = torch.nn.Parameter(
            unpack_from_int32(
                w13_qzeros_2d,
                self.quant_config.weight_bits,
                packed_dim=1,
            )
            .reshape(layer.w13_qzeros.shape[0], layer.w13_qzeros.shape[1], -1)
            .to(layer.w13_scales.dtype),
            requires_grad=False,
        )
        if not self.use_v2_format:
            layer.w13_qzeros += 1

        w2_qzeros_2d = layer.w2_qzeros.data.contiguous().reshape(
            -1, layer.w2_qzeros.shape[-1]
        )
        layer.w2_qzeros = torch.nn.Parameter(
            unpack_from_int32(
                w2_qzeros_2d,
                self.quant_config.weight_bits,
                packed_dim=1,
            )
            .reshape(layer.w2_qzeros.shape[0], layer.w2_qzeros.shape[1], -1)
            .to(layer.w2_scales.dtype),
            requires_grad=False,
        )
        if not self.use_v2_format:
            layer.w2_qzeros += 1

        # ----- w13 -----
        w13_qweight_2d = (
            layer.w13_qweight.data.transpose(-1, -2)
            .contiguous()
            .reshape(-1, layer.w13_qweight.shape[-2])
        )
        w13_qweight_tmp = unpack_from_int32(
            w13_qweight_2d, self.quant_config.weight_bits, packed_dim=1
        )

        if self.quant_config.weight_bits == 4:
            group_size = self.quant_config.group_size
            k_shard_w13 = w13_qweight_tmp.shape[1]

            # Check if the scales are compatible (expanded size must equal K_shard)
            if layer.w13_scales.shape[1] * group_size != k_shard_w13:
                logger.warning_once(
                    f"w13 scales expanded size {layer.w13_scales.shape[1] * group_size} "
                    f"does not match K_shard {k_shard_w13}. Skipping negative-scale correction."
                    f"This may break the accuracy, please try another TP-size or use DeepEP."
                )
                # pack directly
                layer.w13_qweight = torch.nn.Parameter(
                    torch.ops.npu.npu_convert_weight_to_int4pack(
                        w13_qweight_tmp.reshape(
                            layer.w13_qweight.shape[0], layer.w13_qweight.shape[2], -1
                        )
                        .transpose(-1, -2)
                        .contiguous()
                        .reshape(-1, layer.w13_qweight.shape[2])
                        .to(torch.int32)
                    )
                    .reshape(
                        layer.w13_qweight.shape[0], layer.w13_qweight.shape[1] * 8, -1
                    )
                    .contiguous(),
                    requires_grad=False,
                )
            else:
                scale_expanded = layer.w13_scales.data.repeat_interleave(
                    group_size, dim=1
                )
                neg_mask = scale_expanded < 0
                if neg_mask.any():
                    neg_mask = neg_mask.transpose(-1, -2)
                    neg_mask = neg_mask.contiguous().reshape(w13_qweight_tmp.shape)
                    w13_qweight_tmp[neg_mask] = -w13_qweight_tmp[neg_mask]
                    if w13_qweight_tmp.max() > 7:
                        w13_qweight_tmp.clamp_(max=7)
                    layer.w13_scales.data.abs_()

                layer.w13_qweight = torch.nn.Parameter(
                    torch.ops.npu.npu_convert_weight_to_int4pack(
                        w13_qweight_tmp.reshape(
                            layer.w13_qweight.shape[0], layer.w13_qweight.shape[2], -1
                        )
                        .transpose(-1, -2)
                        .contiguous()
                        .reshape(-1, layer.w13_qweight.shape[2])
                        .to(torch.int32)
                    )
                    .reshape(
                        layer.w13_qweight.shape[0], layer.w13_qweight.shape[1] * 8, -1
                    )
                    .contiguous(),
                    requires_grad=False,
                )
        else:
            layer.w13_qweight = torch.nn.Parameter(
                w13_qweight_tmp.reshape(
                    layer.w13_qweight.shape[0], layer.w13_qweight.shape[2], -1
                )
                .transpose(-1, -2)
                .contiguous(),
                requires_grad=False,
            )

        # ----- w2 -----
        w2_qweight_2d = (
            layer.w2_qweight.data.transpose(-1, -2)
            .contiguous()
            .reshape(-1, layer.w2_qweight.shape[-2])
        )
        w2_qweight_tmp = unpack_from_int32(
            w2_qweight_2d, self.quant_config.weight_bits, packed_dim=1
        )

        if self.quant_config.weight_bits == 4:
            group_size = self.quant_config.group_size
            k_shard_w2 = w2_qweight_tmp.shape[1]

            # Check if the scales are compatible
            if layer.w2_scales.shape[1] * group_size != k_shard_w2:
                logger.warning_once(
                    f"w2 scales expanded size {layer.w2_scales.shape[1] * group_size} "
                    f"does not match K_shard {k_shard_w2}. Skipping negative-scale correction."
                    f"This may break the accuracy, please try another TP-size or use DeepEP."
                )
                # pack directly
                layer.w2_qweight = torch.nn.Parameter(
                    torch.ops.npu.npu_convert_weight_to_int4pack(
                        w2_qweight_tmp.reshape(
                            layer.w2_qweight.shape[0], layer.w2_qweight.shape[2], -1
                        )
                        .transpose(-1, -2)
                        .contiguous()
                        .reshape(-1, layer.w2_qweight.shape[2])
                        .to(torch.int32)
                    )
                    .reshape(
                        layer.w2_qweight.shape[0], layer.w2_qweight.shape[1] * 8, -1
                    )
                    .contiguous(),
                    requires_grad=False,
                )
            else:
                scale_expanded = layer.w2_scales.data.repeat_interleave(
                    group_size, dim=1
                )
                neg_mask = scale_expanded < 0
                if neg_mask.any():
                    neg_mask = neg_mask.transpose(-1, -2)
                    neg_mask = neg_mask.contiguous().reshape(w2_qweight_tmp.shape)
                    w2_qweight_tmp[neg_mask] = -w2_qweight_tmp[neg_mask]
                    if w2_qweight_tmp.max() > 7:
                        w2_qweight_tmp.clamp_(max=7)
                    layer.w2_scales.data.abs_()

                layer.w2_qweight = torch.nn.Parameter(
                    torch.ops.npu.npu_convert_weight_to_int4pack(
                        w2_qweight_tmp.reshape(
                            layer.w2_qweight.shape[0], layer.w2_qweight.shape[2], -1
                        )
                        .transpose(-1, -2)
                        .contiguous()
                        .reshape(-1, layer.w2_qweight.shape[2])
                        .to(torch.int32)
                    )
                    .reshape(
                        layer.w2_qweight.shape[0], layer.w2_qweight.shape[1] * 8, -1
                    )
                    .contiguous(),
                    requires_grad=False,
                )
        else:
            layer.w2_qweight = torch.nn.Parameter(
                w2_qweight_tmp.reshape(
                    layer.w2_qweight.shape[0], layer.w2_qweight.shape[2], -1
                )
                .transpose(-1, -2)
                .contiguous(),
                requires_grad=False,
            )

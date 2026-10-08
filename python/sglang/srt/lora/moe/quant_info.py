from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE


class MoeLoraBf16QuantInfo(msgspec.Struct, kw_only=True):
    """W13: [E, S*I, H], gate first when S=2. W2: [E, H, I]."""

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    num_local_experts: int
    intermediate_size: int
    hidden_size: int

    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraBf16QuantInfo:
        return cls(
            w13_weight=base_layer.w13_weight,
            w2_weight=base_layer.w2_weight,
            num_local_experts=int(base_layer.num_local_experts),
            intermediate_size=int(base_layer.w2_weight.shape[2]),
            hidden_size=int(base_layer.w2_weight.shape[1]),
        )


class MoeLoraFp8QuantInfo(msgspec.Struct, kw_only=True):
    """FP8 weights in standard layout, with FP32 inverse block scales."""

    w13_weight: torch.Tensor  # [E_local, S * I, H] float8_e4m3fn
    w13_scale: torch.Tensor  # [E_local, ceil(S*I/block_n), ceil(H/block_k)]
    w2_weight: torch.Tensor  # [E_local, H, I] float8_e4m3fn
    w2_scale: torch.Tensor  # [E_local, ceil(H/block_n), ceil(I/block_k)]
    block_shape: tuple[int, int]
    num_local_experts: int
    intermediate_size: int
    hidden_size: int

    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraFp8QuantInfo:
        quant_info = cls(
            w13_weight=base_layer.w13_weight,
            w13_scale=base_layer.w13_weight_scale_inv,
            w2_weight=base_layer.w2_weight,
            w2_scale=base_layer.w2_weight_scale_inv,
            block_shape=tuple(base_layer.quant_method.weight_block_size),
            num_local_experts=int(base_layer.num_local_experts),
            intermediate_size=int(base_layer.w2_weight.shape[2]),
            hidden_size=int(base_layer.w2_weight.shape[1]),
        )
        _admit_fp8_block_weights(quant_info)
        return quant_info


def _admit_fp8_block_weights(quant_info: MoeLoraFp8QuantInfo) -> None:
    """Validate 128x128 weight blocks and scales once when binding weights."""
    if tuple(quant_info.block_shape) != (128, 128):
        raise NotImplementedError(
            f"MoE LoRA FP8 supports [128, 128] weight blocks, got "
            f"{list(quant_info.block_shape)}"
        )
    for name, value in (
        ("hidden_size", quant_info.hidden_size),
        ("intermediate_size", quant_info.intermediate_size),
    ):
        if value % 128:
            raise NotImplementedError(
                f"MoE LoRA FP8 requires {name} % 128 == 0, got {value}"
            )
    for name, scale, rows, cols in (
        (
            "w13_scale",
            quant_info.w13_scale,
            quant_info.w13_weight.shape[1],
            quant_info.hidden_size,
        ),
        (
            "w2_scale",
            quant_info.w2_scale,
            quant_info.hidden_size,
            quant_info.intermediate_size,
        ),
    ):
        expected = (quant_info.num_local_experts, -(-rows // 128), -(-cols // 128))
        if scale.dtype != torch.float32 or tuple(scale.shape) != expected:
            raise ValueError(
                f"{name} must be fp32 {list(expected)}, got {scale.dtype} "
                f"{tuple(scale.shape)}"
            )


# Marlin's packed weights do not use the standard row-domain layout.
StandardLayoutQuantInfo = MoeLoraBf16QuantInfo | MoeLoraFp8QuantInfo

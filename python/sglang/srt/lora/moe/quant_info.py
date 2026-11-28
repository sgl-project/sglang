from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

        if isinstance(base_layer.quant_method, Fp8MoEMethod):
            w13_scale = base_layer.w13_weight_scale_inv
            w2_scale = base_layer.w2_weight_scale_inv
            block_shape = base_layer.quant_method.weight_block_size
        else:
            # compressed-tensors: no "_inv" suffix; the block shape is on the layer.
            w13_scale = base_layer.w13_weight_scale
            w2_scale = base_layer.w2_weight_scale
            block_shape = base_layer.weight_block_size
        quant_info = cls(
            w13_weight=base_layer.w13_weight,
            w13_scale=w13_scale,
            w2_weight=base_layer.w2_weight,
            w2_scale=w2_scale,
            block_shape=tuple(block_shape),
            num_local_experts=int(base_layer.num_local_experts),
            intermediate_size=int(base_layer.w2_weight.shape[2]),
            hidden_size=int(base_layer.w2_weight.shape[1]),
        )
        _admit_fp8_block_weights(quant_info)
        return quant_info


def admit_k_tiles(scheme: str, quant_info) -> None:
    """Both hidden and intermediate widths must be 128-aligned because each becomes K."""
    for name in ("hidden_size", "intermediate_size"):
        if getattr(quant_info, name) % 128:
            raise NotImplementedError(
                f"{scheme} MoE LoRA requires {name} % 128 == 0, "
                f"got {getattr(quant_info, name)}"
            )


def _admit_fp8_block_weights(quant_info: MoeLoraFp8QuantInfo) -> None:
    if tuple(quant_info.block_shape) != (128, 128):
        raise NotImplementedError(
            f"MoE LoRA FP8 supports [128, 128] weight blocks, got "
            f"{list(quant_info.block_shape)}"
        )
    admit_k_tiles("FP8", quant_info)
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


def nvfp4_activation_scale_mode(base_layer: FusedMoE) -> Literal["static", "per_token"]:
    """The quant config's resolved mode, never inferred from the checkpoint's scales."""
    if base_layer.quant_method.quant_config.use_per_token_activation:
        return "per_token"
    return "static"


StandardLayoutQuantInfo = MoeLoraBf16QuantInfo | MoeLoraFp8QuantInfo


class MoeLoraMarlinQuantInfo(msgspec.Struct, kw_only=True):
    """Prepared Marlin W4A16 or W8A16 tiles for BF16 activations."""

    w13_qweight: torch.Tensor
    w2_qweight: torch.Tensor
    w13_scales: torch.Tensor
    w2_scales: torch.Tensor
    num_local_experts: int
    intermediate_size: int
    hidden_size: int
    gate_up_slices: int
    # NVFP4 only: the per-expert global scales.
    w13_global_scale: torch.Tensor | None = None
    w2_global_scale: torch.Tensor | None = None

    @staticmethod
    def _prepared_fields(base_layer: FusedMoE) -> dict:
        return dict(
            w13_qweight=base_layer.w13_weight,
            w2_qweight=base_layer.w2_weight,
            w13_scales=base_layer.w13_weight_scale,
            w2_scales=base_layer.w2_weight_scale,
            num_local_experts=int(base_layer.num_local_experts),
            intermediate_size=int(base_layer.intermediate_size_per_partition),
            hidden_size=int(base_layer.hidden_size),
            gate_up_slices=2 if base_layer.moe_runner_config.is_gated else 1,
        )


class MoeLoraNvFp4MarlinQuantInfo(MoeLoraMarlinQuantInfo):
    """Group-16 E4M3 scales under per-expert global scales."""

    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraNvFp4MarlinQuantInfo:
        if base_layer.params_dtype != torch.bfloat16:
            raise NotImplementedError("NVFP4 MoE LoRA requires BF16 activations.")
        for name in ("w13_weight", "w2_weight"):
            if getattr(base_layer, name).dtype != torch.int32:
                raise ValueError(
                    f"NVFP4 MoE LoRA requires already-prepared Marlin {name}."
                )
        return cls(
            w13_global_scale=base_layer.w13_weight_scale_2,
            w2_global_scale=base_layer.w2_weight_scale_2,
            **cls._prepared_fields(base_layer),
        )


class MoeLoraMxfp4MarlinQuantInfo(MoeLoraMarlinQuantInfo):
    """Group-32 E8M0 scales without a global scale."""

    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraMxfp4MarlinQuantInfo:
        quant_info = cls(**cls._prepared_fields(base_layer))
        # The loader pads the hidden width to 256; the stages address real columns.
        padded_hidden = quant_info.w13_qweight.shape[1] * 16
        if padded_hidden != quant_info.hidden_size:
            raise NotImplementedError(
                "MXFP4 MoE LoRA requires hidden_size % 256 == 0, got "
                f"{quant_info.hidden_size} (padded to {padded_hidden})"
            )
        return quant_info

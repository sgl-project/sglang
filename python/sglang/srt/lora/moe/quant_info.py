from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Literal

import msgspec
import torch

from sglang.srt.environ import envs

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


class MoeLoraFp8ChannelQuantInfo(msgspec.Struct, kw_only=True):
    """FP32 weight scales per output row and activation scales per token."""

    w13_weight: torch.Tensor  # [E_local, S * I, H] float8_e4m3fn
    w13_scale: torch.Tensor  # [E_local, S * I, 1]
    w2_weight: torch.Tensor  # [E_local, H, I] float8_e4m3fn
    w2_scale: torch.Tensor  # [E_local, H, 1]
    num_local_experts: int
    intermediate_size: int
    hidden_size: int

    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraFp8ChannelQuantInfo:
        return cls(
            w13_weight=base_layer.w13_weight,
            w13_scale=base_layer.w13_weight_scale,
            w2_weight=base_layer.w2_weight,
            w2_scale=base_layer.w2_weight_scale,
            num_local_experts=int(base_layer.num_local_experts),
            intermediate_size=int(base_layer.w2_weight.shape[2]),
            hidden_size=int(base_layer.w2_weight.shape[1]),
        )


class MoeLoraFp8TensorQuantInfo(msgspec.Struct, kw_only=True):
    """One FP32 scale per expert; activations quantize per tensor at GEMM time."""

    w13_weight: torch.Tensor  # [E_local, S * I, H] float8_e4m3fn
    w13_scale: torch.Tensor  # [E_local]
    w2_weight: torch.Tensor  # [E_local, H, I] float8_e4m3fn
    w2_scale: torch.Tensor  # [E_local]
    w13_input_scale: torch.Tensor | None  # one value, or None for dynamic
    w2_input_scale: torch.Tensor | None
    num_local_experts: int
    intermediate_size: int
    hidden_size: int

    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraFp8TensorQuantInfo:
        return cls(
            w13_weight=base_layer.w13_weight,
            w13_scale=base_layer.w13_weight_scale,
            w2_weight=base_layer.w2_weight,
            w2_scale=base_layer.w2_weight_scale,
            w13_input_scale=base_layer.w13_input_scale,
            w2_input_scale=base_layer.w2_input_scale,
            num_local_experts=int(base_layer.num_local_experts),
            intermediate_size=int(base_layer.w2_weight.shape[2]),
            hidden_size=int(base_layer.w2_weight.shape[1]),
        )


class MoeLoraBlockScaledQuantInfo(msgspec.Struct, kw_only=True, dict=True):
    """E4M3 or packed E2M1 pairs; E8M0 scales per 32 (MXFP8, MXFP4) or E4M3 per 16
    (NVFP4). W13: [E, S*I, H / elements_per_byte]; W2: [E, H, I / elements_per_byte];
    scales: checkpoint-form bytes [E, rows, K / sf_vec_size]."""

    w13_weight: torch.Tensor
    w13_scale: torch.Tensor
    w2_weight: torch.Tensor
    w2_scale: torch.Tensor
    elements_per_byte: int
    num_local_experts: int
    intermediate_size: int
    hidden_size: int

    @functools.cached_property
    def packed_scales(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The W13 and W2 block scales in the tcgen05 atom order."""
        from sglang.kernels.ops.lora.moe.quant_blockscaled import pack_scale_atoms

        return pack_scale_atoms(self.w13_scale), pack_scale_atoms(self.w2_scale)

    @classmethod
    def _from_layer(
        cls,
        base_layer: FusedMoE,
        *,
        scheme: str,
        elements_per_byte: int,
        scale_attr: str = "weight_scale_inv",
        sf_vec_size: int = 32,
        **extra,
    ):
        quant_info = cls(
            w13_weight=base_layer.w13_weight,
            w13_scale=getattr(base_layer, f"w13_{scale_attr}").view(torch.uint8),
            w2_weight=base_layer.w2_weight,
            w2_scale=getattr(base_layer, f"w2_{scale_attr}").view(torch.uint8),
            elements_per_byte=elements_per_byte,
            num_local_experts=int(base_layer.num_local_experts),
            intermediate_size=int(base_layer.intermediate_size_per_partition),
            hidden_size=int(base_layer.hidden_size),
            **extra,
        )
        admit_k_tiles(scheme, quant_info)
        for name, scale, rows, k in (
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
            expected = (quant_info.num_local_experts, rows, k // sf_vec_size)
            if tuple(scale.shape) != expected:
                raise ValueError(
                    f"{name} must be scale bytes {list(expected)}, got {tuple(scale.shape)}"
                )
        return quant_info


class MoeLoraMxfp8QuantInfo(MoeLoraBlockScaledQuantInfo):
    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraMxfp8QuantInfo:
        return cls._from_layer(base_layer, scheme="MXFP8", elements_per_byte=1)


class MoeLoraMxfp4QuantInfo(MoeLoraBlockScaledQuantInfo):
    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraMxfp4QuantInfo:
        return cls._from_layer(base_layer, scheme="MXFP4", elements_per_byte=2)


class NvFp4ActivationRecipe(msgspec.Struct, frozen=True, kw_only=True):
    """Choose between rounded amax / 6 and amax / 4 scales in row-scaled units.
    The row denominator is 6 * e4m3_max and E4M3 block scales saturate at 448."""

    e4m3_max: int  # 448 or 256
    err_mode: str  # "MAE" or "MSE"
    err_use_fast_math: bool


def resolve_nvfp4_activation_recipe() -> NvFp4ActivationRecipe | None:
    """The stock per-token recipe; None is flashinfer's default, without 4over6."""
    if not envs.FLASHINFER_NVFP4_4OVER6.get():
        return None
    err_mode = envs.FLASHINFER_NVFP4_4OVER6_ERR_MODE.get().upper()
    if err_mode not in ("MAE", "MSE"):
        raise ValueError(
            f"FLASHINFER_NVFP4_4OVER6_ERR_MODE must be MAE or MSE, got {err_mode!r}"
        )
    return NvFp4ActivationRecipe(
        e4m3_max=256 if envs.FLASHINFER_NVFP4_4OVER6_E4M3_USE_256.get() else 448,
        err_mode=err_mode,
        err_use_fast_math=envs.FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH.get(),
    )


def nvfp4_activation_scale_mode(base_layer: FusedMoE) -> Literal["static", "per_token"]:
    """The quant config's resolved mode, never inferred from the checkpoint's scales."""
    if base_layer.quant_method.quant_config.use_per_token_activation:
        return "per_token"
    return "static"


class MoeLoraNvFp4QuantInfo(MoeLoraBlockScaledQuantInfo):
    """ModelOpt NVFP4 experts; one static activation scale per stage, or one per row."""

    w13_global_scale: torch.Tensor  # fp32 [E, 2] (gate, up) or [E]
    w2_global_scale: torch.Tensor  # fp32 [E]
    # Static mode only; None per-token, so no static scale reaches a per-token path.
    w13_input_scale: float | None
    w2_input_scale: float | None
    act_scale_mode: Literal["static", "per_token"] = "static"
    recipe: NvFp4ActivationRecipe | None = None

    @property
    def activation_denominator(self) -> float:
        """D of the per-token scale ``row_amax / D``."""
        return 6.0 * (448 if self.recipe is None else self.recipe.e4m3_max)

    @functools.cached_property
    def alpha(self) -> tuple[torch.Tensor, torch.Tensor]:
        """FP32 weight scales (times activation scale if static), kept without reciprocals."""
        w13_global = self.w13_global_scale.reshape(self.num_local_experts, -1)
        w2_global = self.w2_global_scale.reshape(-1, 1)
        if self.act_scale_mode == "static":
            w13_global = w13_global * self.w13_input_scale
            w2_global = w2_global * self.w2_input_scale
        return w13_global.contiguous(), w2_global.contiguous()

    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraNvFp4QuantInfo:
        mode = nvfp4_activation_scale_mode(base_layer)
        recipe = None
        if mode == "per_token":
            recipe = resolve_nvfp4_activation_recipe()
            input_scales = dict(w13_input_scale=None, w2_input_scale=None)
        else:
            input_scales = dict(
                w13_input_scale=float(base_layer.w13_input_scale.max()),
                w2_input_scale=float(base_layer.w2_input_scale.max()),
            )
        return cls._from_layer(
            base_layer,
            scheme="NVFP4",
            elements_per_byte=2,
            scale_attr="weight_scale",
            sf_vec_size=16,
            w13_global_scale=base_layer.w13_weight_scale_2.float(),
            w2_global_scale=base_layer.w2_weight_scale_2.float(),
            act_scale_mode=mode,
            recipe=recipe,
            **input_scales,
        )


StandardLayoutQuantInfo = (
    MoeLoraBf16QuantInfo
    | MoeLoraFp8QuantInfo
    | MoeLoraFp8ChannelQuantInfo
    | MoeLoraFp8TensorQuantInfo
    | MoeLoraBlockScaledQuantInfo
)


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


class MoeLoraFp8MarlinQuantInfo(MoeLoraMarlinQuantInfo):
    """E4M3 Marlin tiles with exponent-compensated BF16 scales per K group."""

    group_size: int  # 32 or 128 along K, the same for both stages

    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraFp8MarlinQuantInfo:
        # prepare_moe_fp8_layer_for_marlin: one group size for both stages, widths kept.
        fields = cls._prepared_fields(base_layer)
        group_size = fields["hidden_size"] // fields["w13_scales"].shape[1]
        return cls(group_size=group_size, **fields)

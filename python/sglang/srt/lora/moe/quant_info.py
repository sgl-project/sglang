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
    """FP8 weights in standard layout, with FP32 block dequantization scales."""

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
            # compressed-tensors registers the same multipliers without the
            # DeepSeek "_inv" suffix and keeps the block shape on the layer.
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


class MoeLoraFp8ChannelQuantInfo(msgspec.Struct, kw_only=True):
    """FP8 weights in standard layout with one FP32 scale per output row;
    activations quantize per token at GEMM time (compressed-tensors
    CHANNEL/TOKEN and sglang w8a8_fp8)."""

    w13_weight: torch.Tensor  # [E_local, S * I, H] float8_e4m3fn
    w13_scale: torch.Tensor  # [E_local, S * I, 1]
    w2_weight: torch.Tensor  # [E_local, H, I] float8_e4m3fn
    w2_scale: torch.Tensor  # [E_local, H, 1]
    num_local_experts: int
    intermediate_size: int
    hidden_size: int

    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraFp8ChannelQuantInfo:
        quant_info = cls(
            w13_weight=base_layer.w13_weight,
            w13_scale=base_layer.w13_weight_scale,
            w2_weight=base_layer.w2_weight,
            w2_scale=base_layer.w2_weight_scale,
            num_local_experts=int(base_layer.num_local_experts),
            intermediate_size=int(base_layer.w2_weight.shape[2]),
            hidden_size=int(base_layer.w2_weight.shape[1]),
        )
        for name, scale, rows in (
            ("w13_scale", quant_info.w13_scale, quant_info.w13_weight.shape[1]),
            ("w2_scale", quant_info.w2_scale, quant_info.hidden_size),
        ):
            expected = (quant_info.num_local_experts, rows, 1)
            if scale.dtype != torch.float32 or tuple(scale.shape) != expected:
                raise ValueError(
                    f"{name} must be fp32 {list(expected)}, got {scale.dtype} "
                    f"{tuple(scale.shape)}"
                )
        return quant_info


class MoeLoraFp8TensorQuantInfo(msgspec.Struct, kw_only=True):
    """FP8 weights in standard layout with one FP32 scale per expert;
    activations quantize per tensor at GEMM time, with the checkpoint's
    static scale when it has one (compressed-tensors TENSOR)."""

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
        quant_info = cls(
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
        experts = quant_info.num_local_experts
        for name, scale in (
            ("w13_scale", quant_info.w13_scale),
            ("w2_scale", quant_info.w2_scale),
        ):
            # The scheme folds the gate and up scales into one per expert
            # after loading; the runner binds after that.
            if scale.dtype != torch.float32 or tuple(scale.shape) != (experts,):
                raise ValueError(
                    f"{name} must be fp32 [{experts}], got {scale.dtype} "
                    f"{tuple(scale.shape)}"
                )
        for name, scale in (
            ("w13_input_scale", quant_info.w13_input_scale),
            ("w2_input_scale", quant_info.w2_input_scale),
        ):
            if scale is not None and scale.numel() != 1:
                raise ValueError(
                    f"{name} must hold one value, got {tuple(scale.shape)}"
                )
        return quant_info


class MoeLoraMxQuantInfo(msgspec.Struct, kw_only=True, dict=True):
    """MX block-scaled weights: E4M3 or packed E2M1 pairs with E8M0 scales per 32.

    W13: [E, S*I, H / elements_per_byte]; W2: [E, H, I / elements_per_byte].
    Scales keep the checkpoint form [E, rows, K / 32] as E8M0 bytes. A layer's
    providers share one instance and the tensors it derives on first use.
    """

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
        if base_layer.params_dtype != torch.bfloat16:
            raise NotImplementedError(f"{scheme} MoE LoRA requires BF16 activations.")
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
        # The tcgen05 kernels tile K by 128 and the scale atoms span 4 groups.
        for name, value in (
            ("hidden_size", quant_info.hidden_size),
            ("intermediate_size", quant_info.intermediate_size),
        ):
            if value % 128:
                raise NotImplementedError(
                    f"{scheme} MoE LoRA requires {name} % 128 == 0, got {value}"
                )
        for name, scale, rows in (
            ("w13_scale", quant_info.w13_scale, quant_info.w13_weight.shape[1]),
            ("w2_scale", quant_info.w2_scale, quant_info.hidden_size),
        ):
            k = (
                quant_info.hidden_size
                if name == "w13_scale"
                else quant_info.intermediate_size
            )
            expected = (quant_info.num_local_experts, rows, k // sf_vec_size)
            if tuple(scale.shape) != expected:
                raise ValueError(
                    f"{name} must be E8M0 bytes {list(expected)}, got {tuple(scale.shape)}"
                )
        return quant_info


class MoeLoraMxfp8QuantInfo(MoeLoraMxQuantInfo):
    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraMxfp8QuantInfo:
        if base_layer.w13_weight.dtype != torch.float8_e4m3fn:
            raise NotImplementedError(
                "MXFP8 MoE LoRA requires resident float8_e4m3fn weights"
            )
        return cls._from_layer(base_layer, scheme="MXFP8", elements_per_byte=1)


class MoeLoraMxfp4QuantInfo(MoeLoraMxQuantInfo):
    """Checkpoint-form MXFP4 experts (FP8 configs with ``routed_experts_quant_method``)."""

    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraMxfp4QuantInfo:
        if base_layer.w13_weight.dtype not in (torch.int8, torch.uint8):
            raise NotImplementedError(
                "MXFP4 MoE LoRA on CuTeDSL requires packed checkpoint weights; "
                "a Marlin-repacked layer is unsupported"
            )
        return cls._from_layer(base_layer, scheme="MXFP4", elements_per_byte=2)


class NvFp4ActivationRecipe(msgspec.Struct, frozen=True, kw_only=True):
    """flashinfer's NVFP4 "4over6" activation recipe, as the stock per-token
    path resolves it from the ``FLASHINFER_NVFP4_4OVER6*`` environment.

    ``e4m3_max`` is the normalization reference (the per-token scale is
    ``row_amax / (6 * e4m3_max)``), not a clamp: E4M3 block scales saturate
    at 448 under either value. ``err_mode`` and ``err_use_fast_math`` select
    each block's scale between the ``amax / 6`` and ``amax / 4`` candidates.
    """

    e4m3_max: int  # 448 or 256
    err_mode: str  # "MAE" or "MSE"
    err_use_fast_math: bool

    def __post_init__(self) -> None:
        if self.e4m3_max not in (256, 448):
            raise ValueError(
                f"NVFP4 4over6 e4m3_max must be 256 or 448, got {self.e4m3_max}"
            )
        if self.err_mode not in ("MAE", "MSE"):
            raise ValueError(
                f"NVFP4 4over6 err_mode must be MAE or MSE, got {self.err_mode!r}"
            )

    def describe(self) -> str:
        return (
            f"4over6 with e4m3_max={self.e4m3_max}, err_mode={self.err_mode}, "
            f"err_use_fast_math={self.err_use_fast_math}"
        )


def resolve_nvfp4_activation_recipe() -> NvFp4ActivationRecipe | None:
    """The activation recipe the stock per-token path reads from the
    flashinfer environment at call time. None is flashinfer's default recipe:
    no candidate selection, ``e4m3_max`` 448, per-token scale
    ``row_amax / 2688``."""
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
    """The mode the base layer's quant config resolved (ModelOpt serialized
    checkpoints under their per-token flag, ``nvfp4_online`` always); never
    inferred from the checkpoint's scale values or their absence."""
    if base_layer.quant_method.quant_config.use_per_token_activation:
        return "per_token"
    return "static"


class MoeLoraNvFp4QuantInfo(MoeLoraMxQuantInfo):
    """Canonical ModelOpt NVFP4 experts: E2M1 pairs, E4M3 scales per 16, and
    per-expert global scales. Activations carry one static global scale per
    stage (``static``, the checkpoint's) or one dynamic scale per quantized
    row (``per_token``), whichever mode the base layer resolved."""

    w13_global_scale: torch.Tensor  # fp32 [E, 2] (gate, up) or [E]
    w2_global_scale: torch.Tensor  # fp32 [E]
    # Static mode: the checkpoint's activation scales. Per-token mode: None,
    # so no provider can fold a static scale into a per-token path.
    w13_input_scale: float | None
    w2_input_scale: float | None
    act_scale_mode: Literal["static", "per_token"] = "static"
    # Per-token mode only: the resolved flashinfer recipe, None for the
    # default one (no candidate selection, e4m3_max 448).
    recipe: NvFp4ActivationRecipe | None = None

    def __post_init__(self) -> None:
        scales = (self.w13_input_scale, self.w2_input_scale)
        present = sum(scale is not None for scale in scales)
        if present != (2 if self.act_scale_mode == "static" else 0):
            raise ValueError(
                "static NVFP4 activations take both input scales; per-token "
                f"activations take none ({self.act_scale_mode}: {present} given)"
            )
        if self.act_scale_mode == "static" and self.recipe is not None:
            raise ValueError(
                "an activation recipe applies to per-token activations only"
            )

    @property
    def activation_denominator(self) -> float:
        """D of the per-token scale ``row_amax / D``: ``6 * e4m3_max``."""
        return 6.0 * (448 if self.recipe is None else self.recipe.e4m3_max)

    @functools.cached_property
    def alpha(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-expert FP32 factors of the gate/up and the down accumulator:
        the weight global scales (gate and up rows of W13 may carry separate
        ones), times the activation scale when it is static. Per-token scales
        are applied per row after it. Nothing holds a reciprocal of these."""
        w13_global = self.w13_global_scale.reshape(self.num_local_experts, -1)
        w2_global = self.w2_global_scale.reshape(-1, 1)
        if self.act_scale_mode == "static":
            w13_global = w13_global * self.w13_input_scale
            w2_global = w2_global * self.w2_input_scale
        return w13_global.contiguous(), w2_global.contiguous()

    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraNvFp4QuantInfo:
        if base_layer.w13_weight.dtype != torch.uint8:
            raise NotImplementedError(
                "NVFP4 MoE LoRA on CuTeDSL requires packed checkpoint weights; "
                "a Marlin-repacked layer is unsupported"
            )
        mode = nvfp4_activation_scale_mode(base_layer)
        recipe = None
        if mode == "per_token":
            # The recipe the stock per-token path resolves from the same
            # environment; the quantizers implement every value it can take.
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


# Marlin's packed weights do not use the standard row-domain layout.
StandardLayoutQuantInfo = (
    MoeLoraBf16QuantInfo
    | MoeLoraFp8QuantInfo
    | MoeLoraFp8ChannelQuantInfo
    | MoeLoraFp8TensorQuantInfo
    | MoeLoraMxQuantInfo
)


class MoeLoraMarlinQuantInfo(msgspec.Struct, kw_only=True):
    """Prepared Marlin W4A16 tiles for BF16 activations."""

    w13_qweight: torch.Tensor
    w2_qweight: torch.Tensor
    w13_scales: torch.Tensor
    w2_scales: torch.Tensor
    num_local_experts: int
    intermediate_size: int
    hidden_size: int
    gate_up_slices: int

    @staticmethod
    def _prepared_fields(base_layer: FusedMoE, scheme: str) -> dict:
        if base_layer.params_dtype != torch.bfloat16:
            raise NotImplementedError(f"{scheme} MoE LoRA requires BF16 activations.")
        for name in ("w13_weight", "w2_weight"):
            if getattr(base_layer, name).dtype != torch.int32:
                raise ValueError(
                    f"{scheme} MoE LoRA requires already-prepared Marlin {name}."
                )
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

    w13_global_scale: torch.Tensor
    w2_global_scale: torch.Tensor

    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraNvFp4MarlinQuantInfo:
        return cls(
            w13_global_scale=base_layer.w13_weight_scale_2,
            w2_global_scale=base_layer.w2_weight_scale_2,
            **cls._prepared_fields(base_layer, "NVFP4"),
        )


class MoeLoraMxfp4MarlinQuantInfo(MoeLoraMarlinQuantInfo):
    """Group-32 E8M0 scales without a global scale."""

    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraMxfp4MarlinQuantInfo:
        quant_info = cls(**cls._prepared_fields(base_layer, "MXFP4"))
        # The loader pads the hidden width to 256; the stages address real columns.
        padded_hidden = quant_info.w13_qweight.shape[1] * 16
        if padded_hidden != quant_info.hidden_size:
            raise NotImplementedError(
                "MXFP4 MoE LoRA requires hidden_size % 256 == 0, got "
                f"{quant_info.hidden_size} (padded to {padded_hidden})"
            )
        return quant_info


class MoeLoraFp8MarlinQuantInfo(MoeLoraMarlinQuantInfo):
    """E4M3 weights in Marlin tiles with scales in the activation dtype in 32-
    or 128-wide K groups (channel-wise and per-tensor checkpoints repeat their
    scale over the groups: the kernel's single-group path scales after the K
    accumulation and underflows small products); the FP8 exponent bias is
    folded into the scales."""

    group_size: int  # 32 or 128 along K, the same for both stages

    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraFp8MarlinQuantInfo:
        fields = cls._prepared_fields(base_layer, "FP8")
        hidden, inter = fields["hidden_size"], fields["intermediate_size"]
        # The FP8 repack pads no widths: the tiles keep the layer's geometry.
        if (
            fields["w13_qweight"].shape[1] * 16 != hidden
            or fields["w2_qweight"].shape[1] * 16 != inter
        ):
            raise ValueError(
                "FP8 Marlin tiles must keep the layer's hidden and intermediate "
                f"widths ({hidden}, {inter}), got K tiles "
                f"{fields['w13_qweight'].shape[1]} and {fields['w2_qweight'].shape[1]}"
            )
        groups13, groups2 = fields["w13_scales"].shape[1], fields["w2_scales"].shape[1]
        if groups13 < 2 or groups2 < 2:
            raise NotImplementedError(
                "FP8 Marlin scales must hold at least two K groups per stage: the "
                "kernel reads one group as channel-wise and scales after the K "
                f"accumulation, got {groups13} and {groups2} groups"
            )
        if hidden % groups13 == 0 and hidden // groups13 * groups2 == inter:
            group_size = hidden // groups13
        else:
            raise ValueError(
                f"FP8 Marlin scales must share one K group size, got {groups13} "
                f"groups over {hidden} and {groups2} over {inter}"
            )
        if group_size not in (32, 128):
            raise NotImplementedError(
                f"the Marlin FP8 kernels take 32- or 128-wide K groups, got {group_size}"
            )
        for name in ("w13_scales", "w2_scales"):
            if fields[name].dtype != base_layer.params_dtype:
                raise ValueError(
                    f"FP8 Marlin {name} must be in the activation dtype "
                    f"{base_layer.params_dtype}, got {fields[name].dtype}"
                )
        return cls(group_size=group_size, **fields)

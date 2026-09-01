"""MXFP4 MoE scheme: packed E2M1 weights + per-32 UE8M0 scales, FP8 activations.

Loads `mxfp4-pack-quantized` compressed-tensors checkpoints (e.g. GLM-5.2
DataFree-WMXFP4AFP8-GS32). Two execution backends are supported:

* Marlin (`--moe-runner-backend marlin`): the packed E2M1 bytes are repacked
  into the Marlin MoE layout (`prepare_moe_mxfp4_layer_for_marlin`) and the
  E8M0 scales are normalized to `float8_e8m0fnu`, which the
  `moe_wna16_marlin` kernel consumes natively (bf16 activations).
* Mega-MoE (DeepGEMM SM90 FP4, `--moe-a2a-backend megamoe`): weights are
  transformed via `transform_weights_for_mega_moe_sm90_fp4` and the scales
  ride in an fp32 container (the kernel packs UE8M0 itself).

Checkpoint layout (per expert, before stacking):
    gate/up_proj.weight_packed  uint8  [I, H//2]    two E2M1 nibbles per byte
    gate/up_proj.weight_scale   uint8  [I, H//32]   E8M0 biased exponent
    down_proj.weight_packed     uint8  [H, I//2]
    down_proj.weight_scale      uint8  [H, I//32]

The nibble order (low nibble = even K index) and the E2M1 encoding match
DeepGEMM's `per_token_cast_to_fp4`, so the packed bytes are handed over
untouched -- `MXFP4PackedCompressor` subclasses `NVFP4PackedCompressor` and
reuses its `pack_fp4_to_uint8`. Only the scales need converting: the mega-MoE
kernel consumes UE8M0 values carried in an fp32 container, not the raw E8M0
bytes.

The loaded w13 row order is `[gate; up]` (MergedColumnParallelLinear), which
is exactly the layout the Marlin `silu_and_mul` expects -- no deinterleave is
needed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    MXFP4_PACK_QUANTIZED_FORMAT,
)
from sglang.srt.layers.quantization.mxfp8_block_convert import _ue8m0_to_fp32
from sglang.srt.utils import is_sm90_supported, is_sm100_supported, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsW4A8Mxfp4MoE"]

# Two E2M1 values per byte, and one E8M0 scale per 32 values along K.
MXFP4_PACK_FACTOR = 2
MXFP4_GROUP_SIZE = 32


class CompressedTensorsW4A8Mxfp4MoE(CompressedTensorsMoEScheme):
    """MXFP4 weights (per-32 E8M0 scales) + dynamic FP8 activations, via mega-MoE."""

    def __init__(
        self,
        quant_config: CompressedTensorsConfig,
        weight_quant,
        input_quant,
        quant_format: Optional[str] = None,
    ):
        self.quant_config = quant_config
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.packed_factor = MXFP4_PACK_FACTOR
        self.group_size = weight_quant.group_size

        effective_format = quant_format or self.quant_config.quant_format
        if effective_format != MXFP4_PACK_QUANTIZED_FORMAT:
            raise ValueError(
                "MXFP4 MoE requires mxfp4-pack-quantized format, got "
                f"{effective_format}"
            )
        if not (
            weight_quant.num_bits == 4
            and weight_quant.symmetric
            and self.group_size == MXFP4_GROUP_SIZE
        ):
            raise ValueError(
                "MXFP4 MoE requires symmetric 4-bit group-32 weights, got "
                f"num_bits={weight_quant.num_bits}, symmetric={weight_quant.symmetric}, "
                f"group_size={self.group_size}"
            )

        # `get_min_capability` is never consulted on the MoE path. Marlin has
        # both SM90 and SM100 support, while the dedicated DeepGEMM transform
        # restored here is explicitly SM90-only.
        if get_moe_runner_backend().is_marlin():
            supported = is_sm90_supported() or is_sm100_supported()
            requirement = "SM90 (Hopper) or SM100 (Blackwell)"
        else:
            supported = is_sm90_supported()
            requirement = "SM90 (Hopper)"
        if not supported:
            raise ValueError(
                f"MXFP4 MoE with {get_moe_runner_backend().value} requires "
                f"{requirement}; the selected FP4 kernel is unavailable on "
                "this device."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        # Un-padded partition hidden size. Marlin repacks/pads the intermediate
        # dim in `process_weights_after_loading`, so the buffer created here must
        # stay in checkpoint layout for the loader's narrow-copy fast path.
        self.hidden_size = hidden_size

        # `prepare_moe_mxfp4_layer_for_marlin` derives the activation dtype from
        # `layer.orig_dtype`; the GLM-5.2 family is bf16.
        layer.params_dtype = params_dtype
        layer.orig_dtype = params_dtype

        # `num_experts` is already EP-local and `intermediate_size_per_partition`
        # already TP-sharded; both shard sizes must stay group/pack aligned or the
        # loader would narrow at a misaligned offset without complaining.
        if hidden_size % (self.group_size * self.packed_factor) != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by "
                f"{self.group_size * self.packed_factor}"
            )
        if (
            intermediate_size_per_partition % (self.group_size * self.packed_factor)
            != 0
        ):
            raise ValueError(
                f"intermediate_size_per_partition {intermediate_size_per_partition} must "
                f"be divisible by {self.group_size * self.packed_factor}"
            )

        # Packed E2M1 weights, checkpoint (non-transposed) layout.
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.packed_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.packed_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # E8M0 scales stay uint8 here: the checkpoint stores raw biased
        # exponents, and copying them into a float parameter would convert the
        # byte values numerically instead of preserving them.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )

        w13_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)

        # No `w13_weight_shape` / `w2_weight_shape` placeholders here: unlike the
        # int-based pack-quantized formats, MXFP4PackedCompressor only emits
        # weight_packed and weight_scale, so such params would stay uninitialised
        # and still be picked up as expert weights by EPLB rebalancing.

        layer.is_mxfp4_converted = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Rename to the mega-MoE parameter names and decode E8M0 scales.

        `build_mega_moe_experts_weights` reads `w13_weight` / `w2_weight` /
        `w13_weight_scale_inv` / `w2_weight_scale_inv` by hard-coded name, and
        writes back through those Parameters, so they must be registered under
        exactly those names rather than aliased with `setattr`.
        """
        if layer.is_mxfp4_converted:
            return

        # uint8 -> int8 is a pure bit reinterpretation (DeepGEMM's kPackedFP4).
        w13 = layer.w13_weight_packed.data.view(torch.int8)
        w2 = layer.w2_weight_packed.data.view(torch.int8)

        # Shift-based decode keeps the fp32 mantissa exactly zero, which
        # DeepGEMM's `pack_ue8m0_to_int` asserts on. Computing 2**(v-127) in
        # floating point would drift into subnormals for small exponents.
        w13_sf = _ue8m0_to_fp32(layer.w13_weight_scale.data)
        w2_sf = _ue8m0_to_fp32(layer.w2_weight_scale.data)

        for stale in (
            "w13_weight_packed",
            "w2_weight_packed",
            "w13_weight_scale",
            "w2_weight_scale",
        ):
            delattr(layer, stale)

        layer.register_parameter(
            "w13_weight", torch.nn.Parameter(w13, requires_grad=False)
        )
        layer.register_parameter(
            "w2_weight", torch.nn.Parameter(w2, requires_grad=False)
        )
        layer.register_parameter(
            "w13_weight_scale_inv", torch.nn.Parameter(w13_sf, requires_grad=False)
        )
        layer.register_parameter(
            "w2_weight_scale_inv", torch.nn.Parameter(w2_sf, requires_grad=False)
        )

        # FP8 activations are quantized inside the kernel; no loaded scales.
        layer.a13_scale = None
        layer.a2_scale = None

        layer.is_mxfp4_converted = True

        if get_moe_runner_backend().is_marlin():
            self._build_marlin_weights(layer)
        else:
            self._build_mega_moe_weights(layer)

    def _build_marlin_weights(self, layer: torch.nn.Module) -> None:
        """Repack the checkpoint-layout weights into the Marlin MoE layout.

        `prepare_moe_mxfp4_layer_for_marlin` reads `w13_weight` / `w2_weight`
        (packed int8) plus `w13_weight_scale_inv` / `w2_weight_scale_inv`
        (fp32 per-32 scales) and produces the repacked qweight + permuted
        `float8_e8m0fnu` scales. The loaded row order is `[gate; up]`, which
        is exactly what Marlin's `silu_and_mul` expects, so no deinterleave is
        applied (that helper is only for interleaved GPT-OSS checkpoints).

        When Marlin is selected the mega-MoE weights are never built, so
        `should_use_mega_moe` stays False and the standard FusedMoE path
        (routed through `apply_weights`) is used at inference time.
        """
        from sglang.srt.layers.quantization.marlin_utils import (
            check_moe_marlin_supports_layer,
        )
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            prepare_moe_mxfp4_layer_for_marlin,
        )

        if not is_sm90_supported() and not is_sm100_supported():
            raise RuntimeError("MXFP4 Marlin requires SM90 or SM100.")

        if not check_moe_marlin_supports_layer(layer, 32, allow_tile_padding=True):
            raise RuntimeError("Current MXFP4 MoE layer is not supported by Marlin.")

        prepare_moe_mxfp4_layer_for_marlin(layer)
        layer._mxfp4_backend = "marlin"

    def _build_mega_moe_weights(self, layer: torch.nn.Module) -> None:
        """Hand the weights to DeepGEMM's mega-MoE transform.

        Must run after every in-place weight rewrite and instead of (not before)
        any scale re-layout -- the transform consumes checkpoint-layout per-32
        fp32 scales and does the UE8M0 packing itself.
        """
        from sglang.srt.layers.moe.mega_moe_sm90 import (
            build_sm90_fp4_mega_moe_experts_weights,
        )
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        backend = get_moe_a2a_backend()
        if not backend.is_megamoe():
            raise ValueError(
                "MXFP4 MoE checkpoints are only supported through the mega-MoE "
                f"kernel, but --moe-a2a-backend is '{backend.value}'. Pass "
                "--moe-a2a-backend megamoe, or set "
                "SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1 to have it auto-configured."
            )
        if not is_sm90_supported():
            raise RuntimeError(
                "MXFP4 Mega-MoE currently requires the SM90 FP4 DeepGEMM path. "
                "Use --moe-runner-backend marlin for the MXFP4 Marlin control "
                "or extend build_mega_moe_experts_weights for this architecture."
            )

        build_sm90_fp4_mega_moe_experts_weights(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        if get_moe_runner_backend().is_marlin():
            from sglang.srt.layers.moe.moe_runner import MoeRunner
            from sglang.srt.layers.moe.utils import MoeRunnerBackend

            self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)
        # The mega-MoE path bypasses the runner entirely (DeepseekV2MoE.forward
        # routes straight into forward_mega_moe), so no runner is built there.

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        return self.apply_weights(layer, dispatch_output)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        if get_moe_runner_backend().is_marlin():
            from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
            from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
            from sglang.srt.layers.moe.topk import TopKOutputChecker

            assert TopKOutputChecker.format_is_standard(dispatch_output.topk_output)
            hidden_states = dispatch_output.hidden_states
            if hidden_states.shape[-1] != self.hidden_size:
                hidden_states = torch.nn.functional.pad(
                    hidden_states,
                    (0, self.hidden_size - hidden_states.shape[-1]),
                    mode="constant",
                    value=0.0,
                )

            quant_info = MarlinMoeQuantInfo(
                w13_qweight=layer.w13_weight,
                w2_qweight=layer.w2_weight,
                w13_scales=layer.w13_weight_scale,
                w2_scales=layer.w2_weight_scale,
                w13_g_idx_sort_indices=None,
                w2_g_idx_sort_indices=None,
                weight_bits=4,
                is_k_full=True,
                w13_bias=getattr(layer, "w13_weight_bias", None),
                w2_bias=getattr(layer, "w2_weight_bias", None),
            )
            combine_input = self.runner.run(
                dispatch_output._replace(hidden_states=hidden_states),
                quant_info,
            )
            # ``fused_marlin_moe`` owns routed scaling for every Marlin weight
            # format, including the optimized MXFP4 top-k reduction path.
            return StandardCombineInput(hidden_states=combine_input.hidden_states)

        # Reaching here means the mega-MoE path declined this batch (e.g. the
        # token count exceeded its cap). There is no SM90 MXFP4 grouped GEMM to
        # fall back on, and the weights are already in mega layout, so failing
        # loudly beats returning wrong numbers.
        raise NotImplementedError(
            "MXFP4 MoE weights are prepared exclusively for the DeepGEMM "
            "mega-MoE kernel; there is no fallback grouped-GEMM path. This "
            "batch bypassed forward_mega_moe -- check "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK against the "
            "current batch size."
        )

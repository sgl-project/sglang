# SPDX-License-Identifier: Apache-2.0
"""DSV4 MXFP4 experts on the SM90 mixed-input CUTLASS grouped GEMM.

Sibling of :class:`Mxfp4MarlinMoEMethod`: the DSV4 fp4 experts arrive through
``Fp8Config.get_quant_method``, so selecting ``--moe-runner-backend cutlass_mxfp4``
needs its own wrapper around the fp8 method rather than the ``Mxfp4MoEMethod``
path that GPT-OSS-style checkpoints take.

Unlike marlin this keeps the checkpoint's weight layout untouched -- the
collective decodes E2M1 nibbles in the mainloop -- so the buffers are unpadded
and only the E8M0 scales are repacked at load time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.nn import Module, Parameter

from sglang.kernels.ops.moe.mxfp4_a16_moe_mm import (
    MXFP4_A16_PACKED_SCALES_NUM,
    MXFP4_A16_SCALE_GROUP_SIZE,
    MXFP4_A16_TILE_K,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils import log_info_on_rank0, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)

# The collective leaves each decoded E2M1 code at value * 2^-126, so the repacked scale byte
# carries the checkpoint's biased exponent plus 126.
_CUTLASS_MXFP4_SCALE_BIAS = 126


def _pack_mxfp4_scales_for_cutlass(scales: torch.Tensor) -> torch.Tensor:
    """``[E, N, K/32]`` checkpoint bytes -> ``[E, K/128, N*4]`` with the +126 bias folded in.

    The SM90 collective reads one ``Array<uint8_t, 4>`` per (K tile, N) pair, and its E2M1
    decode skips renormalizing, so a byte ``b`` here scales by ``2^(b - 253)``.
    """
    num_experts, n, num_groups = scales.shape
    packed = MXFP4_A16_PACKED_SCALES_NUM
    if num_groups * MXFP4_A16_SCALE_GROUP_SIZE % MXFP4_A16_TILE_K:
        raise NotImplementedError(
            f"moe_runner_backend=cutlass_mxfp4 tiles K by {MXFP4_A16_TILE_K}, so the "
            f"group count must be a multiple of {packed}, got {num_groups}."
        )
    # The collective widens the scale byte into a bf16 exponent field, where 255 is Inf/NaN,
    # so 254 == 2^1 is the largest scale this decode can carry. Real MXFP4 checkpoints store
    # max_abs/6 per group and never reach it.
    if int(scales.max()) > 254 - _CUTLASS_MXFP4_SCALE_BIAS:
        raise NotImplementedError(
            "moe_runner_backend=cutlass_mxfp4 cannot represent a group scale above 2^1; "
            "use --moe-runner-backend marlin for this checkpoint."
        )
    biased = scales.to(torch.int16).add_(_CUTLASS_MXFP4_SCALE_BIAS).to(torch.uint8)
    return (
        biased.reshape(num_experts, n, num_groups // packed, packed)
        .permute(0, 2, 1, 3)
        .reshape(num_experts, num_groups // packed, n * packed)
        .contiguous()
    )


class Mxfp4CutlassMoEMethod:
    """MXFP4 (E8M0 scales) MoE quantization method on the CUTLASS SM90 grouped GEMM."""

    def __init__(self, fp8_method, prefix: str):
        if not get_platform().is_sm90:
            raise NotImplementedError(
                "moe_runner_backend=cutlass_mxfp4 runs on the SM90 (Hopper) mixed-input "
                "collective; no other architecture is instantiated. Use "
                "--moe-runner-backend marlin on this GPU."
            )
        self._fp8 = fp8_method
        self.prefix = prefix

    def create_moe_runner(self, layer, moe_runner_config):
        from sglang.srt.layers.moe.moe_runner import MoeRunner

        self.runner = MoeRunner(MoeRunnerBackend.CUTLASS_MXFP4, moe_runner_config)

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        layer._dsv4_mxfp4_backend = None  # set in process_weights_after_loading
        if hidden_size % MXFP4_A16_TILE_K or (
            intermediate_size_per_partition % MXFP4_A16_TILE_K
        ):
            raise NotImplementedError(
                "moe_runner_backend=cutlass_mxfp4 needs hidden_size and the per-rank "
                f"intermediate size to be multiples of {MXFP4_A16_TILE_K}, got "
                f"{hidden_size} and {intermediate_size_per_partition}."
            )

        # No padding: the loader's naive copy would otherwise push the [gate; up] split
        # off the buffer's midpoint, and the collective needs no alignment beyond the
        # K tile that the check above already guarantees.
        w13_weight = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        w2_weight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Store loader scales in E8M0; uint8 127 encodes 1.0.
        def _e8m0_ones(*shape: int) -> torch.Tensor:
            return torch.full(shape, 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)

        w13_weight_scale = Parameter(
            _e8m0_ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // MXFP4_A16_SCALE_GROUP_SIZE,
            ),
            requires_grad=False,
        )
        w2_weight_scale = Parameter(
            _e8m0_ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // MXFP4_A16_SCALE_GROUP_SIZE,
            ),
            requires_grad=False,
        )
        w13_weight_scale.format_ue8m0 = False
        w2_weight_scale.format_ue8m0 = False
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, scale_attrs)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_attrs)

    def process_weights_after_loading(self, layer: Module) -> None:
        # Let the FP8 base method handle ROCm normalization, etc.
        self._fp8.process_weights_after_loading(layer)

        log_info_on_rank0(
            logger,
            "Preparing MXFP4 experts for the CUTLASS SM90 grouped GEMM "
            f"(layer: {self.prefix})...",
        )
        layer.w13_weight.data = layer.w13_weight.data.view(torch.uint8)
        layer.w2_weight.data = layer.w2_weight.data.view(torch.uint8)

        # Repack one operand at a time and drop the checkpoint copy before the next,
        # so the transposed scales never coexist with both loaded tensors.
        layer.w13_weight_scale = Parameter(
            _pack_mxfp4_scales_for_cutlass(
                layer.w13_weight_scale_inv.data.view(torch.uint8)
            ),
            requires_grad=False,
        )
        del layer.w13_weight_scale_inv
        layer.w2_weight_scale = Parameter(
            _pack_mxfp4_scales_for_cutlass(
                layer.w2_weight_scale_inv.data.view(torch.uint8)
            ),
            requires_grad=False,
        )
        del layer.w2_weight_scale_inv

        num_experts = layer.w13_weight.shape[0]
        int32 = dict(dtype=torch.int32, device=layer.w13_weight.device)
        # Per-expert grouped GEMM metadata, refilled from topk_ids on every forward.
        layer.cutlass_mxfp4_expert_offsets = torch.empty(num_experts + 1, **int32)
        layer.cutlass_mxfp4_problem_sizes1 = torch.empty((num_experts, 3), **int32)
        layer.cutlass_mxfp4_problem_sizes2 = torch.empty((num_experts, 3), **int32)

        layer._dsv4_mxfp4_backend = "cutlass_mxfp4"
        torch.cuda.empty_cache()

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.moe_runner.cutlass_mxfp4 import (
            CutlassMxfp4MoeQuantInfo,
        )
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        if not TopKOutputChecker.format_is_standard(dispatch_output.topk_output):
            raise ValueError(
                f"Unsupported topk output format: {dispatch_output.topk_output.format}"
            )

        quant_info = CutlassMxfp4MoeQuantInfo(
            w13_weight=layer.w13_weight,
            w13_weight_scale=layer.w13_weight_scale,
            w2_weight=layer.w2_weight,
            w2_weight_scale=layer.w2_weight_scale,
            expert_offsets=layer.cutlass_mxfp4_expert_offsets,
            problem_sizes1=layer.cutlass_mxfp4_problem_sizes1,
            problem_sizes2=layer.cutlass_mxfp4_problem_sizes2,
        )
        return self.runner.run(dispatch_output, quant_info)

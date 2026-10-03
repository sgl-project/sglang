# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsW4A16Mxfp4MoE"]

MXFP4_GROUP_SIZE = 32

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


class CompressedTensorsW4A16Mxfp4MoE(CompressedTensorsMoEScheme):
    """Weight-only MXFP4 MoE via the FP4 Marlin runner: fp4 experts, bf16
    activations.

    Serves a compressed-tensors ``mxfp4-pack-quantized`` checkpoint whose experts
    are stored **split per expert** (``experts.{e}.{gate,up,down}_proj.*``), which
    is a different layout from the fused GPT-OSS mxfp4 that ``Mxfp4MoEMethod``
    and ``Mxfp4MarlinMoEMethod`` handle -- those register ``w13/w2_weight``, so a
    split-expert checkpoint dies on ``KeyError: ...experts.w2_weight_packed``.

    Serves mxfp4a16 (no ``input_activations``) and mxfp4 w4a4, whose activation
    quantization is dropped: the Marlin MoE kernel is weight-only. A w4a4
    checkpoint still ships the input global scales, and the split-expert loader
    raises KeyError on a tensor with no registered destination, so they must be
    registered first and only then dropped.

    Unlike NVFP4 there is no ``weight_global_scale``: MXFP4's E8M0 scales are
    absolute powers of two, so no outer scale is stored and none is passed to the
    runner. That also means the compressed-tensors reciprocal convention (CT
    stores 1/scale where Marlin wants the scale) does not arise here.

    w13 stays in the checkpoint's native [w1; w3] order, which is what
    silu_and_mul in fused_marlin_moe.py expects. _load_w13 reads
    load_up_proj_weight_first off layer.quant_method rather than off the scheme,
    so that order comes from the flag's default and must not be overridden here.
    """

    def __init__(self, has_input_global_scale: bool = False):
        # True for a w4a4 checkpoint being served weight-only.
        self.has_input_global_scale = has_input_global_scale
        self.group_size = MXFP4_GROUP_SIZE

    @classmethod
    def get_min_capability(cls) -> int:
        # FP4 Marlin is weight-only and needs no FP4 tensor cores.
        return 80

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

        layer.params_dtype = params_dtype
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size

        # Two fp4 items are packed per byte along the input dimension.
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
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
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        group_attrs = dict(extra_weight_attrs)
        group_attrs["quant_method"] = FusedMoeWeightScaleSupported.GROUP.value

        # E8M0 exponents arrive as raw uint8 bytes with no float8 dtype attached;
        # _normalize_scale_tensor reinterprets them rather than converting.
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, group_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, group_attrs)

        if self.has_input_global_scale:
            tensor_attrs = dict(extra_weight_attrs)
            tensor_attrs["quant_method"] = FusedMoeWeightScaleSupported.TENSOR.value

            w13_input_global_scale = torch.nn.Parameter(
                torch.empty(num_experts, 2, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_global_scale", w13_input_global_scale)
            set_weight_attrs(w13_input_global_scale, tensor_attrs)

            w2_input_global_scale = torch.nn.Parameter(
                torch.empty(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_global_scale", w2_input_global_scale)
            set_weight_attrs(w2_input_global_scale, tensor_attrs)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            prepare_moe_mxfp4_layer_for_marlin,
        )

        # The activation scales are meaningless for a weight-only kernel.
        if self.has_input_global_scale:
            delattr(layer, "w13_input_global_scale")
            delattr(layer, "w2_input_global_scale")

        # prepare_moe_mxfp4_layer_for_marlin reads `w13/w2_weight`;
        # compressed-tensors uses the `_packed` names.
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")
        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        # The helper derives its working dtype from `orig_dtype`, falling back to
        # the bias dtype and then bfloat16. A compressed-tensors checkpoint has
        # no bias, so state the activation dtype explicitly rather than relying
        # on that fallback.
        layer.orig_dtype = layer.params_dtype

        # check_moe_marlin_supports_layer is deliberately not called: it tests
        # group_size against the GPTQ/AWQ set and reports on the int4 zero-point
        # parameters this layout does not have. The helper below asserts the
        # group size itself; validate shapes explicitly.
        if layer.hidden_size % 128 != 0:
            raise ValueError(
                f"MXFP4 Marlin MoE requires hidden_size % 128 == 0, got "
                f"{layer.hidden_size}."
            )
        if layer.intermediate_size_per_partition % 64 != 0:
            raise ValueError(
                "MXFP4 Marlin MoE requires intermediate_size_per_partition % 64 "
                f"== 0, got {layer.intermediate_size_per_partition}."
            )

        prepare_moe_mxfp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
            build_marlin_moe_quant_info,
        )

        # MXFP4 passes no global scales; build_marlin_moe_quant_info leaves them
        # None, which is what the fused mxfp4 Marlin path does too.
        return self.runner.run(dispatch_output, build_marlin_moe_quant_info(layer))

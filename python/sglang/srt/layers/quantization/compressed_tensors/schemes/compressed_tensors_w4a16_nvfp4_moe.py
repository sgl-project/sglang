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

__all__ = ["CompressedTensorsW4A16Nvfp4MoE"]

NVFP4_GROUP_SIZE = 16

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


class CompressedTensorsW4A16Nvfp4MoE(CompressedTensorsMoEScheme):
    """Weight-only NVFP4 MoE via the FP4 Marlin runner: fp4 experts, bf16/fp16
    activations.

    Serves nvfp4a16 (no ``input_activations``) and, on a pre-Blackwell GPU, an
    nvfp4 w4a4 checkpoint whose activation quantization is dropped. The Marlin
    MoE kernel is weight-only, so the input global scales are dropped after
    loading -- but a w4a4 checkpoint still ships them, and the expert loader
    raises KeyError on a tensor with no registered destination, so they must be
    registered first.

    w13 stays in the checkpoint's native [w1; w3] order, which is what
    silu_and_mul in fused_marlin_moe.py expects. _load_w13 reads
    load_up_proj_weight_first off layer.quant_method rather than off the scheme,
    so that order comes from the flag's default and must not be overridden here.
    """

    def __init__(self, has_input_global_scale: bool = False):
        # True for a w4a4 checkpoint being served weight-only.
        self.has_input_global_scale = has_input_global_scale
        self.group_size = NVFP4_GROUP_SIZE

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

        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=torch.float8_e4m3fn,
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
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, group_attrs)

        tensor_attrs = dict(extra_weight_attrs)
        tensor_attrs["quant_method"] = FusedMoeWeightScaleSupported.TENSOR.value

        # w13 carries one global scale per shard (gate, up); w2 one per expert.
        w13_weight_global_scale = torch.nn.Parameter(
            torch.empty(num_experts, 2, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_weight_global_scale", w13_weight_global_scale)
        set_weight_attrs(w13_weight_global_scale, tensor_attrs)

        w2_weight_global_scale = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_weight_global_scale", w2_weight_global_scale)
        set_weight_attrs(w2_weight_global_scale, tensor_attrs)

        if self.has_input_global_scale:
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
            prepare_moe_nvfp4_layer_for_marlin,
        )

        # The activation scales are meaningless for a weight-only kernel.
        if self.has_input_global_scale:
            delattr(layer, "w13_input_global_scale")
            delattr(layer, "w2_input_global_scale")

        # prepare_moe_nvfp4_layer_for_marlin reads `w13/w2_weight` and
        # `w13/w2_weight_scale_2`; compressed-tensors uses the `_packed` and
        # `_global_scale` names.
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")
        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        # Marlin supports a single shared w1/w3 global scale, so collapse the
        # gate/up columns to the gate scale.
        w13_global_scale = layer.w13_weight_global_scale.data
        if w13_global_scale.dim() > 1:
            if w13_global_scale.shape[1] >= 2 and not torch.allclose(
                w13_global_scale[:, 0], w13_global_scale[:, 1]
            ):
                logger.warning_once(
                    "w1_weight_global_scale must match w3_weight_global_scale. "
                    "Accuracy may be affected."
                )
            w13_global_scale = w13_global_scale[:, 0]
        # compressed-tensors stores the global scale as a divisor (1/scale),
        # while Marlin's weight_scale_2 is the scale itself. Skipping the
        # inversion overflows the bf16 bias multiply to inf, zeroing all logits.
        layer.w13_weight_scale_2 = torch.nn.Parameter(
            (1 / w13_global_scale).contiguous(), requires_grad=False
        )
        delattr(layer, "w13_weight_global_scale")

        layer.w2_weight_scale_2 = torch.nn.Parameter(
            (1 / layer.w2_weight_global_scale.data).contiguous(), requires_grad=False
        )
        delattr(layer, "w2_weight_global_scale")

        # check_moe_marlin_supports_layer is deliberately not called: its
        # `group_size in [-1, 32, 64, 128]` test rejects NVFP4's group_size=16
        # outright, so it would disable this path on every layer. The helper
        # below asserts the group size itself; validate shapes explicitly.
        if layer.hidden_size % 128 != 0:
            raise ValueError(
                f"NVFP4 Marlin MoE requires hidden_size % 128 == 0, got "
                f"{layer.hidden_size}."
            )
        if layer.intermediate_size_per_partition % 64 != 0:
            raise ValueError(
                "NVFP4 Marlin MoE requires intermediate_size_per_partition % 64 "
                f"== 0, got {layer.intermediate_size_per_partition}."
            )

        prepare_moe_nvfp4_layer_for_marlin(layer, group_size=self.group_size)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo

        expert_map = layer.dispatcher.local_expert_mapping
        global_num_experts = (
            layer.dispatcher.num_experts if expert_map is not None else -1
        )
        quant_info = MarlinMoeQuantInfo(
            w13_qweight=layer.w13_weight,
            w2_qweight=layer.w2_weight,
            w13_scales=layer.w13_weight_scale,
            w2_scales=layer.w2_weight_scale,
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=4,
            w13_global_scale=layer.w13_weight_scale_2,
            w2_global_scale=layer.w2_weight_scale_2,
            expert_map=expert_map,
            global_num_experts=global_num_experts,
        )
        return self.runner.run(dispatch_output, quant_info)

# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.utils import route_dropped_experts_to_local
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.utils.common import set_weight_attrs
from sglang.srt.utils import is_xpu

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsW4A16Mxfp4MoE"]

MXFP4_GROUP_SIZE = 32
_is_xpu = is_xpu()

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


class CompressedTensorsW4A16Mxfp4MoE(CompressedTensorsMoEScheme):
    """Split-expert MXFP4 MoE with BF16 activations on CUDA and Intel XPU.

    Serves a compressed-tensors ``mxfp4-pack-quantized`` checkpoint whose experts
    are stored **split per expert** (``experts.{e}.{gate,up,down}_proj.*``), which
    is a different layout from the fused GPT-OSS mxfp4 that ``Mxfp4MoEMethod``
    and ``Mxfp4MarlinMoEMethod`` handle -- those register ``w13/w2_weight``, so a
    split-expert checkpoint dies on ``KeyError: ...experts.w2_weight_packed``.

    Serves mxfp4a16 and mxfp4 w4a4. Both backends keep activations unquantized.
    If the checkpoint stores static activation scales, they are loaded and
    dropped; dynamic activation quantization has no serialized scale.

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

        # Gated experts fuse gate and up into w13; non-gated experts have only up.
        # prepare_moe_mxfp4_layer_for_marlin repacks the same number of shards.
        w13_num_shards = 2 if layer.moe_runner_config.is_gated else 1

        # Two fp4 items are packed per byte along the input dimension.
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
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
                w13_num_shards * intermediate_size_per_partition,
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
                torch.empty(num_experts, w13_num_shards, dtype=torch.float32),
                requires_grad=False,
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
        if _is_xpu:
            logger.info_once("Using Intel XPU grouped GEMM for weight-only MXFP4 MoE")
        else:
            logger.info_once("Using FP4 Marlin for weight-only MXFP4 MoE")
            self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    def get_moe_runner_backend(self, layer):
        return None if _is_xpu else MoeRunnerBackend.MARLIN

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._drop_input_global_scales(layer)

        if _is_xpu:
            return

        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            prepare_moe_mxfp4_layer_for_marlin,
        )

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

    def _drop_input_global_scales(self, layer: torch.nn.Module) -> None:
        if self.has_input_global_scale:
            delattr(layer, "w13_input_global_scale")
            delattr(layer, "w2_input_global_scale")

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        if _is_xpu:
            from sgl_kernel import fused_experts

            from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

            topk_weights, topk_ids, _ = dispatch_output.topk_output
            topk_weights, topk_ids = route_dropped_experts_to_local(
                topk_weights=topk_weights, topk_ids=topk_ids
            )
            return StandardCombineInput(
                hidden_states=fused_experts(
                    dispatch_output.hidden_states,
                    layer.w13_weight_packed,
                    layer.w2_weight_packed,
                    topk_weights,
                    topk_ids,
                    use_mxfp4_w4a16=True,
                    w1_scale=layer.w13_weight_scale,
                    w2_scale=layer.w2_weight_scale,
                    activation=self.moe_runner_config.activation,
                    routed_scaling_factor=self.moe_runner_config.routed_scaling_factor,
                    swiglu_limit=self.moe_runner_config.swiglu_limit,
                )
            )

        return self.runner.run(dispatch_output, self.get_marlin_quant_info(layer))

    def get_marlin_quant_info(self, layer):
        from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
            build_marlin_moe_quant_info,
        )

        return build_marlin_moe_quant_info(layer)


class CompressedTensorsW4A4Mxfp4MegaMoE(CompressedTensorsW4A16Mxfp4MoE):
    """Split-expert MXFP4 with dynamic MXFP4 activations on SM100 MegaMoE."""

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ) -> None:
        if not moe_runner_config.is_gated or moe_runner_config.activation != "silu":
            raise ValueError("Split MXFP4 MegaMoE requires gated SiLU experts.")
        logger.info_once("Using DeepGEMM MegaMoE (mxf4xmxf4) for split MXFP4 MoE")

    def get_moe_runner_backend(self, layer: torch.nn.Module) -> None:
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from deep_gemm import (
            transform_sf_into_required_layout,
            transform_weights_for_mega_moe,
        )

        from sglang.srt.layers.moe.mega_moe import check_mega_moe_shapes

        mma_type = "mxf4xmxf4"
        check_mega_moe_shapes(
            hidden=layer.hidden_size,
            intermediate=layer.intermediate_size_per_partition,
            mma_type=mma_type,
        )

        def pack(weight: torch.Tensor, scale: torch.Tensor):
            num_experts, rows, packed_columns = weight.shape
            scales = transform_sf_into_required_layout(
                scale.view(torch.float8_e8m0fnu).float(),
                mn=rows,
                k=packed_columns * 2,
                recipe=(1, MXFP4_GROUP_SIZE),
                num_groups=num_experts,
                disable_ue8m0_cast=False,
            )
            return weight.view(torch.int8), scales

        layer.mega_l1_weights, layer.mega_l2_weights = transform_weights_for_mega_moe(
            pack(layer.w13_weight_packed.data, layer.w13_weight_scale.data),
            pack(layer.w2_weight_packed.data, layer.w2_weight_scale.data),
            mma_type=mma_type,
        )
        layer.w13_weight_packed.data, layer.w13_weight_scale.data = (
            layer.mega_l1_weights
        )
        layer.w2_weight_packed.data, layer.w2_weight_scale.data = layer.mega_l2_weights
        self._drop_input_global_scales(layer)
        layer._mega_moe_weights_built = True

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        raise RuntimeError(
            "Split MXFP4 MegaMoE must run through the model's MegaMoE forward."
        )

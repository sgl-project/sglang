from __future__ import annotations

import enum
import logging
from enum import Enum
from typing import TYPE_CHECKING

import torch
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.distributed import get_tensor_model_parallel_world_size, get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A8Int8DynamicMoEMethod,
    NPUW4A16Int4DynamicMoEMethod,
    NPUW8A8Int8DynamicMoEMethod,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams, CutlassMoEType
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz, scaled_fp8_quant
from sglang.srt.layers.quantization.fp8_utils import (
    is_blackwell_supported,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.gptq import gptq_marlin_moe_repack
from sglang.srt.layers.quantization.marlin_utils import marlin_moe_permute_scales
from sglang.srt.layers.quantization.utils import (
    all_close_1d,
    per_tensor_dequantize,
    prepare_static_weights_for_trtllm_fp4_moe,
    reorder_w1w3_to_w3w1,
    replace_parameter,
    swizzle_blockscale,
)
from sglang.srt.utils import (
    get_bool_env_var,
    is_cuda,
    is_hip,
    is_npu,
    next_power_of_2,
    set_weight_attrs,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

_is_hip = is_hip()
_is_npu = is_npu()
_is_cuda = is_cuda()

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight


logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsW4A4Nvfp4MoE"]


class CompressedTensorsW4A4Nvfp4MoE(CompressedTensorsScheme):

    def __init__(self, quant_config: CompressedTensorsConfig):
        if not is_blackwell_supported():
            raise ValueError(
                "Current platform does not support NVFP4"
                " quantization. Please use Blackwell and"
                " above."
            )
        self.quant_config = quant_config
        self.group_size = 16
        self.use_flashinfer_trtllm = get_moe_runner_backend().is_flashinfer_trtllm()

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

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
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
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight Scales
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Weight Global Scales
        w13_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, 2, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_weight_global_scale", w13_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)

        w2_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_weight_global_scale", w2_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)

        # Input Global Scales
        w13_input_scale = torch.nn.Parameter(
            torch.empty(num_experts, 2, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_input_global_scale", w13_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_input_global_scale", w2_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # From packed to weight
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")

        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        if self.use_flashinfer_trtllm:
            w, s = reorder_w1w3_to_w3w1(
                layer.w13_weight.data, layer.w13_weight_scale.data, dim=-2
            )
            layer.w13_weight = torch.nn.Parameter(w, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(s, requires_grad=False)

        if not torch.allclose(
            layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
        ):
            logger.warning_once(
                "w1_weight_global_scale must match w3_weight_global_scale. "
                "Accuracy may be affected."
            )

        # Take inverse of global scale saved to disk
        layer.w13_weight_scale_2 = torch.nn.Parameter(
            1 / layer.w13_weight_global_scale[:, 0], requires_grad=False
        )

        layer.w2_weight_scale_2 = torch.nn.Parameter(
            1 / layer.w2_weight_global_scale.data, requires_grad=False
        )

        # w13
        if self.use_flashinfer_trtllm:
            w13_input_global_scale = (
                layer.w13_input_global_scale.min()
                .to(torch.float32)
                .expand(layer.num_local_experts)
            )
        else:
            w13_input_global_scale = layer.w13_input_global_scale.min(dim=1).values.to(
                torch.float32
            )
        layer.g1_alphas = torch.nn.Parameter(
            ((1 / w13_input_global_scale) * layer.w13_weight_scale_2),
            requires_grad=False,
        )

        layer.w13_input_scale_quant = torch.nn.Parameter(
            (w13_input_global_scale), requires_grad=False
        )

        # w2
        if self.use_flashinfer_trtllm:
            w2_input_global_scale = (
                layer.w2_input_global_scale.min()
                .to(torch.float32)
                .expand(layer.num_local_experts)
            )
        else:
            w2_input_global_scale = layer.w2_input_global_scale

        layer.g2_alphas = torch.nn.Parameter(
            ((1 / w2_input_global_scale) * layer.w2_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )

        layer.w2_input_scale_quant = torch.nn.Parameter(
            (w2_input_global_scale), requires_grad=False
        )

        # TensorRT-LLM specific processing
        if self.use_flashinfer_trtllm:
            # Prepare static weights for TRT-LLM kernel
            (
                gemm1_weights_fp4_shuffled,
                gemm1_scales_fp4_shuffled,
                gemm2_weights_fp4_shuffled,
                gemm2_scales_fp4_shuffled,
            ) = prepare_static_weights_for_trtllm_fp4_moe(
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                layer.w2_weight.size(-2),  # hidden_size
                layer.w13_weight.size(-2) // 2,  # intermediate_size
                layer.w13_weight.size(0),  # num_experts
            )
            logger.debug("Finished shuffling weights for TRT-LLM MOE")

            layer.gemm1_weights_fp4_shuffled = torch.nn.Parameter(
                gemm1_weights_fp4_shuffled, requires_grad=False
            )
            layer.gemm2_weights_fp4_shuffled = torch.nn.Parameter(
                gemm2_weights_fp4_shuffled, requires_grad=False
            )
            layer.gemm1_scales_fp4_shuffled = torch.nn.Parameter(
                gemm1_scales_fp4_shuffled, requires_grad=False
            )
            layer.gemm2_scales_fp4_shuffled = torch.nn.Parameter(
                gemm2_scales_fp4_shuffled, requires_grad=False
            )

            # Additional parameter needed for TRT-LLM
            layer.g1_scale_c = torch.nn.Parameter(
                (layer.w2_input_scale_quant * layer.g1_alphas).to(torch.float32),
                requires_grad=False,
            )

            # Clean up weights that won't be used by TRT-LLM
            del layer.w2_weight
            del layer.w2_weight_scale
            del layer.w13_weight
            del layer.w13_weight_scale
        else:
            # swizzle weight scales
            layer.w13_weight_scale = torch.nn.Parameter(
                swizzle_blockscale(layer.w13_weight_scale), requires_grad=False
            )

            layer.w2_weight_scale = torch.nn.Parameter(
                swizzle_blockscale(layer.w2_weight_scale), requires_grad=False
            )

            layer.cutlass_moe_params = CutlassMoEParams(
                CutlassMoEType.BlockscaledFP4,
                layer.w13_weight.device,
                num_experts=layer.num_experts,
                intermediate_size_per_partition=layer.w2_weight.shape[2] * 2,
                hidden_size=layer.w13_weight.shape[2] * 2,
            )

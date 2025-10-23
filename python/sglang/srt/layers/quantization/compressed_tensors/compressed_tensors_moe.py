# Adapted from https://github.com/vllm-project/vllm/tree/v0.8.2/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import enum
import logging
import re
from enum import Enum
from typing import TYPE_CHECKING

try:
    from sgl_kernel import fused_marlin_moe

    FUSED_MARLIN_MOE_AVAILABLE = True
except ImportError:
    FUSED_MARLIN_MOE_AVAILABLE = False

try:
    from kt_kernel import AMXMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False

import torch
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.environ import envs
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    WNA16_SUPPORTED_BITS,
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz, scaled_fp8_quant
from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz
from sglang.srt.layers.quantization.gptq import gptq_marlin_moe_repack
from sglang.srt.layers.quantization.marlin_utils import marlin_moe_permute_scales
from sglang.srt.layers.quantization.utils import (
    all_close_1d,
    per_tensor_dequantize,
    replace_parameter,
)
from sglang.srt.utils import (
    get_bool_env_var,
    get_compiler_backend,
    is_cuda,
    is_hip,
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
_is_cuda = is_cuda()

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.ops.shuffle import shuffle_weight

    from sglang.srt.layers.moe.rocm_moe_utils import rocm_fused_experts_tkw1


if _is_cuda:
    from sgl_kernel import fused_marlin_moe

logger = logging.getLogger(__name__)


def _mask_topk_ids_cpu_experts(topk_ids: torch.Tensor, num_gpu_experts: int):
    """Mask topk_ids >= num_gpu_experts by setting them to -1."""
    topk_ids[topk_ids >= num_gpu_experts] = -1


@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_cpu_expert_ids(topk_ids: torch.Tensor, num_gpu_experts: int):
    """mask CPU expert IDs."""
    _mask_topk_ids_cpu_experts(topk_ids, num_gpu_experts)
    return topk_ids


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


__all__ = [
    "CompressedTensorsMoEMethod",
    "CompressedTensorsW8A8Fp8MoEMethod",
    "CompressedTensorsWNA16MoEMethod",
    "CompressedTensorsWNA16AMXEPMoEMethod",  # for Ktransformers
]


class CompressedTensorsMoEMethod(FusedMoEMethodBase):
    def __new__(cls, *args, **kwargs):
        if cls is CompressedTensorsMoEMethod:
            return super().__new__(cls)
        return super().__new__(cls)

    @staticmethod
    def get_moe_method(
        quant_config: CompressedTensorsConfig,
        layer: torch.nn.Module,
        prefix: str,
    ) -> "CompressedTensorsMoEMethod":
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.

        if envs.SGLANG_KT_MOE_AMX_WEIGHT_PATH.is_set():
            match = re.search(r"(\d+)\.mlp", prefix)
            if not match:
                raise ValueError(
                    f"Unable to extract layer number from prefix '{prefix}'. "
                    f"Expected format: '<layer_number>.mlp'"
                )
            layer_number = int(match.group(1))
            return CompressedTensorsWNA16AMXEPMoEMethod(quant_config, layer_number)

        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get("input_activations")
        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):

            logger.info_once("Using CompressedTensorsWNA16MarlinMoEMethod")
            return CompressedTensorsWNA16MoEMethod(quant_config)
        elif quant_config._is_fp8_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8Fp8MoEMethod(quant_config)
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
            )


class CompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsMoEMethod):

    def __init__(self, quant_config: CompressedTensorsConfig):
        self.quant_config = quant_config
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get("weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations"
        )

        self.static_input_scales = not self.input_quant.dynamic

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

        params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        # per-tensor quantization
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            weight_quant_method = FusedMoeWeightScaleSupported.TENSOR.value
        elif self.weight_quant.strategy == QuantizationStrategy.CHANNEL:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
                requires_grad=False,
            )
            weight_quant_method = FusedMoeWeightScaleSupported.CHANNEL.value
        else:
            raise ValueError(
                f"Unsupported weight quantization strategy: {self.weight_quant.strategy}"
            )

        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update({"quant_method": weight_quant_method})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.static_input_scales:
            assert (
                self.input_quant.strategy == QuantizationStrategy.TENSOR
            ), "Only per-tensor quantization is supported for static input scales"
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module | FusedMoE) -> None:
        # Fp8 moe kernels require a single activation scale.
        # We take the max of all the scales in case they differ.
        if self.static_input_scales:
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None."
                )
            if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                layer.w2_input_scale
            ):
                logger.warning(
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer."
                )
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False
            )
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False
            )

        if is_fp8_fnuz():
            # Normalize the weights and scales
            w13_weight, w13_weight_scale, w13_input_scale = (
                normalize_e4m3fn_to_e4m3fnuz(
                    layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                )
            )
            w2_weight, w2_weight_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
            )
            # Reset the parameter
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(
                w13_weight_scale, requires_grad=False
            )
            if w13_input_scale is not None:
                layer.w13_input_scale = torch.nn.Parameter(
                    w13_input_scale, requires_grad=False
                )
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_weight_scale, requires_grad=False
            )
            if w2_input_scale is not None:
                layer.w2_input_scale = torch.nn.Parameter(
                    w2_input_scale, requires_grad=False
                )
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.num_local_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id],
                    )
                    (
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        _,
                    ) = scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])

                    start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )

        if _use_aiter:
            with torch.no_grad():
                # Pre-shuffle weights
                layer.w13_weight = torch.nn.Parameter(
                    shuffle_weight(layer.w13_weight.data, (16, 16)),
                    requires_grad=False,
                )
                torch.cuda.empty_cache()
                layer.w2_weight = torch.nn.Parameter(
                    shuffle_weight(layer.w2_weight.data, (16, 16)),
                    requires_grad=False,
                )
                torch.cuda.empty_cache()

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        if (
            _use_aiter
            and self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and moe_runner_config.apply_router_weight_on_input
        ):
            topk_weights, topk_ids, _ = topk_output
            output = rocm_fused_experts_tkw1(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=moe_runner_config.activation,
                apply_router_weight_on_input=moe_runner_config.apply_router_weight_on_input,
                use_fp8_w8a8=True,
                per_channel_quant=self.weight_quant.strategy
                == QuantizationStrategy.CHANNEL,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
            )
            return StandardCombineInput(hidden_states=output)
        else:
            quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                use_fp8_w8a8=True,
                per_channel_quant=self.weight_quant.strategy
                == QuantizationStrategy.CHANNEL,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a13_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
            )
            return self.runner.run(dispatch_output, quant_info)


class CompressedTensorsWNA16MoEMethod(CompressedTensorsMoEMethod):

    def __init__(self, quant_config: CompressedTensorsConfig, num_gpu_experts=-1):
        self.quant_config = quant_config
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        config = self.quant_config.target_scheme_map["Linear"].get("weights")
        self.num_bits = config.num_bits
        self.packed_factor = 32 // config.num_bits
        self.strategy = config.strategy
        self.group_size = config.group_size
        self.actorder = config.actorder
        assert config.symmetric, "Only symmetric quantization is supported for MoE"

        if not (
            self.quant_config.quant_format == CompressionFormat.pack_quantized.value
            and self.num_bits in WNA16_SUPPORTED_BITS
        ):
            raise ValueError(
                "For Fused MoE layers, only ",
                f"{CompressionFormat.pack_quantized.value} ",
                "is supported for the following bits: ",
                f"{WNA16_SUPPORTED_BITS}",
            )
        self.num_gpu_experts = num_gpu_experts

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if self.num_gpu_experts != -1:
            num_experts = self.num_gpu_experts

        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update(
            {"is_transposed": True, "quant_method": self.strategy}
        )
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.packed_factor,
                2 * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.packed_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # In the case where we have actorder/g_idx,
        # we do not partition the w2 scales
        load_full_w2 = self.actorder and self.group_size != -1

        if load_full_w2:
            w2_scales_size = intermediate_size_per_partition * layer.moe_tp_size
        else:
            w2_scales_size = intermediate_size_per_partition

        self.is_k_full = (not self.actorder) or layer.moe_tp_size == 1

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        w13_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.ones(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": load_full_w2})

        w2_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        w13_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )

        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None
        layer.marlin_state = GPTQMarlinState.REPACK

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        def replace_tensor(name, new_t):
            # It is important to use resize_() here since it ensures
            # the same buffer is reused
            getattr(layer, name).resize_(new_t.shape)
            getattr(layer, name).copy_(new_t)
            del new_t

        num_experts = layer.w13_weight_g_idx.shape[0]
        device = layer.w13_weight_g_idx.device

        # when running models with grouped act order,
        # resort to g_idx values provided in checkpoint
        if self.actorder == "group":
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_weight_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_weight_g_idx)
            w13_sorted_g_idx = torch.empty_like(layer.w13_weight_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_weight_g_idx)

            for e in range(num_experts):
                w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_weight_g_idx[e]).to(
                    torch.int32
                )
                w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_weight_g_idx[e]).to(
                    torch.int32
                )
                w13_sorted_g_idx[e] = layer.w13_weight_g_idx[e][
                    w13_g_idx_sort_indices[e]
                ]
                w2_sorted_g_idx[e] = layer.w2_weight_g_idx[e][w2_g_idx_sort_indices[e]]

            replace_parameter(layer, "w13_weight_g_idx", w13_sorted_g_idx)
            replace_parameter(layer, "w2_weight_g_idx", w2_sorted_g_idx)
            replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
            replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)

        else:
            layer.w13_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )

        marlin_w13_qweight = gptq_marlin_moe_repack(
            layer.w13_weight_packed,
            layer.w13_g_idx_sort_indices,
            layer.w13_weight_packed.shape[1] * self.packed_factor,
            layer.w13_weight_packed.shape[2],
            self.num_bits,
        )
        replace_parameter(layer, "w13_weight_packed", marlin_w13_qweight)
        marlin_w2_qweight = gptq_marlin_moe_repack(
            layer.w2_weight_packed,
            layer.w2_g_idx_sort_indices,
            layer.w2_weight_packed.shape[1] * self.packed_factor,
            layer.w2_weight_packed.shape[2],
            self.num_bits,
        )
        replace_parameter(layer, "w2_weight_packed", marlin_w2_qweight)
        # Repack scales
        marlin_w13_scales = marlin_moe_permute_scales(
            layer.w13_weight_scale,
            layer.w13_weight_packed.shape[2],
            layer.w13_weight_scale.shape[2],
            self.group_size,
        )
        replace_parameter(layer, "w13_weight_scale", marlin_w13_scales)

        marlin_w2_scales = marlin_moe_permute_scales(
            layer.w2_weight_scale,
            layer.w2_weight_scale.shape[1]
            * (self.group_size if self.group_size != -1 else self.packed_factor),
            layer.w2_weight_scale.shape[2],
            self.group_size,
        )
        replace_parameter(layer, "w2_weight_scale", marlin_w2_scales)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, router_logits = topk_output

        output = fused_marlin_moe(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            router_logits,
            topk_weights,
            topk_ids,
            g_idx1=layer.w13_weight_g_idx,
            g_idx2=layer.w2_weight_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            num_bits=self.num_bits,
            is_k_full=self.is_k_full,
        )
        return StandardCombineInput(hidden_states=output)


class CompressedTensorsWNA16AMXMoEMethod(CompressedTensorsMoEMethod):
    """AMX MoE method using AMXMoEWrapper for CPU inference."""

    def __init__(
        self,
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer_idx,
        num_gpu_experts,
        cpuinfer,
        threadpool_count,
        amx_weight_path,
        chunked_prefill_size,
    ):

        if not KTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "kt_kernel is not installed, to use CompressedTensorsWNA16AMXEPMoEMethod, please install kt_kernel."
            )

        if not FUSED_MARLIN_MOE_AVAILABLE:
            raise ImportError("fused_marlin_moe is not available")

        self.tp_rank = get_tensor_model_parallel_rank()
        self.layer_idx = layer_idx
        self.num_gpu_experts = num_gpu_experts
        self.amx_weight_path = amx_weight_path
        self.chunked_prefill_size = chunked_prefill_size
        self.cpuinfer = cpuinfer
        self.threadpool_count = threadpool_count
        self.amx_wrapper = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.experts_num = num_experts
        self.num_experts_per_tok = extra_weight_attrs.pop("top_k")
        self.hidden_size = hidden_size
        self.moe_intermediate_size = extra_weight_attrs.pop("intermediate_size_full")

        if self.tp_rank != 0:
            return
        self.amx_wrapper = AMXMoEWrapper(
            layer_idx=self.layer_idx,
            num_experts=num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_gpu_experts=self.num_gpu_experts,
            cpuinfer_threads=self.cpuinfer,
            threadpool_count=self.threadpool_count,
            amx_weight_path=self.amx_weight_path,
            chunked_prefill_size=self.chunked_prefill_size,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.tp_rank != 0:
            return

        if self.amx_wrapper is None:
            raise RuntimeError(
                "AMXMoEWrapper not initialized. Call create_weights first."
            )

        torch.cuda.synchronize()
        # Load weights using wrapper
        from sglang.srt.eplb.expert_location_dispatch import (
            get_global_expert_location_metadata,
        )

        physical_to_logical_map_cpu = (
            get_global_expert_location_metadata()
            .physical_to_logical_map_cpu[self.layer_idx]
            .contiguous()
        )
        self.amx_wrapper.load_weights(physical_to_logical_map_cpu)

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> None:
        """Submit AMX inference task asynchronously."""
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        if self.tp_rank != 0 or self.amx_wrapper is None:
            return None

        # Submit forward task using wrapper
        self.amx_wrapper.submit_forward(
            x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream
        )
        return None

    def sync(self, x):
        """Synchronize and retrieve AMX inference results."""
        if self.tp_rank != 0 or self.amx_wrapper is None:
            return torch.zeros_like(x)

        # Sync forward task using wrapper
        return self.amx_wrapper.sync_forward(
            x, torch.cuda.current_stream(x.device).cuda_stream
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        """Execute AMX MoE forward pass synchronously."""
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        if self.tp_rank != 0 or self.amx_wrapper is None:
            return StandardCombineInput(hidden_states=torch.zeros_like(x))

        # Execute forward using wrapper (submit + sync)
        output = self.amx_wrapper.forward(
            x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream
        )
        return StandardCombineInput(hidden_states=output)


def override_config(
    cls,
    num_gpu_experts,
    cpuinfer,
    threadpool_count,
    amx_weight_path,
    amx_method,
    chunked_prefill_size,
):
    """Override MOE configuration via environment variables."""
    # Set environment variables using envs utility class
    if num_gpu_experts is not None:
        envs.SGLANG_KT_MOE_NUM_GPU_EXPERTS.set(num_gpu_experts)
    if cpuinfer is not None:
        envs.SGLANG_KT_MOE_CPUINFER.set(cpuinfer)
    if threadpool_count is not None:
        envs.SGLANG_KT_THREADPOOL_COUNT.set(threadpool_count)
    if amx_weight_path is not None:
        envs.SGLANG_KT_MOE_AMX_WEIGHT_PATH.set(amx_weight_path)
    if amx_method is not None:
        envs.SGLANG_KT_AMX_METHOD.set(amx_method)
    if chunked_prefill_size is not None:
        envs.SGLANG_KT_MOE_CHUNKED_PREFILL_SIZE.set(chunked_prefill_size)


class CompressedTensorsWNA16AMXEPMoEMethod(CompressedTensorsMoEMethod):

    def __init__(
        self,
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer_idx,
    ):
        self.tp_rank = get_tensor_model_parallel_rank()

        if (
            not envs.SGLANG_KT_MOE_NUM_GPU_EXPERTS.is_set()
            or not envs.SGLANG_KT_MOE_CPUINFER.is_set()
            or not envs.SGLANG_KT_MOE_AMX_WEIGHT_PATH.is_set()
        ):
            raise RuntimeError(
                "the following arguments are required: --kt-amx-weight-path, --kt-cpuinfer, --kt-num-gpu-experts"
            )
        self.num_gpu_experts = envs.SGLANG_KT_MOE_NUM_GPU_EXPERTS.value
        cpuinfer = envs.SGLANG_KT_MOE_CPUINFER.value
        threadpool_count = envs.SGLANG_KT_THREADPOOL_COUNT.value
        amx_weight_path = envs.SGLANG_KT_MOE_AMX_WEIGHT_PATH.value
        chunked_prefill_size = envs.SGLANG_KT_MOE_CHUNKED_PREFILL_SIZE.value

        self.AMX_method = CompressedTensorsWNA16AMXMoEMethod(
            quant_config,
            layer_idx,
            self.num_gpu_experts,
            cpuinfer,
            threadpool_count,
            amx_weight_path,
            chunked_prefill_size,
        )
        self.marlin_method = CompressedTensorsWNA16MoEMethod(
            quant_config, self.num_gpu_experts
        )
        self.layer_id = layer_idx

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.global_num_experts = num_experts
        self.AMX_method.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        self.marlin_method.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.AMX_method.process_weights_after_loading(layer)
        self.marlin_method.process_weights_after_loading(layer)

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        """Submit hybrid GPU+CPU MoE task (AMX submission + GPU execution)."""
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, router_logits = topk_output

        # Submit AMX task if on rank 0
        if self.tp_rank == 0:
            self.AMX_method.submit(layer, dispatch_output)

        # Mask CPU expert IDs (>= num_gpu_experts) as -1 so they won't be computed on GPU
        topk_ids = mask_cpu_expert_ids(topk_ids, self.num_gpu_experts)

        # Execute GPU (Marlin) experts
        output = fused_marlin_moe(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            router_logits,
            topk_weights,
            topk_ids,
            g_idx1=layer.w13_weight_g_idx,
            g_idx2=layer.w2_weight_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            num_bits=self.marlin_method.num_bits,
            is_k_full=self.marlin_method.is_k_full,
            global_num_experts=self.global_num_experts,
            expert_map=torch.empty(1, device=x.device),
        )
        return StandardCombineInput(hidden_states=output)

    def sync(self, x):
        """Synchronize and retrieve AMX results."""
        if self.tp_rank != 0:
            return torch.zeros_like(x)
        return self.AMX_method.sync(x)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        """Execute hybrid GPU+CPU MoE forward pass with parallelism."""
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, router_logits = topk_output

        # Step 1: Submit AMX task (non-blocking) if on rank 0
        # This starts CPU computation in parallel
        if self.tp_rank == 0:
            self.AMX_method.submit(layer, dispatch_output)

        # Step 2: Execute GPU (Marlin) experts in parallel with CPU

        # Mask CPU expert IDs (>= num_gpu_experts) as -1 so they won't be computed on GPU
        topk_ids = mask_cpu_expert_ids(topk_ids, self.num_gpu_experts)

        # While GPU computes, CPU is also computing
        output = fused_marlin_moe(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            router_logits,
            topk_weights,
            topk_ids,
            g_idx1=layer.w13_weight_g_idx,
            g_idx2=layer.w2_weight_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            num_bits=self.marlin_method.num_bits,
            is_k_full=self.marlin_method.is_k_full,
            global_num_experts=self.global_num_experts,
            expert_map=torch.empty(1, device=x.device),
        )

        # Step 3: Sync AMX results and combine with GPU results
        if self.tp_rank == 0:
            amx_output = self.AMX_method.sync(x)
            output += amx_output

        return StandardCombineInput(hidden_states=output)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.AMX_method.create_moe_runner(layer, moe_runner_config)

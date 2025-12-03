from __future__ import annotations

import logging
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.amx_utils import _amx_process_weight_after_loading
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer
from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import (
    apply_module_patch,
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_npu,
    set_weight_attrs,
    use_intel_amx_backend,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_npu = is_npu()

if _is_cuda:
    from sgl_kernel import int8_scaled_mm
elif _is_npu:
    import torch_npu

    try:
        from mindie_turbo import _ops as ops
        from mindie_turbo.quantize.quant_utils import quant_per_tensor
    except ImportError:
        useMindIETurbo = False
    else:
        useMindIETurbo = True

    from sgl_kernel_npu.norm.add_rmsnorm_bias import add_rmsnorm_bias

logger = logging.getLogger(__name__)


# func refers to RMSNorm.__init__
def npu_wrapper_rmsnorm_init(func):
    def init(self, hidden_size: int, **extra_args) -> None:
        func(self, hidden_size, **extra_args)
        self.ignore_anti = True
        # The Ascend w8a8_int8 quantization requires adding a bias in rmsnorm
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=False)

    return init


# func refers to RMSNorm.forward_oot
def npu_wrapper_rmsnorm_forward(func):
    def _rmsnorm_forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            x = x.contiguous()
        if residual is not None:
            out, residual_out = add_rmsnorm_bias(
                x,
                residual,
                self.weight.data,
                self.bias,
                self.variance_epsilon,
            )
            return out.to(x.dtype), residual_out

        out = torch_npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]
        out = out + self.bias
        return out.to(x.dtype)

    return _rmsnorm_forward_oot


def npu_fused_experts(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    **kwargs,
):
    w13_offset = kwargs.get("w13_offset", None)
    w2_offset = kwargs.get("w2_offset", None)
    use_wna16 = kwargs.get("use_wna16", False)

    original_shape = hidden_states.shape
    original_dtype = hidden_states.dtype
    scale_dtype = original_dtype if original_dtype == torch.bfloat16 else torch.float32
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]
    row_idx_len = num_tokens * top_k
    row_idx = (
        torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
        .view(top_k, -1)
        .permute(1, 0)
        .contiguous()
    )
    hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch_npu.npu_moe_init_routing(
            hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
        )
    )
    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )
    expert_tokens = expert_tokens.to(torch.int64)
    # gmm1: gate_up_proj
    if not use_wna16:
        hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
        scale_args13 = {
            "scale": [w13_scale.to(scale_dtype)],
            "per_token_scale": [pertoken_scale],
        }
    else:
        scale_args13 = {
            "antiquant_scale": [w13_scale],
            "antiquant_offset": [w13_offset],
        }

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        **scale_args13,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]
    # act_fn: swiglu
    hidden_states = torch_npu.npu_swiglu(hidden_states)
    if not use_wna16:
        hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)

        scale_args2 = {
            "scale": [w2_scale.to(scale_dtype)],
            "per_token_scale": [pertoken_scale],
        }
    else:
        scale_args2 = {"antiquant_scale": [w2_scale], "antiquant_offset": [w2_offset]}
    # gmm2: down_proj
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        **scale_args2,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    final_hidden_states = torch_npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )
    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


class W8A8Int8Config(QuantizationConfig):
    """Config class for W8A8 or W4A16 Quantization.

    - Weight: static, per-channel, symmetric
    - Activation: dynamic, per-token, symmetric
    """

    def __init__(self, quant_config: Dict[str, Any] = {}):
        super().__init__()
        self.quant_description = quant_config
        self.is_dynamic = quant_config.get("is_dynamic", False)
        self.is_moe_w4_dynamic = False
        ignore = cast(List[str], quant_config.get("ignore", []))
        self.ignore = ignore if ignore is not None else []
        packed_modules_mapping = quant_config.get("packed_modules_mapping", {})
        self.packed_modules_mapping = (
            packed_modules_mapping if packed_modules_mapping is not None else {}
        )
        self.target_scheme_map = (
            CompressedTensorsConfig._quantization_scheme_map_from_config(
                config=quant_config
            )
        )
        target = "MoEGMM" if "MoEGMM" in self.target_scheme_map else "Linear"
        target_scheme = self.target_scheme_map.get(target, None)
        if target_scheme is None:
            self.is_moe_w4_dynamic = False
        else:
            weight_quant = target_scheme.get("weights")
            input_quant = target_scheme.get("input_activations")
            self.is_moe_w4_dynamic = self.is_dynamic_token_w4(weight_quant, input_quant)
            self.is_moe_input_quant = input_quant

        if _is_npu:
            # Ascend w8a8_int8 quantization with bias, use wrappers to isolate the effects between models
            for name in self.quant_description.keys():
                if "norm.bias" in name:
                    apply_module_patch(
                        "sglang.srt.layers.layernorm.RMSNorm",
                        "__init__",
                        [npu_wrapper_rmsnorm_init],
                    )
                    apply_module_patch(
                        "sglang.srt.layers.layernorm.RMSNorm",
                        "forward_npu",
                        [npu_wrapper_rmsnorm_forward],
                    )

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return (
            [torch.float16, torch.bfloat16]
            if not _is_npu
            else [torch.int8, torch.float16, torch.bfloat16]
        )

    @classmethod
    def get_min_capability(cls) -> int:
        if _is_npu:
            raise NotImplementedError(
                'NPU hardware does not support "get_min_capability" feature.'
            )
        else:
            return 75

    @classmethod
    def get_name(self) -> str:
        return "w8a8_int8"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        filenames = []
        if _is_npu:
            filenames.append("quant_model_description.json")
        return filenames

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> W8A8Int8Config:
        return cls(config)

    def _get_quant_method_npu(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if should_ignore_layer(
                prefix,
                ignore=self.ignore,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            key = "model"
            if "vision_model" in prefix:
                key = "vision_model"
            elif "visual" in prefix:
                key = "visual"
            packed_modules_mapping_subset = self.packed_modules_mapping.get(key, {})
            prefix_in_quant_config = prefix
            proj_name = prefix.split(".")[-1]
            if proj_name in packed_modules_mapping_subset:
                prefix_in_quant_config = prefix.replace(
                    proj_name, packed_modules_mapping_subset[proj_name][0]
                )
            self.is_dynamic = (
                self.quant_description[prefix_in_quant_config + ".weight"]
                == "W8A8_DYNAMIC"
            )
            if self.is_layer_skipped(prefix, packed_modules_mapping_subset):
                return UnquantizedLinearMethod()
            return (
                NPU_W8A8DynamicLinearMethod(self)
                if self.is_dynamic
                else NPU_W8A8LinearMethod(self)
            )
        elif isinstance(layer, FusedMoE):
            prefix_in_quant_config = prefix + ".0.down_proj.weight"
            is_moe_w4a8_dynamic = (
                self.quant_description.get(prefix_in_quant_config, "STATIC")
                == "W4A8_DYNAMIC"
            )
            if (
                self.is_moe_w4_dynamic and self.is_moe_input_quant is not None
            ) or is_moe_w4a8_dynamic:
                raise ValueError("npu does not support W4A8 currently!")
            elif self.is_moe_w4_dynamic and self.is_moe_input_quant is None:
                return NPU_W4A16MoEMethod(self)
            else:
                return NPU_W8A8MoEMethod(self)
        return None

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        if _is_npu:
            return self._get_quant_method_npu(layer, prefix)
        else:
            from sglang.srt.layers.linear import LinearBase
            from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

            if should_ignore_layer(
                prefix, ignore=self.ignore, fused_mapping=self.packed_modules_mapping
            ):
                return UnquantizedLinearMethod()
            if isinstance(layer, LinearBase):
                return W8A8Int8LinearMethod(self)
            elif isinstance(layer, FusedMoE):
                return W8A8Int8MoEMethod(self)
            return None

    def is_layer_skipped(
        self, prefix: str, fused_mapping: Mapping[str, List[str]] = MappingProxyType({})
    ):
        # adapted from vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped
        proj_name = prefix.split(".")[-1]
        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in fused_mapping[proj_name]
            ]

            is_skipped = None
            for shard_prefix in shard_prefixes:
                is_shard_skipped = (
                    self.quant_description[shard_prefix + ".weight"] == "FLOAT"
                )

                if is_skipped is None:
                    is_skipped = is_shard_skipped
                elif is_shard_skipped != is_skipped:
                    raise ValueError(
                        f"Detected some but not all shards of {prefix} "
                        "are quantized. All shards of fused layers "
                        "to have the same precision."
                    )
        else:
            is_skipped = self.quant_description[prefix + ".weight"] == "FLOAT"

        assert is_skipped is not None
        return is_skipped

    def get_scaled_act_names(self) -> List[str]:
        return []

    def is_dynamic_token_w4(self, weight_quant, input_quant) -> bool:
        is_w4 = weight_quant.num_bits == 4
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
            or weight_quant.strategy == QuantizationStrategy.GROUP.value
        )
        if input_quant is not None:
            is_token = (
                weight_strategy
                and input_quant.strategy == QuantizationStrategy.TOKEN.value
            )
            is_dynamic = not weight_quant.dynamic and input_quant.dynamic
        else:
            is_token = weight_strategy
            is_dynamic = not weight_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_w4 and weight_quant.symmetric and is_token and is_dynamic


class W8A8Int8LinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: W8A8Int8Config):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu:
            assert (
                _is_cpu_amx_available
            ), "W8A8Int8LinearMethod on CPU requires that CPU has AMX support"
            _amx_process_weight_after_loading(layer, ["weight"])
        else:
            layer.weight = Parameter(layer.weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        weight_loader = extra_weight_attrs.get("weight_loader")
        self.logical_widths = output_partition_sizes

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        if use_intel_amx_backend(layer):
            return torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
                x,
                layer.weight,
                layer.weight_scale,
                bias,
                x.dtype,
                True,  # is_vnni
            )
        x_q, x_scale = per_token_quant_int8(x)

        x_q_2d = x_q.view(-1, x_q.shape[-1])
        x_scale_2d = x_scale.view(-1, x_scale.shape[-1])
        output_shape = [*x_q.shape[:-1], layer.weight.shape[1]]

        output = int8_scaled_mm(
            x_q_2d,
            layer.weight,
            x_scale_2d,
            layer.weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )

        return output.view(output_shape)


class W8A8Int8MoEMethod(FusedMoEMethodBase):
    """MoE method for INT8.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.
    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: W8A8Int8Config):
        self.quant_config = quant_config

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

        tp_size = get_tensor_model_parallel_world_size()

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
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
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu:
            assert (
                _is_cpu_amx_available
            ), "W8A8Int8MoEMethod on CPU requires that CPU has AMX support"
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])
        else:
            layer.w13_weight = Parameter(layer.w13_weight, requires_grad=False)
            layer.w2_weight = Parameter(layer.w2_weight, requires_grad=False)
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        if use_intel_amx_backend(layer):
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = topk_output
            x, topk_weights = apply_topk_weights_cpu(
                self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace See [Note] inplace should be False in fused_experts.
                True,  # use_int8_w8a8
                False,  # use_fp8_w8a16
                layer.w13_weight_scale,  # w1_scale
                layer.w2_weight_scale,  # w2_scale
                None,  # block_size
                layer.w13_input_scale,  # a1_scale
                layer.w2_input_scale,  # a2_scale
                True,  # is_vnni
            )
            return StandardCombineInput(hidden_states=output)

        quant_info = TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_int8_w8a8=True,
            per_channel_quant=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )
        return self.runner.run(dispatch_output, quant_info)


class NPU_W8A8LinearMethodImpl:
    """Linear method for NPU W8A8."""

    def __init__(self) -> None:
        # aclnn quant matmul requires to transpose matrix B, set to true by default.
        self.transpose_weight = True

    @staticmethod
    def get_weight(
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> Dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {}
        params_dict["input_scale"] = torch.empty(1, dtype=params_dtype)
        params_dict["input_offset"] = torch.empty(1, dtype=params_dtype)
        return params_dict

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        params_dict["quant_bias"] = torch.empty(output_size, dtype=torch.int32)
        if params_dtype == torch.bfloat16:
            params_dict["deq_scale"] = torch.empty(output_size, dtype=torch.float32)
        elif params_dtype == torch.float16:
            params_dict["deq_scale"] = torch.empty(output_size, dtype=torch.int64)
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # To prevent import loops
        from sglang.srt.layers.linear import RowParallelLinear

        original_dtype = x.dtype
        if original_dtype != torch.int8:
            x = torch_npu.npu_quantize(
                x,
                layer.aclnn_input_scale_reciprocal,
                layer.aclnn_input_offset,
                torch.qint8,
                -1,
                False,
            )
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in Attention TP>1 case)
        if isinstance(layer, RowParallelLinear) and layer.tp_rank > 0:
            quant_bias = None
        else:
            quant_bias = layer.quant_bias
        return torch_npu.npu_quant_matmul(
            x,
            layer.weight,
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=original_dtype,
        )

    def process_weights_after_loading(self, layer):
        expanding_factor = layer.weight.data.shape[1]
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        layer.aclnn_input_scale_reciprocal = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)
        layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 29)


class NPU_W8A8LinearMethodMTImpl:
    """Linear method for NPU W8A8."""

    def __init__(self) -> None:
        self.transpose_weight = True

    @staticmethod
    def get_weight(
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> Dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {}
        params_dict["input_scale"] = torch.empty(1, dtype=params_dtype)
        params_dict["input_offset"] = torch.empty(1, dtype=torch.int8)
        return params_dict

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        params_dict["quant_bias"] = torch.empty(output_size, dtype=torch.int32)
        if params_dtype == torch.bfloat16:
            params_dict["deq_scale"] = torch.empty(output_size, dtype=torch.float32)
        elif params_dtype == torch.float16:
            params_dict["deq_scale"] = torch.empty(output_size, dtype=torch.int64)
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # To prevent import loops
        from sglang.srt.layers.linear import RowParallelLinear

        original_dtype = x.dtype
        if original_dtype != torch.int8:
            x = quant_per_tensor(x, layer.input_scale, layer.input_offset)

        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in Attention TP>1 case)
        if isinstance(layer, RowParallelLinear) and layer.tp_rank > 0:
            quant_bias = None
        else:
            quant_bias = layer.quant_bias

        return ops.quant_matmul(
            x=x, weight=layer.weight, deq_scale=layer.deq_scale, deq_bias=quant_bias
        )

    def process_weights_after_loading(self, layer):
        layer.aclnn_deq_scale = torch.nn.Parameter(
            torch_npu.npu_trans_quant_param(layer.deq_scale.npu()).to(device="npu"),
            requires_grad=False,
        )


class NPU_W8A8LinearMethod(LinearMethodBase):
    """Linear method for NPU quantization.

    This class search for specific quantization
    implementation supported on NPU hardware for linear methods.

    Args:
        quant_config: The NPU quantization config.
    """

    def __init__(self, quantization_config: W8A8Int8Config) -> None:
        self.quantization_config = quantization_config
        self.quant_method = (
            NPU_W8A8LinearMethodMTImpl()
            if useMindIETurbo
            else NPU_W8A8LinearMethodImpl()
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight_dict = self.quant_method.get_weight(
            input_size_per_partition, output_size_per_partition, params_dtype
        )
        for weight_name, weight_param in weight_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})
            layer.register_parameter(weight_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        pertensor_dict = self.quant_method.get_pertensor_param(params_dtype)
        for pertensor_name, pertensor_param in pertensor_dict.items():
            param = PerTensorScaleParameter(
                data=pertensor_param, weight_loader=weight_loader
            )
            # disable warning
            param.ignore_warning = True
            layer.register_parameter(pertensor_name, param)

        perchannel_dict = self.quant_method.get_perchannel_param(
            output_size_per_partition, params_dtype
        )
        for perchannel_name, perchannel_param in perchannel_dict.items():
            param = torch.nn.Parameter(perchannel_param, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(perchannel_name, param)
            set_weight_attrs(param, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.quant_method.apply(layer, x, bias)


class NPU_W8A8DynamicLinearMethodImpl:
    """Linear method for NPU W8A8_DYNAMIC."""

    def __init__(self):
        self.transpose_weight = True

    @staticmethod
    def get_weight(
        input_size: int, output_size: int, params_dtype: torch.dtype
    ) -> Dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        quant_out, dynamic_scale = torch_npu.npu_dynamic_quant(x)
        return torch_npu.npu_quant_matmul(
            quant_out,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale,
            bias=bias,
            output_dtype=original_dtype,
        )

    def process_weights_after_loading(self, layer):
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()
        layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 29)


class NPU_W8A8DynamicLinearMethod(LinearMethodBase):
    """Linear method for NPU quantization.

    This class search for specific quantization
    implementations supported on NPU hardware for linear methods.

    Args:
        quant_config: The NPU quantization config.
    """

    def __init__(self, quantization_config: W8A8Int8Config) -> None:
        self.quantization_config = quantization_config
        self.quant_method = NPU_W8A8DynamicLinearMethodImpl()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight_dict = self.quant_method.get_weight(
            input_size_per_partition, output_size_per_partition, params_dtype
        )
        for weight_name, weight_param in weight_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})
            layer.register_parameter(weight_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        pertensor_dict = self.quant_method.get_pertensor_param(params_dtype)
        for pertensor_name, pertensor_param in pertensor_dict.items():
            param = PerTensorScaleParameter(
                data=pertensor_param, weight_loader=weight_loader
            )
            # disable warning
            param.ignore_warning = True
            layer.register_parameter(pertensor_name, param)

        perchannel_dict = self.quant_method.get_perchannel_param(
            output_size_per_partition, params_dtype
        )
        for perchannel_name, perchannel_param in perchannel_dict.items():
            param = torch.nn.Parameter(perchannel_param, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(perchannel_name, param)
            set_weight_attrs(param, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.quant_method.apply(layer, x, bias)


class NPU_W8A8MoEMethod(FusedMoEMethodBase):
    """MoE method for NPU quantization.

    This class search for specific quantization
    implementations supported on NPU hardware for moe methods.

    Args:
        quant_config: The NPU quantization config.
    """

    def __init__(self, quantization_config: W8A8Int8Config) -> None:
        self.quantization_config = quantization_config
        self.quant_method = self

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.num_experts = num_experts
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        # weight
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
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
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        # scale
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        # offset
        w13_weight_offset = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset)
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)
        w2_weight_offset = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset)
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)

    def release_weight_cache(self, weight: torch.Tensor):
        # .contiguous() introduces additional memory overhead and needs to be released using resize_(0)
        origin_weight = weight.data.transpose(1, 2)
        new_weight = origin_weight.contiguous()
        origin_weight.untyped_storage().resize_(0)
        return new_weight

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_data = self.release_weight_cache(layer.w13_weight.data)
        layer.w13_weight = Parameter(weight_data, requires_grad=False)

        weight_data = self.release_weight_cache(layer.w2_weight.data)
        layer.w2_weight = Parameter(weight_data, requires_grad=False)

        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data.squeeze(-1).contiguous().to(torch.float32),
            requires_grad=False,
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data.squeeze(-1).contiguous(), requires_grad=False
        )
        layer.w13_weight_offset = Parameter(
            layer.w13_weight_offset.data.squeeze(-1).contiguous(), requires_grad=False
        )
        layer.w2_weight_offset = Parameter(
            layer.w2_weight_offset.data.squeeze(-1).contiguous(), requires_grad=False
        )

        if get_bool_env_var("ENABLE_ASCEND_MOE_NZ"):
            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, 29)
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, 29)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)
        output = npu_fused_experts(
            hidden_states=x,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk_ids.shape[1],
        )
        return StandardCombineInput(hidden_states=output)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        # gmm1: gate_up_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.int32,
        )[0]

        # act_fn: swiglu
        hidden_states, swiglu_out_scale = torch_npu.npu_dequant_swiglu_quant(
            x=hidden_states,
            weight_scale=layer.w13_weight_scale,
            activation_scale=hidden_states_scale,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=group_list,
            activate_left=True,
            quant_mode=1,
        )

        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            scale=[layer.w2_weight_scale.to(output_dtype)],
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=output_dtype,
        )[0]
        return hidden_states


class NPU_W4A16MoEMethod(FusedMoEMethodBase):
    """MoE method for NPU W4A16 quantization.

    This class search for specific quantization
    implementations supported on NPU hardware for moe methods.

    Args:
        quant_config: The NPU quantization config.
    """

    def __init__(self, quantization_config: W8A8Int8Config) -> None:
        self.quantization_config = quantization_config
        self.quant_method = self
        self.pack_factor = 8  # weight dtype is int4,  but use int32 to create
        target = (
            "MoEGMM" if "MoEGMM" in quantization_config.target_scheme_map else "Linear"
        )
        if target in quantization_config.target_scheme_map:
            self.group_size = quantization_config.target_scheme_map[target][
                "weights"
            ].group_size
        else:
            self.group_size = 128
        logger.warning_once("NPU_W4A16MoEMethod !!!")

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.num_experts = num_experts
        if (
            extra_weight_attrs.get(
                "intermediate_size_full", intermediate_size_per_partition
            )
            // intermediate_size_per_partition
            > 1
        ):
            quant_method = FusedMoeWeightScaleSupported.GROUP.value
        else:
            quant_method = FusedMoeWeightScaleSupported.CHANNEL.value
        extra_weight_attrs.update({"quant_method": quant_method})
        # weight
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # scale
        weight_scale_dtype = torch.bfloat16
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # offset
        w13_weight_offset = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset)
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)

        w2_weight_offset = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset)
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)

    def pack_to_int32(self, weight: torch.Tensor):
        assert weight.dim() == 3
        if weight.dtype == torch.int32:
            # pack 8 int4 to int32, we use a int32 to represent a int4
            assert (
                weight.shape[-1] % 8 == 0
            ), "the last dim of weight needs to be divided by 8"
            new_weight = torch_npu.npu_convert_weight_to_int4pack(weight.flatten(0, 1))
            new_weight = new_weight.view(weight.shape[0], weight.shape[1], -1)
        elif weight.dtype == torch.int8:
            # pack 4 int8(int4*2) to int32, because in pytorch, we need to use int32 to represent int4
            assert (
                weight.shape[-1] % 4 == 0
            ), "the last dim of weight needs to be divided by 4"
            new_weight = weight.view(torch.int32).contiguous()
        else:
            raise ValueError(f"{weight.dtype=} is not supported !")
        return new_weight

    def unpack_from_int32(
        self,
        value: torch.Tensor,
        num_bits: int,
        shape: torch.Size = None,
        packed_dim=1,
    ) -> torch.Tensor:
        """
        Unpacks a tensor of packed int32 weights into individual int8s, maintaining the
        original bit range.

        Return tensors in int8

        :param value: tensor to unpack
        :param num_bits: number of bits to unpack each data point into
        :param shape: shape to unpack into, used to remove padding
        :returns: unpacked int8 tensor
        """
        if value.dtype is not torch.int32:
            raise ValueError(
                f"Expected {torch.int32} but got {value.dtype}, Aborting unpack."
            )

        if num_bits > 8:
            raise ValueError("Unpacking is only supported for less than 8 bits")

        pack_factor = 32 // num_bits

        # unpack
        mask = (1 << num_bits) - 1

        if packed_dim == 1:
            unpacked = torch.zeros(
                (value.shape[0], value.shape[1] * pack_factor),
                device=value.device,
                dtype=torch.int32,
            )
            for i in range(pack_factor):
                unpacked[:, i::pack_factor] = (value >> (num_bits * i)) & mask

            # remove padding
            if shape is not None:
                original_row_size = int(shape[1])
                unpacked = unpacked[:, :original_row_size]
        else:
            unpacked = torch.zeros(
                (value.shape[0] * pack_factor, value.shape[1]),
                device=value.device,
                dtype=torch.int32,
            )
            for i in range(pack_factor):
                unpacked[i::pack_factor, :] = (value >> (num_bits * i)) & mask

            # remove padding
            original_row_size = int(shape[0])
            unpacked = unpacked[:original_row_size, :]

        # bits are packed in unsigned format, reformat to signed
        # update the value range from unsigned to signed
        offset = pow(2, num_bits) // 2
        unpacked = (unpacked - offset).to(torch.int8)

        return unpacked

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13_weight_scale = layer.w13_weight_scale.data.transpose(-1, -2).contiguous()
        w2_weight_scale = layer.w2_weight_scale.data.transpose(-1, -2).contiguous()
        layer.w13_weight_scale = Parameter(w13_weight_scale, requires_grad=False)
        layer.w2_weight_scale = Parameter(w2_weight_scale, requires_grad=False)

        layer.w13_weight_offset = Parameter(
            layer.w13_weight_offset.data.transpose(-1, -2).contiguous(),
            requires_grad=False,
        )
        layer.w2_weight_offset = Parameter(
            layer.w2_weight_offset.data.transpose(-1, -2).contiguous(),
            requires_grad=False,
        )

        # w = [n, k // 8]  --> [k, n // 8]
        # w13_weight = layer.w13_weight.data.transpose(1, 2).contiguous()
        # w2_weight = layer.w2_weight.data.transpose(1, 2).contiguous()
        unpacked_w13_weight = (
            self.unpack_from_int32(layer.w13_weight.data.flatten(0, 1), 4)
            .view(layer.w13_weight.data.shape[0], layer.w13_weight.data.shape[1], -1)
            .transpose(1, 2)
            .contiguous()
            .int()
        )
        unpacked_w2_weight = (
            self.unpack_from_int32(layer.w2_weight.data.flatten(0, 1), 4)
            .view(layer.w2_weight.data.shape[0], layer.w2_weight.data.shape[1], -1)
            .transpose(1, 2)
            .contiguous()
            .int()
        )

        w13_weight = self.pack_to_int32(unpacked_w13_weight)
        w2_weight = self.pack_to_int32(unpacked_w2_weight)

        layer.w13_weight = Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(w2_weight, requires_grad=False)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)
        output = npu_fused_experts(
            hidden_states=x,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_offset=layer.w13_weight_offset,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_offset=layer.w2_weight_offset,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk_ids.shape[1],
            use_wna16=True,
        )
        return StandardCombineInput(hidden_states=output)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        if hidden_states_scale is None:
            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[layer.w13_weight],
                antiquant_scale=[layer.w13_weight_scale],
                antiquant_offset=[layer.w13_weight_offset],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=output_dtype,
            )[0]

            # act_fn: swiglu
            hidden_states = torch_npu.npu_swiglu(hidden_states)

            # gmm2: down_proj
            out_hidden = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[layer.w2_weight],
                antiquant_scale=[layer.w2_weight_scale],
                antiquant_offset=[layer.w2_weight_offset],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=output_dtype,
            )[0]
        else:
            raise ValueError(
                "when weight is int4, hidden_states only supports non-quant dtype!"
            )

        return out_hidden

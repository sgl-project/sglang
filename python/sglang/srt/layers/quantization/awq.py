# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import (
    is_cuda,
    cpu_has_amx_support,
    is_cpu,
    use_intel_amx_backend,
)
from sglang.srt.layers.amx_utils import _amx_process_packed_qweight_after_loading
_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
import os
SGLANG_USE_CPU_W4A8 = os.getenv("SGLANG_USE_CPU_W4A8", "0") == "1"
if _is_cuda:
    from sgl_kernel import awq_dequantize

logger = logging.getLogger(__name__)


def is_layer_skipped_awq(prefix: str, modules_to_not_convert: List[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


class AWQConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        modules_to_not_convert: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.modules_to_not_convert = modules_to_not_convert or []

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits."
            )
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (
            f"AWQConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_name(self) -> str:
        return "awq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(weight_bits, group_size, zero_point, modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["LinearMethodBase"]:

        if isinstance(layer, LinearBase):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            return AWQLinearMethod(self)
        # elif isinstance(layer, FusedMoE):
        #     return Int4CPUMoEMethod(self)

        return None


class AWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config

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
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        weight_loader = extra_weight_attrs.get("weight_loader")
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu:
            assert (
                _is_cpu_amx_available
            ), "AWQLinearMethod on CPU requires that CPU has AMX support"
            _amx_process_packed_qweight_after_loading(layer, ["qweight", "qzeros", "scales"])
        else:
            layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
            layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
            layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)
        
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_intel_amx_backend(layer):
            if SGLANG_USE_CPU_W4A8:
                return torch.ops.sgl_kernel.da8w4_linear_cpu_with_quant(x, layer.qweight, layer.scales, layer.qzeros, layer.compensation, bias, torch.bfloat16)
            else:
                return torch.ops.sgl_kernel.int4_w4a16_linear(
                    x, layer.qweight, layer.qzeros, layer.scales, bias
                )

        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        out = awq_dequantize(qweight, scales, qzeros)
        out = torch.matmul(reshaped_x, out)

        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)


# class Int4CPUMoEMethod(FusedMoEMethodBase):

#     def __init__(self, quant_config: Int4CPUConfig):
#         self.quant_config = quant_config

#     # vllm.model_executor.layers.quantization.awq_marlin.AWQMoEMethod
#     def create_weights(
#         self,
#         layer: nn.Module,
#         num_experts: int,
#         hidden_size: int,
#         intermediate_size: int,
#         params_dtype: torch.dtype,
#         **extra_weight_attrs,
#     ):
#         extra_weight_attrs.update(
#             {
#                 "is_transposed": True,
#                 "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
#             }
#         )

#         w13_qweight = nn.Parameter(
#             torch.empty(
#                 num_experts,
#                 hidden_size,
#                 2 * intermediate_size // self.quant_config.pack_factor,
#                 dtype=torch.int32,
#             ),
#             requires_grad=False,
#         )
#         layer.register_parameter("w13_qweight", w13_qweight)
#         set_weight_attrs(w13_qweight, extra_weight_attrs)

#         w2_qweight = nn.Parameter(
#             torch.empty(
#                 num_experts,
#                 intermediate_size,
#                 hidden_size // self.quant_config.pack_factor,
#                 dtype=torch.int32,
#             ),
#             requires_grad=False,
#         )
#         layer.register_parameter("w2_qweight", w2_qweight)
#         set_weight_attrs(w2_qweight, extra_weight_attrs)

#         num_groups_w13 = hidden_size // self.quant_config.group_size
#         num_groups_w2 = intermediate_size // self.quant_config.group_size

#         # WEIGHT_SCALES
#         # Allocate 2 scales for w1 and w3 respectively.
#         w13_scales = nn.Parameter(
#             torch.empty(
#                 num_experts, num_groups_w13, intermediate_size * 2, dtype=params_dtype
#             ),
#             requires_grad=False,
#         )
#         layer.register_parameter("w13_scales", w13_scales)
#         set_weight_attrs(w13_scales, extra_weight_attrs)

#         w2_scales = nn.Parameter(
#             torch.empty(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
#             requires_grad=False,
#         )
#         layer.register_parameter("w2_scales", w2_scales)
#         set_weight_attrs(w2_scales, extra_weight_attrs)

#         # WEIGHT_ZERO_POINT
#         # Allocate 2 zero points for w1 and w3 respectively.
#         w13_qzeros = nn.Parameter(
#             torch.empty(
#                 num_experts,
#                 num_groups_w13,
#                 2 * intermediate_size // self.quant_config.pack_factor,
#                 dtype=torch.int32,
#             ),
#             requires_grad=False,
#         )
#         layer.register_parameter("w13_qzeros", w13_qzeros)
#         set_weight_attrs(w13_qzeros, extra_weight_attrs)

#         w2_qzeros = nn.Parameter(
#             torch.empty(
#                 num_experts,
#                 num_groups_w2,
#                 hidden_size // self.quant_config.pack_factor,
#                 dtype=torch.int32,
#             ),
#             requires_grad=False,
#         )
#         layer.register_parameter("w2_qzeros", w2_qzeros)
#         set_weight_attrs(w2_qzeros, extra_weight_attrs)

#     def process_weights_after_loading(self, layer: nn.Module) -> None:
#         w13_qweight, w13_scales_zeros = _autoawq_to_int4pack(
#             layer.w13_qweight.data, layer.w13_qzeros.data, layer.w13_scales.data
#         )
#         del layer.w13_qzeros
#         del layer.w13_scales
#         layer.w13_qweight = nn.Parameter(w13_qweight, requires_grad=False)
#         layer.w13_scales_zeros = nn.Parameter(w13_scales_zeros, requires_grad=False)

#         w2_qweight, w2_scales_zeros = _autoawq_to_int4pack(
#             layer.w2_qweight.data, layer.w2_qzeros.data, layer.w2_scales.data
#         )
#         del layer.w2_qzeros
#         del layer.w2_scales
#         layer.w2_qweight = nn.Parameter(w2_qweight, requires_grad=False)
#         layer.w2_scales_zeros = nn.Parameter(w2_scales_zeros, requires_grad=False)

#         layer.use_intel_amx_backend = False

#     def apply(
#         self,
#         layer: torch.nn.Module,
#         x: torch.Tensor,
#         router_logits: torch.Tensor,
#         top_k: int,
#         renormalize: bool,
#         use_grouped_topk: bool,
#         topk_group: Optional[int] = None,
#         num_expert_group: Optional[int] = None,
#         custom_routing_function: Optional[Callable] = None,
#         correction_bias: Optional[torch.Tensor] = None,
#         activation: str = "silu",
#         inplace: bool = True,
#         no_combine: bool = False,
#     ):
#         from sgl_kernel.cpu import silu_and_mul

#         # Expert selection
#         topk_weights, topk_ids = select_experts(
#             hidden_states=x,
#             router_logits=router_logits,
#             use_grouped_topk=use_grouped_topk,
#             top_k=top_k,
#             renormalize=renormalize,
#             topk_group=topk_group,
#             num_expert_group=num_expert_group,
#             custom_routing_function=custom_routing_function,
#             correction_bias=correction_bias,
#         )

#         # Ref code from https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/e0828e3cc0a03408724b80c3cc92c8e072db8d01/modeling_deepseek.py#L589
#         len_experts = layer.num_experts

#         cnts = topk_ids.new_zeros((topk_ids.shape[0], len_experts))
#         cnts.scatter_(1, topk_ids.to(torch.int64), 1)
#         tokens_per_expert = cnts.sum(dim=0)
#         idxs = topk_ids.view(-1).argsort()

#         sorted_tokens = x[idxs // topk_ids.shape[1]]
#         tokens_per_expert = tokens_per_expert.tolist()

#         if activation == "silu":
#             act = silu_and_mul
#         else:
#             raise ValueError(f"Unsupported activation: {activation=}")

#         outputs = []
#         start_idx = 0
#         for i, num_tokens in enumerate(tokens_per_expert):
#             end_idx = start_idx + num_tokens
#             if num_tokens == 0:
#                 continue
#             tokens_for_this_expert = sorted_tokens[start_idx:end_idx]

#             gate_up = torch._weight_int4pack_mm_for_cpu(
#                 tokens_for_this_expert,
#                 layer.w13_qweight[i],
#                 self.quant_config.group_size,
#                 layer.w13_scales_zeros[i],
#             )
#             gate_up = act(gate_up)
#             expert_out = torch._weight_int4pack_mm_for_cpu(
#                 gate_up,
#                 layer.w2_qweight[i],
#                 self.quant_config.group_size,
#                 layer.w2_scales_zeros[i],
#             )
#             outputs.append(expert_out)
#             start_idx = end_idx

#         outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
#         new_x = torch.empty_like(outs)

#         new_x[idxs] = outs
#         final_out = (
#             new_x.view(*topk_ids.shape, -1)
#             .type(topk_weights.dtype)
#             .mul_(topk_weights.unsqueeze(dim=-1))
#             .sum(dim=1)
#             .type(new_x.dtype)
#         )
#         return final_out
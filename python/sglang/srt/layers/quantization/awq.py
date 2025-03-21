# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List, Optional, Union

import torch

from sglang.srt.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.layers.parameter import PackedvLLMParameter, GroupQuantScaleParameter
from sglang.srt.utils import is_cuda

from sgl_kernel import awq_dequantize

_is_cuda = is_cuda()

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
                f"AWQ, but got {self.weight_bits} bits.")
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (f"AWQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"zero_point={self.zero_point}, "
                f"modules_to_not_convert={self.modules_to_not_convert})")
    
    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_name(self) -> str:
        return "awq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

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
            config, ["modules_to_not_convert"], None)
        return cls(weight_bits, group_size, zero_point, modules_to_not_convert)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["LinearMethodBase"]:
        
        if isinstance(layer, LinearBase):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            return AWQLinearMethod(self)
        return None
    
# class AWQMarlinConfig(QuantizationConfig):
#     """Config class for AWQ Marlin"""

#     if VLLM_AVAILABLE:
#         from vllm.scalar_type import scalar_types

#         # (num_bits, is_sym) -> quant_type
#         TYPE_MAP = {
#             (4, True): scalar_types.uint4b8,
#             (8, True): scalar_types.uint8b128,
#         }
#     else:
#         raise ImportError("vllm is not installed")

#     def __init__(self, weight_bits: int, group_size: int, zero_point: bool,
#                  lm_head_quantized: bool,
#                  modules_to_not_convert: Optional[List[str]],
#                  full_config: Dict[str, Any]) -> None:
#         from vllm.model_executor.layers.quantization.utils.marlin_utils import verify_marlin_supported

#         super().__init__()
#         self.pack_factor = 32 // weight_bits  # packed into int32
#         self.group_size = group_size
#         self.zero_point = zero_point
#         self.lm_head_quantized = lm_head_quantized
#         self.weight_bits = weight_bits
#         self.modules_to_not_convert = modules_to_not_convert or []
#         self.full_config = full_config

#         if self.weight_bits not in self.TYPE_MAP:
#             raise ValueError(f"Unsupported num_bits = {self.weight_bits}. "
#                              f"Supported num_bits = {self.TYPE_MAP.keys()}")

#         self.quant_type = self.TYPE_MAP[self.weight_bits]

#         verify_marlin_supported(self.quant_type,
#                                 group_size=self.group_size,
#                                 has_zp=self.zero_point)

#     def __repr__(self) -> str:
#         return (f"AWQMarlinConfig(quant_type={self.quant_type}, "
#                 f"group_size={self.group_size}, "
#                 f"zero_point={self.zero_point}, "
#                 f"lm_head_quantized={self.lm_head_quantized}, "
#                 f"modules_to_not_convert={self.modules_to_not_convert})")

#     @classmethod
#     def get_name(cls) -> str:
#         return "awq_marlin"

#     @classmethod
#     def get_supported_act_dtypes(cls) -> List[torch.dtype]:
#         return [torch.half, torch.bfloat16]

#     @classmethod
#     def get_min_capability(cls) -> int:
#         return 80

#     @classmethod
#     def get_config_filenames(cls) -> List[str]:
#         return ["quantize_config.json"]

#     @classmethod
#     def from_config(cls, config: Dict[str, Any]) -> "AWQMarlinConfig":
#         weight_bits = cls.get_from_keys(config, ["bits"])
#         group_size = cls.get_from_keys(config, ["group_size"])
#         zero_point = cls.get_from_keys(config, ["zero_point"])
#         lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
#                                                  default=False)
#         modules_to_not_convert = cls.get_from_keys_or(
#             config, ["modules_to_not_convert"], None)
#         return cls(weight_bits, group_size, zero_point, lm_head_quantized,
#                    modules_to_not_convert, config)

#     @classmethod
#     def override_quantization_method(cls, hf_quant_cfg,
#                                      user_quant) -> Optional[str]:
#         can_convert = cls.is_awq_marlin_compatible(hf_quant_cfg)
#         is_valid_user_quant = (user_quant is None or user_quant == "marlin"
#                                or user_quant == "awq_marlin")

#         if can_convert and is_valid_user_quant:
#             msg = ("The model is convertible to {} during runtime."
#                    " Using {} kernel.".format(cls.get_name(), cls.get_name()))
#             logger.info(msg)
#             return cls.get_name()

#         if can_convert and user_quant == "awq":
#             logger.info("Detected that the model can run with awq_marlin"
#                         ", however you specified quantization=awq explicitly,"
#                         " so forcing awq. Use quantization=awq_marlin for"
#                         " faster inference")
#         return None

#     def get_quant_method(self, layer: torch.nn.Module,
#                          prefix: str) -> Optional["QuantizeMethodBase"]:
#         from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

#         if (isinstance(layer, LinearBase) or
#             (isinstance(layer, ParallelLMHead) and self.lm_head_quantized)):
#             if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
#                 return UnquantizedLinearMethod()
#             from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinLinearMethod
#             return AWQMarlinLinearMethod(self)
#         elif isinstance(layer, FusedMoE):
#             from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Method
#             return MoeWNA16Method(self)
#         return None

#     @classmethod
#     def is_awq_marlin_compatible(cls, quant_config: Dict[str, Any]):
#         from vllm.model_executor.layers.quantization.utils.marlin_utils import check_marlin_supported
        
#         # Extract data from quant config.
#         quant_method = quant_config.get("quant_method", "").lower()
#         num_bits = quant_config.get("bits")
#         group_size = quant_config.get("group_size")
#         zero_point = quant_config.get("zero_point")

#         if not _is_cuda:
#             return False

#         if quant_method != "awq":
#             return False

#         # If we cannot find the info needed in the config, cannot convert.
#         if (num_bits is None or group_size is None or zero_point is None):
#             return False

#         if num_bits not in cls.TYPE_MAP:
#             return False

#         return check_marlin_supported(quant_type=cls.TYPE_MAP[num_bits],
#                                       group_size=group_size,
#                                       has_zp=zero_point)

class AWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """
    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

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
            weight_loader=weight_loader)

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
            weight_loader=weight_loader)

        scales = GroupQuantScaleParameter(data=torch.empty(
            input_size_per_partition // self.quant_config.group_size,
            output_size_per_partition,
            dtype=params_dtype,
        ),
                                          input_dim=0,
                                          output_dim=1,
                                          weight_loader=weight_loader)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.qweight = torch.nn.Parameter(layer.qweight.data,
                                           requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data,
                                          requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data,
                                          requires_grad=False)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
        reshaped_x = x.reshape(-1, x.shape[-1])

        out = awq_dequantize(qweight, scales, qzeros)
        out = torch.matmul(reshaped_x, out)
        # # num_tokens >= threshold
        # FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256

        # if FP16_MATMUL_HEURISTIC_CONDITION:
        #     out = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
        #     out = torch.matmul(reshaped_x, out)
        # else:
        #     out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros,
        #                        pack_factor)
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
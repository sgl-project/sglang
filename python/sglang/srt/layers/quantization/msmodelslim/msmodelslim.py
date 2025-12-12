from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import torch
from compressed_tensors.quantization import QuantizationStrategy
from pydantic import BaseModel

# from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
#     NPUW4A8Int4DynamicMoEMethod,
#     NPUW4A16Int4DynamicMoEMethod,
#     NPUW8A8Int8DynamicMoEMethod,
# )
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    _NPULinearMethodBase
    # NPUW8A8Int8DynamicLinearMethod,
    # NPUW8A8Int8LinearMethod,
)
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.msmodelslim.msmodelslim_moe import (
    ModelSlimMoEMethod,
)
from sglang.srt.layers.quantization.msmodelslim.schemes import (
    ModelSlimScheme,
    ModelSlimW8A8Int8,
    ModelSlimW4A4Int4,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    find_matched_target,
    is_activation_quantization_format,
    should_ignore_layer
)
#from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import apply_module_patch

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
        from sgl_kernel_npu.norm.add_rmsnorm_bias import add_rmsnorm_bias

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

        out = torch.ops.npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]
        out = out + self.bias
        return out.to(x.dtype)

    return _rmsnorm_forward_oot


class ModelSlimConfig(QuantizationConfig):
    """
    Config class for ModelSlim Quantization, a NPU-specific quantization type.
    """

    def __init__(self, quant_config: Dict[str, Any] = {}):
        super().__init__()
        self.quant_description = quant_config
        # self.is_dynamic = quant_config.get("is_dynamic", False)
        # self.is_moe_w4_dynamic = False
        ignore = cast(List[str], quant_config.get("ignore", []))
        self.ignore = ignore if ignore is not None else []
        packed_modules_mapping = quant_config.get("packed_modules_mapping", {})
        self.packed_modules_mapping = (
            packed_modules_mapping if packed_modules_mapping is not None else {}
        )
        # self.target_scheme_map = (
        #     CompressedTensorsConfig._quantization_scheme_map_from_config(
        #         config=quant_config
        #     )
        # )
        # target = "MoEGMM" if "MoEGMM" in self.target_scheme_map else "Linear"
        # target_scheme = self.target_scheme_map.get(target, None)
        # if target_scheme is None:
        #     self.is_moe_w4_dynamic = False
        # else:
        #     weight_quant = target_scheme.get("weights")
        #     input_quant = target_scheme.get("input_activations")
        #     self.is_moe_w4_dynamic = self.is_dynamic_token_w4(weight_quant, input_quant)
        #     self.is_moe_input_quant = input_quant

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
    
    def get_linear_method(self) -> ModelSlimLinearMethod:
        return ModelSlimLinearMethod(self)

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_name(cls) -> str:
        return "modelslim"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        filenames = ["quant_model_description.json"]
        return filenames

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> ModelSlimConfig:
        return cls(config)

    def get_quant_method(
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
            # self.is_dynamic = (
            #     self.quant_description[prefix_in_quant_config + ".weight"]
            #     == "W8A8_DYNAMIC"
            # )

            if self.is_layer_skipped(prefix, packed_modules_mapping_subset):
                return UnquantizedLinearMethod()
            scheme = self.get_scheme(layer=layer, layer_name=prefix_in_quant_config)
            if scheme is None:
                raise NotImplementedError("At the moment SGLang on Ascend supports only w4a4 dynamic, w8a8 static/dynamic linear schemes.")
            layer.scheme = scheme
            return (
                ModelSlimLinearMethod(self)
            )
        elif isinstance(layer, FusedMoE):
            return ModelSlimMoEMethod.get_moe_method(self, layer, prefix)
        return None

    def _get_scheme_from_parts(
            self, layer_name: str,
        ) -> ModelSlimScheme:

        quant_type = self.quant_description[layer_name + '.weight']
        if quant_type == "W8A8_DYNAMIC":
            return ModelSlimW8A8Int8(
                quant_config=self.quant_description,
                prefix=layer_name
            )
        elif quant_type == "W4A4_DYNAMIC":
            return ModelSlimW4A4Int4(
                quant_config=self.quant_description,
                prefix=layer_name
            )

            # Detect If Mixed Precision
            # if self._is_wNa16_group_channel(weight_quant, input_quant):
            #     if (
            #         self.quant_format == CompressionFormat.pack_quantized.value
            #         and weight_quant.num_bits in WNA16_SUPPORTED_BITS
            #     ):
            #         return CompressedTensorsWNA16(
            #             num_bits=weight_quant.num_bits,
            #             strategy=weight_quant.strategy,
            #             group_size=weight_quant.group_size,
            #             actorder=weight_quant.actorder,
            #         )
            #     else:
            #         raise ImportError(
            #             "Other method (CompressedTensorsW4A16Sparse24) is not supported now"
            #         )

            #if is_activation_quantization_format(self.quant_format):
                # if self._is_fp8_w8a8(weight_quant, input_quant):
                #     is_fp8_w8a8_supported = self._check_scheme_supported(
                #         CompressedTensorsW8A8Fp8.get_min_capability(), error=False
                #     )
                #     if is_fp8_w8a8_supported:
                #         return CompressedTensorsW8A8Fp8(
                #             strategy=weight_quant.strategy,
                #             is_static_input_scheme=(
                #                 input_quant and not input_quant.dynamic
                #             ),
                #         )
                #     else:
                #         # note: input_quant will be present for converted models;
                #         # will be ignored during inference post loading
                #         return CompressedTensorsW8A16Fp8(
                #             strategy=weight_quant.strategy,
                #             is_static_input_scheme=not input_quant.dynamic,
                #         )

                # # note: input_quant can be None
                # if self._is_fp8_w8a16(weight_quant, input_quant):
                #     is_static_input_scheme = input_quant and not input_quant.dynamic
                #     return CompressedTensorsW8A16Fp8(
                #         strategy=weight_quant.strategy,
                #         is_static_input_scheme=is_static_input_scheme,
                #     )

            #raise NotImplementedError("No msmodelslim compatible scheme was found.")
    
    def get_scheme(
            self, layer: torch.nn.Module, layer_name: Optional[str] = None
        ) -> Optional[ModelSlimScheme]:
            """
            get_scheme method adjusted for modelslim, taken from
            python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py
            """
            # if self.target_scheme_map:
            #     matched_target = find_matched_target(
            #         layer_name=layer_name,
            #         module=layer,
            #         targets=self.target_scheme_map.keys(),
            #         fused_mapping=self.packed_modules_mapping,
            #     )

            #     scheme_dict = self.target_scheme_map[matched_target]
            #     weight_quant = scheme_dict.get("weights")
            #     input_quant = scheme_dict.get("input_activations")
            # else:
                # Find the quant_scheme
            scheme = self._get_scheme_from_parts(  # type: ignore
                # weight_quant=weight_quant,
                # input_quant=input_quant,
                layer_name=layer_name,
            )

            # Ascend doesn't support device capability
            # self._check_scheme_supported(scheme.get_min_capability())
            logger.debug("Using scheme: %s for %s", scheme.__class__.__name__, layer_name)
            return scheme

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

    # def is_dynamic_token_w4(self, weight_quant, input_quant) -> bool:
    #     is_w4 = weight_quant.num_bits == 4
    #     weight_strategy = (
    #         weight_quant.strategy == QuantizationStrategy.TENSOR.value
    #         or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
    #         or weight_quant.strategy == QuantizationStrategy.GROUP.value
    #     )
    #     if input_quant is not None:
    #         is_token = (
    #             weight_strategy
    #             and input_quant.strategy == QuantizationStrategy.TOKEN.value
    #         )
    #         is_dynamic = not weight_quant.dynamic and input_quant.dynamic
    #     else:
    #         is_token = weight_strategy
    #         is_dynamic = not weight_quant.dynamic

    #     # Both symmetric and asymmetric input quantization supported.
    #     # Only symmetric weight quantization supported.
    #     return is_w4 and weight_quant.symmetric and is_token and is_dynamic

    # def _is_static_tensor_w8a8(
    #     self, weight_quant: BaseModel, input_quant: BaseModel
    # ) -> bool:
    #     is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
    #     weight_strategy = (
    #         weight_quant.strategy == QuantizationStrategy.TENSOR.value
    #         or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
    #     )
    #     is_tensor = (
    #         weight_strategy
    #         and input_quant.strategy == QuantizationStrategy.TENSOR.value
    #     )
    #     is_static = not weight_quant.dynamic and not input_quant.dynamic

    #     # Both symmetric and asymmetric input quantization supported.
    #     # Only symmetric weight quantization supported.
    #     return is_8_bits and is_tensor and weight_quant.symmetric and is_static

    # def _is_dynamic_token_w8a8(
    #     self, weight_quant: BaseModel, input_quant: BaseModel
    # ) -> bool:
    #     is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
    #     weight_strategy = (
    #         weight_quant.strategy == QuantizationStrategy.TENSOR.value
    #         or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
    #     )
    #     is_token = (
    #         weight_strategy and input_quant.strategy == QuantizationStrategy.TOKEN.value
    #     )
    #     is_dynamic = not weight_quant.dynamic and input_quant.dynamic

    #     # Both symmetric and asymmetric input quantization supported.
    #     # Only symmetric weight quantization supported.
    #     return is_8_bits and is_token and weight_quant.symmetric and is_dynamic


class ModelSlimLinearMethod(_NPULinearMethodBase):

    def __init__(self, quantization_config: ModelSlimConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

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
        """
        Use the ModelSlimScheme associated with each layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Use the output of create_weights and the CompressedTensorsScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """

        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias)

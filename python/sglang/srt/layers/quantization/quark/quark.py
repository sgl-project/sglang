# SPDX-License-Identifier: Apache-2.0

import fnmatch
from typing import Any, Optional, cast

import torch

import logging
from sglang.srt.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from sglang.srt.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig, QuantizeMethodBase)
from sglang.srt.layers.quantization.quark.schemes import QuarkScheme, QuarkW4A4MXFP4
from sglang.srt.layers.quantization.quark.utils import deep_compare, should_ignore_layer
from vllm.platforms import current_platform

__all__ = ["QuarkLinearMethod"]

logger = logging.getLogger(__name__)


class QuarkConfig(QuantizationConfig):

    def __init__(self,
                 quant_config: dict[str, Any],
                 pack_method: str = "reorder"):
        super().__init__()
        self.quant_config = quant_config
        self.pack_method = pack_method

    def get_linear_method(self) -> "QuarkLinearMethod":
        return QuarkLinearMethod(self)

    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def get_name(self) -> str:
        return "quark"

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        # Check if the layer is skipped for quantization.
        exclude_layers = cast(list[str], self.quant_config.get("exclude"))
        if should_ignore_layer(prefix,
                               ignore=exclude_layers,
                               fused_mapping=self.packed_modules_mapping):
            return UnquantizedLinearMethod()
        
        if isinstance(layer, LinearBase):
            scheme = self.get_scheme(layer=layer, layer_name=prefix)
            layer.scheme = scheme
            return QuarkLinearMethod(self)

        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QuarkConfig":
        export_config = config.get("export")
        if export_config is None:
            raise ValueError("The export key should be included in "
                             "the configurations of Quark quantized model")
        pack_method = cast(str, export_config.get("pack_method"))

        return cls(quant_config=config,
                   pack_method=pack_method)

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def _check_scheme_supported(self,
                                min_capability: int,
                                error: bool = True) -> bool:
        capability_tuple = current_platform.get_device_capability()

        if capability_tuple is not None:
            capability = capability_tuple.to_int()
            supported = capability >= min_capability
            if error and not supported:
                raise RuntimeError(
                    "Quantization scheme is not supported for ",
                    f"the current GPU. Min capability: {min_capability}. ",
                    f"Current capability: {capability}.")
            return supported
        else:
            return False

    def _is_mx_fp4(self, weight_quant: Optional[dict[str, Any]],
                   input_quant: Optional[dict[str, Any]]) -> bool:
        # Confirm weights and input quantized.
        if weight_quant is None or input_quant is None:
            logger.debug("Quark model is not in MX-FP4 format: "
                         "weight_quant or input_quant not set")
            return False

        # Input and weight dtype needs to be fp4.
        if weight_quant.get("dtype") != "fp4" or input_quant.get(
                "dtype") != "fp4":
            logger.debug("Quark model is not in MX-FP4 format: dtype not fp4")
            return False

        # Input and weight qscheme needs to be per group.
        if weight_quant.get("qscheme") != "per_group" or input_quant.get(
                "qscheme") != "per_group":
            logger.debug("Quark model is not in MX-FP4 format: not per_group")
            return False

        # Input and weight group size needs to be 32.
        if weight_quant.get("group_size") != 32 or input_quant.get(
                "group_size") != 32:
            logger.debug(
                "Quark model is not in MX-FP4 format: not group_size=32")
            return False

        # Weights need to use static quantization.
        if weight_quant.get("is_dynamic") is True:
            logger.debug(
                "Quark model is not in MX-FP4 format: not weight static")
            return False

        # Activations need to use dynamic quantization.
        if input_quant.get("is_dynamic") is False:
            logger.debug(
                "Quark model is not in MX-FP4 format: not activation dynamic")
            return False

        # Activations and weight scales need to be in e8m0 format.
        if weight_quant.get("scale_format") != "e8m0" or input_quant.get(
                "scale_format") != "e8m0":
            logger.debug(
                "Quark model is not in MX-FP4 format: not scale_format e8m0")
            return False

        return True

    def _find_matched_config(self, layer_name: str,
                             module: torch.nn.Module) -> dict[str, Any]:

        proj_name = layer_name.split(".")[-1]
        if proj_name in self.packed_modules_mapping:
            shard_proj_names = self.packed_modules_mapping[proj_name]

            # Convert fused_name --> [shard_names]
            shard_names = [
                layer_name.replace(proj_name, shard_proj_name)
                for shard_proj_name in shard_proj_names
            ]
            shard_configs = [
                self._find_matched_config(shard_name, module)
                for shard_name in shard_names
            ]
            if not all(
                    deep_compare(q_config, shard_configs[0])
                    for q_config in shard_configs):
                raise ValueError(
                    f"Found a different quantization configuration for "
                    f"{shard_proj_names} in {layer_name}. vLLM "
                    "requires all to use the same scheme.")
            return shard_configs[0]
        else:
            layer_quant_config = cast(
                dict[str, Any], self.quant_config.get("layer_quant_config"))
            for name_pattern in layer_quant_config:
                if fnmatch.fnmatch(layer_name, name_pattern):
                    return layer_quant_config[name_pattern]

            layer_type = cast(str, type(module))
            layer_type_quant_config = cast(
                dict[str, Any],
                self.quant_config.get("layer_type_quant_config"))
            if layer_type in layer_type_quant_config:
                return layer_type_quant_config[layer_type]

            global_quant_config = cast(
                dict[str, Any], self.quant_config.get("global_quant_config"))
            return global_quant_config

    def _get_scheme_from_config(self, config: dict[str, Any]) -> "QuarkScheme":
        if config.get("output_tensors") or config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with output_tensors "
                "and bias quantized are not supported")
        weight_config = cast(dict[str, Any], config.get("weight"))
        input_config = cast(dict[str, Any], config.get("input_tensors"))

        if self._is_mx_fp4(weight_config, input_config):
            return QuarkW4A4MXFP4(weight_config, input_config)

        raise NotImplementedError("No quark compatible scheme was found. "
                                  f"Weight config: {weight_config}, "
                                  f"Input config: {input_config}")

    def get_scheme(self, layer: torch.nn.Module,
                   layer_name: str) -> "QuarkScheme":

        layer_quant_config = self._find_matched_config(layer_name, layer)

        # Find the quant_scheme
        scheme = self._get_scheme_from_config(layer_quant_config)
        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        self._check_scheme_supported(scheme.get_min_capability())

        return scheme

class QuarkLinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: QuarkConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        """
        Use the CompressedTensorsScheme associated with each layer to create
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
            weight_loader=weight_loader)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None):
        """
        Use the output of create_weights and the CompressedTensorsScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """
        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias)

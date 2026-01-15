"""INT8 quantization implementation for multimodal_gen.

This module provides INT8 quantization support for both weights and activations,
Based on sglang/srt/layers/quantization/blockwise_int8.py implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module

from python.sglang.srt.layers.quantization.int8_utils import (
    apply_w8a8_block_int8_linear,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.base_config import QuantizeConfigBase
from sglang.multimodal_gen.runtime.layers.base_method import QuantizeMethodBase
from sglang.multimodal_gen.runtime.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.models.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = logging.getLogger(__name__)


class Int8Config(QuantizeConfigBase):
    """Configuration for INT8 quantization.

    Supports both weight-only and weight-activation quantization with
    configurable quantization strategies.
    """

    def __init__(
        self,
        is_checkpoint_int8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: List[int] = None,
    ) -> None:
        self.is_checkpoint_int8_serialized = is_checkpoint_int8_serialized
        if is_checkpoint_int8_serialized:
            logger.warning(
                "Detected int8 checkpoint. Please note that the "
                "format is experimental and subject to change."
            )
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not is_checkpoint_int8_serialized:
                raise ValueError(
                    f"The int8 quantization only supports int8-serialized checkpoint for now."
                )
            if len(weight_block_size) != 2:
                raise ValueError(
                    f"The quantization block size of weight must have 2 dimensions, but got {len(weight_block_size)} dimensions."
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    f"The block-wise quantization only supports dynamic activation scheme for now, but got {activation_scheme} activation scheme."
                )
        self.weight_block_size = weight_block_size

    @classmethod
    def get_name(cls) -> str:
        return "int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Int8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_int8_serialized = "int8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        return cls(
            is_checkpoint_int8_serialized=is_checkpoint_int8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from python.sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            if self.weight_block_size is not None:
                return BlockInt8LinearMethod(self)
            raise ValueError(
                f"weight_block_size is None,The int8 quantization only supports block-wise for now."
            )
        return None


class BlockInt8LinearMethod(LinearMethodBase):
    """Linear method for INT8.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic activation scale.

    Limitations:
    Only support block-wise int8 quantization and int8 checkpoint

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Int8Config):
        self.quant_config = quant_config
        assert self.quant_config.weight_block_size is not None
        assert self.quant_config.is_checkpoint_int8_serialized

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
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        tp_size = get_tensor_model_parallel_world_size()

        block_n, block_k = (
            self.quant_config.weight_block_size[0],
            self.quant_config.weight_block_size[1],
        )
        # Required by row parallel
        if tp_size > 1 and input_size // input_size_per_partition == tp_size:
            if input_size_per_partition % block_k != 0:
                raise ValueError(
                    f"Weight input_size_per_partition = "
                    f"{input_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )
        # Required by column parallel or enabling merged weights
        if (tp_size > 1 and output_size // output_size_per_partition == tp_size) or len(
            output_partition_sizes
        ) > 1:
            for output_partition_size in output_partition_sizes:
                if output_partition_size % block_n != 0:
                    raise ValueError(
                        f"Weight output_partition_size = "
                        f"{output_partition_size} is not divisible by "
                        f"weight quantization block_n = {block_n}."
                    )

        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
        weight_dtype = (
            torch.int8
            if self.quant_config.is_checkpoint_int8_serialized
            else params_dtype
        )

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE

        scale = BlockQuantScaleParameter(
            data=torch.empty(
                (output_size_per_partition + block_n - 1) // block_n,
                (input_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale_inv", scale)

        # INPUT ACTIVATION SCALE
        assert self.quant_config.activation_scheme == "dynamic"
        layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: Module) -> None:
        # Block quant doesn't need to process weights after loading
        # Use torch Parameter to avoid cuda graph capturing issue
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(
            layer.weight_scale_inv.data, requires_grad=False
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_w8a8_block_int8_linear(
            input=x,
            weight=layer.weight,
            block_size=self.quant_config.weight_block_size,
            weight_scale=layer.weight_scale_inv,
            input_scale=None,
            bias=bias,
        )

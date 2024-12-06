from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase
from sglang.srt.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    apply_fp8_linear,
    cutlass_fp8_supported,
    requantize_with_max_scale,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)

# Initialize logger for the module
logger = init_logger(__name__)

# Supported activation schemes for the current configuration
ACTIVATION_SCHEMES = ["static"]


class ModelOptFp8Config(QuantizationConfig):
    """Configuration class for ModelOpt FP8 quantization.

    This class handles the configuration of FP8 quantization for ModelOpt,
    including serialization support, supported activation types, and the
    compatibility checks necessary for FP8 functionality.
    """

    def __init__(self, is_checkpoint_fp8_serialized: bool = False) -> None:
        """
        Args:
            is_checkpoint_fp8_serialized (bool): Indicates if the checkpoint
                uses serialized FP8 format. If True, a warning is issued.
        """
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            logger.warning(
                "Detected ModelOpt FP8 checkpoint. Please note that the "
                "format is experimental and subject to change."
            )

    @classmethod
    def get_name(cls) -> str:
        """Returns the name of the quantization method."""
        return "modelopt"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        """Lists the supported activation data types for FP8 quantization."""
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        """Specifies the minimum hardware capability required for FP8."""
        return 89  # Assuming 89 corresponds to specific hardware support (e.g., Hopper GPUs).

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        """Provides the list of expected configuration filenames."""
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelOptFp8Config":
        """Creates a `ModelOptFp8Config` instance from a configuration dictionary.

        Args:
            config (Dict[str, Any]): The configuration dictionary.

        Returns:
            ModelOptFp8Config: The initialized configuration object.

        Raises:
            ValueError: If the quantization method is not FP8.
        """
        quant_config = cls.get_from_keys(config, ["quantization"])
        quant_method = quant_config["quant_algo"]
        is_checkpoint_fp8_serialized = "FP8" in quant_method

        if not is_checkpoint_fp8_serialized:
            raise ValueError(
                "ModelOpt currently only supports static FP8 quantization in "
                "SGLang. Please check the `hf_quant_config.json` file for "
                "your model's quantization configuration."
            )
        return cls(is_checkpoint_fp8_serialized)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        """Returns the quantization method for a given layer.

        Args:
            layer (torch.nn.Module): The module to apply quantization to.
            prefix (str): A name prefix for additional configuration.

        Returns:
            Optional[QuantizeMethodBase]: The quantization method, or None if unsupported.
        """
        if isinstance(layer, LinearBase):
            return ModelOptFp8LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        """Specifies names of activations requiring scaling for quantization.

        Returns:
            List[str]: An empty list, as scaling is not required for FP8 activations.
        """
        return []


class ModelOptFp8LinearMethod(LinearMethodBase):
    """
    Linear method for ModelOpt static FP8 quantization.

    Supports loading FP8 checkpoints with static weight and activation scales.
    Future support may include dynamic scales.

    **Limitations**:
    1. Only supports per-tensor quantization due to `torch._scaled_mm` limitations.
    2. Only supports the `float8_e4m3fn` data type.

    Args:
        quant_config (ModelOptFp8Config): The ModelOpt quantization configuration.
    """

    def __init__(self, quant_config: ModelOptFp8Config):
        super().__init__()
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()

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
        """
        Creates and registers weight, weight scale, and input scale parameters
        for FP8 quantization.

        Args:
            layer (torch.nn.Module): The target layer to which weights will be registered.
            input_size_per_partition (int): Input size for each partition.
            output_partition_sizes (List[int]): List of output sizes for each partition.
            input_size (int): Total input size (unused).
            output_size (int): Total output size (unused).
            params_dtype (torch.dtype): Data type for weights if FP8 is not serialized.
            **extra_weight_attrs: Additional attributes for weight creation, such as loaders.
        """
        del input_size, output_size  # These parameters are unused.

        # Compute sizes and weight type
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_fp8_serialized
            else params_dtype
        )

        # Assign layer attributes
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Register weight parameter
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if self.quant_config.is_checkpoint_fp8_serialized:
            # Register weight scale parameter
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            weight_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("weight_scale", weight_scale)

            # Register input scale parameter
            input_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            input_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Processes weights after loading by requantizing with the maximum scale.

        Args:
            layer (torch.nn.Module): The layer whose weights are processed.
        """
        max_w_scale, quantized_weight = requantize_with_max_scale(
            layer.weight, layer.weight_scale, layer.logical_widths
        )
        layer.weight = Parameter(quantized_weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
        layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Applies the FP8 linear transformation.

        Args:
            layer (torch.nn.Module): The layer containing the weights and scales.
            x (torch.Tensor): The input tensor.
            bias (Optional[torch.Tensor]): An optional bias tensor.

        Returns:
            torch.Tensor: The result of the linear transformation.
        """
        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
        )

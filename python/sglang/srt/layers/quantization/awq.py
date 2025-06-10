# SPDX-License-Identifier: Apache-2.0
import logging
from fractions import Fraction  # Added
from typing import Any, Callable, Dict, List, Optional, Union  # Added Callable, Union

import torch
from torch.nn import Parameter  # Added

from sglang.srt.layers.linear import LinearBase  # Kept
from sglang.srt.layers.linear import LinearMethodBase  # Kept
from sglang.srt.layers.linear import UnquantizedLinearMethod  # Kept

# Removed PackedvLLMParameter, GroupQuantScaleParameter from here, will define AWQ versions
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # Kept
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()
if _is_cuda:
    from sgl_kernel import awq_dequantize

logger = logging.getLogger(__name__)


# Copied and modified from python/sglang/srt/layers/parameter.py


def _adjust_shard_indexes_for_marlin(
    shard_size, shard_offset, marlin_tile_size
):  # Helper for _adjust_shard_indexes_for_packing
    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size


def _adjust_shard_indexes_for_packing(
    shard_size, shard_offset, packed_factor, marlin_tile_size  # Added marlin_tile_size
):
    shard_size = shard_size // packed_factor
    shard_offset = shard_offset // packed_factor
    if marlin_tile_size is not None:  # Added marlin_tile_size condition
        return _adjust_shard_indexes_for_marlin(
            shard_size=shard_size,
            shard_offset=shard_offset,
            marlin_tile_size=marlin_tile_size,
        )
    return shard_size, shard_offset


class BaseAWQParameter(Parameter):
    """
    Base parameter for AWQ linear layers. Extends the torch.nn.parameter
    by taking in a linear weight loader. Will copy the loaded weight
    into the parameter when the provided weight loader is called.
    """

    def __new__(cls, data: torch.Tensor, **kwargs):
        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(self, data: torch.Tensor, weight_loader: Callable):
        self._weight_loader = weight_loader

    @property
    def weight_loader(self):
        return self._weight_loader

    def _assert_and_load(self, loaded_weight: torch.Tensor):
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        self._assert_and_load(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        self._assert_and_load(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._assert_and_load(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._assert_and_load(loaded_weight)


class _ColumnAWQParameter(BaseAWQParameter):
    """
    Private class defining weight loading functionality
    for parameters being loaded into linear layers with column
    parallelism.
    """

    def __init__(self, output_dim: int, **kwargs):
        self._output_dim = output_dim
        super().__init__(**kwargs)

    @property
    def output_dim(self):
        return self._output_dim

    def load_column_parallel_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
    ):
        if not use_presharded_weights:
            shard_size = self.data.shape[self.output_dim]
            loaded_weight = loaded_weight.narrow(
                self.output_dim, tp_rank * shard_size, shard_size
            )
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_offset = kwargs.get("shard_offset")
        shard_size = kwargs.get("shard_size")
        tp_rank = kwargs.get("tp_rank")
        use_presharded_weights = kwargs.get("use_presharded_weights")

        # Adjusted isinstance check
        if (
            isinstance(self, (PackedColumnAWQParameter, PackedAWQParameter))
            and self.packed_dim == self.output_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size
            )

        param_data = self.data
        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
        if not use_presharded_weights:
            loaded_weight = loaded_weight.narrow(
                self.output_dim, tp_rank * shard_size, shard_size
            )
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def load_qkv_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
        **kwargs,
    ):
        shard_offset = kwargs.get("shard_offset")
        shard_size = kwargs.get("shard_size")
        shard_id = kwargs.get("shard_id")
        num_heads = kwargs.get("num_heads")

        # Adjusted isinstance check
        if (
            isinstance(self, (PackedColumnAWQParameter, PackedAWQParameter))
            and self.output_dim == self.packed_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size
            )

        param_data = self.data
        shard_id_val = (
            tp_rank if shard_id == "q" else tp_rank // num_heads
        )  # Renamed shard_id to shard_id_val to avoid conflict
        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
        if not use_presharded_weights:
            loaded_weight = loaded_weight.narrow(
                self.output_dim,
                shard_id_val * shard_size,
                shard_size,  # Used shard_id_val
            )

        assert (
            param_data.shape == loaded_weight.shape
        ), f"{param_data.shape=}, {loaded_weight.shape=}"
        param_data.copy_(loaded_weight)


class RowAWQParameter(BaseAWQParameter):
    """
    Parameter class defining weight_loading functionality
    for parameters being loaded into linear layers with row parallel functionality.
    """

    def __init__(self, input_dim: int, **kwargs):
        self._input_dim = input_dim
        super().__init__(**kwargs)

    @property
    def input_dim(self):
        return self._input_dim

    def load_row_parallel_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
    ):
        if not use_presharded_weights:
            shard_size = self.data.shape[self.input_dim]
            loaded_weight = loaded_weight.narrow(
                self.input_dim, tp_rank * shard_size, shard_size
            )

        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)


class AWQModelWeightParameter(_ColumnAWQParameter, RowAWQParameter):
    """
    Parameter class for linear layer weights. Uses both column and
    row parallelism. (Renamed from ModelWeightParameter)
    """

    pass


class PackedColumnAWQParameter(_ColumnAWQParameter):  # Added this class
    """
    Parameter for model parameters which are packed on disk
    and support column parallelism only. (Renamed from PackedColumnParameter)
    """

    def __init__(
        self,
        packed_factor: Union[int, Fraction],
        packed_dim: int,
        marlin_tile_size: Optional[int] = None,
        **kwargs,
    ):
        self._packed_factor = packed_factor
        self._packed_dim = packed_dim
        self._marlin_tile_size = marlin_tile_size
        super().__init__(**kwargs)

    @property
    def packed_dim(self):
        return self._packed_dim

    @property
    def packed_factor(self):
        return self._packed_factor

    @property
    def marlin_tile_size(self):
        return self._marlin_tile_size

    def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
        return _adjust_shard_indexes_for_packing(  # Uses the local function
            shard_size=shard_size,
            shard_offset=shard_offset,
            packed_factor=self.packed_factor,
            marlin_tile_size=self.marlin_tile_size,
        )


class PackedAWQParameter(AWQModelWeightParameter):
    """
    Parameter for model weights which are packed on disk.
    (Renamed from PackedvLLMParameter)
    """

    def __init__(
        self,
        packed_factor: Union[int, Fraction],
        packed_dim: int,
        marlin_tile_size: Optional[int] = None,
        **kwargs,
    ):
        self._packed_factor = packed_factor
        self._packed_dim = packed_dim
        self._marlin_tile_size = marlin_tile_size
        super().__init__(**kwargs)

    @property
    def packed_dim(self):
        return self._packed_dim

    @property
    def packed_factor(self):
        return self._packed_factor

    @property
    def marlin_tile_size(self):
        return self._marlin_tile_size

    def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
        return _adjust_shard_indexes_for_packing(  # Uses the local function
            shard_size=shard_size,
            shard_offset=shard_offset,
            packed_factor=self.packed_factor,
            marlin_tile_size=self.marlin_tile_size,
        )


# End of copied and modified code from parameter.py


# New FusedAWQLinearMethod
class FusedAWQLinearMethod(LinearMethodBase):
    """
    Linear method for AWQ that handles packing and unpacking internally
    using torch.nn.Parameter for qweight and qzeros.
    """

    def __init__(self, quant_config: "AWQConfig"):  # Forward reference AWQConfig
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
        # For qweight and qzeros, the packing is handled by awq_dequantize,
        # so the shape of the Parameter should be the packed shape.
        packed_output_size_per_partition = (
            output_size_per_partition // self.quant_config.pack_factor
        )

        if output_size_per_partition % self.quant_config.pack_factor != 0:
            # This check might be redundant if the one for qweight/qzeros shape is correct
            raise ValueError("The output size is not aligned with the pack factor.")

        qweight_data = torch.empty(
            input_size_per_partition,
            packed_output_size_per_partition,
            dtype=torch.int32,  # AWQ kernels expect int32 for packed weights
        )
        qweight = Parameter(qweight_data, requires_grad=False)

        qzeros_data = torch.empty(
            input_size_per_partition // self.quant_config.group_size,
            packed_output_size_per_partition,
            dtype=torch.int32,  # AWQ kernels expect int32 for packed zeros
        )
        qzeros = Parameter(qzeros_data, requires_grad=False)

        scales_data = torch.empty(
            input_size_per_partition // self.quant_config.group_size,
            output_size_per_partition,  # Scales are not packed
            dtype=params_dtype,
        )
        scales = Parameter(scales_data, requires_grad=False)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

        # Store quant_config and pack_factor on layer for apply method to access if needed
        # Or pass them to apply method if preferred, but apply signature is fixed by LinearMethodBase
        # Storing on layer is a common pattern.
        layer.quant_config = self.quant_config

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        qzeros = layer.qzeros
        scales = layer.scales

        # pack_factor should be available from quant_config stored on layer or self
        # For FusedAWQLinearMethod, quant_config is stored in self.quant_config
        pack_factor = self.quant_config.pack_factor

        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        if not _is_cuda:
            raise RuntimeError("AWQ dequantization is only supported on CUDA.")

        # The awq_dequantize kernel handles the unpacking of qweight and qzeros.
        # It expects qweight and qzeros in their packed form (int32)
        # and scales in their original form.
        dequantized_weight = awq_dequantize(qweight, scales, qzeros)

        out = torch.matmul(reshaped_x, dequantized_weight)

        if bias is not None:
            out = (
                out + bias
            )  # In-place add can cause issues with autograd if x requires grad
        return out.reshape(out_shape)


# --- Existing AWQConfig and AWQLinearMethod ---
# (Copied from the original python/sglang/srt/layers/quantization/awq.py)


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
            config, ["modules_to_not_convert"], None
        )
        return cls(weight_bits, group_size, zero_point, modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["LinearMethodBase"]:

        if isinstance(layer, LinearBase):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            # This is where one might choose between AWQLinearMethod and FusedAWQLinearMethod
            # For now, let's keep the original AWQLinearMethod as default.
            # The problem description implies creating FusedAWQLinearMethod, not necessarily using it by default.
            # Changing to FusedAWQLinearMethod as per new request.
            return FusedAWQLinearMethod(self)
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
        # Original AWQLinearMethod uses PackedAWQParameter (previously PackedvLLMParameter)
        # and GroupQuantScaleParameter (which I need to define or use an equivalent for AWQ context)
        # For now, I'll use the newly defined PackedAWQParameter and will need a AWQ version of GroupQuantScaleParameter

        # Let's define GroupQuantScaleAWQParameter, similar to GroupQuantScaleParameter from vLLM
        # For simplicity, if GroupQuantScaleParameter was _ColumnvLLMParameter + RowvLLMParameter,
        # then GroupQuantScaleAWQParameter will be _ColumnAWQParameter + RowAWQParameter.
        # Or, it could just be a BaseAWQParameter if no special loading for scales is needed beyond what BaseAWQParameter provides.
        # The original GroupQuantScaleParameter from vLLM parameter.py was:
        # class GroupQuantScaleParameter(_ColumnvLLMParameter, RowvLLMParameter): pass
        # So, I'll define GroupQuantScaleAWQParameter similarly.

        qweight = PackedAWQParameter(  # Uses the new PackedAWQParameter
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

        qzeros = PackedAWQParameter(  # Uses the new PackedAWQParameter
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

        scales = (
            GroupQuantScaleAWQParameter(  # Uses the new GroupQuantScaleAWQParameter
                data=torch.empty(
                    input_size_per_partition // self.quant_config.group_size,
                    output_size_per_partition,
                    dtype=params_dtype,
                ),
                input_dim=0,
                output_dim=1,
                weight_loader=weight_loader,
            )
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # This part might change if PackedAWQParameter handles this differently,
        # but Parameter.data is the typical way.
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (
            qweight.shape[-1] * pack_factor,
        )  # This assumes qweight is packed
        reshaped_x = x.reshape(-1, x.shape[-1])

        if not _is_cuda:
            raise RuntimeError("AWQ dequantization is only supported on CUDA.")

        # awq_dequantize expects tensors. If qweight, qzeros, scales are Parameters,
        # their .data attribute might be needed if the kernel doesn't handle Parameters directly.
        # However, torch functions usually handle Parameters transparently.
        out = awq_dequantize(qweight, scales, qzeros)
        out = torch.matmul(reshaped_x, out)

        if bias is not None:
            out = out + bias  # In-place add can cause issues
        return out.reshape(out_shape)


# Define GroupQuantScaleAWQParameter used in AWQLinearMethod
class GroupQuantScaleAWQParameter(_ColumnAWQParameter, RowAWQParameter):
    """
    Parameter class for weight scales loaded for weights with
    grouped quantization, adapted for AWQ.
    """

    pass

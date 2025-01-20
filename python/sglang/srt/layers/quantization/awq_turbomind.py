import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.nn import Parameter

from sgl_kernel import turbomind
from vllm.model_executor.layers.linear import LinearBase

from sglang.srt.layers.linear import LinearMethodBase, UnquantizedLinearMethod
from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.turbomind_utils import (
    get_u4_slices,
    is_layer_skipped_awq,
    pack_u4_row,
    unpack_awq_gemm,
    verify_turbomind_supported,
)
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.utils import is_cuda, set_weight_attrs

logger = logging.getLogger(__name__)


class AWQTurbomindConfig(QuantizationConfig):
    """Config class for AWQ Turbomind"""

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        lm_head_quantized: bool,
        modules_to_not_convert: Optional[List[str]] = None,
    ) -> None:
        self.pack_factor = 32 // weight_bits  # packed into int32
        self.group_size = group_size
        self.zero_point = zero_point
        self.lm_head_quantized = lm_head_quantized
        self.weight_bits = weight_bits
        self.modules_to_not_convert = modules_to_not_convert or []

        verify_turbomind_supported(self.weight_bits, self.group_size)

    def __repr__(self) -> str:
        return (
            f"AWQTurbomindConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "awq_turbomind"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQTurbomindConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(
            weight_bits,
            group_size,
            zero_point,
            lm_head_quantized,
            modules_to_not_convert,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        can_convert = cls.is_awq_turbomind_compatible(hf_quant_cfg)
        is_valid_user_quant = user_quant is None or user_quant == "awq_turbomind"

        if can_convert and is_valid_user_quant:
            msg = f"The model is convertible to {cls.get_name()} during runtime. Using {cls.get_name()} kernel."
            logger.info(msg)
            return cls.get_name()

        if can_convert and user_quant == "awq":
            logger.info(
                "Detected that the model can run with awq_turbomind"
                ", however you specified quantization=awq explicitly,"
                " so forcing awq. Use quantization=awq_turbomind for"
                " faster inference"
            )
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase) or (
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized
        ):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            return AWQTurbomindLinearMethod(self)

        return None

    @classmethod
    def is_awq_turbomind_compatible(cls, quant_config: Dict[str, Any]):
        if not is_cuda():
            return False

        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        zero_point = quant_config.get("zero_point")

        if quant_method != "awq":
            return False

        # If we cannot find the info needed in the config, cannot convert.
        if num_bits is None or group_size is None or zero_point is None:
            return False

        return verify_turbomind_supported(quant_bit=num_bits, group_size=group_size)

    def get_scaled_act_names(self) -> List[str]:
        return []


class AWQTurbomindLinearMethod(LinearMethodBase):
    """Linear method for AWQ Turbomind.

    Args:
        quant_config: The AWQ Turbomind quantization config.
    """

    def __init__(self, quant_config: AWQTurbomindConfig) -> None:
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
    ) -> None:

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

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

        num_groups = input_size_per_partition // group_size

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_groups,
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
                num_groups,
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

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.num_groups = num_groups

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        qweight_turbomind = unpack_awq_gemm(layer.qweight.data)
        qzeros_turbomind = unpack_awq_gemm(layer.qzeros.data)
        scales_turbomind = layer.scales.data

        qweight_turbomind = pack_u4_row(qweight_turbomind)
        qzeros_turbomind = qzeros_turbomind.to(torch.half)

        device_id = layer.qweight.device.index
        properties = torch.cuda.get_device_properties(device_id)

        def is_16xx_series(name):
            import re

            pattern = r"GTX 16\d\d"
            return bool(re.search(pattern, name))

        simt = is_16xx_series(properties.name)
        qweight_turbomind = qweight_turbomind.contiguous()
        scales_turbomind = scales_turbomind.contiguous()
        qzeros_turbomind = qzeros_turbomind.contiguous()

        self.linear = turbomind.Linear(
            layer.input_size_per_partition,
            layer.output_size_per_partition,
            self.quant_config.weight_bits,
            self.quant_config.group_size,
        )

        self.linear.post_init(
            qweight_turbomind, scales_turbomind, qzeros_turbomind, simt
        )

        layer.qweight = Parameter(qweight_turbomind, requires_grad=False)
        layer.scales = Parameter(scales_turbomind, requires_grad=False)
        layer.qzeros = Parameter(qzeros_turbomind, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = x.view(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (layer.output_size_per_partition,)
        out = torch.empty(
            (x.shape[0], layer.output_size_per_partition),
            dtype=torch.float16,
            device=x.device,
        )
        stream = torch.cuda.current_stream()

        self.linear.forward(x, out, stream.cuda_stream)
        out = torch.from_dlpack(out)
        if bias is not None:
            out.add_(bias)

        return out.view(out_shape)

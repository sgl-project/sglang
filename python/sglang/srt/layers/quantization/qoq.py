from typing import Any, Callable, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.linear import LinearMethodBase
from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()
if _is_cuda:
    from sgl_kernel import qserve_w4a8_per_chn_gemm, qserve_w4a8_per_group_gemm


QoQ_SUPPORTED_WEIGHT_BITS = [4]
QoQ_SUPPORTED_GROUP_SIZES = [-1, 128]


class QoQConfig(QuantizationConfig):
    """Config class for QoQ Quantization.

    - Weight: static, per-channel/group, asymmetric
    - Activation: dynamic, per-token, symmetric

    Reference: https://arxiv.org/abs/2405.04532
    https://github.com/mit-han-lab/omniserve
    """

    def __init__(self, weight_bits: int, group_size: int) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size

        # Verify
        if self.weight_bits not in QoQ_SUPPORTED_WEIGHT_BITS:
            raise ValueError(
                f"QoQ does not support weight_bits = {self.weight_bits}. "
                f"Only weight_bits = {QoQ_SUPPORTED_WEIGHT_BITS} "
                "are supported."
            )
        if self.group_size not in QoQ_SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"QoQ does not support group_size = {self.group_size}. "
                f"Only group_sizes = {QoQ_SUPPORTED_GROUP_SIZES} "
                "are supported."
            )

        # 4 bits packed into 8 bit datatype.
        self.pack_factor = 8 // self.weight_bits

    def __repr__(self) -> str:
        return "QoQConfig(weight_bits={}, group_size={})".format(
            self.weight_bits, self.group_size
        )

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_name(self) -> str:
        return "qoq"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        """List of filenames to search for in the model directory."""
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QoQConfig":
        weight_bits = cls.get_from_keys(config, ["wbits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(weight_bits, group_size)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from sglang.srt.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            return QoQLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class QoQLinearMethod(LinearMethodBase):
    """Linear method for QoQ.

    Args:
        quant_config: The QoQ quantization config.
    """

    def __init__(self, quant_config: QoQConfig):
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

        weight_loader = extra_weight_attrs.get("weight_loader")

        # Validate output_size_per_partition
        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % 32 != 0:
            raise ValueError(
                f"Weight output_size_per_partition = "
                f"{output_size_per_partition} is not divisible by 32."
            )

        # Validate input_size_per_partition
        if input_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                f"Weight input_size_per_partition = "
                f"{input_size_per_partition} is not divisible by "
                f"pack_factor = {self.quant_config.pack_factor}."
            )
        if (
            self.quant_config.group_size != -1
            and input_size_per_partition % self.quant_config.group_size != 0
        ):
            raise ValueError(
                f"Weight input_size_per_partition = "
                f"{input_size_per_partition} is not divisible by "
                f"group_size = {self.quant_config.group_size}."
            )

        qweight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("qweight", qweight)

        s1_scales = ChannelQuantScaleParameter(
            data=torch.empty(output_size_per_partition, dtype=torch.float16),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("s1_scales", s1_scales)

        if self.quant_config.group_size == -1:
            s1_szeros = ChannelQuantScaleParameter(
                data=torch.empty(output_size_per_partition, dtype=torch.float16),
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("s1_szeros", s1_szeros)
        else:
            s2_scales = GroupQuantScaleParameter(
                data=torch.empty(
                    (
                        input_size_per_partition // self.quant_config.group_size,
                        output_size_per_partition,
                    ),
                    dtype=torch.int8,
                ),
                input_dim=0,
                output_dim=1,
                weight_loader=weight_loader,
            )
            layer.register_parameter("s2_scales", s2_scales)

            s2_zeros = GroupQuantScaleParameter(
                data=torch.empty(
                    (
                        input_size_per_partition // self.quant_config.group_size,
                        output_size_per_partition,
                    ),
                    dtype=torch.int8,
                ),
                input_dim=0,
                output_dim=1,
                weight_loader=weight_loader,
            )
            layer.register_parameter("s2_zeros", s2_zeros)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.s1_scales = Parameter(layer.s1_scales.data, requires_grad=False)
        if self.quant_config.group_size == -1:
            layer.s1_szeros = Parameter(layer.s1_szeros.data, requires_grad=False)
        else:
            layer.s2_scales = Parameter(layer.s2_scales.data, requires_grad=False)
            layer.s2_zeros = Parameter(layer.s2_zeros.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        assert x.dtype == torch.float16, "QoQ only supports float16 input now"
        if self.quant_config.group_size == -1:
            x_q, x_scale, x_sum = per_token_quant_int8(
                x, scale_dtype=x.dtype, cal_sum=True
            )
            out = qserve_w4a8_per_chn_gemm(
                x_q, layer.qweight, layer.s1_scales, x_scale, layer.s1_szeros, x_sum
            )
        else:
            x_q, x_scale = per_token_quant_int8(x, scale_dtype=x.dtype)
            out = qserve_w4a8_per_group_gemm(
                x_q,
                layer.qweight,
                layer.s2_zeros,
                layer.s2_scales,
                layer.s1_scales,
                x_scale,
            )
        if bias is not None:
            out = out + bias
        return out

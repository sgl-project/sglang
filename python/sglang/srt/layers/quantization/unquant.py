from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.base_config import LinearMethodBase
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    set_weight_attrs,
    use_intel_amx_backend,
)
from sglang.srt.layers.amx_utils import _amx_process_weight_after_loading

_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

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
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["weight"])

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if use_intel_amx_backend(layer):
            return torch.ops.sgl_kernel.weight_packed_linear(
                x, layer.weight, bias, True  # is_vnni
            )

        return F.linear(x, layer.weight, bias)

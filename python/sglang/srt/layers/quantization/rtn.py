# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright © 2025, Oracle and/or its affiliates.

import logging
import os
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.marlin_utils import (
    apply_rtn_marlin_linear,
    marlin_make_workspace,
)
from sglang.srt.layers.quantization.rtn_utils import (
    fix_weights,
    repack_weights,
    rtn_dequantize,
    rtn_quantize,
)
from sglang.srt.layers.quantization.utils import get_scalar_types, replace_parameter
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)


# By default, use 8 bit as target precision, but it can be overridden by setting the RTN_NUM_BITS envvar
NUM_BITS = os.getenv("RTN_NUM_BITS", "8")


# By default, use group size of 128 parameters, but it can be overridden by setting the RTN_GROUP_SIZE envvar
GROUP_SIZE = os.getenv("RTN_GROUP_SIZE", "128")

ScalarType, scalar_types = get_scalar_types()

_rtn_marlin_workspace = None


class RTNConfig(QuantizationConfig):
    """Config class for RTN."""

    def __init__(
        self,
        weight_bits: int = int(NUM_BITS),
        group_size: int = int(GROUP_SIZE),
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size

        if self.weight_bits != 4 and self.weight_bits != 8:
            raise ValueError(
                "Currently, only 4-bit or 8-bit weight quantization is "
                f"supported for RTN, but got {self.weight_bits} bits."
            )

        self.quant_type = (
            scalar_types.uint8b128 if self.weight_bits == 8 else scalar_types.uint4b8
        )

    def __repr__(self) -> str:
        return (
            f"RTNConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "rtn"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RTNConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(weight_bits, group_size)

    def is_marlin_compatible(self) -> bool:
        from sglang.srt.layers.quantization.gptq import GPTQMarlinConfig
        from sglang.srt.layers.quantization.marlin_utils import check_marlin_supported

        # Check bits/sym
        # RTN is symmetric (centered)
        if (self.weight_bits, True) not in GPTQMarlinConfig.TYPE_MAP:
            return False

        quant_type = GPTQMarlinConfig.TYPE_MAP[(self.weight_bits, True)]
        # Check group size
        return check_marlin_supported(quant_type, self.group_size)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.marlin_utils import (
            check_marlin_supports_layer,
            check_moe_marlin_supports_layer,
        )

        if isinstance(layer, LinearBase):
            if self.is_marlin_compatible() and check_marlin_supports_layer(
                layer, self.group_size
            ):
                return RTNMarlinLinearMethod(self)
            return RTNLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            if self.is_marlin_compatible() and check_moe_marlin_supports_layer(
                layer, self.group_size
            ):
                return RTNMarlinMoEMethod(self)
            return RTNMoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class RTNTensor(torch.Tensor):
    """A Tensor subclass that quantizes on-the-fly via copy_."""

    @staticmethod
    def _base_tensor(data: torch.Tensor) -> torch.Tensor:
        if hasattr(data, "as_subclass"):
            return data.as_subclass(torch.Tensor)
        return data

    @staticmethod
    def __new__(
        cls, data: torch.Tensor, scale: torch.Tensor, quant_config: RTNConfig
    ) -> "RTNTensor":
        base = RTNTensor._base_tensor(data)
        return torch.Tensor._make_subclass(cls, base, False)

    def __init__(
        self, data: torch.Tensor, scale: torch.Tensor, quant_config: RTNConfig
    ) -> None:
        self.scale = scale
        self.quant_config = quant_config
        self._packed_shape = self._base_tensor(data).shape

    def _logical_shape(self) -> torch.Size:
        factor = 1 if self.quant_config.weight_bits == 8 else 2
        batch_present = len(self._packed_shape) == 3
        if batch_present:
            return torch.Size(
                (
                    self._packed_shape[0],
                    self._packed_shape[1] * factor,
                    self._packed_shape[2],
                )
            )
        return torch.Size((self._packed_shape[0] * factor, self._packed_shape[1]))

    @property
    def shape(self):
        return self._logical_shape()

    def size(self, dim: Optional[int] = None):
        shape = self._logical_shape()
        if dim is None:
            return shape
        return shape[dim]

    def dim(self):
        return len(self._logical_shape())

    def narrow(self, dim, start, length):
        factor = 1 if self.quant_config.weight_bits == 8 else 2
        return RTNTensor(
            self._base_tensor(super().narrow(dim, start // factor, length // factor)),
            self.scale.narrow(dim, start, length),
            self.quant_config,
        )

    def __getitem__(self, key):
        return RTNTensor(
            self._base_tensor(super().__getitem__(key)),
            self.scale[key],
            self.quant_config,
        )

    def copy_(self, loaded_weight: torch.Tensor) -> "RTNTensor":
        qweight, weight_scale = rtn_quantize(
            loaded_weight.to(self.device, non_blocking=True),
            self.quant_config.weight_bits,
            self.quant_config.group_size,
        )

        super().copy_(qweight)
        self.scale.data.copy_(weight_scale)
        return self


class RTNParameter(Parameter):
    """A wrapper over Parameter that returns RTNTensor (a wrapper over Tensor)
    when its data is accessed. We need this wrapper for the data loading phase
    only, so we can intercept a weight copying function (torch.Tensor.copy_)
    and apply quantization on-the-fly.
    """

    def __new__(cls, data: torch.Tensor, **kwargs):
        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(
        self, data: torch.Tensor, scale: torch.Tensor, quant_config: RTNConfig
    ) -> None:
        self.scale = scale
        self.quant_config = quant_config
        super().__init__()

    @property
    def data(self):
        return RTNTensor(super().data, self.scale, self.quant_config)


class RTNLinearMethod(LinearMethodBase):
    """Linear method for RTN.

    Args:
        quant_config: The RTN quantization config.
    """

    def __init__(self, quant_config: RTNConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        num_groups_per_col = (
            input_size_per_partition // self.quant_config.group_size
            if self.quant_config.group_size != -1
            else 1
        )

        scale = Parameter(
            torch.empty(
                output_size_per_partition, num_groups_per_col, dtype=params_dtype
            ),
            requires_grad=False,
        )
        factor = 1 if self.quant_config.weight_bits == 8 else 2

        weight = RTNParameter(
            data=torch.empty(
                output_size_per_partition // factor,
                input_size_per_partition,
                dtype=torch.uint8,
            ),
            scale=scale,
            quant_config=self.quant_config,
        )

        layer.register_parameter("weight", weight)
        set_weight_attrs(
            weight,
            {
                **extra_weight_attrs,
                "input_dim": 1,
                "output_dim": 0,
            },
        )

        layer.register_parameter("scale", scale)
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        fix_weights(layer, "weight")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.weight
        scale = layer.scale

        weight = rtn_dequantize(qweight, scale)
        out = F.linear(x, weight)
        del weight
        if bias is not None:
            out.add_(bias)

        return out


class RTNMoEMethod(FusedMoEMethodBase):
    """MoE method for RTN."""

    def __init__(self, quant_config: RTNConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        factor = 1 if self.quant_config.weight_bits == 8 else 2

        # Fused gate_up_proj (column parallel)
        num_groups_per_col = (
            hidden_size // self.quant_config.group_size
            if self.quant_config.group_size != -1
            else 1
        )
        w13_scale = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                num_groups_per_col,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scale", w13_scale)

        w13_weight = RTNParameter(
            data=torch.empty(
                num_experts,
                2 * intermediate_size_per_partition // factor,
                hidden_size,
                dtype=torch.uint8,
            ),
            scale=w13_scale,
            quant_config=self.quant_config,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        num_groups_per_col = (
            intermediate_size_per_partition // self.quant_config.group_size
            if self.quant_config.group_size != -1
            else 1
        )
        w2_scale = Parameter(
            torch.zeros(
                num_experts, hidden_size, num_groups_per_col, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scale", w2_scale)

        w2_weight = RTNParameter(
            data=torch.empty(
                num_experts,
                hidden_size // factor,
                intermediate_size_per_partition,
                dtype=torch.uint8,
            ),
            scale=w2_scale,
            quant_config=self.quant_config,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_bits = self.quant_config.weight_bits
        fix_weights(layer, "w13_weight", weight_bits == 4)
        fix_weights(layer, "w2_weight", weight_bits == 4)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        backend = get_moe_runner_backend()
        if backend.is_auto():
            backend = (
                MoeRunnerBackend.TRITON_KERNELS
                if layer.use_triton_kernels
                else MoeRunnerBackend.TRITON
            )
        elif backend.is_triton_kernels() and not layer.use_triton_kernels:
            logger.warning(
                "RTN MoE requested triton_kernels backend but layer.use_triton_kernels "
                "is False. Falling back to TRITON."
            )
            backend = MoeRunnerBackend.TRITON
        self.runner = MoeRunner(backend, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output,
    ):
        weight_bits = self.quant_config.weight_bits
        group_size = self.quant_config.group_size
        block_shape = None if group_size == -1 else [0, group_size]

        quant_info = TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_int4_w4a16=weight_bits == 4,
            use_int8_w8a16=weight_bits == 8,
            w13_scale=layer.w13_scale,
            w2_scale=layer.w2_scale,
            block_shape=block_shape,
        )

        return self.runner.run(dispatch_output, quant_info)


def init_workspace(device):
    global _rtn_marlin_workspace
    if _rtn_marlin_workspace is None or _rtn_marlin_workspace.device != device:
        _rtn_marlin_workspace = marlin_make_workspace(device, max_blocks_per_sm=4)


class RTNMarlinLinearMethod(RTNLinearMethod):
    """Linear method for RTN with Marlin kernels."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_bits = self.quant_config.weight_bits

        weight, scale = repack_weights(layer.weight, layer.scale, weight_bits)
        replace_parameter(layer, "weight", weight)
        replace_parameter(layer, "scale", scale)

        init_workspace(layer.weight.device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_rtn_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.scale,
            workspace=_rtn_marlin_workspace,
            quant_type=self.quant_config.quant_type,
            output_size_per_partition=layer.output_size_per_partition,
            input_size_per_partition=layer.input_size_per_partition,
            bias=bias,
        )


class RTNMarlinMoEMethod(RTNMoEMethod):
    """MoE method for RTN with Marlin kernels."""

    def __init__(self, quant_config: RTNConfig):
        super().__init__(quant_config)
        self.is_k_full = True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_bits = self.quant_config.weight_bits

        w13_weight, w13_scale = repack_weights(
            layer.w13_weight, layer.w13_scale, weight_bits
        )
        replace_parameter(layer, "w13_weight", w13_weight)
        replace_parameter(layer, "w13_scale", w13_scale)

        w2_weight, w2_scale = repack_weights(
            layer.w2_weight, layer.w2_scale, weight_bits
        )
        replace_parameter(layer, "w2_weight", w2_weight)
        replace_parameter(layer, "w2_scale", w2_scale)

        init_workspace(layer.w13_weight.device)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        assert get_moe_runner_backend().is_auto()
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output,
    ):
        quant_info = MarlinMoeQuantInfo(
            w13_qweight=layer.w13_weight,
            w2_qweight=layer.w2_weight,
            w13_scales=layer.w13_scale,
            w2_scales=layer.w2_scale,
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=self.quant_config.weight_bits,
            w13_g_idx=None,
            w2_g_idx=None,
            is_k_full=self.is_k_full,
        )

        return self.runner.run(dispatch_output, quant_info)

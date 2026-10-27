# SPDX-License-Identifier: Apache-2.0
# Adapted from Tencent-Hunyuan/Hy-Nomas:
# hymm/models/dit/modules/int8_stored_bf16_linear.py
"""Official Nomas groupwise INT8 storage with BF16 linear computation.

Weights stay INT8 between calls, without a BF16 weight cache. Parallel layers
use NomasInt8LinearMethod; ordinary nn.Linear sites use a replicated adapter.
"""

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    LinearMethodBase,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs

GROUP_SIZE = 128
CHECKPOINT_FORMAT = "nomas-int8-bf16-single-v1"


def dequantize_nomas_int8(weight, scales):
    scales = scales.repeat_interleave(GROUP_SIZE, dim=1)
    return (weight.float() * scales[:, : weight.shape[1]]).to(torch.bfloat16)


class NomasInt8LinearMethod(LinearMethodBase):
    """Use the existing parallel linear loaders for groupwise INT8 weights."""

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.bfloat16:
            raise ValueError("Nomas INT8 requires BF16 compute")
        if (
            input_size_per_partition != input_size
            and input_size_per_partition % GROUP_SIZE
        ):
            raise ValueError("Nomas INT8 TP input shards must align to 128 elements")
        out = sum(output_partition_sizes)
        for name, shape, dtype in (
            ("weight", (out, input_size_per_partition), torch.int8),
            (
                "weight_scale",
                (out, (input_size_per_partition + GROUP_SIZE - 1) // GROUP_SIZE),
                torch.float32,
            ),
        ):
            param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            set_weight_attrs(
                param, {"input_dim": 1, "output_dim": 0, "missing_param_init": "error"}
            )
            set_weight_attrs(param, extra_weight_attrs)
            layer.register_parameter(name, param)

    def process_weights_after_loading(self, layer):
        scales = layer.weight_scale
        if not torch.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError("Nomas INT8 scales must be finite and positive")

    def apply(self, layer, x, bias=None):
        if x.dtype != torch.bfloat16:
            if (
                x.is_cuda
                and torch.is_autocast_enabled("cuda")
                and torch.get_autocast_dtype("cuda") == torch.bfloat16
            ):
                x = x.to(torch.bfloat16)
            else:
                raise TypeError("Nomas INT8 linear requires BF16 input")
        return F.linear(
            x, dequantize_nomas_int8(layer.weight, layer.weight_scale), bias
        )


class _NomasInt8ScaleMixin:
    """Keep checkpoint scales in FP32 during component dtype/device moves."""

    def _apply(self, fn, recurse=True):
        original = self.weight_scale.detach()
        result = super()._apply(fn, recurse=recurse)
        if self.weight_scale.dtype != original.dtype:
            self.weight_scale.data = original.to(self.weight_scale.device)
        return result


class NomasInt8ColumnParallelLinear(_NomasInt8ScaleMixin, ColumnParallelLinear):
    """Column-parallel INT8 storage with FP32 scales across component moves."""


class NomasInt8RowParallelLinear(_NomasInt8ScaleMixin, RowParallelLinear):
    """Row-parallel INT8 storage with FP32 scales across component moves."""


class NomasInt8Linear(_NomasInt8ScaleMixin, ReplicatedLinear):
    """Replicated quantized linear with nn.Linear's tensor return contract."""

    def forward(self, x):
        return super().forward(x)[0]

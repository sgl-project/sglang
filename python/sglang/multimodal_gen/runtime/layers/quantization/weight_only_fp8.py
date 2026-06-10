# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.utils import get_group_rank, get_group_size
from sglang.multimodal_gen.runtime.models.utils import set_weight_attrs

FP8_WEIGHT_DTYPE = torch.float8_e4m3fn


def dequantize_rowwise_fp8_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"FP8 linear weight must be 2-D, got shape {weight.shape}")
    if weight_scale.ndim != 1 or weight_scale.shape[0] != weight.shape[0]:
        raise ValueError(
            "FP8 row-wise scale must have shape (out_features,), "
            f"got weight={tuple(weight.shape)} scale={tuple(weight_scale.shape)}"
        )
    return weight.to(dtype) * weight_scale.to(dtype).unsqueeze(1)


class WeightOnlyFP8Linear(nn.Module):
    """Storage-only e4m3 FP8 linear with row-wise dequantization before matmul."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=FP8_WEIGHT_DTYPE),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(self.weight_scale, {"missing_param_init": "error"})
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    out_features, dtype=compute_dtype or torch.get_default_dtype()
                ),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self.compute_dtype or x.dtype
        weight = dequantize_rowwise_fp8_weight(
            self.weight, self.weight_scale, compute_dtype
        )
        bias = self.bias.to(compute_dtype) if self.bias is not None else None
        return F.linear(x.to(compute_dtype), weight, bias)


class WeightOnlyFP8ColumnParallelLinear(nn.Module):
    """Column-parallel e4m3 FP8 linear with row-wise dequantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype | None = None,
        gather_output: bool = True,
        tp_group=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.gather_output = gather_output
        self.tp_group = tp_group or get_tp_group()
        self.tp_size = get_group_size(self.tp_group)
        self.tp_rank = get_group_rank(self.tp_group)
        self.out_features_per_partition = divide(out_features, self.tp_size)
        self.weight = nn.Parameter(
            torch.empty(
                self.out_features_per_partition,
                in_features,
                dtype=FP8_WEIGHT_DTYPE,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            self.weight,
            {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            },
        )
        self.weight_scale = nn.Parameter(
            torch.empty(self.out_features_per_partition, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(
            self.weight_scale,
            {
                "missing_param_init": "error",
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            },
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    self.out_features_per_partition,
                    dtype=compute_dtype or torch.get_default_dtype(),
                ),
                requires_grad=False,
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        output_dim = getattr(param, "output_dim", None)
        if output_dim is not None:
            shard_size = param.data.shape[output_dim]
            loaded_weight = loaded_weight.narrow(
                output_dim, self.tp_rank * shard_size, shard_size
            )
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        assert param.data.shape == loaded_weight.shape
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self.compute_dtype or x.dtype
        weight = dequantize_rowwise_fp8_weight(
            self.weight, self.weight_scale, compute_dtype
        )
        bias = self.bias.to(compute_dtype) if self.bias is not None else None
        output_parallel = F.linear(x.to(compute_dtype), weight, bias)
        if self.gather_output:
            return tensor_model_parallel_all_gather(
                output_parallel, tp_group=self.tp_group
            )
        return output_parallel


def swap_linears_to_weight_only_fp8(module: nn.Module) -> None:
    """Recursively replace nn.Linear with WeightOnlyFP8Linear.

    Ideogram FP8 checkpoints provide ``<linear>.weight_scale`` for every
    quantized linear. Swapping before load lets strict state-dict checks verify
    both the FP8 weight and its row-wise scale.
    """

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            replacement = WeightOnlyFP8Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=child.weight.dtype,
            )
            setattr(module, name, replacement)
        else:
            swap_linears_to_weight_only_fp8(child)

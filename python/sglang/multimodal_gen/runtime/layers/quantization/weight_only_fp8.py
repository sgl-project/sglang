# SPDX-License-Identifier: Apache-2.0

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import sglang.multimodal_gen.envs as envs
from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_tp_group,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.layers.utils import get_group_rank, get_group_size
from sglang.multimodal_gen.runtime.models.utils import set_weight_attrs

FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
W8A8_FP8_GEMM_ENV = "SGLANG_DIFFUSION_ENABLE_W8A8_FP8_GEMM"

logger = logging.getLogger(__name__)
_w8a8_fp8_gemm_warning_logged = False


def _can_apply_fused_w8a8_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    compute_dtype: torch.dtype,
) -> bool:
    return (
        x.device.type == "cuda"
        and weight.device.type == "cuda"
        and weight_scale.device.type == "cuda"
        and not x.is_meta
        and not weight.is_meta
        and not weight_scale.is_meta
        and compute_dtype in (torch.float16, torch.bfloat16)
    )


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


def _apply_srt_w8a8_fp8_linear(*args, **kwargs) -> torch.Tensor:
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

    return apply_fp8_linear(*args, **kwargs)


def _is_cutlass_fp8_supported() -> bool:
    from sglang.srt.layers.quantization.fp8_utils import cutlass_fp8_supported

    return cutlass_fp8_supported()


def _apply_weight_only_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    compute_dtype: torch.dtype,
    enable_fused_w8a8: bool,
) -> torch.Tensor:
    x = x.to(compute_dtype)
    bias = bias.to(compute_dtype) if bias is not None else None
    if enable_fused_w8a8 and _can_apply_fused_w8a8_fp8_linear(
        x, weight, weight_scale, compute_dtype
    ):
        try:
            # The fused kernel uses W8A8 compute; fallback keeps BF16/FP16
            # activations after dequantizing the FP8 weights.
            output = _apply_srt_w8a8_fp8_linear(
                input=x,
                weight=weight.t(),
                weight_scale=weight_scale,
                input_scale=None,
                bias=bias,
                cutlass_fp8_supported=_is_cutlass_fp8_supported(),
            )
            _log_w8a8_fp8_gemm_warning_once()
            return output
        except (ImportError, NotImplementedError):
            pass

    dequant_weight = dequantize_rowwise_fp8_weight(weight, weight_scale, compute_dtype)
    return F.linear(x, dequant_weight, bias)


class WeightOnlyFP8Linear(nn.Module):
    """Storage-only e4m3 FP8 linear with row-wise weight scales."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype | None = None,
        enable_fused_w8a8: bool | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.enable_fused_w8a8 = _resolve_enable_fused_w8a8(enable_fused_w8a8)
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
        return _apply_weight_only_fp8_linear(
            x,
            self.weight,
            self.weight_scale,
            self.bias,
            compute_dtype,
            self.enable_fused_w8a8,
        )


class WeightOnlyFP8ColumnParallelLinear(nn.Module):
    """Column-parallel storage-only e4m3 FP8 linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype | None = None,
        gather_output: bool = True,
        tp_group=None,
        enable_fused_w8a8: bool | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.gather_output = gather_output
        self.enable_fused_w8a8 = _resolve_enable_fused_w8a8(enable_fused_w8a8)
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
        output_parallel = _apply_weight_only_fp8_linear(
            x,
            self.weight,
            self.weight_scale,
            self.bias,
            compute_dtype,
            self.enable_fused_w8a8,
        )
        if self.gather_output:
            return tensor_model_parallel_all_gather(
                output_parallel, tp_group=self.tp_group
            )
        return output_parallel


class WeightOnlyFP8MergedColumnParallelLinear(WeightOnlyFP8ColumnParallelLinear):
    """Column-parallel storage-only FP8 packed linear."""

    def __init__(
        self,
        in_features: int,
        output_sizes: list[int],
        bias: bool = True,
        compute_dtype: torch.dtype | None = None,
        gather_output: bool = False,
        tp_group=None,
        enable_fused_w8a8: bool | None = None,
    ) -> None:
        self.output_sizes = output_sizes
        super().__init__(
            in_features,
            sum(output_sizes),
            bias=bias,
            compute_dtype=compute_dtype,
            gather_output=gather_output,
            tp_group=tp_group,
            enable_fused_w8a8=enable_fused_w8a8,
        )
        assert all(output_size % self.tp_size == 0 for output_size in output_sizes)

    def weight_loader(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        output_dim = getattr(param, "output_dim", None)
        if output_dim is not None:
            shards = []
            current_offset = 0
            for output_size in self.output_sizes:
                loaded_shard = loaded_weight.narrow(
                    output_dim, current_offset, output_size
                )
                shard_size = output_size // self.tp_size
                loaded_shard = loaded_shard.narrow(
                    output_dim, self.tp_rank * shard_size, shard_size
                )
                shards.append(loaded_shard)
                current_offset += output_size
            loaded_weight = torch.cat(shards, dim=output_dim)
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        assert param.data.shape == loaded_weight.shape
        param.data.copy_(loaded_weight)


class WeightOnlyFP8RowParallelLinear(nn.Module):
    """Row-parallel storage-only e4m3 FP8 linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype | None = None,
        input_is_parallel: bool = True,
        reduce_results: bool = True,
        tp_group=None,
        enable_fused_w8a8: bool | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        self.enable_fused_w8a8 = _resolve_enable_fused_w8a8(enable_fused_w8a8)
        self.tp_group = tp_group or get_tp_group()
        self.tp_size = get_group_size(self.tp_group)
        self.tp_rank = get_group_rank(self.tp_group)
        self.in_features_per_partition = divide(in_features, self.tp_size)
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                self.in_features_per_partition,
                dtype=FP8_WEIGHT_DTYPE,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            self.weight,
            {
                "input_dim": 1,
                "weight_loader": self.weight_loader,
            },
        )
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(
            self.weight_scale,
            {
                "missing_param_init": "error",
                "weight_loader": self.weight_loader,
            },
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    out_features, dtype=compute_dtype or torch.get_default_dtype()
                ),
                requires_grad=False,
            )
            set_weight_attrs(self.bias, {"weight_loader": self.weight_loader})
        else:
            self.register_parameter("bias", None)

    def weight_loader(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        input_dim = getattr(param, "input_dim", None)
        if input_dim is not None:
            shard_size = param.data.shape[input_dim]
            loaded_weight = loaded_weight.narrow(
                input_dim, self.tp_rank * shard_size, shard_size
            )
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        assert param.data.shape == loaded_weight.shape
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_is_parallel:
            input_parallel = x
        else:
            input_parallel = split_tensor_along_last_dim(
                x, num_partitions=self.tp_size
            )[self.tp_rank].contiguous()

        compute_dtype = self.compute_dtype or x.dtype
        bias = None if self.tp_rank > 0 else self.bias
        output_parallel = _apply_weight_only_fp8_linear(
            input_parallel,
            self.weight,
            self.weight_scale,
            bias,
            compute_dtype,
            self.enable_fused_w8a8,
        )
        if self.reduce_results and self.tp_size > 1:
            return tensor_model_parallel_all_reduce(
                output_parallel, tp_group=self.tp_group
            )
        return output_parallel


def _resolve_enable_fused_w8a8(value: bool | None) -> bool:
    if value is not None:
        return value
    return envs.SGLANG_DIFFUSION_ENABLE_W8A8_FP8_GEMM


def _log_w8a8_fp8_gemm_warning_once() -> None:
    global _w8a8_fp8_gemm_warning_logged
    if _w8a8_fp8_gemm_warning_logged:
        return
    logger.warning(
        "%s=1 enables W8A8 FP8 GEMM for weight-only FP8 linears; activations "
        "are dynamically quantized to FP8 and outputs may differ from the "
        "official weight-only FP8 path.",
        W8A8_FP8_GEMM_ENV,
    )
    _w8a8_fp8_gemm_warning_logged = True


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

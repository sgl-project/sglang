# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import NamedTuple, Optional, Protocol, runtime_checkable

import torch
from torch import nn


class PackedUe8m0LinearInput(NamedTuple):
    """FP8 activations and packed UE8M0 scales consumed by DeepGEMM."""

    data: torch.Tensor
    scale: torch.Tensor


class PreparedFp8RMSNormInput(NamedTuple):
    """Result of fusing RMSNorm with DeepGEMM activation quantization."""

    linear_input: PackedUe8m0LinearInput
    normalized_input: torch.Tensor


class PackedUe8m0LinearOp(Protocol):
    def __call__(
        self,
        input: PackedUe8m0LinearInput,
        weight: torch.Tensor,
        block_size: list[int],
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...


@runtime_checkable
class SupportsFusedFp8RMSNormInput(Protocol):
    def maybe_prepare_fused_rmsnorm_input(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> Optional[PreparedFp8RMSNormInput]: ...


def has_packed_ue8m0_linear_contract(
    layer: nn.Module,
    *,
    block_quant: bool,
    use_marlin: bool,
    use_mxfp8: bool,
    weight_block_size: Optional[list[int]],
    weight_dtype: torch.dtype,
) -> bool:
    """Check the loaded weight layout required by packed-UE8M0 DeepGEMM."""
    weight = getattr(layer, "weight", None)
    weight_scale = getattr(layer, "weight_scale_inv", None)
    if (
        not block_quant
        or use_marlin
        or use_mxfp8
        or weight_block_size != [128, 128]
        or not isinstance(weight, torch.Tensor)
        or weight.dim() != 2
        or not isinstance(weight_scale, torch.Tensor)
    ):
        return False

    packed_scale_cols = (weight.shape[1] // 128 + 3) // 4
    return bool(
        getattr(layer, "orig_dtype", None) == torch.bfloat16
        and getattr(layer, "input_scale", None) is None
        and weight.dtype == weight_dtype
        and weight.shape[0] % 64 == 0
        and weight.shape[1] % 128 == 0
        and weight_scale.dim() == 2
        and tuple(weight_scale.shape) == (weight.shape[0], packed_scale_cols)
        and weight_scale.dtype == torch.int32
        and weight_scale.device == weight.device
        and weight_scale.stride(0) == 1
        and getattr(weight_scale, "format_ue8m0", False)
    )


def has_fused_rmsnorm_tensor_contract(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    hidden_size: int,
) -> bool:
    """Check shape, dtype, device, and stride requirements of the JIT kernel."""
    return bool(
        x.is_cuda
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.shape[1] == hidden_size
        and x.stride(1) == 1
        and x.stride(0) % 16 == 0
        and tuple(norm_weight.shape) == (hidden_size,)
        and norm_weight.dtype == torch.bfloat16
        and norm_weight.device == x.device
        and norm_weight.stride(0) == 1
    )

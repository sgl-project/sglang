# SPDX-License-Identifier: Apache-2.0
"""
Nunchaku quantized linear methods for SVDQuant integration.

This module provides LinearMethodBase implementations that delegate to Nunchaku's
quantized linear layers, following the SGLang AWQ pattern of using quantization
methods instead of direct layer replacement.
"""

from typing import Any, List, Optional

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import LinearMethodBase
from sglang.multimodal_gen.runtime.models.utils import set_weight_attrs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _check_nunchaku_available() -> bool:
    """Check if nunchaku is installed."""
    try:
        import nunchaku
        return True
    except ImportError:
        return False


class NunchakuSVDQLinearMethod(LinearMethodBase):
    """
    Linear method for Nunchaku SVDQ W4A4 quantization.

    This implements the LinearMethodBase interface using Nunchaku's SVDQW4A4Linear
    layer which provides W4A4 quantization with SVD low-rank decomposition.

    Args:
        precision: Quantization precision ('int4' or 'nvfp4')
        rank: SVD low-rank dimension (default: 32)
        act_unsigned: Use unsigned activation quantization (int4 only)
    """

    def __init__(
        self,
        precision: str = "int4",
        rank: int = 32,
        act_unsigned: bool = False,
    ):
        self.precision = precision
        self.rank = rank
        self.act_unsigned = act_unsigned

        # Set group_size based on precision
        if precision == "nvfp4":
            self.group_size = 16
        else:
            self.group_size = 64

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
        Create quantized weights for SVDQ W4A4 layer.

        For SVDQ W4A4, we create:
        - qweight: Packed quantized weights (int8)
        - wscales: Weight scales
        - smooth_factor: Smoothing factors for activations
        - proj_down, proj_up: Low-rank decomposition matrices
        """
        output_size_per_partition = sum(output_partition_sizes)

        # Packed quantized weights: shape (out_features, in_features // 2)
        qweight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(qweight, {"input_dim": 1, "output_dim": 0})

        # Weight scales: shape (in_features // group_size, out_features)
        num_groups = input_size_per_partition // self.group_size
        if self.precision == "nvfp4":
            scale_dtype = torch.float8_e4m3fn
        else:
            scale_dtype = params_dtype
        wscales = Parameter(
            torch.empty(num_groups, output_size_per_partition, dtype=scale_dtype),
            requires_grad=False,
        )

        # Smooth factor: shape (in_features,)
        smooth_factor = Parameter(
            torch.empty(input_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )

        # Low-rank projections
        proj_down = Parameter(
            torch.empty(input_size_per_partition, self.rank, dtype=params_dtype),
            requires_grad=False,
        )
        proj_up = Parameter(
            torch.empty(output_size_per_partition, self.rank, dtype=params_dtype),
            requires_grad=False,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("wscales", wscales)
        layer.register_parameter("smooth_factor", smooth_factor)
        layer.register_parameter("proj_down", proj_down)
        layer.register_parameter("proj_up", proj_up)

        # Store metadata
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.precision = self.precision
        layer.rank = self.rank
        layer.group_size = self.group_size

        weight_loader = extra_weight_attrs.get("weight_loader")
        if weight_loader is not None:
            set_weight_attrs(qweight, {"weight_loader": weight_loader})
            set_weight_attrs(wscales, {"weight_loader": weight_loader})
            set_weight_attrs(smooth_factor, {"weight_loader": weight_loader})
            set_weight_attrs(proj_down, {"weight_loader": weight_loader})
            set_weight_attrs(proj_up, {"weight_loader": weight_loader})

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Process weights after loading from checkpoint."""
        # Ensure weights are not trainable
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.wscales = Parameter(layer.wscales.data, requires_grad=False)
        layer.smooth_factor = Parameter(layer.smooth_factor.data, requires_grad=False)
        layer.proj_down = Parameter(layer.proj_down.data, requires_grad=False)
        layer.proj_up = Parameter(layer.proj_up.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply SVDQ W4A4 quantized linear transformation.

        Uses Nunchaku's optimized kernels for W4A4 computation with SVD low-rank.
        """
        if not _check_nunchaku_available():
            raise ImportError(
                "nunchaku is required for SVDQ linear method. "
                "Install with: pip install nunchaku"
            )

        from nunchaku.kernels import gemm_w4a4_svd

        # Get quantized parameters
        qweight = layer.qweight
        wscales = layer.wscales
        smooth_factor = layer.smooth_factor
        proj_down = layer.proj_down
        proj_up = layer.proj_up

        # Apply SVDQ W4A4 computation
        # The kernel handles: smooth -> quantize act -> gemm -> dequant + low-rank correction
        out = gemm_w4a4_svd(
            x,
            qweight,
            wscales,
            smooth_factor,
            proj_down,
            proj_up,
            precision=layer.precision,
        )

        if bias is not None:
            out = out + bias

        return out


class NunchakuAWQLinearMethod(LinearMethodBase):
    """
    Linear method for Nunchaku AWQ W4A16 quantization.

    This implements the LinearMethodBase interface using Nunchaku's AWQW4A16Linear
    layer which provides standard W4A16 AWQ quantization.

    Args:
        group_size: Quantization group size (default: 64)
    """

    def __init__(self, group_size: int = 64):
        self.group_size = group_size
        self.pack_factor = 8  # 32 bits / 4 bits per weight

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
        Create quantized weights for AWQ W4A16 layer.

        For AWQ W4A16, we create:
        - qweight: Packed quantized weights (int32)
        - wscales: Weight scales
        - wzeros: Zero points
        """
        output_size_per_partition = sum(output_partition_sizes)

        # Packed quantized weights: shape (out_features // 4, in_features // 2)
        # Each int32 holds 8 x 4-bit values
        qweight = Parameter(
            torch.empty(
                output_size_per_partition // 4,
                input_size_per_partition // 2,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(qweight, {"input_dim": 1, "output_dim": 0})

        # Weight scales: shape (in_features // group_size, out_features)
        num_groups = input_size_per_partition // self.group_size
        wscales = Parameter(
            torch.empty(num_groups, output_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )

        # Zero points: shape (in_features // group_size, out_features)
        wzeros = Parameter(
            torch.empty(num_groups, output_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("wscales", wscales)
        layer.register_parameter("wzeros", wzeros)

        # Store metadata
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.group_size = self.group_size
        layer.pack_factor = self.pack_factor

        weight_loader = extra_weight_attrs.get("weight_loader")
        if weight_loader is not None:
            set_weight_attrs(qweight, {"weight_loader": weight_loader})
            set_weight_attrs(wscales, {"weight_loader": weight_loader})
            set_weight_attrs(wzeros, {"weight_loader": weight_loader})

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Process weights after loading from checkpoint."""
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.wscales = Parameter(layer.wscales.data, requires_grad=False)
        layer.wzeros = Parameter(layer.wzeros.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply AWQ W4A16 quantized linear transformation.

        Uses Nunchaku's optimized kernels for W4A16 computation.
        """
        if not _check_nunchaku_available():
            raise ImportError(
                "nunchaku is required for AWQ linear method. "
                "Install with: pip install nunchaku"
            )

        from nunchaku.kernels import gemm_w4a16

        qweight = layer.qweight
        wscales = layer.wscales
        wzeros = layer.wzeros

        # Apply AWQ W4A16 computation
        out = gemm_w4a16(x, qweight, wscales, wzeros)

        if bias is not None:
            out = out + bias

        return out


class NunchakuTransformerMethod:
    """
    Marker class indicating that the entire transformer should be replaced
    with a Nunchaku quantized version.

    This is used when the quantization is too complex to handle at the
    individual linear layer level (e.g., fused attention with multiple
    quantization schemes).
    """

    def __init__(
        self,
        precision: str = "int4",
        rank: int = 32,
        act_unsigned: bool = False,
        processor: str = "flashattn2",
    ):
        self.precision = precision
        self.rank = rank
        self.act_unsigned = act_unsigned
        self.processor = processor

    def should_replace_transformer(self) -> bool:
        """Indicates that full transformer replacement is needed."""
        return True

    def get_nunchaku_kwargs(self) -> dict:
        """Get kwargs for Nunchaku model loading."""
        return {
            "precision": self.precision,
            "rank": self.rank,
            "processor": self.processor,
        }


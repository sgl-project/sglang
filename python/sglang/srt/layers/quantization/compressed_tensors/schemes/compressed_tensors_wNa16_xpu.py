# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import torch
from compressed_tensors.quantization import ActivationOrdering

from sglang.srt.layers.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.utils import unpack_cols

__all__ = ["XPUCompressedTensorsWNA16"]

# The oneDNN weight-only matmul behind `_weight_int4pack_mm_with_scales_and_zeros`
# is 4-bit only; 8-bit `pack-quantized` checkpoints have no XPU kernel.
XPU_WNA16_SUPPORTED_BITS = [4]


class XPUCompressedTensorsWNA16(CompressedTensorsLinearScheme):
    """W4A16 ``pack-quantized`` on Intel GPUs via oneDNN's int4 matmul.

    ``torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros`` computes
    ``x @ ((nibble - zero_point) * scale).T``, which is exactly the
    compressed-tensors dequant for both 4-bit storage types:

    * ``uint4b8`` (symmetric) stores ``signed + 8``, so the zero point is the
      constant 8.
    * ``uint4`` (asymmetric) stores the raw nibble next to a packed zero point
      that unpacks to the per-group value the kernel subtracts.

    Only the symmetric path is exercised by a checkpoint today (Gemma 4 QAT
    W4A16); the asymmetric path follows from the same identity but is
    unverified for lack of an asymmetric ``pack-quantized`` checkpoint.
    """

    def __init__(
        self,
        strategy: str,
        num_bits: int,
        group_size: Optional[int] = None,
        symmetric: Optional[bool] = True,
        actorder: Optional[ActivationOrdering] = None,
    ):
        if num_bits not in XPU_WNA16_SUPPORTED_BITS:
            raise ValueError(
                f"XPU pack-quantized inference supports num_bits="
                f"{XPU_WNA16_SUPPORTED_BITS}, got {num_bits}."
            )
        if actorder == ActivationOrdering.GROUP:
            raise NotImplementedError(
                "XPU pack-quantized inference does not support activation "
                "reordering (actorder=group): the oneDNN int4 matmul has no "
                "g_idx input."
            )

        self.pack_factor = 32 // num_bits
        self.strategy = strategy
        self.symmetric = symmetric
        self.group_size = -1 if group_size is None else group_size

        if self.group_size == -1 and self.strategy != "channel":
            raise ValueError(
                "Group or channelwise quantization is required, but found no "
                "group size and strategy is not channelwise."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        # `min_capability` ranks NVIDIA SM numbers, which XPU devices do not
        # have. Kernel availability is decided by the scheme dispatch instead.
        return 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)

        if params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "The XPU int4 matmul requires a float16 or bfloat16 activation "
                f"dtype, got {params_dtype}."
            )
        if input_size_per_partition % self.pack_factor != 0:
            raise ValueError(
                f"input_size_per_partition={input_size_per_partition} must be a "
                f"multiple of {self.pack_factor} to unpack the 4-bit weights."
            )

        self.input_size_per_partition = input_size_per_partition
        self.output_size_per_partition = output_size_per_partition

        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = input_size != input_size_per_partition
        # Channelwise scales cover the whole of K, so a row-parallel shard
        # cannot slice them and every rank keeps the full copy.
        partition_scales = not (self.group_size == -1 and row_parallel)

        scales_and_zp_size = input_size // group_size
        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=self.pack_factor,
            packed_dim=1,
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
        )

        weight_scale_args = {
            "weight_loader": weight_loader,
            "data": torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            ),
        }
        zeros_args = {
            "weight_loader": weight_loader,
            "data": torch.zeros(
                output_size_per_partition // self.pack_factor,
                scales_and_zp_size,
                dtype=torch.int32,
            ),
        }

        if not partition_scales:
            weight_scale = ChannelQuantScaleParameter(output_dim=0, **weight_scale_args)
            if not self.symmetric:
                qzeros = PackedColumnParameter(
                    output_dim=0,
                    packed_dim=0,
                    packed_factor=self.pack_factor,
                    **zeros_args,
                )
        else:
            weight_scale = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, **weight_scale_args
            )
            if not self.symmetric:
                qzeros = PackedvLLMParameter(
                    input_dim=1,
                    output_dim=0,
                    packed_dim=0,
                    packed_factor=self.pack_factor,
                    **zeros_args,
                )

        # A 2D array holding the original (pre-packing) weight shape.
        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader
        )

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)
        if not self.symmetric:
            layer.register_parameter("weight_zero_point", qzeros)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        k = self.input_size_per_partition
        n = self.output_size_per_partition
        group_size = self.group_size if self.group_size != -1 else k
        grouped_k = layer.weight_scale.shape[1]

        # `weight_packed` is [N, K/8] int32 with nibble i of a word at bit 4i,
        # so a little-endian byte view is the [N, K/2] uint8 layout the repack
        # helper expects (byte j = nibbles 2j low, 2j+1 high).
        packed = layer.weight_packed.data.contiguous()
        qweight = torch.ops.aten._convert_weight_to_int4pack(
            packed.view(torch.uint8).contiguous(), 8
        )

        # The kernel indexes scales/zero points as [K/group, N].
        scale = layer.weight_scale.data.t().contiguous()

        if self.symmetric:
            # uint4b8: the stored nibble is `signed + 8`.
            zero_point = torch.full(
                (grouped_k, n), 8, dtype=torch.int8, device=qweight.device
            )
        else:
            # [N/8, K/group] packed along N -> [K/group, N] raw nibbles.
            zero_point = (
                unpack_cols(layer.weight_zero_point.data.t(), 4, grouped_k, n)
                .to(torch.int8)
                .contiguous()
            )

        layer.weight_packed = torch.nn.Parameter(qweight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        layer.weight_zero_point = torch.nn.Parameter(zero_point, requires_grad=False)
        # The kernel derives the group count from the scale rows; for the
        # channelwise case that is one group spanning this rank's whole K.
        self.kernel_group_size = group_size

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        out_shape = (*x.shape[:-1], self.output_size_per_partition)

        # The kernel takes a 2D activation and pads N up to its tile width.
        out = torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros(
            x.reshape(-1, x.shape[-1]),
            layer.weight_packed,
            self.kernel_group_size,
            layer.weight_scale,
            layer.weight_zero_point,
        )
        out = out[:, : self.output_size_per_partition].reshape(out_shape)

        if bias is not None:
            out = out + bias
        return out

# SPDX-License-Identifier: Apache-2.0
"""GPTQ int4 dense linear for ROCm (no Marlin).

Weights are repacked once to the ExLlama-shuffled [N, K/8] layout and served
by the Triton W4A16 GEMM in ``gptq_triton``.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.quantization.gptq_triton import (
    gptq_w4a16_skinny_gemm,
    repack_gptq_w4_to_skinny,
)
from sglang.srt.layers.quantization.utils import replace_parameter

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

SUPPORTED_GROUP_SIZES = {-1, 32, 64, 128, 256}

SYM_ZERO_POINT = 8


class GPTQTritonLinearKernel:
    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config
        self.use_v2_format = getattr(quant_config, "checkpoint_format", "") == "gptq_v2"
        # Set by GPTQLinearScheme for row-parallel layers; unused here.
        self.use_shuffle = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.quant_config.weight_bits != 4:
            raise NotImplementedError(
                "GPTQ on ROCm without Marlin supports 4-bit weights only, "
                f"got {self.quant_config.weight_bits}-bit."
            )
        if self.quant_config.desc_act:
            raise NotImplementedError(
                "GPTQ on ROCm without Marlin does not support desc_act=True."
            )
        group_size = self.quant_config.group_size
        if group_size not in SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"GPTQ on ROCm requires group_size in {SUPPORTED_GROUP_SIZES}, "
                f"got {group_size}."
            )

        qweight = layer.qweight.data
        k = qweight.shape[0] * 8

        # Like Marlin, ignore qzeros for sym checkpoints: some AutoRound exports
        # (e.g. Qwen3.6 MTP tensors) store them without the v1 "zero - 1" offset.
        if getattr(self.quant_config, "sym", False):
            qzeros = None
        else:
            # GPTQ qzeros are [G, N/8] with sequential nibbles; v1 stores zero - 1.
            packed_zeros = layer.qzeros.data
            if not self.use_v2_format:
                nibbles = [((packed_zeros >> s) + 1) & 0xF for s in range(0, 32, 4)]
                packed_zeros = functools.reduce(
                    torch.bitwise_or,
                    (z << s for z, s in zip(nibbles, range(0, 32, 4))),
                )
            sym_word = sum(SYM_ZERO_POINT << s for s in range(0, 32, 4))
            sym_word -= 1 << 32  # 0x88888888 as a signed int32
            if bool((packed_zeros == sym_word).all()):
                qzeros = None
            else:
                qzeros = packed_zeros.t().contiguous()  # [N/8, G]

        replace_parameter(layer, "qweight", repack_gptq_w4_to_skinny(qweight, k))
        replace_parameter(layer, "scales", layer.scales.data.t().contiguous())
        if qzeros is None:
            del layer.qzeros
            layer.qzeros = None
        else:
            replace_parameter(layer, "qzeros", qzeros)
        if hasattr(layer, "g_idx"):
            del layer.g_idx

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n = layer.qweight.shape[0]
        out_shape = x.shape[:-1] + (n,)
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out = gptq_w4a16_skinny_gemm(
            x_2d,
            layer.qweight,
            layer.scales,
            self.quant_config.group_size,
            layer.qzeros,
        )
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)

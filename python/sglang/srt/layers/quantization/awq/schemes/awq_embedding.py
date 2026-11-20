# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from sglang.srt.layers.parameter import ModelWeightParameter

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.awq.awq import AWQConfig, AWQMarlinConfig

__all__ = ["AWQEmbeddingScheme"]


class AWQEmbeddingScheme:
    """Row-major INT4 token embeddings, dequantized after lookup."""

    def __init__(self, quant_config: AWQConfig | AWQMarlinConfig) -> None:
        settings = quant_config.embedding_quantization
        if (settings["bits"], settings["layout"], settings["zero_point"]) != (
            4,
            "row_major",
            True,
        ):
            raise ValueError("AWQ embeddings require row_major INT4 with zero points")
        self.group_size = settings["group_size"]

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        params_dtype: torch.dtype,
        weight_loader,
    ) -> None:
        self.params_dtype = params_dtype
        num_rows = sum(output_partition_sizes)
        num_groups = input_size_per_partition // self.group_size
        for name, width, dtype in (
            ("qweight", input_size_per_partition // 8, torch.int32),
            ("qzeros", num_groups // 8, torch.int32),
            ("scales", num_groups, torch.float16),
        ):
            layer.register_parameter(
                name,
                ModelWeightParameter(
                    data=torch.empty(num_rows, width, dtype=dtype),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                ),
            )
        layer.register_buffer(
            "shifts", torch.arange(0, 32, 4, dtype=torch.int32), persistent=False
        )

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        packed = F.embedding(input_, layer.qweight)
        weight = ((packed.unsqueeze(-1) >> layer.shifts) & 15).flatten(-2)
        packed_zeros = F.embedding(input_, layer.qzeros)
        zeros = ((packed_zeros.unsqueeze(-1) >> layer.shifts) & 15).flatten(-2)
        scales = F.embedding(input_, layer.scales).float()
        # Preserve checkpoint FP16 scales until the final activation cast.
        weight = weight.unflatten(-1, (-1, self.group_size)).float()
        output = (weight - zeros.float().unsqueeze(-1)) * scales.unsqueeze(-1)
        return output.flatten(-2).to(self.params_dtype)

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from sglang.kernels.ops.embeddings.awq_embedding import awq_embedding
from sglang.srt.layers.parameter import ModelWeightParameter
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.awq.awq import AWQConfig, AWQMarlinConfig

__all__ = ["AWQEmbeddingScheme"]

_is_cuda = is_cuda()


class AWQEmbeddingScheme:
    """Row-major INT4 token embeddings, dequantized after lookup."""

    def __init__(self, quant_config: AWQConfig | AWQMarlinConfig) -> None:
        settings = quant_config.embedding_quantization
        if (
            settings["bits"] != 4
            or settings["layout"] != "row_major"
            or not settings["zero_point"]
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
        output_size_per_partition = sum(output_partition_sizes)
        num_groups = input_size_per_partition // self.group_size

        qweight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 8,
                dtype=torch.int32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        qzeros = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                num_groups // 8,
                dtype=torch.int32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        scales = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                num_groups,
                dtype=torch.float16,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)
        layer.register_buffer(
            "shifts", torch.arange(0, 32, 4, dtype=torch.int32), persistent=False
        )

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        if _is_cuda and input_.is_cuda:
            return awq_embedding(
                input_ids=input_,
                qweight=layer.qweight,
                qzeros=layer.qzeros,
                scales=layer.scales,
                group_size=self.group_size,
                dtype=self.params_dtype,
            )
        packed = F.embedding(input_, layer.qweight)
        weight = ((packed.unsqueeze(-1) >> layer.shifts) & 15).flatten(-2)
        packed_zeros = F.embedding(input_, layer.qzeros)
        zeros = ((packed_zeros.unsqueeze(-1) >> layer.shifts) & 15).flatten(-2)
        scales = F.embedding(input_, layer.scales).float()
        # Preserve checkpoint FP16 scales until the final activation cast.
        weight = weight.unflatten(-1, (-1, self.group_size)).float()
        output = (weight - zeros.float().unsqueeze(-1)) * scales.unsqueeze(-1)
        return output.flatten(-2).to(self.params_dtype)

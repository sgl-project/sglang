# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .gguf_linear import GGUFLinearScheme

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.gguf.gguf import GGUFConfig

__all__ = ["GGUFEmbeddingScheme", "GGUFAscendEmbeddingScheme"]


class GGUFEmbeddingScheme(GGUFLinearScheme):
    def _init_kernel(self, quant_config: "GGUFConfig"):
        from sglang.srt.hardware_backend.gpu.quantization.gguf_kernels import (
            GGUFEmbeddingKernel,
        )

        return GGUFEmbeddingKernel(quant_config)

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return self.kernel.embedding(layer, x)


class GGUFAscendEmbeddingScheme(GGUFEmbeddingScheme):
    def _init_kernel(self, quant_config: "GGUFConfig"):
        from sglang.srt.hardware_backend.npu.quantization.gguf_kernels import (
            GGUFAscendEmbeddingKernel,
        )

        return GGUFAscendEmbeddingKernel(quant_config)

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return self.kernel.embedding(layer, x)

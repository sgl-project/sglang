# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.hardware_backend.cpu.quantization.awq_kernels import (
    AWQIntelAMXLinearKernel,
    AWQIntelAMXMoEKernel,
)
from sglang.srt.layers.moe import MoeRunnerConfig

from .awq_linear import AWQLinearScheme
from .awq_moe import AWQMoEScheme

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.awq.awq import AWQConfig

__all__ = ["AWQIntelAMXLinearScheme", "AWQIntelAMXMoEScheme"]


class AWQIntelAMXLinearScheme(AWQLinearScheme):
    """Linear scheme for AWQ on Intel CPU with AMX."""

    def _init_kernel(self, quant_config: AWQConfig):
        return AWQIntelAMXLinearKernel(quant_config)


class AWQIntelAMXMoEScheme(AWQMoEScheme):
    """MoE scheme for AWQ on Intel CPU with AMX."""

    def _init_kernel(self, quant_config: AWQConfig):
        return AWQIntelAMXMoEKernel(quant_config)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.kernel.create_moe_runner(layer, moe_runner_config)

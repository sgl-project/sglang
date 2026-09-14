# SPDX-License-Identifier: Apache-2.0

from sglang.srt.layers.quantization.base_scheme import BaseLinearScheme, BaseMoEScheme

__all__ = ["GPTQLinearSchemeBase", "GPTQMoESchemeBase"]


class GPTQLinearSchemeBase(BaseLinearScheme):
    """Base class for GPTQ linear schemes. The contract is BaseLinearScheme's."""


class GPTQMoESchemeBase(BaseMoEScheme):
    """Base class for GPTQ MoE schemes. The contract is BaseMoEScheme's."""

# SPDX-License-Identifier: Apache-2.0

from sglang.srt.layers.quantization.base_scheme import BaseLinearScheme, BaseMoEScheme

__all__ = ["AWQLinearSchemeBase", "AWQMoESchemeBase"]


class AWQLinearSchemeBase(BaseLinearScheme):
    """Base class for AWQ linear schemes. The contract is BaseLinearScheme's."""


class AWQMoESchemeBase(BaseMoEScheme):
    """Base class for AWQ MoE schemes. The contract is BaseMoEScheme's."""

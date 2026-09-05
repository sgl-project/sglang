# SPDX-License-Identifier: Apache-2.0

from sglang.srt.layers.quantization.base_scheme import BaseLinearScheme, BaseMoEScheme

__all__ = ["QuarkLinearScheme", "QuarkMoEScheme"]


class QuarkLinearScheme(BaseLinearScheme):
    """
    Base class for the linear schemes supported by Quark. The contract is
    BaseLinearScheme's.
    """


class QuarkMoEScheme(BaseMoEScheme):
    """
    Base class for the MoE schemes supported by Quark. The contract is
    BaseMoEScheme's.
    """

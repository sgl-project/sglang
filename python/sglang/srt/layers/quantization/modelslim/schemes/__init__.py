# SPDX-License-Identifier: Apache-2.0

# NOTE: Import order is critical to avoid circular dependency.
# modelslim_mxfp8 imports ModelSlimLinearScheme from this package,
# so the base class must be imported first.
# isort: off
from .modelslim_scheme import ModelSlimLinearScheme, ModelSlimMoEScheme
from .modelslim_mxfp8 import ModelSlimMXFP8Scheme
from .modelslim_mxfp4_w4a8 import ModelSlimMXFP4W4A8Scheme

# isort: on
from .modelslim_w4a4_int4 import ModelSlimW4A4Int4
from .modelslim_w4a4_int4_moe import ModelSlimW4A4Int4MoE
from .modelslim_w4a8_int8 import ModelSlimW4A8Int8
from .modelslim_w4a8_int8_moe import ModelSlimW4A8Int8MoE
from .modelslim_w8a8_int8 import ModelSlimW8A8Int8
from .modelslim_w8a8_int8_moe import ModelSlimW8A8Int8MoE

__all__ = [
    "ModelSlimLinearScheme",
    "ModelSlimMoEScheme",
    "ModelSlimMXFP8Scheme",
    "ModelSlimMXFP4W4A8Scheme",
    "ModelSlimW8A8Int8",
    "ModelSlimW4A4Int4",
    "ModelSlimW4A4Int4MoE",
    "ModelSlimW4A8Int8",
    "ModelSlimW4A8Int8MoE",
    "ModelSlimW8A8Int8MoE",
]

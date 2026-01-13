# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from enum import Enum

LIBRARY_NAME = "sonicmoe"
TENSORMAP = "tensormap"


class KernelBackendMoE(Enum):
    scattermoe = "scattermoe"
    torch = "torch"
    sonicmoe = "sonicmoe"


class ActivationType(Enum):
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    REGLU = "reglu"

    RELU_SQ = "relu_sq"
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"


def is_glu(activation_type: ActivationType):
    return activation_type in [
        ActivationType.SWIGLU,
        ActivationType.REGLU,
        ActivationType.GEGLU,
    ]

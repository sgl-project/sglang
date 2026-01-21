# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import os
from contextlib import contextmanager

_IS_USING_QUACK_GEMM = os.getenv("USE_QUACK_GEMM", "0") == "1"


@contextmanager
def enable_quack_gemm(enable: bool = True):
    global _IS_USING_QUACK_GEMM

    previous_value = _IS_USING_QUACK_GEMM
    _IS_USING_QUACK_GEMM = enable

    yield

    _IS_USING_QUACK_GEMM = previous_value


def is_using_quack_gemm() -> bool:
    return _IS_USING_QUACK_GEMM

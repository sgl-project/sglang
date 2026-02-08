# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

from .cutlass_dsl import (
    Constexpr,
    as_numeric,
    min,
    max,
    and_,
    or_,
    all_,
    any_,
    not_,
    all_,
    any_,
    select_,
    # Control-flow without AST pre-processor
    if_generate,
    for_generate,
    LoopUnroll,
    while_generate,
    yield_out,
    # Control-flow with AST pre-processor
    range_constexpr,
    range_dynamic,
    const_expr,
    dynamic_expr,
    # Data types
    dtype,  # Provides conversions to types inheriting from NumericType
    DSLRuntimeError,
    JitArgAdapterRegistry,
    # Construction utilities for user-defined classes
    extract_mlir_values,
    new_from_mlir_values,
)

from .cute.typing import *

# Utilities not belonging to CuTe
from . import utils as utils

# Used as internal symbol
from . import cutlass_dsl as _dsl

# Aliases
LaunchConfig = _dsl.BaseDSL.LaunchConfig
register_jit_arg_adapter = _dsl.JitArgAdapterRegistry.register_jit_arg_adapter
gpu = _dsl.cutlass_gpu
cuda = _dsl.cuda_helpers

CACHE_FILE = "compiled_cache.db"

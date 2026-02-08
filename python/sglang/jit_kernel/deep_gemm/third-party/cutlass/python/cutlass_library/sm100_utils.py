#################################################################################################
#
# Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Utilities for enumerating CUTLASS library SM100 kernels
"""

import argparse
import enum
from itertools import product
import math
import logging
import os.path
import shutil
import sys
import copy
from typing import Any, Optional, Sequence, Tuple, List, Union, Callable

try:
  import builtins
  if hasattr(builtins, "CUTLASS_IGNORE_PACKAGE") and CUTLASS_IGNORE_PACKAGE == True:
    raise ImportError("Disabling attempt to import cutlass_library")
  from cutlass_library.library import *
except ImportError:
  from library import *

#### Step 0: define levels

# One integer level controls multiple "generators" and how many
# combinations they generate. That is the "global" level.
# "Generators" are WGMMA shapes, MMA multipliers, cluster sizes, and
# anything that is eventually involved in the Cartesian product
# which yields our kernel configurations.
# For simplicity, each generator defines their own levels, 
# starting from 0. As a rule we assume 10 or fewer levels, making
# their level a digit.
# The "global" level simply stacks these digits and represents them
# as a single integer.
# 
# For example, level 500 indicates cluster sizes are at level 5, MMA
# multipliers are at level 0, and WGMMA shapes are at level 0 as well.
#
# Here we define the global level to generator level mappings.


def get_tcgen05_level_from_global_level(global_level: int):
    return global_level % 10

def get_mma_level_from_global_level(global_level: int):
    return (global_level // 10) % 10


def get_cluster_level_from_global_level(global_level: int):
    return (global_level // 100) % 10


def get_pruning_level_from_global_level(global_level: int):
    return (global_level // 1000) % 10


#### Step 1: generate MMA instruction shapes based on levels

try:
    from .sm100_shapes import *
except:
    from sm100_shapes import *

###########

def generate_tf32_math_instructions_sm100(level: int):
    """
    Generate all TensorOp math instructions for TF32 MMA that are supported by SM100 at or above the given level.

    Args:
        level: The global level to generate math instructions for.

    Returns:
        A tuple of two lists of MathInstruction objects. 
        The first list contains the math instructions for 1SM, and the second list contains the math instructions for 2SM.
    """
    tcgen05_level = get_tcgen05_level_from_global_level(level)
    math_instructions_1sm = []
    math_instructions_2sm = []

    shapes_1sm = [
        shape for shape, min_level in SM100_MMA_SHAPES_TF32_DENSE_1SM.items() if tcgen05_level >= min_level
    ]
    shapes_2sm = [
        shape for shape, min_level in SM100_MMA_SHAPES_TF32_DENSE_2SM.items() if tcgen05_level >= min_level
    ]

    for shape in shapes_1sm:
        math_instructions_1sm.append(
          MathInstruction(
              shape,
              DataType.tf32, DataType.tf32, DataType.f32,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add)
        )

    for shape in shapes_2sm:
        math_instructions_2sm.append(
          MathInstruction(
              shape,
              DataType.tf32, DataType.tf32, DataType.f32,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add)
        )
 
    return math_instructions_1sm, math_instructions_2sm

def generate_16b_math_instructions_sm100(level: int):
    """
    Generate all TensorOp math instructions for 16b MMA that are supported by SM100 at or above the given level.

    Args:
        level: The global level to generate math instructions for.

    Returns:
        A tuple of two lists of MathInstruction objects. 
        The first list contains the math instructions for 1SM, and the second list contains the math instructions for 2SM.
    """
    tcgen05_level = get_tcgen05_level_from_global_level(level)
    math_instructions_1sm = []
    math_instructions_2sm = []

    shapes_1sm = [
        shape for shape, min_level in SM100_MMA_SHAPES_16b_DENSE_1SM.items() if tcgen05_level >= min_level
    ]
    shapes_2sm = [
        shape for shape, min_level in SM100_MMA_SHAPES_16b_DENSE_2SM.items() if tcgen05_level >= min_level
    ]

    for shape in shapes_1sm:
        math_instructions_1sm.append(
          MathInstruction(
              shape,
              DataType.f16, DataType.f16, DataType.f16,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add)
        )
        math_instructions_1sm.append(
          MathInstruction(
              shape,
              DataType.f16, DataType.f16, DataType.f32,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add)
        )
        math_instructions_1sm.append(
          MathInstruction(
              shape,
              DataType.bf16, DataType.bf16, DataType.f32,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add)
        )


    for shape in shapes_2sm:
        math_instructions_2sm.append(
          MathInstruction(
              shape,
              DataType.f16, DataType.f16, DataType.f16,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add)
        )
        math_instructions_2sm.append(
          MathInstruction(
              shape,
              DataType.f16, DataType.f16, DataType.f32,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add)
        )
        math_instructions_2sm.append(
          MathInstruction(
              shape,
              DataType.bf16, DataType.bf16, DataType.f32,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add)
        )
 
    return math_instructions_1sm, math_instructions_2sm


def generate_fp8_math_instructions_sm100(level: int, enable_runtime_dtype = True, enable_compile_time_dtype = True):
    """
    Generate all TensorOp math instructions for FP8 MMA that are supported by SM100 at or above the given level.

    Args:
        level: The global level to generate math instructions for.
        enable_runtime_dtype: Whether to generate runtime dtype math instructions.
        enable_compile_time_dtype: Whether to generate compile time dtype math instructions.

    Returns:
        A tuple of two lists of MathInstruction objects. 
        The first list contains the math instructions for 1SM, and the second list contains the math instructions for 2SM.
    """

    tcgen05_level = get_tcgen05_level_from_global_level(level)
    pruning_level = get_pruning_level_from_global_level(level)
    math_instructions_1sm = []
    math_instructions_2sm = []

    shapes_1sm = [
        shape for shape, min_level in SM100_MMA_SHAPES_F8F6F4_DENSE_1SM.items() if tcgen05_level >= min_level
    ]
    shapes_2sm = [
        shape for shape, min_level in SM100_MMA_SHAPES_F8F6F4_DENSE_2SM.items() if tcgen05_level >= min_level
    ]

    for shape in shapes_1sm:
        if enable_runtime_dtype:
            math_instructions_1sm.append(
              MathInstruction(
                  shape,
                  DataType.f8, DataType.f8, DataType.f32,
                  OpcodeClass.TensorOp,
                  MathOperation.multiply_add)
            )
        if enable_compile_time_dtype:    
            math_instructions_1sm.append(
              MathInstruction(
                  shape,
                  DataType.e4m3, DataType.e4m3, DataType.f32,
                  OpcodeClass.TensorOp,
                  MathOperation.multiply_add)
            )
            math_instructions_1sm.append(
              MathInstruction(
                  shape,
                  DataType.e5m2, DataType.e4m3, DataType.f32,
                  OpcodeClass.TensorOp,
                  MathOperation.multiply_add)
            )
            math_instructions_1sm.append(
              MathInstruction(
                  shape,
                  DataType.e4m3, DataType.e5m2, DataType.f32,
                  OpcodeClass.TensorOp,
                  MathOperation.multiply_add)
            )
            if pruning_level >= 2:
                math_instructions_1sm.append(
                  MathInstruction(
                      shape,
                      DataType.e5m2, DataType.e5m2, DataType.f32,
                      OpcodeClass.TensorOp,
                      MathOperation.multiply_add)
                )

    for shape in shapes_2sm:
        if enable_runtime_dtype:
            math_instructions_2sm.append(
              MathInstruction(
                  shape,
                  DataType.f8, DataType.f8, DataType.f32,
                  OpcodeClass.TensorOp,
                  MathOperation.multiply_add)
            )
        if enable_compile_time_dtype:    
            math_instructions_2sm.append(
              MathInstruction(
                  shape,
                  DataType.e4m3, DataType.e4m3, DataType.f32,
                  OpcodeClass.TensorOp,
                  MathOperation.multiply_add)
            )
            math_instructions_2sm.append(
              MathInstruction(
                  shape,
                  DataType.e5m2, DataType.e4m3, DataType.f32,
                  OpcodeClass.TensorOp,
                  MathOperation.multiply_add)
            )
            math_instructions_2sm.append(
              MathInstruction(
                  shape,
                  DataType.e4m3, DataType.e5m2, DataType.f32,
                  OpcodeClass.TensorOp,
                  MathOperation.multiply_add)
            )
            if pruning_level >= 2:
                math_instructions_2sm.append(
                  MathInstruction(
                      shape,
                      DataType.e5m2, DataType.e5m2, DataType.f32,
                      OpcodeClass.TensorOp,
                      MathOperation.multiply_add)
                )

    return math_instructions_1sm, math_instructions_2sm

def generate_f8f6f4_math_instructions_sm100(level: int, enable_runtime_dtype = True, enable_compile_time_dtype = True):
    """
    Generate all TensorOp math instructions for FP8 FP6 and FP4 MMA that are supported by SM100 at or above the given level.

    Args:
        level: The global level to generate math instructions for.
        enable_runtime_dtype: Whether to generate runtime dtype math instructions.
        enable_compile_time_dtype: Whether to generate compile time dtype math instructions.

    Returns:
        A tuple of two lists of MathInstruction objects. 
        The first list contains the math instructions for 1SM, and the second list contains the math instructions for 2SM.
    """

    tcgen05_level = get_tcgen05_level_from_global_level(level)
    math_instructions_1sm = []
    math_instructions_2sm = []

    shapes_1sm = [
        shape for shape, min_level in SM100_MMA_SHAPES_F8F6F4_DENSE_1SM.items() if tcgen05_level >= min_level
    ]
    shapes_2sm = [
        shape for shape, min_level in SM100_MMA_SHAPES_F8F6F4_DENSE_2SM.items() if tcgen05_level >= min_level
    ]

    for shape in shapes_1sm:
        if enable_runtime_dtype:

            runtime_types = [ DataType.f8, DataType.f6, DataType.f4 ]

            for a_type, b_type in product(runtime_types, repeat=2):
                math_instructions_1sm.append(
                  MathInstruction(
                      shape,
                      a_type, b_type, DataType.f32,
                      OpcodeClass.TensorOp,
                      MathOperation.multiply_add)
                )

        if enable_compile_time_dtype:
            compile_time_types = [ DataType.e4m3, DataType.e5m2, DataType.e3m2, DataType.e2m1 ]

            for a_type, b_type in product(compile_time_types, repeat=2):
                math_instructions_1sm.append(
                  MathInstruction(
                      shape,
                      a_type, b_type, DataType.f32,
                      OpcodeClass.TensorOp,
                      MathOperation.multiply_add)
                )


    for shape in shapes_2sm:
        if enable_runtime_dtype:

            runtime_types = [ DataType.f8, DataType.f6, DataType.f4 ]

            for a_type, b_type in product(runtime_types, repeat=2):
                math_instructions_2sm.append(
                  MathInstruction(
                      shape,
                      a_type, b_type, DataType.f32,
                      OpcodeClass.TensorOp,
                      MathOperation.multiply_add)
                )

        if enable_compile_time_dtype:
            compile_time_types = [ DataType.e4m3, DataType.e5m2, DataType.e3m2, DataType.e2m1 ]

            for a_type, b_type in product(compile_time_types, repeat=2):
                math_instructions_2sm.append(
                  MathInstruction(
                      shape,
                      a_type, b_type, DataType.f32,
                      OpcodeClass.TensorOp,
                      MathOperation.multiply_add)
                )

    return math_instructions_1sm, math_instructions_2sm

def generate_mxf8f6f4_math_instructions_sm100(level: int, enable_runtime_dtype = True, enable_compile_time_dtype = True):
    """
    Generate all BlockScaledTensorOp math instructions for MXFP8, MXFP6, and MXFP4 MMA that are supported by SM100 at or above the given level.

    Args:
        level: The global level to generate math instructions for.
        enable_runtime_dtype: Whether to generate runtime dtype math instructions.
        enable_compile_time_dtype: Whether to generate compile time dtype math instructions.

    Returns:
        A tuple of two lists of MathInstruction objects. 
        The first list contains the math instructions for 1SM, and the second list contains the math instructions for 2SM.
    """

    tcgen05_level = get_tcgen05_level_from_global_level(level)
    pruning_level = get_pruning_level_from_global_level(level)

    math_instructions_1sm = []
    math_instructions_2sm = []

    shapes_1sm = [
        shape for shape, min_level in SM100_MMA_SHAPES_MXF8F6F4_DENSE_1SM.items() if tcgen05_level >= min_level
    ]
    shapes_2sm = [
        shape for shape, min_level in SM100_MMA_SHAPES_MXF8F6F4_DENSE_2SM.items() if tcgen05_level >= min_level
    ]

    for shape in shapes_1sm:
        if enable_runtime_dtype:

            runtime_types = [ DataType.f8, DataType.f6, DataType.f4 ]

            for a_type, b_type in product(runtime_types, repeat=2):

                if pruning_level < 2 and ((a_type == DataType.f8 or b_type == DataType.f8)):
                    continue

                math_instructions_1sm.append(
                  MathInstruction(
                      shape,
                      a_type, b_type, DataType.f32,
                      OpcodeClass.BlockScaledTensorOp,
                      MathOperation.multiply_add,
                      DataType.ue8m0)
                )

        if enable_compile_time_dtype:
            compile_time_types = [ DataType.e4m3, 
                                   DataType.e5m2, 
                                   DataType.e3m2, 
                                   DataType.e2m3,
                                   DataType.e2m1 ]

            for a_type, b_type in product(compile_time_types, repeat=2):
                math_instructions_1sm.append(
                  MathInstruction(
                      shape,
                      a_type, b_type, DataType.f32,
                      OpcodeClass.BlockScaledTensorOp,
                      MathOperation.multiply_add,
                      DataType.ue8m0)
                )


    for shape in shapes_2sm:
        if enable_runtime_dtype:

            runtime_types = [ DataType.f8, DataType.f6, DataType.f4 ]

            for a_type, b_type in product(runtime_types, repeat=2):

                if pruning_level < 2 and ((a_type == DataType.f8 or b_type == DataType.f8)):
                    continue

                math_instructions_2sm.append(
                  MathInstruction(
                      shape,
                      a_type, b_type, DataType.f32,
                      OpcodeClass.BlockScaledTensorOp,
                      MathOperation.multiply_add,
                      DataType.ue8m0)
                )

        if enable_compile_time_dtype:
            compile_time_types = [ DataType.e4m3, 
                                   DataType.e5m2, 
                                   DataType.e3m2, 
                                   DataType.e2m3,
                                   DataType.e2m1 ]

            for a_type, b_type in product(compile_time_types, repeat=2):
                math_instructions_2sm.append(
                  MathInstruction(
                      shape,
                      a_type, b_type, DataType.f32,
                      OpcodeClass.BlockScaledTensorOp,
                      MathOperation.multiply_add,
                      DataType.ue8m0)
                )

    return math_instructions_1sm, math_instructions_2sm

def generate_mxf4nvf4_math_instructions_sm100(level: int, enable_runtime_dtype = True, enable_compile_time_dtype = True):
    """
    Generate all BlockScaledTensorOp math instructions for MXFP4 and MXFP4 MMA that are supported by SM100 at or above the given level.

    Args:
        level: The global level to generate math instructions for.
        enable_runtime_dtype: Whether to generate runtime dtype math instructions.
        enable_compile_time_dtype: Whether to generate compile time dtype math instructions.

    Returns:
        A tuple of two lists of MathInstruction objects. 
        The first list contains the math instructions for 1SM, and the second list contains the math instructions for 2SM.
    """
    tcgen05_level = get_tcgen05_level_from_global_level(level)
    math_instructions_1sm = []
    math_instructions_2sm = []

    shapes_1sm = [
        shape for shape, min_level in SM100_MMA_SHAPES_MXF4NVF4_DENSE_1SM.items() if tcgen05_level >= min_level
    ]
    shapes_2sm = [
        shape for shape, min_level in SM100_MMA_SHAPES_MXF4NVF4_DENSE_2SM.items() if tcgen05_level >= min_level
    ]

    for shape in shapes_1sm:
        if enable_runtime_dtype:

            runtime_types = [ DataType.f4 ]

            for a_type, b_type in product(runtime_types, repeat=2):
                math_instructions_1sm.append(
                  MathInstruction(
                      shape,
                      a_type, b_type, DataType.f32,
                      OpcodeClass.BlockScaledTensorOp,
                      MathOperation.multiply_add,
                      DataType.ue8m0)
                )
                math_instructions_1sm.append(
                  MathInstruction(
                      shape,
                      a_type, b_type, DataType.f32,
                      OpcodeClass.BlockScaledTensorOp,
                      MathOperation.multiply_add,
                      DataType.ue4m3)
                )


        if enable_compile_time_dtype:
            compile_time_types = [ DataType.e2m1, 
                                 ]

            for a_type, b_type in product(compile_time_types, repeat=2):
                math_instructions_1sm.append(
                  MathInstruction(
                      shape,
                      a_type, b_type, DataType.f32,
                      OpcodeClass.BlockScaledTensorOp,
                      MathOperation.multiply_add,
                      DataType.ue8m0)
                )
                math_instructions_1sm.append(
                  MathInstruction(
                      shape,
                      a_type, b_type, DataType.f32,
                      OpcodeClass.BlockScaledTensorOp,
                      MathOperation.multiply_add,
                      DataType.ue4m3)
                )


    for shape in shapes_2sm:
        if enable_runtime_dtype:

            runtime_types = [ DataType.f4 ]

            for a_type, b_type in product(runtime_types, repeat=2):
                math_instructions_2sm.append(
                  MathInstruction(
                      shape,
                      a_type, b_type, DataType.f32,
                      OpcodeClass.BlockScaledTensorOp,
                      MathOperation.multiply_add,
                      DataType.ue8m0)
                )
                math_instructions_2sm.append(
                  MathInstruction(
                      shape,
                      a_type, b_type, DataType.f32,
                      OpcodeClass.BlockScaledTensorOp,
                      MathOperation.multiply_add,
                      DataType.ue4m3)
                )


        if enable_compile_time_dtype:
            compile_time_types = [ DataType.e2m1, 
                                 ]

            for a_type, b_type in product(compile_time_types, repeat=2):
                math_instructions_2sm.append(
                  MathInstruction(
                      shape,
                      a_type, b_type, DataType.f32,
                      OpcodeClass.BlockScaledTensorOp,
                      MathOperation.multiply_add,
                      DataType.ue8m0)
                )
                math_instructions_2sm.append(
                  MathInstruction(
                      shape,
                      a_type, b_type, DataType.f32,
                      OpcodeClass.BlockScaledTensorOp,
                      MathOperation.multiply_add,
                      DataType.ue4m3)
                )


    return math_instructions_1sm, math_instructions_2sm


def generate_cluster_shapes_sm100(level: int, change_priority_func : Union[Callable, None] = None):
    """
    Generate all cluster shapes for SM100 at or above the given level.

    Args:
        level: The global level to generate cluster shapes for.

    Returns:
        A tuple of two lists of cluster shapes. 
        The first list contains the cluster shapes for 1SM, and the second list contains the cluster shapes for 2SM.
    """
    cluster_level = get_cluster_level_from_global_level(level)

    assert cluster_level >= 4

    if change_priority_func is not None:
        SM100_CLUSTER_SHAPES_1SM_CPY = copy.deepcopy(SM100_CLUSTER_SHAPES_1SM)
        SM100_CLUSTER_SHAPES_2SM_CPY = copy.deepcopy(SM100_CLUSTER_SHAPES_2SM)
        change_priority_func(SM100_CLUSTER_SHAPES_1SM_CPY, SM100_CLUSTER_SHAPES_2SM_CPY)
        shapes_1sm = [
            list(shape) for shape, min_level in SM100_CLUSTER_SHAPES_1SM_CPY.items() if cluster_level >= min_level
        ]
        shapes_2sm = [
            list(shape) for shape, min_level in SM100_CLUSTER_SHAPES_2SM_CPY.items() if cluster_level >= min_level
        ]

        return shapes_1sm, shapes_2sm
   
    else:

        shapes_1sm = [
            list(shape) for shape, min_level in SM100_CLUSTER_SHAPES_1SM.items() if cluster_level >= min_level
        ]
        shapes_2sm = [
            list(shape) for shape, min_level in SM100_CLUSTER_SHAPES_2SM.items() if cluster_level >= min_level
        ]

        return shapes_1sm, shapes_2sm

#################################################################################################
#
# Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Utilities for enumerating CUTLASS library SM90 kernels
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
from typing import Any, Optional, Sequence, Tuple, List

try:
  import builtins
  if hasattr(builtins, "CUTLASS_IGNORE_PACKAGE") and CUTLASS_IGNORE_PACKAGE == True:
    raise ImportError("Disabling attempt to import cutlass_library")
  from cutlass_library.library import *
except ImportError:
  from library import *

# NOTE: this is a duplicate of CudaToolkitVersionSatisfies in generator.py
def CudaToolkitVersionSatisfies(semantic_ver_string, major, minor, patch = 0):

  # by default, use the latest CUDA Toolkit version
  cuda_version = [11, 0, 132]

  # Update cuda_version based on parsed string
  if semantic_ver_string != '':
    for i, x in enumerate([int(x) for x in semantic_ver_string.split('.')[:3]]):
      if i < len(cuda_version):
        cuda_version[i] = x
      else:
        cuda_version.append(x)
  return cuda_version >= [major, minor, patch]

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


def get_wgmma_level_from_global_level(global_level: int):
    return global_level % 10


def get_mma_level_from_global_level(global_level: int):
    return (global_level // 10) % 10


def get_cluster_level_from_global_level(global_level: int):
    return (global_level // 100) % 10


def get_pruning_level_from_global_level(global_level: int):
    return (global_level // 1000) % 10


#### Step 1: generate MMA instruction shapes based on levels

try:
    from .sm90_shapes import (
        SM90_MMA_MULTIPLIERS,
        SM90_CLUSTER_SIZES,
        SM90_WGMMA_SHAPES_TF32_DENSE,
        SM90_WGMMA_SHAPES_FP16_BF16_DENSE,
        SM90_WGMMA_SHAPES_FP8_DENSE,
        SM90_WGMMA_SHAPES_INT8_DENSE,
    )
except:
    from sm90_shapes import (
        SM90_MMA_MULTIPLIERS,
        SM90_CLUSTER_SIZES,
        SM90_WGMMA_SHAPES_TF32_DENSE,
        SM90_WGMMA_SHAPES_FP16_BF16_DENSE,
        SM90_WGMMA_SHAPES_FP8_DENSE,
        SM90_WGMMA_SHAPES_INT8_DENSE,
    )


def generate_tf32_math_instruction_shapes_sm90(level: int):
    assert isinstance(level, int) and level >= 0
    filtered_list_of_wgmma_shapes = [
        wgmma_shape for wgmma_shape, min_level in SM90_WGMMA_SHAPES_TF32_DENSE.items() if level >= min_level
    ]
    return filtered_list_of_wgmma_shapes

def generate_fp16_bf16_math_instruction_shapes_sm90(level: int):
    assert isinstance(level, int) and level >= 0
    filtered_list_of_wgmma_shapes = [
        wgmma_shape for wgmma_shape, min_level in SM90_WGMMA_SHAPES_FP16_BF16_DENSE.items() if level >= min_level
    ]
    return filtered_list_of_wgmma_shapes

def generate_fp8_math_instruction_shapes_sm90(level: int):
    assert isinstance(level, int) and level >= 0
    filtered_list_of_wgmma_shapes = [
        wgmma_shape for wgmma_shape, min_level in SM90_WGMMA_SHAPES_FP8_DENSE.items() if level >= min_level
    ]
    return filtered_list_of_wgmma_shapes

def generate_int8_math_instruction_shapes_sm90(level: int):
    assert isinstance(level, int) and level >= 0
    filtered_list_of_wgmma_shapes = [
        wgmma_shape for wgmma_shape, min_level in SM90_WGMMA_SHAPES_INT8_DENSE.items() if level >= min_level
    ]
    return filtered_list_of_wgmma_shapes

def generate_mixed_dtype_math_instructions_shapes_sm90(wgmma_level: int, a_type: DataType, b_type: DataType):
    # DataTypeSize are in the unit of bits
    a_bytes = DataTypeSize[a_type] // 8
    b_bytes = DataTypeSize[b_type] // 8
    if a_bytes == 4 or b_bytes == 4:
        return generate_tf32_math_instruction_shapes_sm90(wgmma_level)
    elif a_bytes == 2 or b_bytes == 2:
        return generate_fp16_bf16_math_instruction_shapes_sm90(wgmma_level)
    else:
        return generate_fp8_math_instruction_shapes_sm90(wgmma_level)

###########

def generate_tf32_math_instructions_sm90(level: int):
    wgmma_level = get_wgmma_level_from_global_level(level)
    math_instructions = []
    for math_instruction_shape in generate_tf32_math_instruction_shapes_sm90(wgmma_level):
        math_instructions.append(
          MathInstruction(
              math_instruction_shape,
              DataType.tf32, DataType.tf32, DataType.f32,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add)
        )
    return math_instructions

def generate_fp16_bf16_math_instructions_sm90(level: int):
    wgmma_level = get_wgmma_level_from_global_level(level)
    math_instructions = []
    for math_instruction_shape in generate_fp16_bf16_math_instruction_shapes_sm90(wgmma_level):
        math_instructions += [
          MathInstruction(
              math_instruction_shape,
              DataType.f16, DataType.f16, DataType.f16,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add),
          MathInstruction(
              math_instruction_shape,
              DataType.f16, DataType.f16, DataType.f32,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add),
          MathInstruction(
              math_instruction_shape,
              DataType.bf16, DataType.bf16, DataType.f32,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add),
        ]
    return math_instructions

def generate_fp8_math_instructions_sm90(level: int):
    wgmma_level = get_wgmma_level_from_global_level(level)
    math_instructions = []
    for math_instruction_shape in generate_fp8_math_instruction_shapes_sm90(wgmma_level):
        math_instructions += [
          MathInstruction(
              math_instruction_shape,
              DataType.e4m3, DataType.e4m3, DataType.f32,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add),
          MathInstruction(
              math_instruction_shape,
              DataType.e4m3, DataType.e5m2, DataType.f32,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add),
          MathInstruction(
              math_instruction_shape,
              DataType.e5m2, DataType.e4m3, DataType.f32,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add),
          MathInstruction(
              math_instruction_shape,
              DataType.e5m2, DataType.e5m2, DataType.f32,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add),
        ]
    return math_instructions

def generate_mixed_dtype_math_instructions_sm90(level: int, types_of_a_b_acc: List[Tuple[DataType, DataType, DataType]]):
    wgmma_level = get_wgmma_level_from_global_level(level)
    math_instructions = []
    for a_type, b_type, acc_type in types_of_a_b_acc:
        math_instruction_shapes = generate_mixed_dtype_math_instructions_shapes_sm90(wgmma_level, a_type, b_type)
        for math_instruction_shape in math_instruction_shapes:
            math_instructions += [
                MathInstruction(
                    math_instruction_shape,
                    a_type, b_type, acc_type,
                    OpcodeClass.TensorOp,
                    MathOperation.multiply_add
                ),
            ]
    return math_instructions

def generate_int8_math_instructions_sm90(level: int):
    wgmma_level = get_wgmma_level_from_global_level(level)
    math_instructions = []
    for math_instruction_shape in generate_int8_math_instruction_shapes_sm90(wgmma_level):
        math_instructions += [
          MathInstruction(
              math_instruction_shape,
              DataType.s8, DataType.s8, DataType.s32,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add),
          MathInstruction(
              math_instruction_shape,
              DataType.u8, DataType.u8, DataType.s32,
              OpcodeClass.TensorOp,
              MathOperation.multiply_add),
        ]
    return math_instructions

def make_sparse_math_instructions(math_instructions):
    sparse_instructions = []
    for inst in math_instructions:
        if inst.opcode_class == OpcodeClass.TensorOp:
            sparse_instructions.append(MathInstruction(
                (inst.instruction_shape[0], inst.instruction_shape[1], inst.instruction_shape[2] * 2),
                inst.element_a, inst.element_b, inst.element_accumulator,
                OpcodeClass.SparseTensorOp,
                inst.math_operation),)
    return sparse_instructions


#### Step 2: generate tile descriptions from math instruction shapes

def is_tile_desc_valid(tile_description):
    if tile_description.minimum_compute_capability != 90 or tile_description.maximum_compute_capability != 90:
        return False

    element_a, element_b, element_accum = (
        tile_description.math_instruction.element_a,
        tile_description.math_instruction.element_b,
        tile_description.math_instruction.element_accumulator
    )

    cluster_size, cta_shape = (
        tile_description.cluster_shape,
        tile_description.threadblock_shape,
    )
    grid_size = (
        cta_shape[0] * cluster_size[0] +
        cta_shape[1] * cluster_size[1] +
        cta_shape[2] * cluster_size[2]
    )
    num_ctas_in_cluster = cluster_size[0] * cluster_size[1] * cluster_size[2]
    cluster_shape = (
        cluster_size[0] * cta_shape[0],
        cluster_size[1] * cta_shape[1],
        cluster_size[2] * cta_shape[2]
    )

    FP32_TYPES = [DataType.f32, DataType.tf32]
    FP16_TYPES = [DataType.f16, DataType.bf16]
    is_fp32 = element_a in FP32_TYPES and element_b in FP32_TYPES
    is_fp16 = element_a in FP16_TYPES and element_b in FP16_TYPES

    # Maximum number of CTAs per cluster is 8 for Hopper, but up to 16 is
    # allowed for non portable clusters.
    if num_ctas_in_cluster > 16 or num_ctas_in_cluster < 1:
        return False

    if grid_size < 1:
        return False

    # SM90 WGMMA shapes are always 64 across M, therefore
    # CTA shape across M must always be a multiple of 64.
    if cta_shape[0] < 64 or cta_shape[0] % 64 != 0:
        return False

    # The minimum WGMMA shape across N is 8, and increments
    # vary across different dtypes, but they're never smaller
    # than 8. The minimum CTA shape allowed across N though is 16.
    if cta_shape[1] < 16 or cta_shape[1] % 8 != 0:
        return False

    # SM90 WGMMA shapes across K are always 8 for 32 bit dense
    # operations, 16 for 16 bit, and 32 for 8 bit. In any case,
    # the CTA shape across K should be a multiple of 8 and at least
    # twice the WGMMA shape across K.
    if cta_shape[2] < 16 or cta_shape[2] % 8 != 0:
        return False

    # Minimum of 2 stages (very rough heuristic that may filter out valid kernel configs)
    if (cluster_shape[0] >= 128 or cluster_shape[1] >= 128) and cluster_shape[2] >= 256:
        return False

    if is_fp32 and (cluster_shape[0] >= 128 or cluster_shape[1] >= 128) and cluster_shape[2] >= 128:
        return False

    if is_fp32 and cluster_shape[0] >= 256 and cluster_shape[1] >= 256 and cluster_shape[2] >= 64:
        return False

    if is_fp16 and cluster_shape[0] >= 256 and cluster_shape[1] >= 256 and cluster_shape[2] >= 128:
        return False

    # CTA shape upper bound: <256, 256, 256>
    if cta_shape[0] > 256 or cta_shape[1] > 256 or cta_shape[2] > 256:
        return False

    return True

def get_mma_multipliers(level: int):
    assert isinstance(level, int) and level >= 0
    mma_level = get_mma_level_from_global_level(level)
    return [
        mma_mul for mma_mul, mma_min_level in SM90_MMA_MULTIPLIERS.items() if mma_level >= mma_min_level
    ]

def get_cluster_sizes(level: int, is_aligned: bool):
    if not is_aligned:
        return [(1, 1, 1)]
    assert isinstance(level, int) and level >= 0
    cluster_level = get_cluster_level_from_global_level(level)
    return [
        cluster_size for cluster_size, cluster_min_level in SM90_CLUSTER_SIZES.items() if cluster_level >= cluster_min_level
    ]

def generate_tile_descriptions_sm90(math_instructions, is_aligned: bool, level: int):
    tile_descriptions = set()
    mma_multipliers, cluster_sizes = get_mma_multipliers(level), get_cluster_sizes(level, is_aligned)
    for math_inst, mma_mul, cluster_size in product(math_instructions, mma_multipliers, cluster_sizes):

        # generator can stamp out duplicate kernels, because it doesn't explicitly set instruction
        # shape for SM90 kernels, and the 3.X collective API doesn't directly expose them when using
        # the auto kernel schedule.

        math_inst_stub = copy.deepcopy(math_inst)
        math_inst_stub.instruction_shape = [0, 0, 0]

        tile_desc = TileDescription(
            threadblock_shape=[
                math_inst.instruction_shape[0] * mma_mul[0],
                math_inst.instruction_shape[1] * mma_mul[1],
                math_inst.instruction_shape[2] * mma_mul[2]
            ],
            stages=0,
            warp_count=[4, 1, 1],
            math_instruction=math_inst_stub,
            min_compute=90,
            max_compute=90,
            cluster_shape=cluster_size)
        # For sparse kernels K-tile is twice as large (due to 2x MMA-K size)
        # Reduce it to same size as dense to afford more smem stages
        if math_inst.opcode_class == OpcodeClass.SparseTensorOp:
            tile_desc.threadblock_shape[2] = tile_desc.threadblock_shape[2] // 2
        if is_tile_desc_valid(tile_desc):
            tile_descriptions.add(tile_desc)

    return tile_descriptions

#### Step 3: map tile description to valid schedules

def is_tile_desc_compatible_with_cooperative(tile_description):
    # Cooperative kernels require a minimum CTA-M of 128
    return tile_description.threadblock_shape[0] % 128 == 0


def can_tile_desc_use_shmem_in_epilogue(tile_description, data_types):
    dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc, dtype_epi = (
        data_types["a_type"],
        data_types["b_type"],
        data_types["c_type"],
        data_types["d_type"],
        data_types["acc_type"],
        data_types["epi_type"]
    )
    mn = tile_description.threadblock_shape[0] * tile_description.threadblock_shape[1]
    bitsize_c, bitsize_d = DataTypeSize[dtype_c], DataTypeSize[dtype_d]

    shmem_bits_c, shmem_bits_d = bitsize_c * mn, bitsize_d * mn
    shmem_bits_total = shmem_bits_c + shmem_bits_d
    # Magic number: 2^20
    # Existing logic suggested that tile shape 256x128 (or 128x256)
    # would run out of shmem if D is FP32, and source is needed.
    # That would be 256 * 128 * 32 == 2^21 (~262 KB), which is over the limit.
    # Hopper's max shmem size is 228 KB, and 2^20 ~= 131 KB.
    # Since epilogue can't possibly use ALL of the shmem available
    # we can just settle on 2^20 bits (~ 131 KB) being the upper bound
    # we would allow for epilogue.
    # This can be different for non-persistent kernels where epilogue and
    # mainloop shmem is shared.
    if shmem_bits_total > 2 ** 20:
        return False

    return True


def get_valid_schedules(tile_description, cuda_version, is_aligned, data_types, layout,
                        instantiation_level, enable_fp8_fast_acc=True, gemm_kind=GemmKind.Universal3x):
    # Level 0: prune according to existing generator.py behavior
    # Level >= 1: no pruning
    level = get_pruning_level_from_global_level(instantiation_level)
    schedules = []
    stream_k_schedules = []

    if not is_tile_desc_valid(tile_description):
        return schedules, stream_k_schedules

    FP16_TYPES = [DataType.f16, DataType.bf16]
    is_fp16 = data_types["a_type"] in FP16_TYPES and data_types["b_type"] in FP16_TYPES

    FP8_TYPES = [DataType.e4m3, DataType.e5m2]
    is_fp8 = data_types["a_type"] in FP8_TYPES and data_types["b_type"] in FP8_TYPES
    can_do_fp8_fast_accum = is_fp8 and enable_fp8_fast_acc

    FP32_TYPES = [DataType.f32, DataType.tf32]
    is_fp32 = data_types["a_type"] in FP32_TYPES and data_types["b_type"] in FP32_TYPES
    requires_transposed_epilogue = is_fp32 and layout[0][0] == LayoutType.RowMajor and layout[1][0] == LayoutType.RowMajor

    can_do_cooperative = is_tile_desc_compatible_with_cooperative(tile_description)
    can_do_tma_epilogue = is_aligned and not requires_transposed_epilogue and can_tile_desc_use_shmem_in_epilogue(tile_description, data_types)

    default_epilogue = EpilogueScheduleType.NoSmemWarpSpecialized if not requires_transposed_epilogue else EpilogueScheduleType.EpilogueTransposed
    auto_epilogue = EpilogueScheduleType.ScheduleAuto if not requires_transposed_epilogue else EpilogueScheduleType.EpilogueTransposed

    cta_m, cta_n, cta_k = (
        tile_description.threadblock_shape[0],
        tile_description.threadblock_shape[1],
        tile_description.threadblock_shape[2]
    )
    c_type = data_types["c_type"]
    d_type = data_types["d_type"]
    is_void_c = c_type == DataType.void

    # Filter out invalid kernels
    is_nt = layout[0][0] == LayoutType.ColumnMajor and layout[1][0] == LayoutType.RowMajor
    is_tn = layout[0][0] == LayoutType.RowMajor and layout[1][0] == LayoutType.ColumnMajor
    is_nn = layout[0][0] == LayoutType.ColumnMajor and layout[1][0] == LayoutType.ColumnMajor

    # static_assert(size<0>(SmemLayoutB{}) % WarpgroupTileSize == 0,
    #   "Copy size must evenly divide SMEM tile.");
    if is_fp32 and is_nt and (cta_n % cta_k != 0):
        return [], []

    # static_assert(!TransposeB || (cutlass::bits_to_bytes((size<1>(SmemLayoutB{}) * sizeof_bits<InternalElementB>::value))) == 128,
    # "SmemLayoutB K must be 128bytes to be transposed.")
    if is_fp32 and is_nt and cta_k != 32:
        return [], []

    # Static assert failure when instantiating SmemLayoutB
    if is_fp32 and (is_tn or is_nn) and (cta_n % cta_k != 0):
        return [], []

    grouped = is_grouped(gemm_kind)
    if grouped:
        # the following cases are unsupported by grouped GEMM
        if not is_aligned:
            return [], []
        if requires_transposed_epilogue:
            return [], []

    # Early pruning
    if level < 1:
        # Don't stamp out FP16/BF16 kernels smaller than or equal to 64x128x64
        if is_fp16 and cta_m <= 64 and cta_n <= 128 and cta_k <= 64:
            return [], []

        # FP8 configs with CTA tile larger than or equal to 256x128x128 limit data types and schedules
        is_large_fp8_tile = is_fp8 and cta_m >= 256 and cta_n >= 128 and cta_k >= 128
        if is_large_fp8_tile:
            # Only void-C, and only FP8 outputs allowed
            if not is_void_c or d_type not in FP8_TYPES:
                return [], []
            if CudaToolkitVersionSatisfies(cuda_version, 12, 1) and can_do_cooperative and can_do_tma_epilogue:
                schedules = []
                if is_blockwise(gemm_kind):
                    schedules.append(
                        [
                            to_grouped_schedule(KernelScheduleType.BlockwiseTmaWarpSpecializedCooperative, grouped),
                            to_grouped_schedule(EpilogueScheduleType.TmaWarpSpecializedCooperative, grouped)
                        ])
                else:
                    schedules.append(
                        [
                            to_grouped_schedule(KernelScheduleType.TmaWarpSpecializedCooperative, grouped),
                            to_grouped_schedule(EpilogueScheduleType.TmaWarpSpecializedCooperative, grouped)
                        ])
                    schedules.append(
                        [
                            to_grouped_schedule(KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum, grouped),
                            to_grouped_schedule(EpilogueScheduleType.TmaWarpSpecializedCooperative, grouped)
                        ])
                return schedules, []
            return [], []

        if is_fp8 and not is_large_fp8_tile:
            valid_dtypes_for_c = [DataType.f32, DataType.bf16, DataType.f16, DataType.void]
            # Prune all configs with fp8 source, and all configs with non-fp8 output
            # that have different dtypes for source and output.
            if c_type not in valid_dtypes_for_c or (d_type not in FP8_TYPES and c_type != d_type):
                return [], []

        # FP32/TF32 kernels don't stamp out void-C
        if is_fp32 and is_void_c:
            return [], []

    # Void-c only makes a difference for TMA epilogues
    if is_void_c and not can_do_tma_epilogue:
        return [], []

    # For mixed input data types
    a_type_size = DataTypeSize[data_types["a_type"]]
    b_type_size = DataTypeSize[data_types["b_type"]]
    if a_type_size != b_type_size and CudaToolkitVersionSatisfies(cuda_version, 12, 1):
        schedules = []
        stream_k_schedules = []
        epilogue_schedule = EpilogueScheduleType.TmaWarpSpecialized
        if a_type_size > b_type_size:
            epilogue_schedule = EpilogueScheduleType.EpilogueTransposed
        
        if not is_blockwise(gemm_kind):
            schedules.append([
                KernelScheduleType.TmaWarpSpecialized,
                epilogue_schedule
            ])
            schedules.append([
                KernelScheduleType.TmaWarpSpecializedPingpong,
                epilogue_schedule
            ])
        if cta_m >= 128:
            if a_type_size > b_type_size:
                epilogue_schedule = EpilogueScheduleType.EpilogueTransposed
            else:
                epilogue_schedule = EpilogueScheduleType.TmaWarpSpecializedCooperative
            if is_blockwise(gemm_kind):
                schedules.append([
                    KernelScheduleType.BlockwiseTmaWarpSpecializedCooperative,
                    epilogue_schedule
                ])
            else:
                schedules.append([
                    KernelScheduleType.TmaWarpSpecializedCooperative,
                    epilogue_schedule
                ])
                stream_k_schedules.append([
                    KernelScheduleType.TmaWarpSpecializedCooperative,
                    epilogue_schedule
                ])
        return schedules, stream_k_schedules

    if not is_aligned and not is_blockwise(gemm_kind):
        schedules = [[KernelScheduleType.CpAsyncWarpSpecialized,
                    default_epilogue]]
        stream_k_schedules = []

        if CudaToolkitVersionSatisfies(cuda_version, 12, 1) and can_do_cooperative:
            schedules.append([
                KernelScheduleType.CpAsyncWarpSpecializedCooperative,
                default_epilogue
            ])
            stream_k_schedules.append([
                KernelScheduleType.CpAsyncWarpSpecializedCooperative,
                default_epilogue
            ])

        return schedules, stream_k_schedules

    schedules = []
    # Pruning: emit Void-C and Grouped kernels with persistent kernels only
    if (level >= 1 or not is_void_c) and not grouped and not is_blockwise(gemm_kind):
        # Pruning: don't stamp out fp8 kernels with auto schedule
        if not is_fp8:
            schedules.append([KernelScheduleType.ScheduleAuto, auto_epilogue])
        schedules.append([KernelScheduleType.TmaWarpSpecialized, default_epilogue])
    stream_k_schedules = []
    
    if CudaToolkitVersionSatisfies(cuda_version, 12, 0):
        if can_do_tma_epilogue:
            assert not requires_transposed_epilogue
            # Inconsistency: fp8 pingpong only gets stamped out with fast accum
            if (not is_fp8 or level >= 1) and not is_blockwise(gemm_kind):
                schedules.append([
                    to_grouped_schedule(KernelScheduleType.TmaWarpSpecializedPingpong, grouped),
                    to_grouped_schedule(EpilogueScheduleType.TmaWarpSpecialized, grouped)
                ])
            if can_do_fp8_fast_accum:
                schedules.append([
                    to_grouped_schedule(KernelScheduleType.TmaWarpSpecializedPingpongFP8FastAccum, grouped),
                    to_grouped_schedule(EpilogueScheduleType.TmaWarpSpecialized, grouped)
                ])

    if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
        # Pruning: don't stamp out fp8 ping-pong kernel with non-tma epilogue
        if not is_fp8 or level >= 1:
            if not is_blockwise(gemm_kind):
                schedules.append([to_grouped_schedule(KernelScheduleType.TmaWarpSpecializedPingpong, grouped), to_grouped_schedule(default_epilogue, grouped)])
            else:
                schedules.append([to_grouped_schedule(KernelScheduleType.BlockwiseTmaWarpSpecializedPingpong, grouped), to_grouped_schedule(default_epilogue, grouped)])

        if can_do_fp8_fast_accum:
            if not grouped:
                schedules.append([KernelScheduleType.TmaWarpSpecializedFP8FastAccum, default_epilogue])
            schedules.append([to_grouped_schedule(KernelScheduleType.TmaWarpSpecializedPingpongFP8FastAccum, grouped), to_grouped_schedule(default_epilogue, grouped)])

        if can_do_cooperative:
            if is_blockwise(gemm_kind):
                schedules.append([
                    to_grouped_schedule(KernelScheduleType.BlockwiseTmaWarpSpecializedCooperative, grouped),
                    to_grouped_schedule(default_epilogue, grouped)
                ])
                stream_k_schedules.append([
                    KernelScheduleType.BlockwiseTmaWarpSpecializedCooperative,
                    default_epilogue
                ])
            else:
                schedules.append([
                    to_grouped_schedule(KernelScheduleType.TmaWarpSpecializedCooperative, grouped),
                    to_grouped_schedule(default_epilogue, grouped)
                ])
                stream_k_schedules.append([
                    KernelScheduleType.TmaWarpSpecializedCooperative,
                    default_epilogue
                ])
            if can_do_fp8_fast_accum:
                schedules.append([
                    to_grouped_schedule(KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum, grouped),
                    to_grouped_schedule(default_epilogue, grouped)
                ])
                stream_k_schedules.append([
                    KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum,
                    default_epilogue
                ])

        # persistent kernels with TMA epilogues
        if can_do_tma_epilogue:
            assert not requires_transposed_epilogue
            if can_do_cooperative:
                if is_blockwise(gemm_kind):
                    schedules.append([
                        to_grouped_schedule(KernelScheduleType.BlockwiseTmaWarpSpecializedCooperative, grouped),
                        to_grouped_schedule(EpilogueScheduleType.TmaWarpSpecializedCooperative, grouped)
                    ])
                    stream_k_schedules.append([
                        KernelScheduleType.BlockwiseTmaWarpSpecializedCooperative,
                        EpilogueScheduleType.TmaWarpSpecializedCooperative
                    ])
                else:
                    schedules.append([
                        to_grouped_schedule(KernelScheduleType.TmaWarpSpecializedCooperative, grouped),
                        to_grouped_schedule(EpilogueScheduleType.TmaWarpSpecializedCooperative, grouped)
                    ])
                    stream_k_schedules.append([
                        KernelScheduleType.TmaWarpSpecializedCooperative,
                        EpilogueScheduleType.TmaWarpSpecializedCooperative
                    ])
                if can_do_fp8_fast_accum:
                    schedules.append([
                        to_grouped_schedule(KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum, grouped),
                        to_grouped_schedule(EpilogueScheduleType.TmaWarpSpecializedCooperative, grouped)
                    ])
                    stream_k_schedules.append([
                        KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum,
                        EpilogueScheduleType.TmaWarpSpecializedCooperative
                    ])
    # Grouped GEMM do not support Stream-K scheduler
    if grouped:
        return schedules, []
    return schedules, stream_k_schedules


#### Misc: helpers

def generate_data_types_from_math_instruction(math_instruction, element_source = None, element_dest = None, element_epilogue = None):
    element_a, element_b = math_instruction.element_a, math_instruction.element_b
    element_accumulator = math_instruction.element_accumulator
    element_c = element_source or element_accumulator
    element_d = element_dest or element_accumulator
    element_epilogue = element_epilogue or element_accumulator
    data_types = {
        "a_type"   : element_a,
        "b_type"   : element_b,
        "c_type"   : element_c,
        "d_type"   : element_d,
        "acc_type" : element_accumulator,
        "epi_type" : element_epilogue
    }
    return data_types

def fix_alignments(data_types, layout, alignment_bits = 128):
    operand_keys = ["a_type", "b_type", "c_type"]
    operands_to_fix = ["c_type"]
    new_layout = []
    assert len(layout) == len(operand_keys)
    for i, k in enumerate(operand_keys):
        assert k in data_types and data_types[k] in DataTypeSize
        dtype = data_types[k]
        dtype_size_bits = DataTypeSize[dtype]

        layout_type = layout[i][0]
        layout_alignment = layout[i][1]

        # Don't modify alignment if dtype's been changed to void
        if k in operands_to_fix and dtype_size_bits >= 1:
            layout_alignment = alignment_bits // dtype_size_bits

        new_layout.append([layout_type, layout_alignment])

    return new_layout

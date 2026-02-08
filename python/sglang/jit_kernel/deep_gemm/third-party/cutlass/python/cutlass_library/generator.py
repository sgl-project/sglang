#################################################################################################
#
# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Utilities for enumerating CUTLASS library kernels
"""

import argparse
import enum
from itertools import chain, product
import logging
import os.path
import shutil
import sys
import copy
from typing import Any, Dict, Optional, Sequence, Tuple

_LOGGER = logging.getLogger(__name__)

def logging_prefix(indent_level: int = 0) -> str:
  """String prefix for start of each debug log entry"""
  prefix = '*** '
  indent = '  '
  return f"{prefix}{indent_level * indent}"

def log_debug_line(line: str, indent_level: int = 0) -> None:
  """Log one line of debug output"""
  prefix = logging_prefix(indent_level)
  _LOGGER.debug(prefix + line)

# Certain usecases of cutlass_library nearly always prefer to run as scripts with
# relative imports, rather than via an installed Python package. An example of this
# is using CUTLASS's CMake system to generate a library of kernels to be profiled.
# To make it easy to use these use cases when an existing installation of cutlass_library
# exists, this global flag can be set to true (via command-line arguments) to ensure
# that package-based installations are not used.

# Create a temporary argument parser to check only for the availability of the
# --disable-cutlass-package-imports argument, which controls whether package-based
# imports are disabled.
def _add_package_disablement_flag(argparser):
  argparser.add_argument("--disable-cutlass-package-imports", action='store_true', required=False,
                     help="Disable use of cutlass_library from Python package")

_parser = argparse.ArgumentParser()
_add_package_disablement_flag(_parser)
_args, _ = _parser.parse_known_args()

# Add `CUTLASS_IGNORE_PACKAGE` to `builtins` so that it is visible for gating future
# imports without requiring importing another module. Ideally, we would just place this
# as a global variable in a module to that could be imported and checked (e.g.,
# utils.CUTLASS_IGNORE_PACKAGE). However, this raises the issue of determining
# where this module should be sourced (from the cutlass_library package or from
# a relative import), which is the problem this variable is being used to solve in the
# first place.
import builtins
builtins.CUTLASS_IGNORE_PACKAGE = _args.disable_cutlass_package_imports

try:
  if CUTLASS_IGNORE_PACKAGE:
    raise ImportError("Disabling attempt to import cutlass_library")
  from cutlass_library.library import *
  from cutlass_library.manifest import *
  from cutlass_library.heuristics import *
  from cutlass_library.emit_kernel_listing import emit_gemm_kernel_testlist 
except ImportError:
  from library import *
  from manifest import *
  from heuristics import *
  from emit_kernel_listing import emit_gemm_kernel_testlist 
###################################################################################################

#
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

# From cuda 13.0, Thor SM is renumbered from 101 to 110
def ThorSMRenumbering(cuda_version):
  return 110 if CudaToolkitVersionSatisfies(cuda_version, 13, 0) else 101

###################################################################################################
###################################################################################################

#
def EpilogueAlignment(max_alignment, tile, epilogue_steps = 8):
  ''' Helper to compute the maximum alignment of the epilogue '''

  def product(X, identity = 1):
    result = identity
    for item in X:
      result *= item
    return result

  elements_per_thread = product(tile.threadblock_shape[:-1]) // product(tile.warp_count) // 32 // epilogue_steps
  return min(max_alignment, elements_per_thread)

def DefaultSwizzlingFunctor():
    return SwizzlingFunctor.Identity8
    # To use StreamK decomposition for basic GEMMs, set `swizzling_functor = SwizzlingFunctor.StreamK`

#
def CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = DefaultSwizzlingFunctor()):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:

            # If alignment is a tuple or a list, then we have different alignments for A and B
            alignment_a = alignment if isinstance(alignment, int) else alignment[0]
            alignment_b = alignment if isinstance(alignment, int) else alignment[1]
            alignment_c = min(8, alignment_a) if isinstance(alignment, int) else alignment[2]

            A = TensorDescription(element_a, layout[0], alignment_a, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment_b, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GemmOperation(GemmKind.Universal, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

# Generates 3.0 API based GemmUniversal API kernels. Alignment constraints are folded in with layouts
def CreateGemmUniversal3xOperator(
    manifest, layouts, tile_descriptions, data_types,
    schedules = [[KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto]],
    complex_transforms=None,
    epilogue_functor=EpilogueFunctor.LinearCombination,
    swizzling_functor=SwizzlingFunctor.Identity1,
    tile_schedulers=[TileSchedulerType.Default],
    gemm_kind=GemmKind.Universal3x):

  if type(data_types) is dict:
    data_types = [data_types]

  for s in schedules:
    assert(len(s) == 2)

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none), ]

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    if len(tile_descriptions) == 0:
      return operations
    tile_descriptions = [tile_descriptions[0]]

  combinations = product(layouts, tile_descriptions, data_types, complex_transforms, schedules, tile_schedulers)
  for layout, tile_description, data_type, complex_transform, schedules, tile_scheduler in combinations:
    kernel_schedule, epilogue_schedule = schedules
    A = TensorDescription(
        data_type["a_type"], layout[0][0], layout[0][1], complex_transform[0])
    B = TensorDescription(
        data_type["b_type"], layout[1][0], layout[1][1], complex_transform[1])

    C = TensorDescription(data_type["c_type"], layout[2][0], layout[2][1])
    D = TensorDescription(data_type["d_type"], layout[2][0], layout[2][1])

    gemm_op_extra_args = {}
    element_compute = data_type.get("epi_type", data_type["acc_type"])

    if "sf_type" in data_type:
      gemm_op_extra_args["ScaleFactorA"] = data_type["sf_type"]
      gemm_op_extra_args["ScaleFactorB"] = data_type["sf_type"]
      gemm_op_extra_args["ScaleFactorD"] = { "tensor": TensorDescription(data_type["sfd_type"]["type"], data_type["sfd_type"]["layout"]),
                                             "vector_size" : data_type["sfd_type"]["vector_size"]}
      assert is_block_scaled(gemm_kind)
    
    if tile_description.explicit_vector_sizes != None:
      assert len(tile_description.explicit_vector_sizes) == 3
      gemm_op_extra_args["ScaleFactorMVecSize"] = tile_description.explicit_vector_sizes[0]
      gemm_op_extra_args["ScaleFactorNVecSize"] = tile_description.explicit_vector_sizes[1]
      gemm_op_extra_args["ScaleFactorKVecSize"] = tile_description.explicit_vector_sizes[2]
      assert is_blockwise(gemm_kind)
    else:
      assert not is_blockwise(gemm_kind)

    A_dtype = data_type["a_type"]
    B_dtype = data_type["b_type"]
    A_dtype_bits = DataTypeSize[A_dtype]
    B_dtype_bits = DataTypeSize[B_dtype]
    is_A_dtype_narrow = A_dtype_bits < B_dtype_bits
    if is_A_dtype_narrow:
      narrow_dtype, wide_dtype = (A_dtype, B_dtype)
      narrow_dtype_bits, wide_dtype_bits = (A_dtype_bits, B_dtype_bits)
    else:
      narrow_dtype, wide_dtype = (B_dtype, A_dtype)
      narrow_dtype_bits, wide_dtype_bits = (B_dtype_bits, A_dtype_bits)

    mixed_input_modes = [None]
    if narrow_dtype_bits != wide_dtype_bits:
      if narrow_dtype == DataType.s4 and (wide_dtype == DataType.e4m3 or wide_dtype == DataType.e5m2):
        mixed_input_modes = [MixedInputMode.ScaleOnly]
      else:
        mixed_input_modes = [MixedInputMode.ConvertOnly, MixedInputMode.ScaleOnly, MixedInputMode.ScaleWithZeroPoint]

    mixed_input_shuffle_options = [False]
    if (mixed_input_modes[0] is not None) and (wide_dtype_bits == 16) and (narrow_dtype_bits == 4 or narrow_dtype_bits == 8):
      mixed_input_shuffle_options = [False, True]

    for mixed_input_mode, mixed_input_shuffle in product(mixed_input_modes, mixed_input_shuffle_options):
      operation = GemmOperation(
          gemm_kind, tile_description.minimum_compute_capability,
          tile_description, A, B, C, element_compute, epilogue_functor, swizzling_functor, D,
          kernel_schedule, epilogue_schedule, tile_scheduler,
          mixed_input_mode=mixed_input_mode, mixed_input_shuffle=mixed_input_shuffle, **gemm_op_extra_args)
      manifest.append(operation)
      operations.append(operation)

  return operations

# Generates 3.0 API based GemmUniversal API kernels. Alignment constraints are folded in with layouts
def CreateSparseGemmUniversal3xOperator(
    manifest, layouts, tile_descriptions, data_types,
    schedules = [[KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto]],
    complex_transforms=None,
    epilogue_functor=EpilogueFunctor.LinearCombination,
    swizzling_functor=SwizzlingFunctor.Identity1,
    tile_schedulers=[TileSchedulerType.Default]):

  if type(data_types) is dict:
    data_types = [data_types]

  for s in schedules:
    assert(len(s) == 2)

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none), ]

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0]]

  combinations = product(layouts, tile_descriptions, data_types, complex_transforms, schedules, tile_schedulers)
  for layout, tile_description, data_type, complex_transform, schedules, tile_scheduler in combinations:
    kernel_schedule, epilogue_schedule = schedules
    A = TensorDescription(
        data_type["a_type"], layout[0][0], layout[0][1], complex_transform[0])
    B = TensorDescription(
        data_type["b_type"], layout[1][0], layout[1][1], complex_transform[1])

    # Currently assume tensor C/D have same layout requirement.
    C = TensorDescription(data_type["c_type"], layout[2][0], layout[2][1])
    D = TensorDescription(data_type["d_type"], layout[2][0], layout[2][1])

    element_compute = data_type.get("epi_type", data_type["acc_type"])

    operation = GemmOperation(
        GemmKind.SparseUniversal3x, tile_description.minimum_compute_capability,
        tile_description, A, B, C, element_compute, epilogue_functor, swizzling_functor, D,
        kernel_schedule, epilogue_schedule, tile_scheduler)

    manifest.append(operation)
    operations.append(operation)

  return operations

#
def CreateSparseGemmOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  gemm_kinds = [GemmKind.Sparse]

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:

            alignment_c = min(8, alignment)

            A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GemmOperation(GemmKind.Sparse, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

#
def CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  gemm_kinds = [GemmKind.PlanarComplex, GemmKind.PlanarComplexArray]

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for gemm_kind in gemm_kinds:
    for layout in layouts:
      for tile_description in tile_descriptions:
        for alignment in alignment_constraints:
          for complex_transform in complex_transforms:

            alignment_c = min(8, alignment)

            A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            manifest.append(GemmOperation(gemm_kind, \
              tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue))
  return

#
def CreateGemmGroupedOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:

            alignment_c = min(8, alignment)

            A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GroupedGemmOperation(GemmKind.Grouped, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

#
def CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, data_type, \
  alignment_constraints, blas_mode, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  element_a, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for fill_mode in fill_modes:
      for tile_description in tile_descriptions:
        for alignment in alignment_constraints:

          # SERK supported layouts (RowMajor, ColumnMajor) with no conjugation
          complex_transform = ComplexTransform.none

          # HERK supported layouts (RowMajor + conj, ColumnMajor)
          if blas_mode == BlasMode.hermitian and layout[0] == LayoutType.RowMajor:
            complex_transform = ComplexTransform.conj

          alignment_c = 1 # Alignment only applies to A in SYRK

          A = TensorDescription(element_a, layout[0], alignment, complex_transform)
          C = SymmetricTensorDescription(element_c, layout[1], fill_mode, alignment_c)

          # Rank-K update
          new_operation = RankKOperation(RankKKind.Universal, tile_description.minimum_compute_capability, \
            tile_description, A, C, element_epilogue, epilogue_functor, swizzling_functor, blas_mode)

          manifest.append(new_operation)
          operations.append(new_operation)

          # Rank-2K update
          new_operation = Rank2KOperation(RankKKind.Universal, tile_description.minimum_compute_capability, \
            tile_description, A, C, element_epilogue, epilogue_functor, swizzling_functor, blas_mode)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

#
def CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for side_mode in side_modes:
      for fill_mode in fill_modes:
        for diag_type in diag_types:
          for tile_description in tile_descriptions:
            for alignment in alignment_constraints:
              for complex_transform in complex_transforms:

                  alignment_c = min(8, alignment)

                  A = TriangularTensorDescription(element_a, layout[0], side_mode, fill_mode, diag_type,
                                                  alignment, complex_transform)
                  B = TensorDescription(element_b, layout[1], alignment)
                  C = TensorDescription(element_c, layout[2], alignment_c)

                  new_operation = TrmmOperation(TrmmKind.Universal, tile_description.minimum_compute_capability, \
                    tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

                  manifest.append(new_operation)
                  operations.append(new_operation)

  return operations

#
def CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, data_type, \
  alignment_constraints, blas_mode, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  element_a, element_b, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for side_mode in side_modes:
      for fill_mode in fill_modes:
        for tile_description in tile_descriptions:
          for alignment in alignment_constraints:

            # SYMM supported layouts (RowMajor, ColumnMajor) with no conjugation
            complex_transform = ComplexTransform.none

            alignment_a = 1 # No vectorized access for the triangular matrix
            alignment_c = min(8, alignment)

            A = SymmetricTensorDescription(element_a, layout[0], fill_mode, alignment_a, complex_transform, side_mode)
            # tensor A and B have same data type and layout
            B = TensorDescription(element_b, layout[0], alignment)
            C = TensorDescription(element_c, layout[1], alignment_c)

            # SYMM/HEMM update
            new_operation = SymmOperation(SymmKind.Universal, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor, blas_mode)

            manifest.append(new_operation)
            operations.append(new_operation)

            # SYMM/HEMM update
            new_operation = SymmOperation(SymmKind.Universal, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor, blas_mode)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

###########################################################################################################
#   ConvolutionOperator support variations
#        ____________________________________________________________________
#         ConvolutionalOperator |      Analytic          |    Optimized
#        ____________________________________________________________________
#        |       Fprop          |     (strided)          |    (strided)
#        |       Dgrad          |     (strided, unity*)  |    (strided, unity)
#        |       Wgrad          |     (strided)          |    (strided)
#        ____________________________________________________________________
#
# Note :  Operator marked (*) are supported but not generated to keep the instantiated kernel count low
###########################################################################################################
# Convolution for 2D operations
def CreateConv2dOperator(manifest, layout, tile_descriptions, data_type, alignment_constraints, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], \
  epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4):

  element_a, element_b, element_c, element_epilogue = data_type

  # one exceptional case

  # iterator algorithm (analytic and optimized)
  iterator_algorithms = [IteratorAlgorithm.Analytic, IteratorAlgorithm.Optimized]

  # by default, only generate the largest tile size, largest alignment, and optimized iterator
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]
    iterator_algorithms = [IteratorAlgorithm.Optimized]

  operations = []

  for tile in tile_descriptions:
    for alignment in alignment_constraints:

      alignment_c = min(8, alignment)

      A = TensorDescription(element_a, layout[0], alignment)
      B = TensorDescription(element_b, layout[1], alignment)
      C = TensorDescription(element_c, layout[2], alignment_c)

      swizzling_functor_ = swizzling_functor

      #
      # Conv2d Fprop
      #
      if ConvKind.Fprop in conv_kinds:

        # Strided support for Analytic and Optimized Fprop
        for iterator_algorithm in iterator_algorithms:
          new_operations = [
            # None grouped kernel
            Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
              A, B, C, element_epilogue, StrideSupport.Unity, epilogue_functor, swizzling_functor_),
          ]

          # Instance group conv kernel
          if tile.math_instruction.opcode_class == OpcodeClass.TensorOp and A.layout == LayoutType.TensorNHWC and \
            tile.minimum_compute_capability >= 80:
            # SingleGroup kernel
            new_operations.append(Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
              A, B, C, element_epilogue, StrideSupport.Unity, epilogue_functor, swizzling_functor_, group_mode=GroupMode.SingleGroup))

            # Analytic iterator supports MultipleGroup mode
            if iterator_algorithm == IteratorAlgorithm.Analytic:
              new_operations.append(Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
                A, B, C, element_epilogue, StrideSupport.Unity, epilogue_functor, swizzling_functor_, group_mode=GroupMode.MultipleGroup))

          for new_operation in new_operations:
            manifest.append(new_operation)
            operations.append(new_operation)

      #
      # Conv2d Dgrad
      #
      if ConvKind.Dgrad in conv_kinds:

        # Unity stride for Analytic and Optimized Dgrad
        for iterator_algorithm in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Dgrad, iterator_algorithm, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Unity, epilogue_functor, swizzling_functor_)

          manifest.append(new_operation)
          operations.append(new_operation)

        # Strided support for Analytic Dgrad
        # strided dgrad uses a special threadblock swizzle
        # note that SwizzlingFunctor.StridedDgradHorizontal might be
        # better for problem sizes with large activation channel count
        swizzling_functor_strided_dgrad_ = SwizzlingFunctor.StridedDgradIdentity1

        if IteratorAlgorithm.Analytic in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Dgrad, IteratorAlgorithm.Analytic, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_strided_dgrad_)

          manifest.append(new_operation)
          operations.append(new_operation)

        # Strided support for Optimized Dgrad
        if IteratorAlgorithm.Optimized in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Dgrad, IteratorAlgorithm.Optimized, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_strided_dgrad_)

          manifest.append(new_operation)
          operations.append(new_operation)

      #
      # Conv2d Wgrad
      #
      if ConvKind.Wgrad in conv_kinds:

        # Strided support for Analytic and Optimized Wgrad
        for iterator_algorithm in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Wgrad, iterator_algorithm, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

# Convolution for 2D operations specialized for few channels
def CreateConv2dFixedChannelsOperator(manifest, layout, tile_descriptions, data_type, channel_counts, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], \
  epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4):

  element_a, element_b, element_c, element_epilogue = data_type

  # one exceptional case

  # iterator algorithm (analytic and optimized)
  iterator_algorithms = [IteratorAlgorithm.FixedChannels,]

  # by default, only generate the largest tile size, largest alignment, and optimized iterator
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    channel_counts = [channel_counts[0],]

  operations = []



  for tile in tile_descriptions:
    for channel_count in channel_counts:

      alignment_c = EpilogueAlignment(channel_count, tile)

      A = TensorDescription(element_a, layout[0], channel_count)
      B = TensorDescription(element_b, layout[1], channel_count)
      C = TensorDescription(element_c, layout[2], alignment_c)

      swizzling_functor_ = swizzling_functor

      #
      # Conv2d Fprop
      #
      if ConvKind.Fprop in conv_kinds:

        # Strided support for Analytic and Optimized Fprop
        for iterator_algorithm in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

# Convolution for 2D operations specialized for few channels
def CreateConv2dFewChannelsOperator(manifest, layout, tile_descriptions, data_type, channel_counts, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], \
  epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4):

  element_a, element_b, element_c, element_epilogue = data_type

  # one exceptional case

  # iterator algorithm (analytic and optimized)
  iterator_algorithms = [IteratorAlgorithm.FewChannels,]

  # by default, only generate the largest tile size, largest alignment, and optimized iterator
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    channel_counts = [channel_counts[0],]

  operations = []

  for tile in tile_descriptions:
    for channel_count in channel_counts:

      alignment_c = EpilogueAlignment(channel_count, tile)

      A = TensorDescription(element_a, layout[0], channel_count)
      B = TensorDescription(element_b, layout[1], channel_count)
      C = TensorDescription(element_c, layout[2], alignment_c)

      swizzling_functor_ = swizzling_functor

      #
      # Conv2d Fprop
      #
      if ConvKind.Fprop in conv_kinds:

        # Strided support for Analytic and Optimized Fprop
        for iterator_algorithm in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

# Convolution for 3D operations
def CreateConv3dOperator(manifest, layout, tile_descriptions, data_type, alignment, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], epilogue_functor = EpilogueFunctor.LinearCombination):

  element_a, element_b, element_c, element_epilogue = data_type

  # one exceptional case
  alignment_c = min(8, alignment)

  # iterator algorithm (analytic and optimized)
  iterator_algorithms = [IteratorAlgorithm.Analytic, IteratorAlgorithm.Optimized]

  # by default, only generate the largest tile size and optimized iterators
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    iterator_algorithms = [IteratorAlgorithm.Optimized]

  operations = []

  # All tile sizes for Conv3dFprop and Conv3dWgrad
  for tile in tile_descriptions:
    A = TensorDescription(element_a, layout, alignment)
    B = TensorDescription(element_b, layout, alignment)
    C = TensorDescription(element_c, layout, alignment_c)

    #
    # Conv3d Fprop
    #
    if ConvKind.Fprop in conv_kinds:
      # Strided support for Analytic and Optimized Fprop
      for iterator_algorithm in iterator_algorithms:
        new_operation = Conv3dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
                                        A, B, C, element_epilogue, StrideSupport.Strided)
        manifest.append(new_operation)
        operations.append(new_operation)
    #
    # Conv3d Wgrad
    #
    if ConvKind.Wgrad in conv_kinds:

      # Strided support for Analytic and Optimized Wgrad
      for iterator_algorithm in iterator_algorithms:
        new_operation = Conv3dOperation(ConvKind.Wgrad, iterator_algorithm, tile.minimum_compute_capability, tile,\
          A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor)
        manifest.append(new_operation)
        operations.append(new_operation)

  # All tile sizes for Conv3dDgrad
  for tile in tile_descriptions:

    A = TensorDescription(element_a, layout, alignment)
    B = TensorDescription(element_b, layout, alignment)
    C = TensorDescription(element_c, layout, alignment_c)

    #
    # Conv3d Dgrad
    #
    if ConvKind.Dgrad in conv_kinds:
      # Unity stride for Optimized Dgrad
      new_operation = Conv3dOperation(ConvKind.Dgrad, IteratorAlgorithm.Optimized, tile.minimum_compute_capability, tile,\
        A, B, C, element_epilogue, StrideSupport.Unity, epilogue_functor)

      manifest.append(new_operation)
      operations.append(new_operation)

      # Strided support for Analytic Dgrad
      # Conv3dDgrad has a naive strided support which does not cut down redundant MMAs
      new_operation = Conv3dOperation(ConvKind.Dgrad, IteratorAlgorithm.Analytic, tile.minimum_compute_capability, tile,\
        A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor)

      manifest.append(new_operation)
      operations.append(new_operation)

  return operations

# Convolution for Depthwise 2d conv
def CreateDepthwiseConv2dOperator(manifest, layout, tile_descriptions, data_type, alignment_constraints, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], \
  epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4):

  element_a, element_b, element_c, element_epilogue = data_type

  # iterator algorithm (FixedStrideDilation, Optimized)
  iterator_algorithms = [IteratorAlgorithm.FixedStrideDilation, IteratorAlgorithm.Optimized]

  # by default, only generate the largest tile size, largest alignment, and optimized iterator
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  operations = []

  for tile in tile_descriptions:
    for alignment in alignment_constraints:

      alignment_c = min(8, alignment)

      A = TensorDescription(element_a, layout[0], alignment)
      B = TensorDescription(element_b, layout[1], alignment)
      C = TensorDescription(element_c, layout[2], alignment_c)

      swizzling_functor_ = swizzling_functor

      if ConvKind.Fprop in conv_kinds:

        # Strided support for Optimized and FixedStridedDilation Depthwise Conv
        for iterator_algorithm in iterator_algorithms:
          stride_support = StrideSupport.Strided
          if iterator_algorithm == IteratorAlgorithm.FixedStrideDilation:
              if tile.stride == [-1, -1] or tile.dilation == [-1,-1]:
                continue
              stride_support = StrideSupport.Fixed

          if iterator_algorithm == IteratorAlgorithm.Optimized:
              if tile.stride != [-1, -1] or tile.dilation != [-1,-1]:
                continue
          new_operation = Conv2dOperation(ConvKind.Fprop,
                                          iterator_algorithm,
                                          tile.minimum_compute_capability,
                                          tile,
                                          A, B, C,
                                          element_epilogue,
                                          stride_support,
                                          epilogue_functor,
                                          swizzling_functor_,
                                          group_mode=GroupMode.Depthwise)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

class ConvOperation3x:
  """All parameters of a CUTLASS 3 convolution operation.

  Unlike CUTLASS 2 convolutions, CUTLASS 3 convolutions do not
  distinguish between 2-D and 3-D convolutions by kernel class name.
  Instead, for CUTLASS 3 convolutions, the tensor layouts encode
  whether the convolution is 2-D or 3-D.  Thus, this class deduces
  the OperationKind (either Conv2d or Conv3d) from the layouts,
  rather than taking it as a constructor parameter.
  """
  def __init__(self,
               conv_kind: ConvKind,
               tile_description: TileDescription,
               A: TensorDescription,
               B: TensorDescription,
               C: TensorDescription,
               element_compute: Optional[DataType] = None,
               D: Optional[TensorDescription] = None,
               kernel_schedule: KernelScheduleType = KernelScheduleType.ScheduleAuto,
               epilogue_schedule: EpilogueScheduleType = EpilogueScheduleType.ScheduleAuto,
               tile_scheduler: TileSchedulerType = TileSchedulerType.Default,
               log_indent_level: int = 1):
    log_debug_line(f'ConvOperation3x::init: conv_kind: {conv_kind}', log_indent_level)
    log_indent_level = log_indent_level + 1

    self.conv_kind = conv_kind
    self.tile_description = tile_description
    self.A = A
    self.B = B
    self.C = C
    self.element_compute = C.element if element_compute is None else element_compute
    self.kernel_schedule = kernel_schedule
    self.epilogue_schedule = epilogue_schedule

    self.arch = tile_description.minimum_compute_capability
    self.tile_scheduler = tile_scheduler
    if D == None:
      self.D = C
    else:
      self.D = D

    self.is_3x = True
    self.group_mode = GroupMode.NoneGroup # CUTLASS 3 convolutions currently aren't grouped

    operation_kind = None
    for layout in (A.layout, B.layout, C.layout):
      assert(isinstance(layout, LayoutType))
      new_operation_kind = convolution_tensor_layout_type_to_operation_kind(layout)
      if operation_kind is None:
        operation_kind = new_operation_kind
      else: # CUTLASS 3 convolutions don't permit mixing 2-D and 3-D layouts.
        assert(operation_kind == new_operation_kind)
    assert(operation_kind is not None)
    self.operation_kind = operation_kind

  def __str__(self):
    return f"ConvOperation3x: operation_kind={self.operation_kind}, conv_kind={self.conv_kind}, tile_description={self.tile_description}"

  def is_complex(self):
    complex_operators = [
      MathOperation.multiply_add_complex,
      MathOperation.multiply_add_complex_gaussian,
      MathOperation.multiply_add_complex_fast_f32
    ]
    return self.tile_description.math_instruction.math_operation in complex_operators

  def is_mixed_input(self):
    return self.A.element != self.B.element

  def accumulator_type(self):
    accum = self.tile_description.math_instruction.element_accumulator
    if self.is_complex():
      return get_complex_from_real(accum)
    return accum

  def short_math_name(self):
    if self.tile_description.math_instruction.math_operation == MathOperation.multiply_add_complex_gaussian:
      return "g%s" % ShortDataTypeNames[self.accumulator_type()]
    return ShortDataTypeNames[self.accumulator_type()]

  def core_name(self):
    ''' The basic operation kind is prefixed with a letter indicating the accumulation type. '''

    inst_shape = ''
    inst_operation = ''
    intermediate_type = ''

    math_operations_map = {
      MathOperation.xor_popc: 'xor',
      MathOperation.and_popc: 'and',
    }

    tensor_ops = [
      OpcodeClass.TensorOp,
      OpcodeClass.WmmaTensorOp,
      OpcodeClass.SparseTensorOp,
      OpcodeClass.BlockScaledTensorOp, 
    ]

    is_tensor_op = self.tile_description.math_instruction.opcode_class in tensor_ops

    if is_tensor_op:

      math_op = self.tile_description.math_instruction.math_operation
      math_op_string = math_operations_map[math_op] if math_op in math_operations_map.keys() else ''

      if self.tile_description.math_instruction.element_a != self.A.element and \
        self.tile_description.math_instruction.element_a != self.tile_description.math_instruction.element_accumulator:
        intermediate_type = DataTypeNames[self.tile_description.math_instruction.element_a]

    return "%s%s%s" % (math_op_string, intermediate_type, ConvKindNames[self.conv_kind])

  def extended_name(self):
    '''Generates a string representing the MMA atom. Assumes accumulator type is C type.'''
    extended_name = "{core_name}_{element_a}{layout_a}_{element_b}{layout_b}_{element_acc}_{element_c}_{element_d}{layout_c}".format(
      element_a = DataTypeNames[self.A.element],
      layout_a = ShortLayoutTypeNames[self.A.layout],
      element_b = DataTypeNames[self.B.element],
      layout_b = ShortLayoutTypeNames[self.B.layout],
      element_acc = DataTypeNames[self.accumulator_type()],
      element_c = DataTypeNames[self.C.element],
      layout_c = ShortLayoutTypeNames[self.C.layout],
      element_d = DataTypeNames[self.D.element],
      core_name = self.core_name())

    return extended_name

  # Generates a short string representing underlying kernel schedule type
  def kernel_schedule_name(self):
    return KernelScheduleSuffixes[self.kernel_schedule]

  # Generates a short string representing underlying epilogue schedule type
  def epilogue_schedule_name(self):
    return EpilogueScheduleSuffixes[self.epilogue_schedule]
  
  # Generate a short string representing the operation class
  def opcode_class_name(self):
    return OpcodeClassNames[self.tile_description.math_instruction.opcode_class]

  # Generates the full kernel function name
  def configuration_name(self):
    ''' The full function name indicates architecture, extended name, tile size, and layout. '''
    kernel_name_template = "cutlass3x_sm{ar}_{op}_{ex}{ct}{cs}_{l}_align{al}{t}{k}{e}"
    return kernel_name_template.format(
        ar = self.arch,
        op = self.opcode_class_name(),
        ex = self.extended_name(),
        ct = '_' + 'x'.join([str(i) for i in self.tile_description.tile_shape]) if self.tile_description.tile_shape[0] > 0 else "",
        cs = '_' + 'x'.join([str(i) for i in self.tile_description.cluster_shape]),
        l = self.tile_description.stages,
        al = str(max(self.A.alignment, self.B.alignment)),
        t = TileSchedulerSuffixes[self.tile_scheduler],
        k = self.kernel_schedule_name(),
        e = self.epilogue_schedule_name())

  def procedural_name(self):
    return self.configuration_name()

def convolution_tensor_layout_type_to_operation_kind(layout: LayoutType) -> OperationKind:
  if layout == LayoutType.TensorNHWC or layout == LayoutType.TensorKCSR:
    return OperationKind.Conv2d
  elif layout == LayoutType.TensorNDHWC or layout == LayoutType.TensorKCSRT:
    return OperationKind.Conv3d
  else:
    raise RuntimeError(f'LayoutType {layout} does not have a corresponding OperationKind')

def CreateConvOperator3x(manifest: Manifest,
                         dims_and_alignments: Sequence[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]],
                         tile_descriptions: Sequence[Sequence[TileDescription]],
                         data_types,
                         schedule_pairs: Sequence[Tuple[KernelScheduleType, KernelScheduleType]] = \
                           [(KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto)],
                         complex_transforms: Optional[Sequence[ComplexTransform]] = None,
                         tile_schedulers: Sequence[TileSchedulerType] = [TileSchedulerType.Default],
                         conv_kind: ConvKind = ConvKind.Fprop,
                         log_indent_level: int = 1):
  """
  Create zero or more CUTLASS 3 two-dimensional convolution operators.

  Create a CUTLASS 3 two-dimensional convolution operator
  for all feasible combinations of the input parameters.
  Add the operators to the manifest.

  dims_and_alignments: 3-level list.  Each outer list term is a list [A, B, C].
    Each inner list (A, B, or C) has the form [num_spatial_dimensions, alignment].
    Both are integers; the first is the number of spatial dimensions
    (currently, only 2 or 3 are supported), and the second is the byte alignment.
    We deduce the operation_kind (either OperationKind.Conv2d or OperationKind.Conv3d)
    from num_spatial_dimensions.

  This function doesn't take layouts, unlike the GEMM functions.
  CUTLASS 3 convolutions currently support three input layouts:

  * TensorNWC for 1-D convolutions,
  * TensorNHWC for 2-D convolutions, and
  * TensorNDHWC for 3-D convolutions.

  Output (C and D) layouts are the same as input layouts,
  except for Wgrad convolutions, where the layouts are

  * TensorKCS for 1-D convolutions,
  * TensorKCSR for 2-D convolutions, and
  * TensorKCSRT for 3-D convolutions.

  The output layouts are completely constrained by the input layouts
  and the convolution kind.

  tile_descriptions: 2-level list.
    Outer level has one list per math instruction.
    Inner level has one TileDescription for each cluster shape.

  data_types: Either a single data_type dictionary, or a list of them.
    Keys: 'a_type', 'b_type', 'c_type', 'd_type', 'acc_type', 'epi_type'

  complex_transforms: Optional list of pairs.
    First element of each pair is the complex transform for A, and
    second element of each pair is the complex transform for B.

  schedule_pairs: [(kernel_schedule, epilogue_schedule), ...]

  conv_kind: Convolution kind (Fprop, Dgrad, or Wgrad).
  """
  log_debug_line('CreateConvOperator3x', log_indent_level)
  log_indent_level = log_indent_level + 1
  log_debug_line(f'conv_kind: {conv_kind}', log_indent_level)

  for triple in dims_and_alignments:
    assert(isinstance(triple, tuple) or isinstance(triple, list))
    assert(len(triple) == 3)

    spatial_dimensionality = None # to be determined by loop below

    for entry in triple: # [A, B, C]
      assert(len(entry) == 2)
      [dim, alignment] = entry
      assert(type(dim) is int)
      assert(dim == 2 or dim == 3)
      assert(type(alignment) is int)
      assert(alignment > 0)
      if spatial_dimensionality is None:
        spatial_dimensionality = dim
      else:
        # A, B, and C need to have the same spatial dimensionality
        assert(spatial_dimensionality == dim)

  def input_and_output_layouts(spatial_dim: int, kind: ConvKind) -> Tuple[LayoutType, LayoutType]:
    if spatial_dim == 1:
      input_layout = LayoutType.TensorNWC
      if kind == ConvKind.Wgrad:
        output_layout = LayoutType.TensorKCS
      else:
        output_layout = input_layout
    elif spatial_dim == 2:
      input_layout = LayoutType.TensorNHWC
      if kind == ConvKind.Wgrad:
        output_layout = LayoutType.TensorKCSR
      else:
        output_layout = input_layout
    elif spatial_dim == 3:
      input_layout = LayoutType.TensorNDHWC
      if kind == ConvKind.Wgrad:
        output_layout = LayoutType.TensorKCSRT
      else:
        output_layout = input_layout
    else:
      assert(False)
    return (input_layout, output_layout)

  def dims_to_layouts(A_B_C: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> \
      Tuple[Tuple[LayoutType, int], Tuple[LayoutType, int], Tuple[LayoutType, int]]:
    [A, B, C] = A_B_C
    [spatial_dim, alignment] = A
    [input_layout, output_layout] = input_and_output_layouts(spatial_dim, conv_kind)
    return ((input_layout, A[1]),
            (input_layout, B[1]),
            (output_layout, C[1]))

  # layouts: list of triples (A, B, C).
  # Each of A, B, and C has the form [layout, alignment].
  layouts = [dims_to_layouts(A_B_C) for A_B_C in dims_and_alignments]

  if type(data_types) is dict:
    data_types = [data_types]

  for s in schedule_pairs:
    assert(len(s) == 2)

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none)]

  # product produces a one-pass generator, so the loop must call it anew each time.
  def make_combinations():
    return product(
      layouts,
      tile_descriptions,
      data_types,
      complex_transforms,
      schedule_pairs,
      tile_schedulers
    )

  operations = []
  for layout_triple, tile_description, data_type, complex_transform_pair, schedule_pair, tile_scheduler in make_combinations():
    A_layout, A_alignment = layout_triple[0]
    A_xform = complex_transform_pair[0]
    B_layout, B_alignment = layout_triple[1]
    B_xform = complex_transform_pair[1]
    C_layout, C_alignment = layout_triple[2]
    D_layout = C_layout
    D_alignment = C_alignment

    A = TensorDescription(data_type["a_type"], A_layout, A_alignment, A_xform)
    B = TensorDescription(data_type["b_type"], B_layout, B_alignment, B_xform)
    C = TensorDescription(data_type["c_type"], C_layout, C_alignment)
    D = TensorDescription(data_type["d_type"], D_layout, D_alignment)
    element_compute = data_type.get("epi_type", data_type["acc_type"])
    kernel_schedule, epilogue_schedule = schedule_pair

    operation = ConvOperation3x(conv_kind=conv_kind,
                                tile_description=tile_description,
                                A=A,
                                B=B,
                                C=C,
                                element_compute=element_compute,
                                D=D,
                                kernel_schedule=kernel_schedule,
                                epilogue_schedule=epilogue_schedule,
                                tile_scheduler=tile_scheduler,
                                log_indent_level=log_indent_level)
    log_debug_line(f'Created ConvOperation3x: {str(operation)}', log_indent_level)
    manifest.append(operation)
    operations.append(operation)

  return operations

###################################################################################################
###################################################################################################

#
def GenerateSM50_Simt(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f64, DataType.f64, DataType.f64,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 50
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 8], 2, [1, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    if math_inst.element_a == DataType.f32:
      conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
      CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

#
def GenerateSM50_Simt_complex(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add_complex),
  ]

  min_cc = 50
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128,  64, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 8], 2, [1, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      DataType.cf32,
      DataType.cf32,
      DataType.cf32,
      DataType.cf32,
    ]


    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

#
def GenerateSM50(manifest, cuda_version):
  GenerateSM50_Simt(manifest, cuda_version)
  GenerateSM50_Simt_complex(manifest, cuda_version)

###################################################################################################
###################################################################################################

#
def GenerateSM60_Simt(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 60
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 8], 2, [1, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)
#
def GenerateSM60_Simt_DepthwiseConv2d(manifest, cuda_version):

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 60
  max_cc = 1024

  alignment_constraints = [8,]

  filter_3x3 = [3, 3]
  filter_5x5 = [5, 5]

  # [stride_h, stride_w]
  # [-1, -1] means all stride size.
  strides = [[-1,-1], [1, 1], [2, 2]]
  # [dilation_h, dilation_w]
  # [-1, -1] means all dilation size.
  dilations = [[-1,-1], [1, 1], [2, 2]]

  #groups per thread block
  g16 = 16
  g32 = 32
  g64 = 64

  #output shape per thread block
  npq_1x4x4 = [1, 4, 4]
  npq_1x8x8 = [1, 8, 8]
  npq_1x10x10 = [1, 10, 10]

  tile_descriptions = []
  for math_inst in math_instructions:
    for stride, dilation in product(strides, dilations):
      tile_descriptions.extend([
        # filter3x3               ThreadBlock_output, filter, stage, warp
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g32], filter_3x3, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g64], filter_3x3, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g16], filter_3x3, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),

        Direct2dConvFixedStrideDilationTileDescription(npq_1x10x10+[g64], filter_3x3, 2, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),

        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g32], filter_3x3, 4, stride, dilation, [4, 1, 1],  math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g64], filter_3x3, 4,  stride, dilation,[4, 1, 1], math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g16], filter_3x3, 4, stride, dilation, [4, 1, 1],  math_inst, min_cc, max_cc),

        # filter5x5               ThreadBlock_output, filter, stage, warp
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g32], filter_5x5, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g64], filter_5x5, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g16], filter_5x5, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),

        Direct2dConvFixedStrideDilationTileDescription(npq_1x10x10+[g64], filter_5x5, 2, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),

        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g32], filter_5x5, 4, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g64], filter_5x5, 4, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g16], filter_5x5, 4, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc)
      ])

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateDepthwiseConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

#
def GenerateSM60(manifest, cuda_version):
  GenerateSM60_Simt(manifest, cuda_version)
  GenerateSM60_Simt_DepthwiseConv2d(manifest, cuda_version)

###################################################################################################
###################################################################################################

#
def GenerateSM61_Simt(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 4],                                      \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 61
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 32], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 32], 2, [1, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]
    data_type_mixed = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_a,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)
#

#
def GenerateSM61(manifest, cuda_version):
  GenerateSM61_Simt(manifest, cuda_version)

###################################################################################################
###################################################################################################

#
def GenerateSM70_TensorOp_884(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 1):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 4],                                      \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [8, 8, 4],                                      \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 70
  max_cc = 75

  alignment_constraints = [8, 4, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)

      CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, alignment_constraints)

#
def GenerateSM70_PlanarComplexTensorOp_884(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 1):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 4],                                      \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [8, 8, 4],                                      \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 70
  max_cc = 75

  alignment_constraints = [8, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, complex_transforms)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, complex_transforms)


#
def GenerateSM70_WmmaTensorOp_161616(manifest, cuda_version):

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 16, 16],                                   \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.WmmaTensorOp,                       \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 16, 16],                                   \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.WmmaTensorOp,                       \
      MathOperation.multiply_add),
  ]

  min_cc = 70
  max_cc = 1024

  alignment_constraints = [8,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)

#
##################################################################################################
#

def GenerateSM70(manifest, cuda_version):
  GenerateSM70_TensorOp_884(manifest, cuda_version)
  GenerateSM70_PlanarComplexTensorOp_884(manifest, cuda_version)

  # To limit build size, WMMA GEMMs are disabled for now.
  #
  #GenerateSM70_WmmaTensorOp_161616(manifest, cuda_version)

###################################################################################################
###################################################################################################

#
def GenerateSM75_TensorOp_1688_FewChannels(manifest, cuda_version, math_inst):

  min_cc = 75
  max_cc = 1024

  tile_descriptions = [
    TileDescription([128,  64, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([256,  64, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64,  64, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 64], 2, [2, 2, 2], math_inst, min_cc, max_cc),
  ]

  data_type = [
    math_inst.element_a,
    math_inst.element_b,
    math_inst.element_accumulator,
    math_inst.element_accumulator,
  ]

  conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)

  CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions, data_type, [4, 8])
  CreateConv2dFewChannelsOperator(manifest, conv_layout, tile_descriptions, data_type, [1, 2, 4])

  # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
  if math_inst.element_a != math_inst.element_accumulator:

    data_type_mixed = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_a,
      math_inst.element_accumulator,
    ]

    CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, [4, 8])
    CreateConv2dFewChannelsOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, [1, 2, 4])

#
def GenerateSM75_TensorOp_1688(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 8],                                     \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 8],                                     \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 75
  max_cc = 1024

  alignment_constraints = [8, 4, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64], 2, [1, 2, 2], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)

    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)

      CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, alignment_constraints)

    # Separate generator for 'few channels' specializations
    GenerateSM75_TensorOp_1688_FewChannels(manifest, cuda_version, math_inst)

#

#
def GenerateSM75_PlanarComplexTensorOp_1688(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 8],                                     \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 8],                                     \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 75
  max_cc = 1024

  alignment_constraints = [8, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([ 64, 128, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, complex_transforms)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, complex_transforms)

#
def GenerateSM75_TensorOp_8816_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 16],                                     \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [8, 8, 16],                                     \
      DataType.u8, DataType.u8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 75
  max_cc = 90

  alignment_constraints = [16,]
  alignment_constraints_small_channels = [16, 8, 4]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 64], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 64], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 64], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  32, 64], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 256, 64], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 64], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  32, 64], 2, [2, 1, 1], math_inst, min_cc, max_cc),

      TileDescription([256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 32], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  32, 32], 2, [2, 1, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      DataType.s32,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombination)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        DataType.f32,
      ]

      operations = []

      operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

      operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
        data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

      operations += CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions,
        data_type_mixed, alignment_constraints_small_channels, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

      operations += CreateConv2dFewChannelsOperator(manifest, conv_layout, tile_descriptions,
        data_type_mixed, alignment_constraints_small_channels, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

      for op in operations:
        if op.tile_description.threadblock_shape[1] >= 128:
          op.C.alignment = 16
        else:
          op.C.alignment = 8

#

#
def GenerateSM75_TensorOp_8816_Interleaved(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.ColumnMajorInterleaved32, LayoutType.RowMajorInterleaved32, LayoutType.ColumnMajorInterleaved32),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 16],                                     \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [8, 8, 16],                                     \
      DataType.u8, DataType.u8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 75
  max_cc = 90

  alignment_constraints = [16,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 64], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 64], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 64], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type_mixed = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_a,
      DataType.f32,
    ]

    operations = CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

    conv_layout = (LayoutType.TensorNC32HW32, LayoutType.TensorC32RSK32, LayoutType.TensorNC32HW32)

    operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

    for op in operations:
      op.C.alignment = 8
#

#
def GenerateSM75_TensorOp_8832_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 32],                                     \
      DataType.s4, DataType.s4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [8, 8, 32],                                     \
      DataType.u4, DataType.u4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 75
  max_cc = 89

  alignment_constraints = [32,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 128], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      DataType.s32,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombination)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        DataType.f32,
      ]

      operations = []

      operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

      operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
        data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

      for op in operations:
        if op.tile_description.threadblock_shape[1] >= 128:
          op.C.alignment = 16
        elif op.tile_description.threadblock_shape[1] == 64:
          op.C.alignment = 8
        else:
          op.C.alignment = 8

#

#
def GenerateSM75_TensorOp_8832_Interleaved(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.ColumnMajorInterleaved64, LayoutType.RowMajorInterleaved64, LayoutType.ColumnMajorInterleaved64),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 32],                                     \
      DataType.s4, DataType.s4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [8, 8, 32],                                     \
      DataType.u4, DataType.u4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 75
  max_cc = 89

  alignment_constraints = [32,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 128], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        DataType.f32,
      ]

      operations = CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

      conv_layout = (LayoutType.TensorNC64HW64, LayoutType.TensorC64RSK64, LayoutType.TensorNC64HW64)

      operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
        data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

      for op in operations:
        op.C.alignment = 16
#

#
def GenerateSM75_TensorOp_88128(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 128],                                   \
      DataType.b1, DataType.b1, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.xor_popc),
  ]

  min_cc = 75
  max_cc = {
    MathOperation.xor_popc: 89,
    MathOperation.and_popc: 90
  }

  alignment_constraints = [128,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 512], 2, [4, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128, 256, 512], 2, [2, 4, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128, 128, 512], 2, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64, 256, 512], 2, [1, 4, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([256,  64, 512], 2, [4, 1, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64, 128, 512], 2, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128,  64, 512], 2, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64,  64, 512], 2, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
    ]

    data_type = [DataType.b1, DataType.b1, DataType.s32, DataType.s32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

#

#
def GenerateSM75_WmmaTensorOp_161616(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 16, 16],                                   \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.WmmaTensorOp,                       \
      MathOperation.multiply_add),
  ]

  min_cc = 75
  max_cc = 1024

  alignment_constraints = [16,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      DataType.f32,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        DataType.f32,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)
#

#
def GenerateSM75_Simt_complex(manifest, cuda_version):
  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add_complex),
  ]

  min_cc = 75
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 8], 5, [4, 2, 1], math_inst, min_cc, max_cc)
    ]
    data_type = [
      DataType.cf32,
      DataType.cf32,
      DataType.cf32,
      DataType.cf32
    ]

    complex_transforms = [
      (ComplexTransform.none, ComplexTransform.none),
      (ComplexTransform.conj, ComplexTransform.none),
      (ComplexTransform.none, ComplexTransform.conj),
      (ComplexTransform.conj, ComplexTransform.conj)
    ]

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

def GenerateSM75(manifest, cuda_version):
  GenerateSM75_TensorOp_1688(manifest, cuda_version)
  GenerateSM75_PlanarComplexTensorOp_1688(manifest, cuda_version)
  GenerateSM75_TensorOp_8816_TN(manifest, cuda_version)
  GenerateSM75_TensorOp_8816_Interleaved(manifest, cuda_version)
  GenerateSM75_TensorOp_8832_TN(manifest, cuda_version)
  GenerateSM75_TensorOp_8832_Interleaved(manifest, cuda_version)
  GenerateSM75_TensorOp_88128(manifest, cuda_version)
  #GenerateSM75_WmmaTensorOp_161616(manifest, cuda_version)
  GenerateSM75_Simt_complex(manifest, cuda_version)


###################################################################################################
###################################################################################################

#
def GenerateSM80_TensorOp_16816(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.bf16, DataType.bf16, DataType.f32,     \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [8, 4, 2]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 64],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 64],  3, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    CreateGemmGroupedOperator(manifest, layouts, tile_descriptions, data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
    CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions, data_type, [4, 8])
    CreateConv3dOperator(manifest, LayoutType.TensorNDHWC, tile_descriptions, data_type, 8)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)

      CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, alignment_constraints)
      CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, [4, 8])
      CreateConv3dOperator(manifest, LayoutType.TensorNDHWC, tile_descriptions, data_type_mixed, 8)
#

#
def GenerateSM80_SparseTensorOp_16832(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 1):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.RowMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.bf16, DataType.bf16, DataType.f32,     \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [8]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([ 64, 128,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128,  64],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256,  64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128,  64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64,  64],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256,  64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64,  64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64,  64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)

#

#
def GenerateSM80_PlanarComplexTensorOp_16816(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.bf16, DataType.bf16, DataType.f32,     \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [8, ]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([ 64, 128, 32], 3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, complex_transforms)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, complex_transforms)

#
def GenerateSM80_TensorOp_16816_mixed_input_upcast_a(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  # Upcast on Operand A
  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.s8, DataType.f16, DataType.f32,        \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.u8, DataType.f16, DataType.f32,        \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.s8, DataType.bf16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.u8, DataType.bf16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.s8, DataType.f16, DataType.f16,        \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.u8, DataType.f16, DataType.f16,        \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
  ]

  min_cc = 80
  max_cc = 1024

  # For mixed-input alignment constraints are a list of lists, where the
  # inner list contains the alignment constraints for operands/matrices
  # [[alignA, alignB, alignC],..]
  alignment_constraints = [[16, 8, 8],]

  for math_inst in math_instructions:
    tile_descriptions = [
      # 128x128
      TileDescription([128, 128, 64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      # 128x64
      TileDescription([128, 64, 64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 64, 64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 64, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      # 128x32
      TileDescription([128, 32, 64],  9, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 32, 64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      # 128x16
      TileDescription([128, 16, 64],  5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 16, 64],  3, [2, 1, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    # streamk uses more regs which can cause spill for the biggest warp tile size when the accumulators are 32bit.
    operations = CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination, SwizzlingFunctor.Identity8)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_b != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_b,
        math_inst.element_accumulator,
      ]

      operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombination, SwizzlingFunctor.Identity8)

    for op in operations:
      if (DataTypeSize[op.C.element] == 16) and \
         (op.tile_description.threadblock_shape[1] <= 32):
        op.C.alignment = 4

#
def GenerateSM80_TensorOp_16816_mixed_input_upcast_b(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.s8, DataType.f32,        \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.u8, DataType.f32,        \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.bf16, DataType.s8, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.bf16, DataType.u8, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.s8, DataType.f16,        \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.u8, DataType.f16,        \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
  ]

  min_cc = 80
  max_cc = 1024

  # For mixed-input alignment constraints are a list of lists, where the
  # inner list contains the alignment constraints for operands/matrices
  # [[alignA, alignB, alignC],..]
  alignment_constraints = [[8, 16, 8],]

  for math_inst in math_instructions:
    tile_descriptions = [
      # 128x128
      TileDescription([128, 128, 64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      # 128x64
      TileDescription([128, 64, 64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 64, 64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 64, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      # 128x32
      TileDescription([128, 32, 64],  9, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 32, 64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 32, 32],  9, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 32, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      # 128x16
      TileDescription([128, 16, 64],  5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 16, 64],  3, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 16, 32],  9, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 16, 32],  5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 16, 32],  3, [2, 1, 1], math_inst, min_cc, max_cc),
      # 256x16
      TileDescription([256, 16, 32],  5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 16, 32],  3, [2, 1, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    # streamk uses more regs which can cause spill for the biggest warp tile size when the accumulators are 32bit.
    operations = CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination, SwizzlingFunctor.Identity8)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombination, SwizzlingFunctor.Identity8)

    for op in operations:
      if op.tile_description.threadblock_shape[1] <= 32:
        op.C.alignment = 4

#
def GenerateSM80_TensorOp_16832_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.u8, DataType.u8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 80
  max_cc = 1024
  smem_usage = 164

  alignment_constraints = [16,]
  alignment_constraints_small_channels = [16, 8, 4]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128,  64],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256,  64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64,  64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256,  64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  32,  64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 256,  64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128,  64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32,  64],  6, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128,  64],  6, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64,  64], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  32, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [math_inst.element_a, math_inst.element_b, math_inst.element_accumulator, DataType.s32]
    data_type_mixed = [math_inst.element_a, math_inst.element_b, math_inst.element_a, DataType.f32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombination)

    operations = []

    operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

    operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

    operations += CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints_small_channels, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

    operations += CreateConv2dFewChannelsOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints_small_channels, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

    for op in operations:
      if op.tile_description.threadblock_shape[1] >= 128:
        if op.tile_description.threadblock_shape[0] == 32:
          op.C.alignment = 8
        else:
          op.C.alignment = 16
      else:
        op.C.alignment = 8

#

def GenerateSM80_TensorOp_16832_TN_mixed_input_upcast_a(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  # Upcast on Operand A
  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.s4, DataType.s8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
  ]

  min_cc = 80
  max_cc = 1024

  # For mixed-input alignment constraints are a list of lists, where the 
  # inner list contains the alignment constraints for operands/matrices 
  # [[alignA, alignB, alignC],..]
  alignment_constraints = [[32, 16, 4],]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128,  64],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256,  64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64,  64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256,  64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 256,  64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128,  64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  32, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    # streamk uses more regs which can cause spill for the biggest warp tile size when the accumulators are 32bit.
    operations = CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination, SwizzlingFunctor.Identity8)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. S8 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
      alignment_constraints = [[32, 16, 16],]

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_b,
        DataType.f32
      ]

      operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp, SwizzlingFunctor.Identity8)

    for op in operations:
      if op.tile_description.threadblock_shape[1] >= 128:
        if op.tile_description.threadblock_shape[0] == 32:
          op.C.alignment = 8
        else:
          op.C.alignment = 16
      else:
        op.C.alignment = 8
#

#
def GenerateSM80_TensorOp_16832_TN_mixed_input_upcast_b(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  # Upcast on Operand B
  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.s8, DataType.s4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
  ]

  min_cc = 80
  max_cc = 1024

  # For mixed-input alignment constraints are a list of lists, where the 
  # inner list contains the alignment constraints for operands/matrices 
  # [[alignA, alignB, alignC],..]
  alignment_constraints = [[16, 32, 4],]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128,  64],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256,  64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64,  64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256,  64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  32,  64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128,  64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32,  64],  6, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  32, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    # streamk uses more regs which can cause spill for the biggest warp tile size when the accumulators are 32bit.
    operations = CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination, SwizzlingFunctor.Identity8)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. S8 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
      alignment_constraints = [[16, 32, 16],]

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        DataType.f32,
      ]

      operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp, SwizzlingFunctor.Identity8)

    for op in operations:
      if op.tile_description.threadblock_shape[1] >= 128:
        if op.tile_description.threadblock_shape[0] == 32:
          op.C.alignment = 8
        else:
          op.C.alignment = 16
      else:
        op.C.alignment = 8
#

#
def GenerateSM80_SparseTensorOp_16864_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 1):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
  ]

  math_inst =                                         \
    MathInstruction(                                  \
      [16, 8, 64],                                    \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [16,]

  tile_descriptions = [
    TileDescription([128,  64, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 128, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([256,  64, 128],  3, [4, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 128],  6, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64,  64, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128,  64, 256],  4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64,  64, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.s8, DataType.s8, DataType.s32, DataType.s32]
  data_type_mixed = [DataType.s8, DataType.s8, DataType.s8, DataType.f32]

  CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)

  operations = []

  operations += CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
    data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

  for op in operations:
    if op.tile_description.threadblock_shape[1] >= 128:
      op.C.alignment = 16
    else:
      op.C.alignment = 8
#

#
def GenerateSM80_TensorOp_16832_Interleaved(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajorInterleaved32, LayoutType.RowMajorInterleaved32, LayoutType.ColumnMajorInterleaved32),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.u8, DataType.u8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [16,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 64],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 64], 10, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type_mixed = [math_inst.element_a, math_inst.element_b, math_inst.element_a, DataType.f32]

    operations = CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

    conv_layout = (LayoutType.TensorNC32HW32, LayoutType.TensorC32RSK32, LayoutType.TensorNC32HW32)

    operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

    for op in operations:
      op.C.alignment = 8
#

#
def GenerateSM80_TensorOp_16864_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 64],                                    \
      DataType.s4, DataType.s4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [16, 8, 64],                                    \
      DataType.u4, DataType.u4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 80
  max_cc = 1024
  alignment_constraints = [32,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 128],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 256],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 256],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 256],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 256],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 256],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 256],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [math_inst.element_a, math_inst.element_b, math_inst.element_accumulator, DataType.s32]
    data_type_mixed = [math_inst.element_a, math_inst.element_b, math_inst.element_a, DataType.f32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)

    operations = []

    operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombination)

    operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

    for op in operations:
      if op.tile_description.threadblock_shape[1] >= 128:
        op.C.alignment = 16
      elif op.tile_description.threadblock_shape[1] == 64:
        op.C.alignment = 8
      else:
        op.C.alignment = 8
#

#
def GenerateSM80_SparseTensorOp_168128_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 1):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
  ]

  math_inst =                                         \
    MathInstruction(                                  \
      [16, 8, 128],                                    \
      DataType.s4, DataType.s4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate)

  min_cc = 80
  max_cc = 1024
  alignment_constraints = [32,]

  tile_descriptions = [
    TileDescription([ 64,  64, 256],  4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([256,  64, 256],  3, [4, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 128, 256],  3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 256, 256],  3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 256, 256],  4, [1, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128,  64, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 256],  6, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 128, 512],  3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128,  64, 512],  4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 512],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64,  64, 512],  3, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.s4, DataType.s4, DataType.s32, DataType.s32]
  data_type_mixed = [DataType.s4, DataType.s4, DataType.s4, DataType.f32]

  CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)

  operations = []

  operations += CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
    data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

  for op in operations:
    if op.tile_description.threadblock_shape[1] > 128:
      op.C.alignment = 16
    else:
      op.C.alignment = 8
#

#
def GenerateSM80_TensorOp_16864_Interleaved(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
      (LayoutType.ColumnMajorInterleaved64, LayoutType.RowMajorInterleaved64, LayoutType.ColumnMajorInterleaved64),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 64],                                    \
      DataType.s4, DataType.s4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [16, 8, 64],                                    \
      DataType.u4, DataType.u4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 80
  max_cc = 1024
  alignment_constraints = [32,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  6, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type_mixed = [math_inst.element_a, math_inst.element_b, math_inst.element_a, DataType.f32]

    operations = []

    operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

    conv_layout = (LayoutType.TensorNC64HW64, LayoutType.TensorC64RSK64, LayoutType.TensorNC64HW64)

    operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

    for op in operations:
      op.C.alignment = 16
#

#
def GenerateSM80_TensorOp_168256(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 256],                                   \
      DataType.b1, DataType.b1, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.xor_popc),
    MathInstruction(                                  \
      [16, 8, 256],                                   \
      DataType.b1, DataType.b1, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.and_popc),
  ]

  min_cc = 80
  max_cc = {
    MathOperation.xor_popc: 89,
    MathOperation.and_popc: 90
  }

  alignment_constraints = [128,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128,  512],  3, [4, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128, 256,  512],  3, [2, 4, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([256,  64,  512],  4, [4, 1, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64, 256,  512],  4, [1, 4, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128, 128,  512],  5, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128,  64,  512],  6, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64, 128,  512],  6, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64,  64,  512], 10, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([256, 128, 1024],  3, [4, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128, 256, 1024],  3, [2, 4, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([256,  64, 1024],  4, [4, 1, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64, 256, 1024],  4, [1, 4, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128, 128, 1024],  4, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128,  64, 1024],  3, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64, 128, 1024],  3, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64,  64, 1024],  5, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
    ]

    data_type = [DataType.b1, DataType.b1, DataType.s32, DataType.s32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

#

#
def GenerateSM80_TensorOp_1688(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                      \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,     \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [4, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 16],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 16],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 16], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64,  128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    data_type_mixed = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_a,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)

    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, alignment_constraints)
#

#
def GenerateSM80_TensorOp_1688_fast_math(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                      \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,     \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add),
    MathInstruction(                                      \
      [16, 8, 8],                                         \
      DataType.f16, DataType.f16, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_f16),
    MathInstruction(                                      \
      [16, 8, 8],                                         \
      DataType.bf16, DataType.bf16, DataType.f32,       \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_bf16),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [4, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 16],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 16],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 16], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

#
def GenerateSM80_TensorOp_1688_fast_fp32_math(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                      \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [4, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 16],  4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 16],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 16],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 16],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

def GenerateSM80_TensorOp_1688_fast_fp32_math_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst = MathInstruction(                            \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_fast_f32)

  min_cc = 80
  max_cc = 1024

  tile_descriptions = [
    TileDescription([128, 64, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [
    DataType.cf32, DataType.cf32, DataType.cf32, DataType.cf32
  ]

  alignment_constraints = [1,]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)


#
def GenerateSM80_SparseTensorOp_16816_fast_math(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 1):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.RowMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor),
  ]

  math_instructions = [
    MathInstruction(                                      \
      [16, 8, 16],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,     \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [4]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]

    CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)
#

#
def GenerateSM80_TensorOp_1688_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst = MathInstruction(                  \
    [16, 8, 8],                                 \
    DataType.tf32, DataType.tf32, DataType.f32,   \
    OpcodeClass.TensorOp,                       \
    MathOperation.multiply_add_complex)

  min_cc = 80
  max_cc = 1024

  tile_descriptions = [
    TileDescription([128, 128, 16], 4, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 4, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 4, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [
    DataType.cf32, DataType.cf32, DataType.cf32, DataType.cf32
  ]

  alignment_constraints = [1,]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM80_TensorOp_1688_rank_k(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_instructions = [
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1, 2, 4]  # Alignment only applies to A in SYRK

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      #TileDescription([256,  64, 16],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 256, 16],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64,  64, 16], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      #TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64,  64, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32]

    CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM80_TensorOp_1688_rank_k_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_instructions = [
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 64, 16], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 128, 16], 4, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([64, 32, 16], 4, [2, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([32, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      DataType.cf32, DataType.cf32, DataType.cf32
    ]

    alignment_constraints = [1,]

    # SYRK
    CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.symmetric)

    # HERK
    CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM80_TensorOp_1688_trmm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_instructions = [
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1, 2, 4]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 16],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 16],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 16], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      #TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64,  64, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]

    CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
      data_type, alignment_constraints)
#

#
def GenerateSM80_TensorOp_1688_trmm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_instructions = [
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 64, 16], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 128, 16], 4, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 32, 16], 4, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([32, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      DataType.cf32, DataType.cf32, DataType.cf32, DataType.cf32
    ]

    alignment_constraints = [1,]

    complex_transforms = [
      ComplexTransform.none, ComplexTransform.conj,
    ]

    CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
      data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM80_TensorOp_1688_symm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  # A and B have same layouts
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_instructions = [
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [
    1, 2, 4
  ]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      #TileDescription([256,  64, 16],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 256, 16],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64,  64, 16], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      #TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64,  64, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]

    CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM80_TensorOp_1688_symm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_instructions = [
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 64, 16], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 128, 16], 4, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([64, 32, 16], 4, [2, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([32, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      DataType.cf32, DataType.cf32, DataType.cf32, DataType.cf32
    ]

    alignment_constraints = [1,]

    # SYMM
    CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.symmetric)

    # HEMM
    CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM80_TensorOp_884(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 64, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 256, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 32, 16], 3, [4, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 256, 16], 3, [1, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints)
#

#
def GenerateSM80_TensorOp_884_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64,  8 ], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  128, 8 ], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  64,  8 ], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  32,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  64,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  32,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16,  32,  8 ], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  16,  8 ], 4, [2, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64,  16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  128, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  64,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  32,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  64,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  32,  16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16,  32,  16], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  16,  16], 3, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)

#
def GenerateSM80_TensorOp_884_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM80_TensorOp_884_rank_k(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64]

  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM80_TensorOp_884_rank_k_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64]

  # SYRK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HERK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)

#

#
def GenerateSM80_TensorOp_884_rank_k_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [ComplexTransform.none,]

  # SYRK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HERK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM80_TensorOp_884_trmm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints)
#

#
def GenerateSM80_TensorOp_884_trmm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    ComplexTransform.none, ComplexTransform.conj,
  ]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#


#
def GenerateSM80_TensorOp_884_trmm_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    ComplexTransform.none, ComplexTransform.conj,
  ]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM80_TensorOp_884_symm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM80_TensorOp_884_symm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  # SYMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HEMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM80_TensorOp_884_symm_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [ComplexTransform.none,]

  # SYMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HEMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#

###################################################################################################

#
def GenerateSM80_Simt_f32(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 8], 5, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 8], 5, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 8], 5, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 8], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 8], 4, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 8], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 8], 5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 8], 5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 8], 5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 8], 5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 8], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#


#
def GenerateSM80_Simt_f64(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f64, DataType.f64, DataType.f64,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 8], 5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 8], 5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 8], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)
#


##################################################################################################
#
def GenerateSM80_Simt_complex(manifest, cuda_version):
  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add_complex),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  data_type = [
    DataType.cf32,
    DataType.cf32,
    DataType.cf32,
    DataType.cf32
  ]

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  for math_inst in math_instructions:

    tile_descriptions = [
      TileDescription([128, 128, 8], 5, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 8], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, alignment_constraints, complex_transforms)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

###################################################################################################

#
def GenerateSM80(manifest, cuda_version):
  GenerateSM80_TensorOp_16816(manifest, cuda_version)
  GenerateSM80_SparseTensorOp_16832(manifest, cuda_version)
  GenerateSM80_PlanarComplexTensorOp_16816(manifest, cuda_version)
  GenerateSM80_TensorOp_1688(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_fast_math(manifest, cuda_version)
  GenerateSM80_SparseTensorOp_16816_fast_math(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_complex(manifest, cuda_version)
  # 3xTF32
  GenerateSM80_TensorOp_1688_fast_fp32_math(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_fast_fp32_math_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_rank_k(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_rank_k_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_trmm(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_trmm_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_symm(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_symm_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_884(manifest, cuda_version)
  GenerateSM80_TensorOp_884_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_884_complex_gaussian(manifest, cuda_version)
  GenerateSM80_TensorOp_884_rank_k(manifest, cuda_version)
  GenerateSM80_TensorOp_884_rank_k_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_884_rank_k_complex_gaussian(manifest, cuda_version)
  GenerateSM80_TensorOp_884_trmm(manifest, cuda_version)
  GenerateSM80_TensorOp_884_trmm_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_884_trmm_complex_gaussian(manifest, cuda_version)
  GenerateSM80_TensorOp_884_symm(manifest, cuda_version)
  GenerateSM80_TensorOp_884_symm_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_884_symm_complex_gaussian(manifest, cuda_version)
  GenerateSM80_TensorOp_16816_mixed_input_upcast_a(manifest, cuda_version)
  GenerateSM80_TensorOp_16816_mixed_input_upcast_b(manifest, cuda_version)
  GenerateSM80_TensorOp_16832_TN(manifest, cuda_version)
  GenerateSM80_TensorOp_16832_TN_mixed_input_upcast_a(manifest, cuda_version)
  GenerateSM80_TensorOp_16832_TN_mixed_input_upcast_b(manifest, cuda_version)
  GenerateSM80_SparseTensorOp_16864_TN(manifest, cuda_version)
  GenerateSM80_TensorOp_16832_Interleaved(manifest, cuda_version)
  GenerateSM80_TensorOp_16864_TN(manifest, cuda_version)
  GenerateSM80_SparseTensorOp_168128_TN(manifest, cuda_version)
  GenerateSM80_TensorOp_16864_Interleaved(manifest, cuda_version)
  GenerateSM80_TensorOp_168256(manifest, cuda_version)
  GenerateSM80_Simt_f32(manifest, cuda_version)
  GenerateSM80_Simt_f64(manifest, cuda_version)
  GenerateSM80_Simt_complex(manifest, cuda_version)

###################################################################################################

def GenerateSM89_TensorOp_16832_fp8(manifest, element_acc):
  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor)
  ]

  math_instructions = [
    MathInstruction(
      [16, 8, 32],
      DataType.e4m3, DataType.e4m3, element_acc,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [16, 8, 32],
      DataType.e4m3, DataType.e5m2, element_acc,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [16, 8, 32],
      DataType.e5m2, DataType.e4m3, element_acc,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [16, 8, 32],
      DataType.e5m2, DataType.e5m2, element_acc,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [16, 8, 32],
      DataType.e4m3, DataType.e4m3, element_acc,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add_fast_accum),
    MathInstruction(
      [16, 8, 32],
      DataType.e4m3, DataType.e5m2, element_acc,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add_fast_accum),
    MathInstruction(
      [16, 8, 32],
      DataType.e5m2, DataType.e4m3, element_acc,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add_fast_accum),
    MathInstruction(
      [16, 8, 32],
      DataType.e5m2, DataType.e5m2, element_acc,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add_fast_accum),
  ]

  min_cc = 89
  max_cc = 100
  alignment_constraints = [16,]
  alignment_constraints_small_channels = [16, 8, 4]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128,  64],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128,  64],  6, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256,  64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256,  64],  6, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64,  64],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64,  64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256,  64],  3, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256,  64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  32, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  32,  64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 256,  64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128,  64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128,  64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128,  64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64,  64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64,  64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64,  64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128,  64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128,  64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128,  64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32,  64],  6, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128,  64],  6, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64,  64], 10, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_types = [
      [
        math_inst.element_a,
        math_inst.element_b,
        DataType.f32,
        math_inst.element_accumulator
      ],
      [
        math_inst.element_a,
        math_inst.element_b,
        DataType.bf16,
        math_inst.element_accumulator
      ],
    ]

    operations = []
    for data_type in data_types:
      operations += CreateGemmOperator(manifest, layouts, tile_descriptions, data_type,
        alignment_constraints, None, EpilogueFunctor.LinearCombination)

      conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
      operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
        data_type, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombination)

      operations += CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions,
        data_type, alignment_constraints_small_channels, [ConvKind.Fprop], EpilogueFunctor.LinearCombination)

    for op in operations:
      if op.tile_description.threadblock_shape[1] >= 128:
        if op.tile_description.threadblock_shape[0] == 32:
          op.C.alignment = 8
        else:
          op.C.alignment = 16
      else:
        op.C.alignment = 8

def GenerateSM89_TensorOp_16832_fp8_fp32acc(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 4):
    return

  GenerateSM89_TensorOp_16832_fp8(manifest, DataType.f32)

def GenerateSM89_TensorOp_16832_fp8_fp16acc(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 8):
    return

  GenerateSM89_TensorOp_16832_fp8(manifest, DataType.f16)

#
def GenerateSM89_SparseTensorOp_16864_fp8(manifest, cuda_version):

  if (
    not CudaToolkitVersionSatisfies(cuda_version, 12, 4)
  ):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor)
  ]

  math_instructions = [
    MathInstruction(
      [16, 8, 64],
      DataType.e4m3, DataType.e4m3, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [16, 8, 64],
      DataType.e4m3, DataType.e5m2, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [16, 8, 64],
      DataType.e5m2, DataType.e4m3, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [16, 8, 64],
      DataType.e5m2, DataType.e5m2, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [16, 8, 64],
      DataType.e4m3, DataType.e4m3, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add_fast_accum),
    MathInstruction(
      [16, 8, 64],
      DataType.e4m3, DataType.e5m2, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add_fast_accum),
    MathInstruction(
      [16, 8, 64],
      DataType.e5m2, DataType.e4m3, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add_fast_accum),
    MathInstruction(
      [16, 8, 64],
      DataType.e5m2, DataType.e5m2, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add_fast_accum),
  ]

  min_cc = 89
  max_cc = 89

  alignment_constraints = [16,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128,  64, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 256],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_types = [
      [
        math_inst.element_a,
        math_inst.element_b,
        DataType.f32,
        math_inst.element_accumulator
      ],
    ]

    operations = []
    for data_type in data_types:
      operations += CreateSparseGemmOperator(manifest, layouts, tile_descriptions, data_type,
        alignment_constraints, None, EpilogueFunctor.LinearCombination)

    for op in operations:
      if op.tile_description.threadblock_shape[1] >= 128:
        op.C.alignment = 16
      else:
        op.C.alignment = 8

###################################################################################################

#
def GenerateSM89(manifest, cuda_version):
  GenerateSM89_TensorOp_16832_fp8_fp32acc(manifest, cuda_version)
  GenerateSM89_TensorOp_16832_fp8_fp16acc(manifest, cuda_version)
  GenerateSM89_SparseTensorOp_16864_fp8(manifest, cuda_version)

###################################################################################################


try:
    from .sm90_utils import (
        generate_fp16_bf16_math_instructions_sm90,
        generate_tf32_math_instructions_sm90,
        generate_int8_math_instructions_sm90,
        generate_fp8_math_instructions_sm90,
        generate_mixed_dtype_math_instructions_sm90,
        make_sparse_math_instructions,
        generate_tile_descriptions_sm90,
        get_valid_schedules,
        generate_data_types_from_math_instruction,
        fix_alignments,
    )
except ImportError:
    from sm90_utils import (
        generate_fp16_bf16_math_instructions_sm90,
        generate_tf32_math_instructions_sm90,
        generate_int8_math_instructions_sm90,
        generate_fp8_math_instructions_sm90,
        generate_mixed_dtype_math_instructions_sm90,
        make_sparse_math_instructions,
        generate_tile_descriptions_sm90,
        get_valid_schedules,
        generate_data_types_from_math_instruction,
        fix_alignments,
    )

def GenerateSM90_TensorOp_16b_WGMMA_gemm(manifest, cuda_version, gemm_kind=GemmKind.Universal3x):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 3 if is_grouped(gemm_kind) else 0):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=100, default_level=131, exhaustive_level=9992)
  is_aligned = True

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 8], [LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    8], [LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 1]],
  ]

  math_instructions = generate_fp16_bf16_math_instructions_sm90(instantiation_level)
  tile_descriptions = generate_tile_descriptions_sm90(
      math_instructions=math_instructions,
      is_aligned=is_aligned,
      level=instantiation_level)

  for tile_desc in tile_descriptions:
    math_inst = tile_desc.math_instruction
    data_type_w_source = generate_data_types_from_math_instruction(math_inst)
    data_type_wo_source = generate_data_types_from_math_instruction(math_inst, element_source=DataType.void)
    data_types = [data_type_w_source, data_type_wo_source]

    # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
        data_type_mixed_w_source = generate_data_types_from_math_instruction(
            math_inst,
            element_source=math_inst.element_a,
            element_dest=math_inst.element_a
        )
        data_type_mixed_wo_source = generate_data_types_from_math_instruction(
            math_inst,
            element_source=DataType.void,
            element_dest=math_inst.element_a
        )
        data_types.append(data_type_mixed_w_source)
        data_types.append(data_type_mixed_wo_source)

    for layout in layouts:
        for data_type in data_types:
            layout = fix_alignments(data_type, layout, alignment_bits=128)

            schedules, stream_k_schedules = get_valid_schedules(
              tile_description=tile_desc,
              cuda_version=cuda_version,
              is_aligned=is_aligned,
              data_types=data_type,
              instantiation_level=instantiation_level,
              layout=layout,
              gemm_kind=gemm_kind,
            )

            if len(schedules):
              CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type, schedules, gemm_kind=gemm_kind)
              if len(stream_k_schedules):
                assert CudaToolkitVersionSatisfies(cuda_version, 12, 1)
                CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type,
                                              stream_k_schedules,
                                              tile_schedulers=[TileSchedulerType.StreamK])


def GenerateSM90_TensorOp_16b_WGMMA_alignx_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=100, default_level=101, exhaustive_level=9992)
  is_aligned = False

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    4], [LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 4], [LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    2], [LayoutType.ColumnMajor, 2], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    2], [LayoutType.RowMajor,    2], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 2], [LayoutType.ColumnMajor, 2], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 2], [LayoutType.RowMajor,    2], [LayoutType.ColumnMajor, 1]],
  ]

  math_instructions = generate_fp16_bf16_math_instructions_sm90(instantiation_level)
  tile_descriptions = generate_tile_descriptions_sm90(
      math_instructions=math_instructions,
      is_aligned=is_aligned,
      level=instantiation_level)

  for tile_desc in tile_descriptions:
    math_inst = tile_desc.math_instruction
    data_type_w_source = generate_data_types_from_math_instruction(math_inst)
    data_types = [data_type_w_source]

    # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
        data_type_mixed_w_source = generate_data_types_from_math_instruction(
            math_inst,
            element_source=math_inst.element_a,
            element_dest=math_inst.element_a
        )
        data_types.append(data_type_mixed_w_source)

    for layout in layouts:
        for data_type in data_types:
            layout = fix_alignments(data_type, layout, alignment_bits=128)

            schedules, stream_k_schedules = get_valid_schedules(
              tile_description=tile_desc,
              cuda_version=cuda_version,
              is_aligned=is_aligned,
              data_types=data_type,
              instantiation_level=instantiation_level,
              layout=layout,
            )

            if len(schedules):
              CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type, schedules)
              if len(stream_k_schedules):
                assert CudaToolkitVersionSatisfies(cuda_version, 12, 1)
                CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type,
                                              stream_k_schedules,
                                              tile_schedulers=[TileSchedulerType.StreamK])

def GenerateSM90_SparseTensorOp_16b_WGMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 2):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=100, default_level=131, exhaustive_level=9992)
  is_aligned = True

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 8], [LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,   16], [LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,   16], [LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 1]],
  ]

  math_instructions = make_sparse_math_instructions(generate_fp16_bf16_math_instructions_sm90(instantiation_level))
  tile_descriptions = generate_tile_descriptions_sm90(
      math_instructions=math_instructions,
      is_aligned=is_aligned,
      level=instantiation_level)

  for tile_desc in tile_descriptions:
    math_inst = tile_desc.math_instruction
    data_type_w_source = generate_data_types_from_math_instruction(math_inst)
    data_type_wo_source = generate_data_types_from_math_instruction(math_inst, element_source=DataType.void)
    data_types = [data_type_w_source, data_type_wo_source]

    # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
        data_type_mixed_w_source = generate_data_types_from_math_instruction(
            math_inst,
            element_source=math_inst.element_a,
            element_dest=math_inst.element_a
        )
        data_type_mixed_wo_source = generate_data_types_from_math_instruction(
            math_inst,
            element_source=DataType.void,
            element_dest=math_inst.element_a
        )
        data_types.append(data_type_mixed_w_source)
        data_types.append(data_type_mixed_wo_source)

    for layout in layouts:
        for data_type in data_types:
            layout = fix_alignments(data_type, layout, alignment_bits=128)

            schedules, stream_k_schedules = get_valid_schedules(
              tile_description=tile_desc,
              cuda_version=cuda_version,
              is_aligned=is_aligned,
              data_types=data_type,
              instantiation_level=instantiation_level,
              layout=layout,
            )

            if len(schedules):
              CreateSparseGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type, schedules)
              if len(stream_k_schedules):
                assert CudaToolkitVersionSatisfies(cuda_version, 12, 1)
                CreateSparseGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type,
                                                    stream_k_schedules,
                                                    tile_schedulers=[TileSchedulerType.StreamK])


def GenerateSM90_TensorOp_tf32_WGMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=120, default_level=121, exhaustive_level=9992)
  is_aligned = True

  # layouts for ABC and their alignments
  layouts = [
    [[LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 4]],
    [[LayoutType.RowMajor,    4], [LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 4]],
    [[LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 4]],
    [[LayoutType.ColumnMajor, 4], [LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 4]],
  ]

  math_instructions = generate_tf32_math_instructions_sm90(instantiation_level)
  tile_descriptions = generate_tile_descriptions_sm90(
      math_instructions=math_instructions,
      is_aligned=is_aligned,
      level=instantiation_level)

  for tile_desc in tile_descriptions:
    math_inst = tile_desc.math_instruction

    for layout in layouts:
        data_type_tf32 = generate_data_types_from_math_instruction(math_inst)
        data_type_tf32_wo_source = generate_data_types_from_math_instruction(math_inst, element_source=DataType.void)
        data_type_f32 = copy.deepcopy(data_type_tf32)
        data_type_f32_wo_source = copy.deepcopy(data_type_tf32_wo_source)
        data_type_f32["a_type"] = DataType.f32
        data_type_f32["b_type"] = DataType.f32
        data_type_f32["epi_type"] = DataType.f32
        data_type_f32_wo_source["a_type"] = DataType.f32
        data_type_f32_wo_source["b_type"] = DataType.f32
        data_type_f32_wo_source["epi_type"] = DataType.f32
        data_types = [data_type_tf32, data_type_f32, data_type_tf32_wo_source, data_type_f32_wo_source]

        for data_type in data_types:
            layout = fix_alignments(data_type, layout, alignment_bits=128)

            schedules, stream_k_schedules = get_valid_schedules(
              tile_description=tile_desc,
              cuda_version=cuda_version,
              is_aligned=is_aligned,
              data_types=data_type,
              instantiation_level=instantiation_level,
              layout=layout,
            )

            if len(schedules):
              CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type, schedules)
              if len(stream_k_schedules):
                assert CudaToolkitVersionSatisfies(cuda_version, 12, 1)
                CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type,
                                              stream_k_schedules,
                                              tile_schedulers=[TileSchedulerType.StreamK])


def GenerateSM90_TensorOp_tf32_WGMMA_alignx_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=100, default_level=101, exhaustive_level=9992)
  is_aligned = False

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.RowMajor,    2], [LayoutType.ColumnMajor, 2], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    2], [LayoutType.RowMajor,    2], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 2], [LayoutType.ColumnMajor, 2], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 2], [LayoutType.RowMajor,    2], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    1], [LayoutType.ColumnMajor, 1], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    1], [LayoutType.RowMajor,    1], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 1], [LayoutType.ColumnMajor, 1], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 1], [LayoutType.RowMajor,    1], [LayoutType.ColumnMajor, 1]],
  ]

  math_instructions = generate_tf32_math_instructions_sm90(instantiation_level)
  tile_descriptions = generate_tile_descriptions_sm90(
      math_instructions=math_instructions,
      is_aligned=is_aligned,
      level=instantiation_level)

  for tile_desc in tile_descriptions:
    math_inst = tile_desc.math_instruction

    for layout in layouts:
        # Inconsistency: TF32 does not stamp out void-C
        data_type_tf32 = generate_data_types_from_math_instruction(math_inst)
        data_type_f32 = copy.deepcopy(data_type_tf32)
        data_type_f32["a_type"] = DataType.f32
        data_type_f32["b_type"] = DataType.f32
        data_type_f32["epi_type"] = DataType.f32
        for data_type in [data_type_tf32, data_type_f32]:
            # Inconsistency: alignments aren't fixed in TF32 / alignx
            # layout = fix_alignments(data_type, layout, alignment_bits=128)

            schedules, stream_k_schedules = get_valid_schedules(
              tile_description=tile_desc,
              cuda_version=cuda_version,
              is_aligned=is_aligned,
              data_types=data_type,
              instantiation_level=instantiation_level,
              layout=layout,
            )

            if len(schedules):
              CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type, schedules)
              if len(stream_k_schedules):
                assert CudaToolkitVersionSatisfies(cuda_version, 12, 1)
                CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type,
                                              stream_k_schedules,
                                              tile_schedulers=[TileSchedulerType.StreamK])


def GenerateSM90_SparseTensorOp_tf32_WGMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 2):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=120, default_level=121, exhaustive_level=9992)
  is_aligned = True

  # layouts for ABC and their alignments
  layouts = [
    [[LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 4]],
  ]

  math_instructions = make_sparse_math_instructions(generate_tf32_math_instructions_sm90(instantiation_level))
  tile_descriptions = generate_tile_descriptions_sm90(
      math_instructions=math_instructions,
      is_aligned=is_aligned,
      level=instantiation_level)

  for tile_desc in tile_descriptions:
    math_inst = tile_desc.math_instruction

    for layout in layouts:
        data_type_tf32 = generate_data_types_from_math_instruction(math_inst)
        data_type_tf32_wo_source = generate_data_types_from_math_instruction(math_inst, element_source=DataType.void)
        data_type_f32 = copy.deepcopy(data_type_tf32)
        data_type_f32_wo_source = copy.deepcopy(data_type_tf32_wo_source)
        data_type_f32["a_type"] = DataType.f32
        data_type_f32["b_type"] = DataType.f32
        data_type_f32["epi_type"] = DataType.f32
        data_type_f32_wo_source["a_type"] = DataType.f32
        data_type_f32_wo_source["b_type"] = DataType.f32
        data_type_f32_wo_source["epi_type"] = DataType.f32
        data_types = [data_type_tf32, data_type_f32, data_type_tf32_wo_source, data_type_f32_wo_source]

        for data_type in data_types:
            layout = fix_alignments(data_type, layout, alignment_bits=128)

            schedules, stream_k_schedules = get_valid_schedules(
              tile_description=tile_desc,
              cuda_version=cuda_version,
              is_aligned=is_aligned,
              data_types=data_type,
              instantiation_level=instantiation_level,
              layout=layout,
            )

            if len(schedules):
              CreateSparseGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type, schedules)
              if len(stream_k_schedules):
                assert CudaToolkitVersionSatisfies(cuda_version, 12, 1)
                CreateSparseGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type,
                                                    stream_k_schedules,
                                                    tile_schedulers=[TileSchedulerType.StreamK])


def GenerateSM90_TensorOp_int8_WGMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=100, default_level=111, exhaustive_level=9992)
  is_aligned = True

  # layouts for ABC and their alignments
  layouts = [
    [[LayoutType.RowMajor, 16], [LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 16]],
  ]

  math_instructions = generate_int8_math_instructions_sm90(instantiation_level)
  tile_descriptions = generate_tile_descriptions_sm90(
      math_instructions=math_instructions,
      is_aligned=is_aligned,
      level=instantiation_level)

  for tile_desc in tile_descriptions:
    math_inst = tile_desc.math_instruction
    data_type_w_source = generate_data_types_from_math_instruction(math_inst)
    data_type_wo_source = generate_data_types_from_math_instruction(math_inst, element_source=DataType.void)
    data_type_int8_output = generate_data_types_from_math_instruction(
        math_inst,
        element_source=DataType.s8,
        element_dest=math_inst.element_a,
        element_epilogue=DataType.f32
    )
    data_types = [data_type_w_source, data_type_wo_source, data_type_int8_output]

    for layout in layouts:
        for data_type in data_types:
            layout = fix_alignments(data_type, layout, alignment_bits=128)

            schedules, stream_k_schedules = get_valid_schedules(
              tile_description=tile_desc,
              cuda_version=cuda_version,
              is_aligned=is_aligned,
              data_types=data_type,
              instantiation_level=instantiation_level,
              layout=layout,
            )

            if len(schedules):
              CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type, schedules)
              if len(stream_k_schedules):
                assert CudaToolkitVersionSatisfies(cuda_version, 12, 1)
                CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type,
                                              stream_k_schedules,
                                              tile_schedulers=[TileSchedulerType.StreamK])


def GenerateSM90_TensorOp_int8_WGMMA_alignx_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=100, default_level=111, exhaustive_level=9992)
  is_aligned = False

  # layouts for ABC and their alignments
  layouts = [
    [[LayoutType.RowMajor,  8], [LayoutType.ColumnMajor,  8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,  4], [LayoutType.ColumnMajor,  4], [LayoutType.ColumnMajor, 1]],
  ]

  math_instructions = generate_int8_math_instructions_sm90(instantiation_level)
  tile_descriptions = generate_tile_descriptions_sm90(
      math_instructions=math_instructions,
      is_aligned=is_aligned,
      level=instantiation_level)

  for tile_desc in tile_descriptions:
    math_inst = tile_desc.math_instruction
    data_type_w_source = generate_data_types_from_math_instruction(math_inst)
    data_type_int8_output = generate_data_types_from_math_instruction(
        math_inst,
        element_source=DataType.s8,
        element_dest=math_inst.element_a,
        element_epilogue=DataType.f32
    )
    data_types = [data_type_w_source, data_type_int8_output]

    for layout in layouts:
        for data_type in data_types:
            layout = fix_alignments(data_type, layout, alignment_bits=128)

            schedules, stream_k_schedules = get_valid_schedules(
              tile_description=tile_desc,
              cuda_version=cuda_version,
              is_aligned=is_aligned,
              data_types=data_type,
              instantiation_level=instantiation_level,
              layout=layout,
            )

            if len(schedules):
              CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type, schedules)
              if len(stream_k_schedules):
                assert CudaToolkitVersionSatisfies(cuda_version, 12, 1)
                CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type,
                                              stream_k_schedules,
                                              tile_schedulers=[TileSchedulerType.StreamK])


def GenerateSM90_SparseTensorOp_int8_WGMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 2):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=100, default_level=111, exhaustive_level=9992)
  is_aligned = True

  # layouts for ABC and their alignments
  layouts = [
    [[LayoutType.RowMajor, 32], [LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 16]],
  ]

  math_instructions = make_sparse_math_instructions(generate_int8_math_instructions_sm90(instantiation_level))
  tile_descriptions = generate_tile_descriptions_sm90(
      math_instructions=math_instructions,
      is_aligned=is_aligned,
      level=instantiation_level)

  for tile_desc in tile_descriptions:
    math_inst = tile_desc.math_instruction
    # s8.u8 and u8.s8 wgmma variants require PTX 8.4
    if math_inst.element_a != math_inst.element_b and not CudaToolkitVersionSatisfies(cuda_version, 12, 4):
      continue
    data_type_w_source = generate_data_types_from_math_instruction(math_inst)
    data_type_wo_source = generate_data_types_from_math_instruction(math_inst, element_source=DataType.void)
    data_type_int8_output = generate_data_types_from_math_instruction(
        math_inst,
        element_source=DataType.s8,
        element_dest=math_inst.element_a,
        element_epilogue=DataType.f32
    )
    data_types = [data_type_w_source, data_type_wo_source, data_type_int8_output]

    for layout in layouts:
        for data_type in data_types:
            layout = fix_alignments(data_type, layout, alignment_bits=128)

            schedules, stream_k_schedules = get_valid_schedules(
              tile_description=tile_desc,
              cuda_version=cuda_version,
              is_aligned=is_aligned,
              data_types=data_type,
              instantiation_level=instantiation_level,
              layout=layout,
            )

            if len(schedules):
              CreateSparseGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type, schedules)
              if len(stream_k_schedules):
                assert CudaToolkitVersionSatisfies(cuda_version, 12, 1)
                CreateSparseGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type,
                                                    stream_k_schedules,
                                                    tile_schedulers=[TileSchedulerType.StreamK])


def GenerateSM90_TensorOp_fp8_WGMMA_gemm(manifest, cuda_version, gemm_kind=GemmKind.Universal3x):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 3 if is_grouped(gemm_kind) else 0):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=20, default_level=121, exhaustive_level=9992)
  is_aligned = True

  # layouts for ABC and their alignments
  layouts = [
    [[LayoutType.RowMajor, 16], [LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 1]],  # TN Layout
  ]

  math_instructions = generate_fp8_math_instructions_sm90(instantiation_level)
  tile_descriptions = generate_tile_descriptions_sm90(
      math_instructions=math_instructions,
      is_aligned=is_aligned,
      level=instantiation_level)

  for tile_desc in tile_descriptions:
    math_inst = tile_desc.math_instruction
    data_types = []
    fp8_types = [DataType.e4m3, DataType.e5m2]
    valid_types_for_d = [DataType.f32, DataType.bf16, DataType.f16, DataType.e4m3, DataType.e5m2]
    valid_types_for_c = copy.deepcopy(valid_types_for_d)
    valid_types_for_c.append(DataType.void)
    for c_type, d_type in product(valid_types_for_c, valid_types_for_d):
        data_types.append(
            generate_data_types_from_math_instruction(
                math_inst,
                element_source=c_type,
                element_dest=d_type,
            )
        )
    else:
        for d_type in valid_types_for_d:
            data_types.append(
                generate_data_types_from_math_instruction(
                    math_inst,
                    element_source=DataType.void,
                    element_dest=d_type,
                )
            )

    for layout in layouts:
        for data_type in data_types:
            # Inconsistency: alignments aren't fixed in FP8
            # layout = fix_alignments(data_type, layout, alignment_bits=128)

            schedules, stream_k_schedules = get_valid_schedules(
              tile_description=tile_desc,
              cuda_version=cuda_version,
              is_aligned=is_aligned,
              data_types=data_type,
              instantiation_level=instantiation_level,
              layout=layout,
              gemm_kind=gemm_kind,
            )

            if len(schedules):
              CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type, schedules, gemm_kind=gemm_kind)
              if len(stream_k_schedules):
                assert CudaToolkitVersionSatisfies(cuda_version, 12, 1)
                CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type,
                                              stream_k_schedules,
                                              tile_schedulers=[TileSchedulerType.StreamK])

def GenerateSM90_TensorOp_fp8_WGMMA_gemm_with_blockwise(manifest, cuda_version, gemm_kind=GemmKind.BlockwiseUniversal3x):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 3 if is_grouped(gemm_kind) else 0):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=20, default_level=121, exhaustive_level=9992)
  is_aligned = True

  # layouts for ABC and their alignments
  layouts = [
    [[LayoutType.RowMajor, 16], [LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 1]],  # TN Layout
  ]

  math_instructions = generate_fp8_math_instructions_sm90(instantiation_level)
  tile_descriptions_ = generate_tile_descriptions_sm90(
      math_instructions=math_instructions,
      is_aligned=is_aligned,
      level=instantiation_level)

  tile_descriptions = list()

  for desc in tile_descriptions_:
    desc.explicit_vector_sizes = [1, desc.tile_shape[1], desc.tile_shape[2]]
    tile_descriptions.append(copy.deepcopy(desc))
    desc.explicit_vector_sizes = [desc.tile_shape[0], desc.tile_shape[1], desc.tile_shape[2]]
    tile_descriptions.append(copy.deepcopy(desc))
    desc.explicit_vector_sizes = [desc.tile_shape[0], desc.tile_shape[1], desc.tile_shape[2]]
    tile_descriptions.append(copy.deepcopy(desc))
    desc.explicit_vector_sizes = [1, 1, desc.tile_shape[2]]
    tile_descriptions.append(copy.deepcopy(desc))

  for tile_desc in tile_descriptions:
    math_inst = tile_desc.math_instruction
    data_types = []
    fp8_types = [DataType.e4m3, DataType.e5m2]
    valid_types_for_d = [DataType.f32, DataType.bf16, DataType.f16, DataType.e4m3, DataType.e5m2]
    valid_types_for_c = copy.deepcopy(valid_types_for_d)
    valid_types_for_c.append(DataType.void)
    for c_type, d_type in product(valid_types_for_c, valid_types_for_d):
        data_types.append(
            generate_data_types_from_math_instruction(
                math_inst,
                element_source=c_type,
                element_dest=d_type,
            )
        )
    else:
        for d_type in valid_types_for_d:
            data_types.append(
                generate_data_types_from_math_instruction(
                    math_inst,
                    element_source=DataType.void,
                    element_dest=d_type,
                )
            )

    for layout in layouts:
        for data_type in data_types:
            # Inconsistency: alignments aren't fixed in FP8
            # layout = fix_alignments(data_type, layout, alignment_bits=128)

            schedules, stream_k_schedules = get_valid_schedules(
              tile_description=tile_desc,
              cuda_version=cuda_version,
              is_aligned=is_aligned,
              data_types=data_type,
              instantiation_level=instantiation_level,
              layout=layout,
              gemm_kind=gemm_kind,
              enable_fp8_fast_acc=False,
            )

            if len(schedules):
              CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type, schedules, gemm_kind=gemm_kind)
              if len(stream_k_schedules):
                assert CudaToolkitVersionSatisfies(cuda_version, 12, 1)
                CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type,
                                              stream_k_schedules,
                                              tile_schedulers=[TileSchedulerType.StreamK],
                                              gemm_kind=gemm_kind)



def GenerateSM90_TensorOp_fp8_WGMMA_alignx_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=0, default_level=101, exhaustive_level=9992)
  is_aligned = False

  # layouts for ABC and their alignments
  layouts = [
    [[LayoutType.RowMajor, 8], [LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 1]],  # TN Layout
    [[LayoutType.RowMajor, 4], [LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 1]],  # TN Layout
  ]

  math_instructions = generate_fp8_math_instructions_sm90(instantiation_level)
  tile_descriptions = generate_tile_descriptions_sm90(
      math_instructions=math_instructions,
      is_aligned=is_aligned,
      level=instantiation_level)

  for tile_desc in tile_descriptions:
    math_inst = tile_desc.math_instruction
    data_types = [generate_data_types_from_math_instruction(math_inst)]
    fp8_types = [DataType.e4m3, DataType.e5m2]
    valid_types_for_d = [DataType.f32, DataType.bf16, DataType.f16, DataType.e4m3, DataType.e5m2]
    valid_types_for_c = copy.deepcopy(valid_types_for_d)
    valid_types_for_c.append(DataType.void)
    for c_type, d_type in product(valid_types_for_c, valid_types_for_d):
        data_types.append(
            generate_data_types_from_math_instruction(
                math_inst,
                element_source=c_type,
                element_dest=d_type,
            )
        )

    for layout in layouts:
        for data_type in data_types:
            # Inconsistency: alignments aren't fixed in FP8
            # layout = fix_alignments(data_type, layout, alignment_bits=128)

            schedules, stream_k_schedules = get_valid_schedules(
              tile_description=tile_desc,
              cuda_version=cuda_version,
              is_aligned=is_aligned,
              data_types=data_type,
              instantiation_level=instantiation_level,
              layout=layout,
            )

            if len(schedules):
              CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type, schedules)
              if len(stream_k_schedules):
                assert CudaToolkitVersionSatisfies(cuda_version, 12, 1)
                CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type,
                                              stream_k_schedules,
                                              tile_schedulers=[TileSchedulerType.StreamK])

def GenerateSM90_TensorOp_mixed_dtype_WGMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 1):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=20, default_level=121, exhaustive_level=9999)
  is_aligned = True

  # layouts for ABC, their alignments will be fixed later based on the data type
  layouts = [
    [[LayoutType.RowMajor, 16], [LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 16]],
  ]

  valid_types_for_a_b_acc = [
    (DataType.e4m3, DataType.f16, DataType.f32),
    (DataType.e4m3, DataType.bf16, DataType.f32),
    (DataType.e5m2, DataType.f16, DataType.f32),
    (DataType.e5m2, DataType.bf16, DataType.f32),
    (DataType.s8, DataType.f16, DataType.f32),
    (DataType.s8, DataType.bf16, DataType.f32),
    (DataType.u8, DataType.f16, DataType.f32),
    (DataType.u8, DataType.bf16, DataType.f32),
    (DataType.s4, DataType.f16, DataType.f32),
    (DataType.s4, DataType.bf16, DataType.f32),
    (DataType.s4, DataType.e4m3, DataType.f32),
    (DataType.s4, DataType.e5m2, DataType.f32),
    (DataType.u4, DataType.f16, DataType.f32),
    (DataType.u4, DataType.bf16, DataType.f32),
    (DataType.u2, DataType.f16, DataType.f32),
    (DataType.u2, DataType.bf16, DataType.f32),
    (DataType.s2, DataType.f16, DataType.f32),
    (DataType.s2, DataType.bf16, DataType.f32),
  ]
  # Note: For sizeof(a_type) > sizeof(b_type), some generated kernels might crash due to a compiler bug. Disable it for now.
  #swapped_valid_types_for_a_b_acc = [(b_type, a_type, acc_type) for a_type, b_type, acc_type in valid_types_for_a_b_acc]
  #valid_types_for_a_b_acc = valid_types_for_a_b_acc + swapped_valid_types_for_a_b_acc

  math_instructions = generate_mixed_dtype_math_instructions_sm90(instantiation_level, valid_types_for_a_b_acc)

  valid_types_for_d = [DataType.f32, DataType.bf16, DataType.f16, DataType.e4m3, DataType.e5m2]
  valid_types_for_c = copy.deepcopy(valid_types_for_d)

  tile_descriptions = generate_tile_descriptions_sm90(
    math_instructions=math_instructions,
    is_aligned=is_aligned,
    level=instantiation_level)

  for tile_desc in tile_descriptions:
    math_inst = tile_desc.math_instruction
    data_types = []

    # Limit C/D types to avoid a giant number of instantiations.
    # A typical use case for mixed dtype in DL is weight quantization (tensor A),
    # therefore we can limit the output type to that of activation (tensor B).
    valid_types_for_c = [math_inst.element_b]
    valid_types_for_d = [math_inst.element_b]

    for c_type, d_type in product(valid_types_for_c, valid_types_for_d):
      data_types.append(
        generate_data_types_from_math_instruction(
          math_inst,
          element_source=c_type,
          element_dest=d_type,
        )
      )

    for layout in layouts:
      for data_type in data_types:
        # Fix alignments, DataTypeSize are in the unit of bits
        alignment_bits = 128
        layout[0][1] = alignment_bits // DataTypeSize[data_type['a_type']]
        layout[1][1] = alignment_bits // DataTypeSize[data_type['b_type']]
        layout[2][1] = alignment_bits // DataTypeSize[data_type['c_type']]

        schedules, stream_k_schedules = get_valid_schedules(
          tile_description=tile_desc,
          cuda_version=cuda_version,
          is_aligned=is_aligned,
          data_types=data_type,
          instantiation_level=instantiation_level,
          layout=layout,
        )

        if len(schedules):
          CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type, schedules)
          if len(stream_k_schedules):
            assert CudaToolkitVersionSatisfies(cuda_version, 12, 1)
            CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type,
                                          stream_k_schedules,
                                          tile_schedulers=[TileSchedulerType.StreamK])


def GenerateSM90_SparseTensorOp_fp8_WGMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 2):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=20, default_level=121, exhaustive_level=9992)
  is_aligned = True

  # layouts for ABC and their alignments
  layouts = [
    [[LayoutType.RowMajor, 32], [LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 1]],  # TN Layout
  ]

  math_instructions = make_sparse_math_instructions(generate_fp8_math_instructions_sm90(instantiation_level))
  tile_descriptions = generate_tile_descriptions_sm90(
      math_instructions=math_instructions,
      is_aligned=is_aligned,
      level=instantiation_level)

  for tile_desc in tile_descriptions:
    math_inst = tile_desc.math_instruction
    data_types = []
    fp8_types = [DataType.e4m3, DataType.e5m2]
    valid_types_for_d = [DataType.f32, DataType.bf16, DataType.f16, DataType.e4m3, DataType.e5m2]
    valid_types_for_c = copy.deepcopy(valid_types_for_d)
    valid_types_for_c.append(DataType.void)
    for c_type, d_type in product(valid_types_for_c, valid_types_for_d):
        data_types.append(
            generate_data_types_from_math_instruction(
                math_inst,
                element_source=c_type,
                element_dest=d_type,
            )
        )
    else:
        for d_type in valid_types_for_d:
            data_types.append(
                generate_data_types_from_math_instruction(
                    math_inst,
                    element_source=DataType.void,
                    element_dest=d_type,
                )
            )

    for layout in layouts:
        for data_type in data_types:
            # Inconsistency: alignments aren't fixed in FP8
            # layout = fix_alignments(data_type, layout, alignment_bits=128)

            schedules, stream_k_schedules = get_valid_schedules(
              tile_description=tile_desc,
              cuda_version=cuda_version,
              is_aligned=is_aligned,
              data_types=data_type,
              instantiation_level=instantiation_level,
              layout=layout,
            )

            if len(schedules):
              CreateSparseGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type, schedules)
              if len(stream_k_schedules):
                assert CudaToolkitVersionSatisfies(cuda_version, 12, 1)
                CreateSparseGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type,
                                                    stream_k_schedules,
                                                    tile_schedulers=[TileSchedulerType.StreamK])


def GenerateSM90_TensorOp_1684(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst = MathInstruction(
      [16, 8, 4],
      DataType.f64, DataType.f64, DataType.f64,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 64, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 256, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 32, 16], 3, [4, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 256, 16], 3, [1, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateGemmOperator(manifest, layouts, tile_descriptions,
    data_type, alignment_constraints)

#

#
def GenerateSM90_TensorOp_1684_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64,  8 ], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  128, 8 ], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  64,  8 ], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  32,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  64,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  32,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16,  32,  8 ], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  16,  8 ], 4, [2, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64,  16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  128, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  64,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  32,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  64,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  32,  16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16,  32,  16], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  16,  16], 3, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM90_TensorOp_1684_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM90_TensorOp_1684_rank_k(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64]

  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM90_TensorOp_1684_rank_k_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64]

  # SYRK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HERK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)

#

#
def GenerateSM90_TensorOp_1684_rank_k_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [ComplexTransform.none,]

  # SYRK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HERK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM90_TensorOp_1684_trmm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints)
#

#
def GenerateSM90_TensorOp_1684_trmm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    ComplexTransform.none, ComplexTransform.conj,
  ]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#


#
def GenerateSM90_TensorOp_1684_trmm_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    ComplexTransform.none, ComplexTransform.conj,
  ]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM90_TensorOp_1684_symm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM90_TensorOp_1684_symm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  # SYMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HEMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM90_TensorOp_1684_symm_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [ComplexTransform.none,]

  # SYMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HEMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#



# Blackwell SM 100 generators

try:
    import cutlass_library.sm100_utils
    from cutlass_library.sm100_utils import (
      generate_tf32_math_instructions_sm100,
      generate_16b_math_instructions_sm100,
      generate_f8f6f4_math_instructions_sm100,
      generate_mxf8f6f4_math_instructions_sm100,
      generate_mxf4nvf4_math_instructions_sm100,
      generate_fp8_math_instructions_sm100,
      generate_cluster_shapes_sm100,
      get_pruning_level_from_global_level
    )
except ImportError:
    import sm100_utils
    from sm100_utils import (
      generate_tf32_math_instructions_sm100,
      generate_16b_math_instructions_sm100,
      generate_f8f6f4_math_instructions_sm100,
      generate_mxf8f6f4_math_instructions_sm100,
      generate_mxf4nvf4_math_instructions_sm100,
      generate_fp8_math_instructions_sm100,
      generate_cluster_shapes_sm100,
      get_pruning_level_from_global_level
    )

###################################################################################################

def get_tma_alignment_elt(data_type : DataType, is_f8f6f4 : bool = True ) -> int:
  if DataTypeSize[data_type] < 8 and is_f8f6f4:
    return int(128)
  return int(16 * 8 / DataTypeSize[data_type])

sm100_cluster_shape_1sm = [
  [4,4,1]
  , DynamicClusterShape
]

sm100_cluster_shape_2sm = [
  # cluster_m % 2 == 0 for 2sm
  [4,4,1]
  , DynamicClusterShape
]

def GenerateSM100_TensorOp_32b_UMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 8):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=490, default_level=490, exhaustive_level=9999)

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 4]],
    [[LayoutType.ColumnMajor, 4], [LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 4]],
    [[LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 4]],
    [[LayoutType.RowMajor,    4], [LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 4]],
    [[LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 4], [LayoutType.RowMajor,    4]],
    [[LayoutType.ColumnMajor, 4], [LayoutType.RowMajor,    4], [LayoutType.RowMajor,    4]],
    [[LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 4], [LayoutType.RowMajor,    4]],
    [[LayoutType.RowMajor,    4], [LayoutType.RowMajor,    4], [LayoutType.RowMajor,    4]],
  ]

  data_types = [
    {
      "a_type"   : DataType.f32,
      "b_type"   : DataType.f32,
      "c_type"   : DataType.f32,
      "d_type"   : DataType.f32,
      "acc_type" : DataType.f32,
      "epi_type" : DataType.f32,
    },
    {
      "a_type"   : DataType.f32,
      "b_type"   : DataType.f32,
      "c_type"   : DataType.void,
      "d_type"   : DataType.f32,
      "acc_type" : DataType.f32,
      "epi_type" : DataType.f32,
    },
  ]

  thor_sm = ThorSMRenumbering(cuda_version)

  min_cc = 100
  max_cc = thor_sm

  math_instructions_1sm, math_instructions_2sm = generate_tf32_math_instructions_sm100(instantiation_level)

  cluster_shapes_1sm, cluster_shapes_2sm = generate_cluster_shapes_sm100(instantiation_level)

  if thor_sm in manifest.compute_capabilities_baseline :
    if [4,4,1] in cluster_shapes_1sm :
      cluster_shapes_1sm.remove([4,4,1])
    if [4,4,1] in cluster_shapes_2sm :
      cluster_shapes_2sm.remove([4,4,1])

  tile_schedulers = [
    TileSchedulerType.Default, TileSchedulerType.StreamK
  ]

  # 1xSM MMA kernels
  for math_inst in math_instructions_1sm:
    tile_descriptions = []
    for cluster_shape in cluster_shapes_1sm:
      multiplier_1sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else cluster_shape
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_1sm[0],
          math_inst.instruction_shape[1]     * multiplier_1sm[1],
          math_inst.instruction_shape[2] * 4 * multiplier_1sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_types,
      [[KernelScheduleType.TmaWarpSpecialized1SmSm100, EpilogueScheduleType.TmaWarpSpecialized1Sm]],
      tile_schedulers=tile_schedulers)

  # 2xSM MMA kernels
  for math_inst in math_instructions_2sm:
    tile_descriptions = []
    for cluster_shape in cluster_shapes_2sm:
      multiplier_2sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else (cluster_shape[0] // 2, cluster_shape[1], cluster_shape[2])
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_2sm[0],
          math_inst.instruction_shape[1]     * multiplier_2sm[1],
          math_inst.instruction_shape[2] * 4 * multiplier_2sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    if math_inst.instruction_shape[0] == 128:
      epi_schedule = EpilogueScheduleType.TmaWarpSpecialized2Sm
    else:
      epi_schedule = EpilogueScheduleType.ScheduleAuto

    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_types,
      [[KernelScheduleType.TmaWarpSpecialized2SmSm100, epi_schedule]], tile_schedulers=tile_schedulers)

def GenerateSM100_TensorOp_16b_UMMA_gemm(manifest, cuda_version, gemm_kind=GemmKind.Universal3x):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 8):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=490, default_level=490, exhaustive_level=9999)

  # layouts for ABC and their alignments. C alignment will be set later based on output type
  layouts = [
    [[LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 0]],
    [[LayoutType.ColumnMajor, 8], [LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 0]],
    [[LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 0]],
    [[LayoutType.RowMajor,    8], [LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 0]],
    [[LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 8], [LayoutType.RowMajor,    0]],
    [[LayoutType.ColumnMajor, 8], [LayoutType.RowMajor,    8], [LayoutType.RowMajor,    0]],
    [[LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 8], [LayoutType.RowMajor,    0]],
    [[LayoutType.RowMajor,    8], [LayoutType.RowMajor,    8], [LayoutType.RowMajor,    0]],
  ]

  thor_sm = ThorSMRenumbering(cuda_version)

  math_instructions_1sm, math_instructions_2sm = generate_16b_math_instructions_sm100(instantiation_level)
  
  min_cc = 100
  max_cc = thor_sm
  grouped = is_grouped(gemm_kind)

  cluster_shapes_1sm, cluster_shapes_2sm = generate_cluster_shapes_sm100(instantiation_level)

  if thor_sm in manifest.compute_capabilities_baseline :
    if [4,4,1] in cluster_shapes_1sm :
      cluster_shapes_1sm.remove([4,4,1])
    if [4,4,1] in cluster_shapes_2sm :
      cluster_shapes_2sm.remove([4,4,1])

  tile_schedulers = [
    TileSchedulerType.Default, TileSchedulerType.StreamK
  ]

  # 1xSM MMA kernels
  for math_inst in math_instructions_1sm:
    tile_descriptions = []
    for cluster_shape in cluster_shapes_1sm:
      multiplier_1sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else cluster_shape
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_1sm[0],
          math_inst.instruction_shape[1]     * multiplier_1sm[1],
          math_inst.instruction_shape[2] * 4 * multiplier_1sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : math_inst.element_accumulator,
        "d_type"   : math_inst.element_accumulator,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : math_inst.element_accumulator,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator,
      },
    ]
    # Set alignment d based on Destination format.
    for layout in layouts:
      layout[2][1] = 128 // DataTypeSize[data_types[0]["d_type"]]

    kernel_schedule = KernelScheduleType.TmaWarpSpecialized1SmSm100 if not grouped else KernelScheduleType.PtrArrayTmaWarpSpecialized1SmSm100
    epi_schedule = EpilogueScheduleType.TmaWarpSpecialized1Sm if not grouped else EpilogueScheduleType.PtrArrayTmaWarpSpecialized1Sm
    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_types,
      [[kernel_schedule, epi_schedule]],
      tile_schedulers=tile_schedulers, gemm_kind=gemm_kind)

    # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
      data_types_mixed = [
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : math_inst.element_a,
          "d_type"   : math_inst.element_a,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : math_inst.element_accumulator,
        },
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : DataType.void,
          "d_type"   : math_inst.element_a,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : math_inst.element_accumulator,
        },
      ]
      # Set alignment d based on Destination format.
      for layout in layouts:
        layout[2][1] = 128 // DataTypeSize[data_types_mixed[0]["d_type"]]

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_types_mixed,
        [[kernel_schedule, epi_schedule]],
        tile_schedulers=tile_schedulers, gemm_kind=gemm_kind)

  # 2xSM MMA kernels
  for math_inst in math_instructions_2sm:
    tile_descriptions = []
    for cluster_shape in cluster_shapes_2sm:
      multiplier_2sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else (cluster_shape[0] // 2, cluster_shape[1], cluster_shape[2])
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_2sm[0],
          math_inst.instruction_shape[1]     * multiplier_2sm[1],
          math_inst.instruction_shape[2] * 4 * multiplier_2sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : math_inst.element_accumulator,
        "d_type"   : math_inst.element_accumulator,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : math_inst.element_accumulator,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator,
      },
    ]
    # Set alignment d based on Destination format.
    for layout in layouts:
      layout[2][1] = 128 // DataTypeSize[data_types[0]["d_type"]]

    if grouped:
      epi_schedule = EpilogueScheduleType.PtrArrayTmaWarpSpecialized2Sm
    elif math_inst.instruction_shape[0] == 128:
      epi_schedule = EpilogueScheduleType.TmaWarpSpecialized2Sm
    else:
      epi_schedule = EpilogueScheduleType.ScheduleAuto
    kernel_schedule = to_grouped_schedule(KernelScheduleType.TmaWarpSpecialized2SmSm100, grouped)

    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_types,
      [[kernel_schedule, epi_schedule]], tile_schedulers=tile_schedulers, gemm_kind=gemm_kind)

    # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
      data_types_mixed = [
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : math_inst.element_a,
          "d_type"   : math_inst.element_a,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : math_inst.element_accumulator,
        },
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : DataType.void,
          "d_type"   : math_inst.element_a,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : math_inst.element_accumulator,
        },
      ]
      # Set alignment d based on Destination format.
      for layout in layouts:
        layout[2][1] = 128 // DataTypeSize[data_types_mixed[0]["d_type"]]

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_types_mixed,
        [[kernel_schedule, epi_schedule]], tile_schedulers=tile_schedulers, gemm_kind=gemm_kind)

def GenerateSM100_TensorOp_fp8_UMMA_gemm(manifest, cuda_version, gemm_kind=GemmKind.Universal3x):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 8):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=591 , default_level=591 , exhaustive_level=9999)

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 0]],
    [[LayoutType.ColumnMajor, 16], [LayoutType.RowMajor,    16], [LayoutType.ColumnMajor, 0]], 
    [[LayoutType.RowMajor,    16], [LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 0]],
    [[LayoutType.RowMajor,    16], [LayoutType.RowMajor,    16], [LayoutType.ColumnMajor, 0]],
    [[LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 16], [LayoutType.RowMajor,    0]],
    [[LayoutType.ColumnMajor, 16], [LayoutType.RowMajor,    16], [LayoutType.RowMajor,    0]],
    [[LayoutType.RowMajor,    16], [LayoutType.ColumnMajor, 16], [LayoutType.RowMajor,    0]],
    [[LayoutType.RowMajor,    16], [LayoutType.RowMajor,    16], [LayoutType.RowMajor,    0]],
  ]

  thor_sm = ThorSMRenumbering(cuda_version)

  min_cc = 100
  max_cc = thor_sm

  epi_type = DataType.f32
  grouped = is_grouped(gemm_kind)

  math_instructions_1sm, math_instructions_2sm = generate_fp8_math_instructions_sm100(instantiation_level, enable_runtime_dtype=not grouped)

  cluster_shapes_1sm, cluster_shapes_2sm = generate_cluster_shapes_sm100(instantiation_level)

  if thor_sm in manifest.compute_capabilities_baseline :
    if [4,4,1] in cluster_shapes_1sm :
      cluster_shapes_1sm.remove([4,4,1])
    if [4,4,1] in cluster_shapes_2sm :
      cluster_shapes_2sm.remove([4,4,1])

  tile_schedulers = [
    TileSchedulerType.Default, TileSchedulerType.StreamK
  ]

  # 1xSM MMA kernels
  for math_inst in math_instructions_1sm:
    tile_descriptions = []
    for cluster_shape in cluster_shapes_1sm:
      multiplier_1sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else cluster_shape
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_1sm[0],
          math_inst.instruction_shape[1]     * multiplier_1sm[1],
          math_inst.instruction_shape[2] * 4 * multiplier_1sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.f16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e4m3,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.bf16,
        "d_type"   : DataType.bf16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.bf16,
        "d_type"   : DataType.e4m3,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.bf16,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f32,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.bf16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e4m3,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      }
    ]

    # Set alignment d based on Destination format.
    for layout in layouts:
      layout[2][1] = 128 // DataTypeSize[data_types[0]["d_type"]]

    for data_type in data_types:
      if ( data_type["a_type"] == DataType.e4m3 ) and ( data_type["b_type"] == DataType.e4m3 ) and\
         ( data_type["d_type"] == DataType.e5m2 ):
        continue
      kernel_schedule = to_grouped_schedule(KernelScheduleType.TmaWarpSpecialized1SmSm100, grouped)
      epi_schedule = to_grouped_schedule(EpilogueScheduleType.TmaWarpSpecialized1Sm, grouped)
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
        [[kernel_schedule, epi_schedule]],
        tile_schedulers=tile_schedulers, gemm_kind=gemm_kind)

  # 2xSM MMA kernels

  for math_inst in math_instructions_2sm:
    tile_descriptions = []
    for cluster_shape in cluster_shapes_2sm:
      multiplier_2sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else (cluster_shape[0] // 2, cluster_shape[1], cluster_shape[2])
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_2sm[0],
          math_inst.instruction_shape[1]     * multiplier_2sm[1],
          math_inst.instruction_shape[2] * 4 * multiplier_2sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.f16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e4m3,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.bf16,
        "d_type"   : DataType.bf16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.bf16,
        "d_type"   : DataType.e4m3,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.bf16,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f32,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.bf16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e4m3,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      }
    ]

    # Set alignment d based on Destination format.
    for layout in layouts:
      layout[2][1] = 128 // DataTypeSize[data_types[0]["d_type"]]

    for data_type in data_types:
      if ( data_type["a_type"] == DataType.e4m3 ) and ( data_type["b_type"] == DataType.e4m3 ) and\
         ( data_type["d_type"] == DataType.e5m2 ):
        continue

      if grouped:
        epi_schedule = EpilogueScheduleType.PtrArrayTmaWarpSpecialized2Sm
      elif math_inst.instruction_shape[0] == 128:
        epi_schedule = EpilogueScheduleType.TmaWarpSpecialized2Sm
      else:
        epi_schedule = EpilogueScheduleType.ScheduleAuto
      kernel_schedule = to_grouped_schedule(KernelScheduleType.TmaWarpSpecialized2SmSm100, grouped)

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
      [[kernel_schedule, epi_schedule]], tile_schedulers=tile_schedulers, gemm_kind=gemm_kind)

def GenerateSM100_TensorOp_fp8_UMMA_gemm_with_blockwise(manifest, cuda_version, gemm_kind=GemmKind.BlockwiseUniversal3x):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 8):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=593, default_level=593, exhaustive_level=9999)

  grouped = is_grouped(gemm_kind)

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 0]],
    [[LayoutType.ColumnMajor, 16], [LayoutType.RowMajor,    16], [LayoutType.ColumnMajor, 0]], 
    [[LayoutType.RowMajor,    16], [LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 0]],
    [[LayoutType.RowMajor,    16], [LayoutType.RowMajor,    16], [LayoutType.ColumnMajor, 0]],
    [[LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 16], [LayoutType.RowMajor,    0]],
    [[LayoutType.ColumnMajor, 16], [LayoutType.RowMajor,    16], [LayoutType.RowMajor,    0]],
    [[LayoutType.RowMajor,    16], [LayoutType.ColumnMajor, 16], [LayoutType.RowMajor,    0]],
    [[LayoutType.RowMajor,    16], [LayoutType.RowMajor,    16], [LayoutType.RowMajor,    0]],
  ]

  min_cc = 100
  max_cc = 100
  epi_type = DataType.f32

  pruning_level = get_pruning_level_from_global_level(instantiation_level)

  math_instructions_1sm, math_instructions_2sm = generate_fp8_math_instructions_sm100(instantiation_level, enable_compile_time_dtype=grouped or pruning_level >= 1, enable_runtime_dtype=not grouped)

  cluster_shapes_1sm, cluster_shapes_2sm = generate_cluster_shapes_sm100(instantiation_level)

  tile_schedulers = [
    TileSchedulerType.Default,
  ]

  # 1xSM MMA kernels
  for math_inst in math_instructions_1sm:
    tile_descriptions = []
    for cluster_shape in cluster_shapes_1sm:
      multiplier_1sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else cluster_shape
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_1sm[0],
          math_inst.instruction_shape[1]     * multiplier_1sm[1],
          math_inst.instruction_shape[2] * 4 * multiplier_1sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape,
          [math_inst.instruction_shape[0], math_inst.instruction_shape[1], 
           math_inst.instruction_shape[2] * 4]))
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_1sm[0],
          math_inst.instruction_shape[1]     * multiplier_1sm[1],
          math_inst.instruction_shape[2] * 4 * multiplier_1sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape,
          [1, math_inst.instruction_shape[1], 
           math_inst.instruction_shape[2] * 4]))
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_1sm[0],
          math_inst.instruction_shape[1]     * multiplier_1sm[1],
          math_inst.instruction_shape[2] * 4 * multiplier_1sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape,
          [math_inst.instruction_shape[0], 1, 
           math_inst.instruction_shape[2] * 4]))

    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.f16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.bf16,
        "d_type"   : DataType.bf16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f32,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.bf16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
    ]

    # Set alignment d based on Destination format.
    for layout in layouts:
      layout[2][1] = 128 // DataTypeSize[data_types[0]["d_type"]]

    is_runtime_datatype = lambda runtime_datatype: runtime_datatype in (DataType.f4, DataType.f6, DataType.f8)
    for data_type in data_types:
      if ( data_type["a_type"] == DataType.e4m3 ) and ( data_type["b_type"] == DataType.e4m3 ) and\
         ( data_type["d_type"] == DataType.e5m2 ):
        continue

      is_runtime_datatype_a = is_runtime_datatype(data_type["a_type"])
      is_runtime_datatype_b = is_runtime_datatype(data_type["d_type"])

      # A/B datatypes should be both static or dynamic
      if (is_runtime_datatype_a != is_runtime_datatype_b):
        continue

      kernel_schedule = to_grouped_schedule(KernelScheduleType.BlockwiseTmaWarpSpecialized1SmSm100, grouped)
      epi_schedule = to_grouped_schedule(EpilogueScheduleType.TmaWarpSpecialized1Sm, grouped)
      epi_schedule_nosmem = to_grouped_schedule(EpilogueScheduleType.BlockwiseNoSmemWarpSpecialized1Sm, grouped)
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
        [[kernel_schedule, epi_schedule], [kernel_schedule, epi_schedule_nosmem]],
        tile_schedulers=tile_schedulers, gemm_kind=gemm_kind)

def GenerateSM100_TensorOp_mixed_8bits_UMMA_gemm(manifest, cuda_version, gemm_kind=GemmKind.Universal3x):

  # SM100 MMA with mixed F4/F6/F8 inputs + without block scale
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=590, default_level=590, exhaustive_level=9999)

  grouped = is_grouped(gemm_kind)

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.RowMajor,    -1], [LayoutType.ColumnMajor, -1], [LayoutType.RowMajor, -1]],
  ]

  math_instructions_1sm, math_instructions_2sm = generate_f8f6f4_math_instructions_sm100(instantiation_level, enable_runtime_dtype=not grouped)

  def change_priority_func(shapes_1sm, shapes_2sm):
    shapes_1sm[(1,2,1)] = 6
    shapes_1sm[(1,4,1)] = 6
    shapes_2sm[(2,2,1)] = 6
    shapes_2sm[(2,4,1)] = 6
    shapes_2sm[(4,2,1)] = 6

  cluster_shapes_1sm, cluster_shapes_2sm = generate_cluster_shapes_sm100(instantiation_level, change_priority_func)

  tile_schedulers = [
    TileSchedulerType.Default, TileSchedulerType.StreamK
  ]

  thor_sm = ThorSMRenumbering(cuda_version)

  min_cc = 100
  max_cc = thor_sm

  epi_type = DataType.f32

  is_runtime_datatype = lambda runtime_datatype: runtime_datatype in (DataType.f4, DataType.f6, DataType.f8)

  if thor_sm in manifest.compute_capabilities_baseline :
    if [4,4,1] in cluster_shapes_1sm :
      cluster_shapes_1sm.remove([4,4,1])
    if [4,4,1] in cluster_shapes_2sm :
      cluster_shapes_2sm.remove([4,4,1])

  # 1xSM MMA kernels
  for math_inst in math_instructions_1sm:
    tile_descriptions = []
    for cluster_shape in cluster_shapes_1sm:
      multiplier_1sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else cluster_shape
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_1sm[0],
          math_inst.instruction_shape[1]     * multiplier_1sm[1],
          math_inst.instruction_shape[2] * 4 * multiplier_1sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    kernel_data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f32,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      }
      ]

    for kernel_data_type in kernel_data_types:
      # Filter out some kernel
      if ( kernel_data_type["a_type"] == DataType.e4m3 ) and ( kernel_data_type["b_type"] == DataType.e4m3 ) and\
         ( kernel_data_type["d_type"] == DataType.e5m2 ):
        continue

      # Update layout alignment
      # alignment for d might be different for each kernel_data_type
      layouts_copy = copy.deepcopy(layouts)
      for layout in layouts_copy:
        # alignment for a
        layout[0][1] = get_tma_alignment_elt(kernel_data_type["a_type"])
        # alignment for b
        layout[1][1] = get_tma_alignment_elt(kernel_data_type["b_type"])
        # alignment for d
        layout[2][1] = get_tma_alignment_elt(kernel_data_type["d_type"])

      CreateGemmUniversal3xOperator(manifest, layouts_copy, tile_descriptions, [kernel_data_type],
        [[KernelScheduleType.TmaWarpSpecialized1SmSm100, EpilogueScheduleType.TmaWarpSpecialized1Sm]], tile_schedulers=tile_schedulers)

  for math_inst in math_instructions_2sm:
    tile_descriptions = []
    for cluster_shape in cluster_shapes_2sm:
      multiplier_2sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else (cluster_shape[0] // 2, cluster_shape[1], cluster_shape[2])
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_2sm[0],
          math_inst.instruction_shape[1]     * multiplier_2sm[1],
          math_inst.instruction_shape[2] * 4 * multiplier_2sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    kernel_data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f32,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
      }
      ]

    for kernel_data_type in kernel_data_types:
      # Filter some kernel
      if ( kernel_data_type["a_type"] == DataType.e4m3 ) and ( kernel_data_type["b_type"] == DataType.e4m3 ) and\
         ( kernel_data_type["d_type"] == DataType.e5m2 ):
        continue

      # Update layout alignment
      # alignment for d might be different for each kernel_data_type
      layouts_copy = copy.deepcopy(layouts)
      for layout in layouts_copy:
        # alignment for a
        layout[0][1] = get_tma_alignment_elt(kernel_data_type["a_type"])
        # alignment for b
        layout[1][1] = get_tma_alignment_elt(kernel_data_type["b_type"])
        # alignment for d
        layout[2][1] = get_tma_alignment_elt(kernel_data_type["d_type"])

      if math_inst.instruction_shape[0] == 128:
        CreateGemmUniversal3xOperator(manifest, layouts_copy, tile_descriptions, [kernel_data_type],
          [[KernelScheduleType.TmaWarpSpecialized2SmSm100, EpilogueScheduleType.TmaWarpSpecialized2Sm]], tile_schedulers=tile_schedulers)
      else:
        CreateGemmUniversal3xOperator(manifest, layouts_copy, tile_descriptions, [kernel_data_type],
          [[KernelScheduleType.TmaWarpSpecialized2SmSm100, EpilogueScheduleType.ScheduleAuto]], tile_schedulers=tile_schedulers)

def GenerateSM100_TensorOp_mixed_8bits_UMMA_gemm_with_block_scaled(manifest, cuda_version, gemm_kind=GemmKind.BlockScaledUniversal3x):

  # SM100 MMA with mixed F4/F6/F8 inputs + block scale
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 8):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=590, default_level=590, exhaustive_level=9999)

  grouped = is_grouped(gemm_kind)

  layouts = [
    [[LayoutType.RowMajor,    128], [LayoutType.ColumnMajor, 128], [LayoutType.RowMajor,    0]],
    [[LayoutType.RowMajor,    128], [LayoutType.ColumnMajor, 128], [LayoutType.ColumnMajor, 0]],
    [[LayoutType.ColumnMajor, 128], [LayoutType.RowMajor,    128], [LayoutType.RowMajor,    0]],
  ]

  math_instructions_1sm, math_instructions_2sm = generate_mxf8f6f4_math_instructions_sm100(instantiation_level, enable_runtime_dtype=not grouped)

  def change_priority_func(shapes_1sm, shapes_2sm):
    shapes_1sm[(1,2,1)] = 6
    shapes_1sm[(1,4,1)] = 6
    shapes_2sm[(2,2,1)] = 6
    shapes_2sm[(2,4,1)] = 6
    shapes_2sm[(4,2,1)] = 6

  cluster_shapes_1sm, cluster_shapes_2sm = generate_cluster_shapes_sm100(instantiation_level, change_priority_func)

  ab_types  = [
    DataType.f4, DataType.f6,
    DataType.e2m1, 
    DataType.e2m3, 
    DataType.e3m2,
    DataType.e5m2,
    DataType.e4m3,
  ]

  acc_types = [ DataType.f32 ]

  def tile_schedulers(sfdtype):
    # Only use the stream-K scheduler for non-void SFD to limit kernel count. When SFD is void,
    # the epilogue is the traditional linear combination, for which we already have tests with stream-K.
    if sfdtype["type"] == DataType.void or grouped:
      return [TileSchedulerType.Default]
    else:
      return [TileSchedulerType.Default, TileSchedulerType.StreamK]

  thor_sm = ThorSMRenumbering(cuda_version)

  min_cc = 100
  max_cc = thor_sm

  epi_type = DataType.f32

  is_runtime_datatype = lambda runtime_datatype: runtime_datatype in (DataType.f4, DataType.f6, DataType.f8)

  if thor_sm in manifest.compute_capabilities_baseline :
    if [4,4,1] in cluster_shapes_1sm :
      cluster_shapes_1sm.remove([4,4,1])
    if [4,4,1] in cluster_shapes_2sm :
      cluster_shapes_2sm.remove([4,4,1])

  # 1xSM MMA kernels
  for math_inst in math_instructions_1sm:
    assert math_inst.opcode_class == OpcodeClass.BlockScaledTensorOp
    tile_descriptions = []
    for cluster_shape in cluster_shapes_1sm:
      multiplier_1sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else cluster_shape
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_1sm[0],
          math_inst.instruction_shape[1]     * multiplier_1sm[1],
          math_inst.instruction_shape[2] * 4 * multiplier_1sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.bf16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e3m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      }]

    # Set alignment d based on Destination format.
    for layout in layouts:
      layout[2][1] = 128 // DataTypeSize[data_types[0]["d_type"]]

    for data_type in data_types:
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
        [[to_grouped_schedule(KernelScheduleType.Mxf8f6f4TmaWarpSpecialized1SmSm100, grouped), to_grouped_schedule(EpilogueScheduleType.TmaWarpSpecialized1Sm, grouped)]]
        , tile_schedulers = tile_schedulers(data_type["sfd_type"]), gemm_kind=gemm_kind)

  for math_inst in math_instructions_2sm:
    assert math_inst.opcode_class == OpcodeClass.BlockScaledTensorOp
    tile_descriptions = []
    for cluster_shape in cluster_shapes_2sm:
      multiplier_2sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else (cluster_shape[0] // 2, cluster_shape[1], cluster_shape[2])
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_2sm[0],
          math_inst.instruction_shape[1]     * multiplier_2sm[1],
          math_inst.instruction_shape[2] * 4 * multiplier_2sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.bf16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 32, "layout" : LayoutType.RowMajor}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e3m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 32, "layout" : LayoutType.RowMajor}
      },
    ]

    # Set alignment d based on Destination format.
    for data_type in data_types:
      for layout in layouts:
        # alignment for a
        layout[0][1] = get_tma_alignment_elt(data_type["a_type"])
        # alignment for b
        layout[1][1] = get_tma_alignment_elt(data_type["b_type"])
        # alignment for d
        layout[2][1] = get_tma_alignment_elt(data_type["d_type"])
        for tile in tile_descriptions:
          math_inst = tile.math_instruction
          # Filter some kernels that does not meet the alignment requirements.
          if layout[0][0] == LayoutType.ColumnMajor:
            if math_inst.instruction_shape[0] // 2 % layout[0][1] != 0:
              continue
          else:
            if tile.threadblock_shape[2] // tile.cluster_shape[2] % layout[0][1] != 0:
              continue
  
          if layout[1][0] == LayoutType.RowMajor:
            if math_inst.instruction_shape[1] // 2 % layout[1][1] != 0:
              continue
          else:
            if tile.threadblock_shape[2] // tile.cluster_shape[2] % layout[1][1] != 0:
              continue
          
          if grouped:
            CreateGemmUniversal3xOperator(manifest, [layout], [tile], [data_type],
              [[to_grouped_schedule(KernelScheduleType.Mxf8f6f4TmaWarpSpecialized2SmSm100, grouped), to_grouped_schedule(EpilogueScheduleType.TmaWarpSpecialized2Sm, grouped)]]
              , tile_schedulers = tile_schedulers(data_type["sfd_type"]), gemm_kind=gemm_kind)
          elif math_inst.instruction_shape[0] == 128:
            CreateGemmUniversal3xOperator(manifest, [layout], [tile], [data_type],
              [[KernelScheduleType.Mxf8f6f4TmaWarpSpecialized2SmSm100, EpilogueScheduleType.TmaWarpSpecialized2Sm]]
              , tile_schedulers = tile_schedulers(data_type["sfd_type"]), gemm_kind=gemm_kind)
          else:
            CreateGemmUniversal3xOperator(manifest, [layout], [tile], [data_type],
              [[KernelScheduleType.Mxf8f6f4TmaWarpSpecialized2SmSm100, EpilogueScheduleType.ScheduleAuto]]
              , tile_schedulers = tile_schedulers(data_type["sfd_type"]), gemm_kind=gemm_kind)



def GenerateSM100_TensorOp_fp4_UMMA_gemm_with_block_scaled(manifest, cuda_version, gemm_kind=GemmKind.BlockScaledUniversal3x):
  # SM100 MMA with F4 + block scale
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 8):
    return

  instantiation_level = manifest.get_instantiation_level(pruned_level=591, default_level=591, exhaustive_level=9999)

  grouped = is_grouped(gemm_kind)

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.RowMajor,    32], [LayoutType.ColumnMajor, 32], [LayoutType.RowMajor,    0]],
    [[LayoutType.RowMajor,    32], [LayoutType.ColumnMajor, 32], [LayoutType.ColumnMajor, 0]],
  ]

  math_instructions_1sm, math_instructions_2sm = generate_mxf4nvf4_math_instructions_sm100(instantiation_level, enable_runtime_dtype=not grouped)

  def change_priority_func(shapes_1sm, shapes_2sm):
    shapes_1sm[(1,2,1)] = 6
    shapes_1sm[(1,4,1)] = 6
    shapes_2sm[(2,2,1)] = 6
    shapes_2sm[(2,4,1)] = 6
    shapes_2sm[(4,2,1)] = 6

  cluster_shapes_1sm, cluster_shapes_2sm = generate_cluster_shapes_sm100(instantiation_level, change_priority_func=change_priority_func)

  acc_types = [ DataType.f32 ] # Accumulator is always 32 bits for block scaled MMA instructions

  def tile_schedulers(sfdtype):
    # Only use the stream-K scheduler for non-void SFD to limit kernel count. When SFD is void,
    # the epilogue is the traditional linear combination, for which we already have tests with stream-K.
    if sfdtype["type"] == DataType.void or grouped:
      return [TileSchedulerType.Default]
    else:
      return [TileSchedulerType.Default, TileSchedulerType.StreamK]

  thor_sm = ThorSMRenumbering(cuda_version)

  min_cc = 100
  max_cc = thor_sm

  epi_type = DataType.f32

  is_runtime_datatype = lambda runtime_datatype: runtime_datatype in (DataType.f4, DataType.f6, DataType.f8)

  if thor_sm in manifest.compute_capabilities_baseline :
    if [4,4,1] in cluster_shapes_1sm :
      cluster_shapes_1sm.remove([4,4,1])
    if [4,4,1] in cluster_shapes_2sm :
      cluster_shapes_2sm.remove([4,4,1])

  # 1xSM MMA kernels
  for math_inst in math_instructions_1sm:
    assert math_inst.opcode_class == OpcodeClass.BlockScaledTensorOp
    tile_descriptions = []
    for cluster_shape in cluster_shapes_1sm:
      multiplier_1sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else cluster_shape
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_1sm[0],
          math_inst.instruction_shape[1]     * multiplier_1sm[1],
          math_inst.instruction_shape[2] * 4 * multiplier_1sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))
      assert math_inst.instruction_shape[2] * 4 == 256

    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.bf16,
        "d_type"   : DataType.bf16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e2m1,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 32, "layout" : LayoutType.RowMajor}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e2m1,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 16, "layout" : LayoutType.RowMajor}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e2m1,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 16, "layout" : LayoutType.RowMajor}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e2m1,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 32, "layout" : LayoutType.RowMajor}
      }
    ]

    # Set alignment d based on Destination format.
    for layout in layouts:
      layout[2][1] = 128 // DataTypeSize[data_types[0]["d_type"]]

    for layout in layouts:
      for data_type in data_types:
        if (data_type["sfd_type"]["type"] != DataType.void) and (data_type["d_type"] == DataType.e2m1) and (layout[2][0] == LayoutType.RowMajor):
          data_type["sfd_type"]["layout"] = layout[2][0] # For FP4 output , the scalefactor layout is same layout as D layout.
        if (data_type["sfd_type"]["type"] != DataType.void) and (data_type["d_type"] == DataType.e2m1) and (layout[2][0] == LayoutType.ColumnMajor):
            continue

        # E2M1 x E2M1, vector size 32, E8
        # E2M1 x E2M1, vector size 16, UE4M3
        isFp4 = math_inst.element_scale_factor == DataType.ue8m0 and  math_inst.element_a == DataType.e2m1 and math_inst.element_b == DataType.e2m1
        epi_schedule = to_grouped_schedule(EpilogueScheduleType.TmaWarpSpecialized1Sm, grouped)
        epi_nosmem_schedule = to_grouped_schedule(EpilogueScheduleType.NoSmemWarpSpecialized1Sm, grouped)
        nvfp4_kernel_schedule = to_grouped_schedule(KernelScheduleType.Nvf4TmaWarpSpecialized1SmSm100, grouped)
        fp4_kernel_schedule = to_grouped_schedule(KernelScheduleType.Mxf4TmaWarpSpecialized1SmSm100, grouped)

        nvfp4_schedules = [[nvfp4_kernel_schedule, epi_schedule], [nvfp4_kernel_schedule, epi_nosmem_schedule]]
        fp4_schedules   = [[fp4_kernel_schedule, epi_schedule], [fp4_kernel_schedule, epi_nosmem_schedule]]
        CreateGemmUniversal3xOperator(manifest, [layout], tile_descriptions, data_type, nvfp4_schedules
          , tile_schedulers=tile_schedulers(data_type["sfd_type"]), gemm_kind=gemm_kind
          )
        if isFp4:
          CreateGemmUniversal3xOperator(manifest, [layout], tile_descriptions, data_type, fp4_schedules
          , tile_schedulers=tile_schedulers(data_type["sfd_type"]), gemm_kind=gemm_kind
          )

  for math_inst in math_instructions_2sm:
    assert math_inst.opcode_class == OpcodeClass.BlockScaledTensorOp
    tile_descriptions = []
    for cluster_shape in cluster_shapes_2sm:
      multiplier_2sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else (cluster_shape[0] // 2, cluster_shape[1], cluster_shape[2])
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_2sm[0],
          math_inst.instruction_shape[1]     * multiplier_2sm[1],
          math_inst.instruction_shape[2] * 4 * multiplier_2sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.bf16,
        "d_type"   : DataType.bf16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e2m1,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 32, "layout" : LayoutType.RowMajor}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e2m1,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 16, "layout" : LayoutType.RowMajor}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e2m1,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 16, "layout" : LayoutType.RowMajor}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e2m1,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 32, "layout" : LayoutType.RowMajor}
      }
    ]

    # Set alignment d based on Destination format.
    for layout in layouts:
      layout[2][1] = 128 // DataTypeSize[data_types[0]["d_type"]]

    for layout in layouts:
      for data_type in data_types:
        if (data_type["sfd_type"]["type"] != DataType.void) and (data_type["d_type"] == DataType.e2m1) and (layout[2][0] == LayoutType.RowMajor):
          data_type["sfd_type"]["layout"] = layout[2][0] # For FP4 output , the scalefactor layout is same layout as D layout.
        if (data_type["sfd_type"]["type"] != DataType.void) and (data_type["d_type"] == DataType.e2m1) and (layout[2][0] == LayoutType.ColumnMajor):
            continue

        # E2M1 x E2M1, vector size 32, E8
        isFp4 = math_inst.element_scale_factor == DataType.ue8m0 and  math_inst.element_a == DataType.e2m1 and math_inst.element_b == DataType.e2m1

        epi_schedule = EpilogueScheduleType.ScheduleAuto if not grouped else EpilogueScheduleType.PtrArrayTmaWarpSpecialized2Sm
        epi_nosmem_schedule = to_grouped_schedule(EpilogueScheduleType.NoSmemWarpSpecialized2Sm, grouped)
        nvfp4_kernel_schedule = to_grouped_schedule(KernelScheduleType.Nvf4TmaWarpSpecialized2SmSm100, grouped)
        fp4_kernel_schedule = to_grouped_schedule(KernelScheduleType.Mxf4TmaWarpSpecialized2SmSm100, grouped)

        nvfp4_schedules = [[nvfp4_kernel_schedule, epi_schedule], [nvfp4_kernel_schedule, epi_nosmem_schedule]]
        fp4_schedules   = [[fp4_kernel_schedule, epi_schedule], [fp4_kernel_schedule, epi_nosmem_schedule]]
        CreateGemmUniversal3xOperator(manifest, [layout], tile_descriptions, data_type, nvfp4_schedules
          , tile_schedulers=tile_schedulers(data_type["sfd_type"]), gemm_kind=gemm_kind)
        if isFp4:
          CreateGemmUniversal3xOperator(manifest, [layout], tile_descriptions, data_type, fp4_schedules
          , tile_schedulers=tile_schedulers(data_type["sfd_type"]), gemm_kind=gemm_kind)

def GenerateSM103_TensorOp_fp4_ultra_UMMA_gemm_with_block_scaled(manifest, cuda_version, gemm_kind=GemmKind.BlockScaledUniversal3x):
  # SM100 MMA with F4 + block scale
  if not CudaToolkitVersionSatisfies(cuda_version, 13, 0):
    return

  grouped = is_grouped(gemm_kind)

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.RowMajor,    32], [LayoutType.ColumnMajor, 32], [LayoutType.RowMajor,    0]],
    [[LayoutType.RowMajor,    32], [LayoutType.ColumnMajor, 32], [LayoutType.ColumnMajor, 0]],
  ]

  instruction_sizes_1sm = [
    [128, 128, 96], 
  ]

  instruction_sizes_2sm = [
    [256, 128, 96], 
    [256, 192, 96],
    [256, 256, 96]
  ]

  ab_types  = [
    DataType.f4,
    DataType.e2m1, 
  ]

  sf_types  = [
    DataType.ue4m3,
    DataType.ue8m0
  ]

  acc_types = [ DataType.f32 ] # Accumulator is always 32 bits for block scaled MMA instructions

  def tile_schedulers(sfdtype):
    # Only use the stream-K scheduler for non-void SFD to limit kernel count. When SFD is void,
    # the epilogue is the traditional linear combination, for which we already have tests with stream-K.
    if grouped:
      return [TileSchedulerType.Default]
    if sfdtype["type"] == DataType.void:
      return [TileSchedulerType.Default]
    else:
      return [TileSchedulerType.Default, TileSchedulerType.StreamK]

  min_cc = 103
  max_cc = 103
  epi_type = DataType.f32

  math_instructions_1sm = []

  is_runtime_datatype = lambda runtime_datatype: runtime_datatype in (DataType.f4, DataType.f6, DataType.f8)

  for instr_size, a_type, b_type, sf_type, acc_type in product(instruction_sizes_1sm, ab_types, ab_types, sf_types, acc_types):
    is_runtime_datatype_a = is_runtime_datatype(a_type)
    is_runtime_datatype_b = is_runtime_datatype(b_type)

    # A/B datatypes should be both static or dynamic
    if (is_runtime_datatype_a != is_runtime_datatype_b):
      continue

    math_instructions_1sm.append(
      MathInstruction(
        instr_size,
        a_type, b_type, acc_type,
        OpcodeClass.BlockScaledTensorOp,
        MathOperation.multiply_add,
        sf_type)
    )

  math_instructions_2sm = []

  for instr_size, a_type, b_type, sf_type, acc_type in product(instruction_sizes_2sm, ab_types, ab_types, sf_types, acc_types):
    is_runtime_datatype_a = is_runtime_datatype(a_type)
    is_runtime_datatype_b = is_runtime_datatype(b_type)

    # A/B datatypes should be both static or dynamic
    if (is_runtime_datatype_a != is_runtime_datatype_b):
      continue

    math_instructions_2sm.append(
      MathInstruction(
        instr_size,
        a_type, b_type, acc_type,
        OpcodeClass.BlockScaledTensorOp,
        MathOperation.multiply_add,
        sf_type)
    )

  cluster_shapes_1sm = [
    [1,1,1],
    # [1,2,1],
    [2,1,1],
    # [1,4,1],
    [4,4,1],
    DynamicClusterShape
  ]

  # 1xSM MMA kernels
  for math_inst in math_instructions_1sm:
    tile_descriptions = []
    for cluster_shape in cluster_shapes_1sm:
      multiplier_1sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else cluster_shape
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_1sm[0],
          math_inst.instruction_shape[1]     * multiplier_1sm[1],
          768],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.bf16,
        "d_type"   : DataType.bf16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e2m1,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 32, "layout" : LayoutType.RowMajor}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e2m1,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 16, "layout" : LayoutType.RowMajor}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e2m1,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 16, "layout" : LayoutType.RowMajor}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e2m1,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 32, "layout" : LayoutType.RowMajor}
      }
    ]

    # Set alignment d based on Destination format.
    for layout in layouts:
      for data_type in data_types:
        # Set alignment d based on Destination format.
        if DataTypeSize[data_type["c_type"]] == 0 :
          layout[2][1] = 256 // DataTypeSize[data_type["d_type"]]
        else:
          layout[2][1] = min(256 // DataTypeSize[data_type["d_type"]], 256 // DataTypeSize[data_type["c_type"]])
        
        if data_type["sfd_type"]["type"] != DataType.void and (data_type["d_type"] == DataType.e2m1) and (layout[2][0] == LayoutType.RowMajor):
          data_type["sfd_type"]["layout"] = layout[2][0] # For FP4 output , the scalefactor layout is same layout as D layout.
        if (data_type["sfd_type"]["type"] != DataType.void) and (data_type["d_type"] == DataType.e2m1) and (layout[2][0] == LayoutType.ColumnMajor):
            continue
        #   E2M1 x E2M1, vector size 32, E8
        isFp4 = math_inst.element_scale_factor == DataType.ue8m0 and  math_inst.element_a == DataType.e2m1 and math_inst.element_b == DataType.e2m1

        epilogue_1sm_schedule = to_grouped_schedule(EpilogueScheduleType.NoSmemWarpSpecialized1Sm, grouped)

        nvfp4_schedule                  = [to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized1SmVs16Sm103, grouped), epilogue_1sm_schedule]              
        nvfp4_schedule_disable_prefetch = [to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized1SmVs16Sm103DisablePrefetch, grouped), epilogue_1sm_schedule]                
        nvfp4_schedule_tma_prefetch     = [to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized1SmVs16Sm103TmaPrefetch, grouped), epilogue_1sm_schedule]
        fp4_schedule                    = [to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized1SmVs32Sm103, grouped), epilogue_1sm_schedule]
        fp4_schedule_disable_prefetch   = [to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized1SmVs32Sm103DisablePrefetch, grouped), epilogue_1sm_schedule]
        fp4_schedule_tma_prefetch       = [to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized1SmVs32Sm103TmaPrefetch, grouped), epilogue_1sm_schedule]
        nvfp4_schedules = [nvfp4_schedule, nvfp4_schedule_disable_prefetch, nvfp4_schedule_tma_prefetch]
        fp4_schedules   = [fp4_schedule, fp4_schedule_disable_prefetch, fp4_schedule_tma_prefetch]

        CreateGemmUniversal3xOperator(manifest, [layout], tile_descriptions, data_type, 
                                      nvfp4_schedules, tile_schedulers=tile_schedulers(data_type["sfd_type"]), gemm_kind=gemm_kind)
        if isFp4:
          CreateGemmUniversal3xOperator(manifest, [layout], tile_descriptions, data_type,
                                        fp4_schedules, tile_schedulers=tile_schedulers(data_type["sfd_type"]), gemm_kind=gemm_kind)

  cluster_shapes_2sm = [
    [2,1,1],
    # [2,2,1],
    # [2,4,1],
    [4,1,1],
    # [4,2,1],
    [4,4,1],
    DynamicClusterShape
  ]

  for math_inst in math_instructions_2sm:
    tile_descriptions = []
    for cluster_shape in cluster_shapes_2sm:
      multiplier_2sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else (cluster_shape[0] // 2, cluster_shape[1], cluster_shape[2])
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_2sm[0],
          math_inst.instruction_shape[1]     * multiplier_2sm[1],
          math_inst.instruction_shape[2] * 8 * multiplier_2sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.bf16,
        "d_type"   : DataType.bf16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e2m1,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 32, "layout" : LayoutType.RowMajor}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e2m1,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 16, "layout" : LayoutType.RowMajor}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e2m1,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 16, "layout" : LayoutType.RowMajor}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e2m1,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 32, "layout" : LayoutType.RowMajor}
      }
    ]

    # Set alignment d based on Destination format.
    for layout in layouts:
      for data_type in data_types:
        # Set alignment d based on Destination format.
        if DataTypeSize[data_type["c_type"]] == 0 :
          layout[2][1] = 256 // DataTypeSize[data_type["d_type"]]
        else:
          layout[2][1] = min(256 // DataTypeSize[data_type["d_type"]], 256 // DataTypeSize[data_type["c_type"]])
        
        if data_type["sfd_type"]["type"] != DataType.void and (data_type["d_type"] == DataType.e2m1) and (layout[2][0] == LayoutType.RowMajor):
          data_type["sfd_type"]["layout"] = layout[2][0] # For FP4 output , the scalefactor layout is same layout as D layout.
        if (data_type["sfd_type"]["type"] != DataType.void) and (data_type["d_type"] == DataType.e2m1) and (layout[2][0] == LayoutType.ColumnMajor):
            continue
        #   E2M1 x E2M1, vector size 32, E8
        isFp4 = math_inst.element_scale_factor == DataType.ue8m0 and  math_inst.element_a == DataType.e2m1 and math_inst.element_b == DataType.e2m1

        epilogue_2sm_schedule = to_grouped_schedule(EpilogueScheduleType.NoSmemWarpSpecialized2Sm, grouped)

        nvfp4_schedule                  = [to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized2SmVs16Sm103, grouped), epilogue_2sm_schedule]              
        nvfp4_schedule_disable_prefetch = [to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized2SmVs16Sm103DisablePrefetch, grouped), epilogue_2sm_schedule]                
        nvfp4_schedule_tma_prefetch     = [to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized2SmVs16Sm103TmaPrefetch, grouped), epilogue_2sm_schedule]
        fp4_schedule                    = [to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized2SmVs32Sm103, grouped), epilogue_2sm_schedule]
        fp4_schedule_disable_prefetch   = [to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized2SmVs32Sm103DisablePrefetch, grouped), epilogue_2sm_schedule]
        fp4_schedule_tma_prefetch       = [to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized2SmVs32Sm103TmaPrefetch, grouped), epilogue_2sm_schedule]
        nvfp4_schedules = [nvfp4_schedule, nvfp4_schedule_disable_prefetch, nvfp4_schedule_tma_prefetch]
        fp4_schedules   = [fp4_schedule, fp4_schedule_disable_prefetch, fp4_schedule_tma_prefetch]

        CreateGemmUniversal3xOperator(manifest, [layout], tile_descriptions, data_type, 
                                      nvfp4_schedules, tile_schedulers=tile_schedulers(data_type["sfd_type"]), gemm_kind=gemm_kind)
        if isFp4:
          CreateGemmUniversal3xOperator(manifest, [layout], tile_descriptions, data_type,
                                        fp4_schedules, tile_schedulers=tile_schedulers(data_type["sfd_type"]), gemm_kind=gemm_kind)


def GenerateSM100_TensorOp_int8_UMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 8):
    return

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 0]],
    [[LayoutType.ColumnMajor, 16], [LayoutType.RowMajor,    16], [LayoutType.ColumnMajor, 0]],
    [[LayoutType.RowMajor,    16], [LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 0]],
    [[LayoutType.RowMajor,    16], [LayoutType.RowMajor,    16], [LayoutType.ColumnMajor, 0]],
    [[LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 16], [LayoutType.RowMajor,    0]],
    [[LayoutType.ColumnMajor, 16], [LayoutType.RowMajor,    16], [LayoutType.RowMajor,    0]],
    [[LayoutType.RowMajor,    16], [LayoutType.ColumnMajor, 16], [LayoutType.RowMajor,    0]],
    [[LayoutType.RowMajor,    16], [LayoutType.RowMajor,    16], [LayoutType.RowMajor,    0]],
  ]

  thor_sm = ThorSMRenumbering(cuda_version)

  min_cc = 100
  max_cc = thor_sm

  epi_type = DataType.f32

  math_instructions_1sm = [
    MathInstruction(
      [64, 128, 32],
      DataType.s8, DataType.s8, DataType.s32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [128, 128, 32],
      DataType.s8, DataType.s8, DataType.s32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [128, 256, 32],
      DataType.s8, DataType.s8, DataType.s32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add)]

  cluster_shapes_1sm = [[1,2,1], [2,1,1], [1,1,1], [1,4,1], [4,4,1]
                        , DynamicClusterShape
                       ]

  if thor_sm in manifest.compute_capabilities_baseline :
    cluster_shapes_1sm = [[1,2,1], [2,1,1], [1,1,1], [1,4,1]
                          , DynamicClusterShape
                         ]                    

  tile_schedulers = [
    TileSchedulerType.Default, TileSchedulerType.StreamK
  ]

  # 1xSM MMA kernels
  for math_inst in math_instructions_1sm:
    tile_descriptions = []
    for cluster_shape in cluster_shapes_1sm:
      multiplier_1sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else cluster_shape
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_1sm[0],
          math_inst.instruction_shape[1]     * multiplier_1sm[1],
          math_inst.instruction_shape[2] * 4 * multiplier_1sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : math_inst.element_accumulator,
        "d_type"   : math_inst.element_accumulator,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : math_inst.element_accumulator,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator,
      },
    ]
    # Set alignment d based on Destination format.
    for layout in layouts:
      layout[2][1] = 128 // DataTypeSize[data_types[0]["d_type"]]

    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_types,
      [[KernelScheduleType.TmaWarpSpecialized1SmSm100, EpilogueScheduleType.TmaWarpSpecialized1Sm]],
      tile_schedulers=tile_schedulers)

    # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
      data_types_mixed = [
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : math_inst.element_a,
          "d_type"   : math_inst.element_a,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : epi_type,
        },
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : DataType.void,
          "d_type"   : math_inst.element_a,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : epi_type,
        },
      ]
      # Set alignment d based on Destination format.
      for layout in layouts:
        layout[2][1] = 128 // DataTypeSize[data_types_mixed[0]["d_type"]]

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_types_mixed,
        [[KernelScheduleType.TmaWarpSpecialized1SmSm100, EpilogueScheduleType.TmaWarpSpecialized1Sm]],
        tile_schedulers=tile_schedulers)

  # 2xSM MMA kernels
  math_instructions_2sm = [
    MathInstruction(
      [128, 128, 32],
      DataType.s8, DataType.s8, DataType.s32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [128, 256, 32],
      DataType.s8, DataType.s8, DataType.s32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [256, 128, 32],
      DataType.s8, DataType.s8, DataType.s32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [256, 256, 32],
      DataType.s8, DataType.s8, DataType.s32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
  ]

  cluster_shapes_2sm = [[2,1,1], [2,2,1], [2,4,1], [4,1,1], [4,2,1], [4,4,1]
                        , DynamicClusterShape
                       ]

  if thor_sm in manifest.compute_capabilities_baseline :
    cluster_shapes_2sm = [[2,1,1], [2,2,1], [2,4,1], [4,1,1], [4,2,1]
                          , DynamicClusterShape
                         ]

  for math_inst in math_instructions_2sm:
    tile_descriptions = []
    for cluster_shape in cluster_shapes_2sm:
      multiplier_2sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else (cluster_shape[0] // 2, cluster_shape[1], cluster_shape[2])
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_2sm[0],
          math_inst.instruction_shape[1]     * multiplier_2sm[1],
          math_inst.instruction_shape[2] * 4 * multiplier_2sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : math_inst.element_accumulator,
        "d_type"   : math_inst.element_accumulator,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator,
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : math_inst.element_accumulator,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator,
      },
    ]
    # Set alignment d based on Destination format.
    for layout in layouts:
      layout[2][1] = 128 // DataTypeSize[data_types[0]["d_type"]]

    if math_inst.instruction_shape[0] == 128:
      epi_schedule = EpilogueScheduleType.TmaWarpSpecialized2Sm
    else:
      epi_schedule = EpilogueScheduleType.ScheduleAuto

    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_types,
      [[KernelScheduleType.TmaWarpSpecialized2SmSm100, epi_schedule]], tile_schedulers=tile_schedulers)

    # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
      data_types_mixed = [
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : math_inst.element_a,
          "d_type"   : math_inst.element_a,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : epi_type,
        },
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : DataType.void,
          "d_type"   : math_inst.element_a,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : epi_type,
        },
      ]
      # Set alignment d based on Destination format.
      for layout in layouts:
        layout[2][1] = 128 // DataTypeSize[data_types[0]["d_type"]]

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_types_mixed,
        [[KernelScheduleType.TmaWarpSpecialized2SmSm100, epi_schedule]], tile_schedulers=tile_schedulers)


def GenerateSM100_SparseTensorOp_32b_UMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  # layouts for ABC and their alignments.
  layouts = [
    # Alignment requirement will be over-write below
    [[LayoutType.RowMajor, -1], [LayoutType.ColumnMajor, -1], [LayoutType.RowMajor, -1]],
  ]

  thor_sm = ThorSMRenumbering(cuda_version)

  min_cc = 100
  max_cc = thor_sm

  tile_schedulers = [
    TileSchedulerType.Default, TileSchedulerType.StreamK
  ]

  kernel_data_types = [
    # void_c
    {
      "a_type"   : DataType.f32,
      "b_type"   : DataType.f32,
      "c_type"   : DataType.void,
      "d_type"   : DataType.f32,
      "acc_type" : DataType.f32,
      "epi_type" : DataType.f32,
    },
    # none void_c
    {
      "a_type"   : DataType.f32,
      "b_type"   : DataType.f32,
      "c_type"   : DataType.f32,
      "d_type"   : DataType.f32,
      "acc_type" : DataType.f32,
      "epi_type" : DataType.f32,
    },
  ]

  math_instructions_1sm = [
    MathInstruction(
      [128, 128, 16],
      DataType.tf32, DataType.tf32, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [128, 256, 16],
      DataType.tf32, DataType.tf32, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
  ]

  math_instructions_2sm = [
    MathInstruction(
      [256, 128, 16],
      DataType.tf32, DataType.tf32, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [256, 256, 16],
      DataType.tf32, DataType.tf32, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
  ]

  # 1xSM MMA kernels
  for math_inst in math_instructions_1sm:
    tile_descriptions = []
    for cluster_shape in sm100_cluster_shape_1sm:
      if thor_sm in manifest.compute_capabilities_baseline :
        if cluster_shape == [4,4,1] :
          continue
      multiplier_1sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else cluster_shape
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_1sm[0],
          math_inst.instruction_shape[1]     * multiplier_1sm[1],
          math_inst.instruction_shape[2] * 2 * multiplier_1sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    for kernel_data_type in kernel_data_types:
      # Update layout alignment
      # alignment for d might be different for each kernel_data_type
      layouts_copy = copy.deepcopy(layouts)
      for layout in layouts_copy:
        # alignment for a, 2 for sparsity
        layout[0][1] = get_tma_alignment_elt(kernel_data_type["a_type"]) * ( 2 if layout[0][0] == LayoutType.RowMajor else 1)
        # alignment for b
        layout[1][1] = get_tma_alignment_elt(kernel_data_type["b_type"])
        # alignment for d
        layout[2][1] = get_tma_alignment_elt(kernel_data_type["d_type"])

      CreateSparseGemmUniversal3xOperator(manifest, layouts_copy, tile_descriptions, [kernel_data_type],
        [[KernelScheduleType.SparseTmaWarpSpecialized1SmSm100, EpilogueScheduleType.TmaWarpSpecialized1Sm]],
        tile_schedulers=tile_schedulers)

  # 2xSM MMA kernels
  for math_inst in math_instructions_2sm:
    tile_descriptions = []
    for cluster_shape in sm100_cluster_shape_2sm:
      if thor_sm in manifest.compute_capabilities_baseline :
        if cluster_shape == [4,4,1] :
          continue
      multiplier_2sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else (cluster_shape[0] // 2, cluster_shape[1], cluster_shape[2])
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_2sm[0],
          math_inst.instruction_shape[1]     * multiplier_2sm[1],
          math_inst.instruction_shape[2] * 2 * multiplier_2sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    for kernel_data_type in kernel_data_types:
      # Update layout alignment
      # alignment for d might be different for each kernel_data_type
      layouts_copy = copy.deepcopy(layouts)
      for layout in layouts_copy:
        # alignment for a, 2 for sparsity
        layout[0][1] = get_tma_alignment_elt(kernel_data_type["a_type"]) * ( 2 if layout[0][0] == LayoutType.RowMajor else 1)
        # alignment for b
        layout[1][1] = get_tma_alignment_elt(kernel_data_type["b_type"])
        # alignment for d
        layout[2][1] = get_tma_alignment_elt(kernel_data_type["d_type"])

      CreateSparseGemmUniversal3xOperator(manifest, layouts_copy, tile_descriptions, [kernel_data_type],
        [[KernelScheduleType.SparseTmaWarpSpecialized2SmSm100, EpilogueScheduleType.TmaWarpSpecialized2Sm]],
        tile_schedulers=tile_schedulers)

def GenerateSM100_SparseTensorOp_16b_UMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  # layouts for ABC and their alignments.
  layouts = [
    # Alignment requirement will be over-write below
    [[LayoutType.RowMajor, -1], [LayoutType.ColumnMajor, -1], [LayoutType.RowMajor, -1]],
  ]

  thor_sm = ThorSMRenumbering(cuda_version)

  min_cc = 100
  max_cc = thor_sm

  tile_schedulers = [
    TileSchedulerType.Default, TileSchedulerType.StreamK
  ]

  kernel_data_types = [
    # void_c
    {
      "a_type"   : DataType.f16,
      "b_type"   : DataType.f16,
      "c_type"   : DataType.void,
      "d_type"   : DataType.f16,
      "acc_type" : DataType.f32,
      "epi_type" : DataType.f32,
    },
    # none void_c
    {
      "a_type"   : DataType.f16,
      "b_type"   : DataType.f16,
      "c_type"   : DataType.f16,
      "d_type"   : DataType.f16,
      "acc_type" : DataType.f32,
      "epi_type" : DataType.f32,
    },
  ]

  math_instructions_1sm = [
    MathInstruction(
      [128, 128, 32],
      DataType.f16, DataType.f16, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [128, 256, 32],
      DataType.f16, DataType.f16, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
  ]

  math_instructions_2sm = [
    MathInstruction(
      [256, 128, 32],
      DataType.f16, DataType.f16, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [256, 256, 32],
      DataType.f16, DataType.f16, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
  ]

  # 1xSM MMA kernels
  for math_inst in math_instructions_1sm:
    tile_descriptions = []
    for cluster_shape in sm100_cluster_shape_1sm:
      if thor_sm in manifest.compute_capabilities_baseline :
        if cluster_shape == [4,4,1] :
          continue
      multiplier_1sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else cluster_shape
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_1sm[0],
          math_inst.instruction_shape[1]     * multiplier_1sm[1],
          math_inst.instruction_shape[2] * 2 * multiplier_1sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    for kernel_data_type in kernel_data_types:
      # Update layout alignment
      # alignment for d might be different for each kernel_data_type
      layouts_copy = copy.deepcopy(layouts)
      for layout in layouts_copy:
        # alignment for a, 2 for sparsity
        layout[0][1] = get_tma_alignment_elt(kernel_data_type["a_type"]) * ( 2 if layout[0][0] == LayoutType.RowMajor else 1)
        # alignment for b
        layout[1][1] = get_tma_alignment_elt(kernel_data_type["b_type"])
        # alignment for d
        layout[2][1] = get_tma_alignment_elt(kernel_data_type["d_type"])

      CreateSparseGemmUniversal3xOperator(manifest, layouts_copy, tile_descriptions, [kernel_data_type],
        [[KernelScheduleType.SparseTmaWarpSpecialized1SmSm100, EpilogueScheduleType.TmaWarpSpecialized1Sm]],
        tile_schedulers=tile_schedulers)

  # 2xSM MMA kernels
  for math_inst in math_instructions_2sm:
    tile_descriptions = []
    for cluster_shape in sm100_cluster_shape_2sm:
      if thor_sm in manifest.compute_capabilities_baseline :
        if cluster_shape == [4,4,1] :
          continue
      multiplier_2sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else (cluster_shape[0] // 2, cluster_shape[1], cluster_shape[2])
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_2sm[0],
          math_inst.instruction_shape[1]     * multiplier_2sm[1],
          math_inst.instruction_shape[2] * 2 * multiplier_2sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    for kernel_data_type in kernel_data_types:
      # Update layout alignment
      # alignment for d might be different for each kernel_data_type
      layouts_copy = copy.deepcopy(layouts)
      for layout in layouts_copy:
        # alignment for a, 2 for sparsity
        layout[0][1] = get_tma_alignment_elt(kernel_data_type["a_type"]) * ( 2 if layout[0][0] == LayoutType.RowMajor else 1)
        # alignment for b
        layout[1][1] = get_tma_alignment_elt(kernel_data_type["b_type"])
        # alignment for d
        layout[2][1] = get_tma_alignment_elt(kernel_data_type["d_type"])

      CreateSparseGemmUniversal3xOperator(manifest, layouts_copy, tile_descriptions, [kernel_data_type],
        [[KernelScheduleType.SparseTmaWarpSpecialized2SmSm100, EpilogueScheduleType.TmaWarpSpecialized2Sm]],
        tile_schedulers=tile_schedulers)

def GenerateSM100_SparseTensorOp_int8_UMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  # layouts for ABC and their alignments.
  layouts = [
    # Alignment requirement will be over-write below
    [[LayoutType.RowMajor, -1], [LayoutType.ColumnMajor, -1], [LayoutType.RowMajor, -1]],
  ]

  thor_sm = ThorSMRenumbering(cuda_version)

  min_cc = 100
  max_cc = thor_sm

  tile_schedulers = [
    TileSchedulerType.Default, TileSchedulerType.StreamK
  ]

  kernel_data_types = [
    # void_c
    {
      "a_type"   : DataType.s8,
      "b_type"   : DataType.s8,
      "c_type"   : DataType.void,
      "d_type"   : DataType.s8,
      "acc_type" : DataType.f32,
      "epi_type" : DataType.f32,
    },
    # none void_c
    {
      "a_type"   : DataType.s8,
      "b_type"   : DataType.s8,
      "c_type"   : DataType.s8,
      "d_type"   : DataType.s8,
      "acc_type" : DataType.f32,
      "epi_type" : DataType.f32,
    },
  ]

  math_instructions_1sm = [
    MathInstruction(
      [128, 128, 64],
      DataType.s8, DataType.s8, DataType.s32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [128, 256, 64],
      DataType.s8, DataType.s8, DataType.s32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add)]

  math_instructions_2sm = [
    MathInstruction(
      [256, 128, 64],
      DataType.s8, DataType.s8, DataType.s32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [256, 256, 64],
      DataType.s8, DataType.s8, DataType.s32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
  ]

  # 1xSM MMA kernels
  for math_inst in math_instructions_1sm:
    tile_descriptions = []
    for cluster_shape in sm100_cluster_shape_1sm:
      if thor_sm in manifest.compute_capabilities_baseline :
        if cluster_shape == [4,4,1] :
          continue
      multiplier_1sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else cluster_shape
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_1sm[0],
          math_inst.instruction_shape[1]     * multiplier_1sm[1],
          math_inst.instruction_shape[2] * 2 * multiplier_1sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    for kernel_data_type in kernel_data_types:
      # Update layout alignment
      # alignment for d might be different for each kernel_data_type
      layouts_copy = copy.deepcopy(layouts)
      for layout in layouts_copy:
        # alignment for a, 2 for sparsity
        layout[0][1] = get_tma_alignment_elt(kernel_data_type["a_type"]) * ( 2 if layout[0][0] == LayoutType.RowMajor else 1)
        # alignment for b
        layout[1][1] = get_tma_alignment_elt(kernel_data_type["b_type"])
        # alignment for d
        layout[2][1] = get_tma_alignment_elt(kernel_data_type["d_type"])

      CreateSparseGemmUniversal3xOperator(manifest, layouts_copy, tile_descriptions, [kernel_data_type],
        [[KernelScheduleType.SparseTmaWarpSpecialized1SmSm100, EpilogueScheduleType.TmaWarpSpecialized1Sm]],
        tile_schedulers=tile_schedulers)

  # 2xSM MMA kernels
  for math_inst in math_instructions_2sm:
    tile_descriptions = []
    for cluster_shape in sm100_cluster_shape_2sm:
      if thor_sm in manifest.compute_capabilities_baseline :
        if cluster_shape == [4,4,1] :
          continue
      multiplier_2sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else (cluster_shape[0] // 2, cluster_shape[1], cluster_shape[2])
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_2sm[0],
          math_inst.instruction_shape[1]     * multiplier_2sm[1],
          math_inst.instruction_shape[2] * 2 * multiplier_2sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    for kernel_data_type in kernel_data_types:
      # Update layout alignment
      # alignment for d might be different for each kernel_data_type
      layouts_copy = copy.deepcopy(layouts)
      for layout in layouts_copy:
        # alignment for a, 2 for sparsity
        layout[0][1] = get_tma_alignment_elt(kernel_data_type["a_type"]) * ( 2 if layout[0][0] == LayoutType.RowMajor else 1)
        # alignment for b
        layout[1][1] = get_tma_alignment_elt(kernel_data_type["b_type"])
        # alignment for d
        layout[2][1] = get_tma_alignment_elt(kernel_data_type["d_type"])

      CreateSparseGemmUniversal3xOperator(manifest, layouts_copy, tile_descriptions, [kernel_data_type],
        [[KernelScheduleType.SparseTmaWarpSpecialized2SmSm100, EpilogueScheduleType.TmaWarpSpecialized2Sm]],
        tile_schedulers=tile_schedulers)

def GenerateSM100_SparseTensorOp_fp8_UMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  # layouts for ABC and their alignments.
  layouts = [
    # Alignment requirement will be over-write below
    [[LayoutType.RowMajor, -1], [LayoutType.ColumnMajor, -1], [LayoutType.RowMajor, -1]],
  ]

  thor_sm = ThorSMRenumbering(cuda_version)

  min_cc = 100
  max_cc = thor_sm

  tile_schedulers = [
    TileSchedulerType.Default, TileSchedulerType.StreamK
  ]

  kernel_data_types = [
    # NOTE: a/b type in kernel will be overwrite below.
    #* void_c
    # f8_f8_f32_void_f16
    {
      "a_type"   : DataType.e4m3,
      "b_type"   : DataType.e4m3,
      "c_type"   : DataType.void,
      "d_type"   : DataType.f16,
      "acc_type" : DataType.f32,
      "epi_type" : DataType.f32,
    },
    #* non-void_c
    # f8_f8_f32_f16_f8
    {
      "a_type"   : DataType.e4m3,
      "b_type"   : DataType.e4m3,
      "c_type"   : DataType.f16,
      "d_type"   : DataType.e4m3,
      "acc_type" : DataType.f32,
      "epi_type" : DataType.f32,
    },
  ]

  math_instructions_1sm = [
    # Runtime DType
    MathInstruction(
      [128, 128, 64],
      DataType.f8, DataType.f8, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [128, 256, 64],
      DataType.f8, DataType.f8, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
  ]

  math_instructions_2sm = [
    # Runtime DType
    MathInstruction(
      [256, 128, 64],
      DataType.f8, DataType.f8, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [256, 256, 64],
      DataType.f8, DataType.f8, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
  ]

  # 1xSM MMA kernels
  for math_inst in math_instructions_1sm:
    tile_descriptions = []
    for cluster_shape in sm100_cluster_shape_1sm:
      if thor_sm in manifest.compute_capabilities_baseline :
        if cluster_shape == [4,4,1] :
          continue
      multiplier_1sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else cluster_shape
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_1sm[0],
          math_inst.instruction_shape[1]     * multiplier_1sm[1],
          math_inst.instruction_shape[2] * 2 * multiplier_1sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    for kernel_data_type in kernel_data_types:
      # Update input AB type
      kernel_data_type["a_type"] = math_inst.element_a
      kernel_data_type["b_type"] = math_inst.element_b

      # Update layout alignment
      # alignment for d might be different for each kernel_data_type
      layouts_copy = copy.deepcopy(layouts)
      for layout in layouts_copy:
        # alignment for a, 2 for sparsity
        layout[0][1] = get_tma_alignment_elt(kernel_data_type["a_type"]) * ( 2 if layout[0][0] == LayoutType.RowMajor else 1)
        # alignment for b
        layout[1][1] = get_tma_alignment_elt(kernel_data_type["b_type"])
        # alignment for d
        layout[2][1] = get_tma_alignment_elt(kernel_data_type["d_type"])

      CreateSparseGemmUniversal3xOperator(manifest, layouts_copy, tile_descriptions, [kernel_data_type],
        [[KernelScheduleType.SparseTmaWarpSpecialized1SmSm100, EpilogueScheduleType.TmaWarpSpecialized1Sm]],
        tile_schedulers=tile_schedulers)

  # 2xSM MMA kernels
  for math_inst in math_instructions_2sm:
    tile_descriptions = []
    for cluster_shape in sm100_cluster_shape_2sm:
      if thor_sm in manifest.compute_capabilities_baseline :
        if cluster_shape == [4,4,1] :
          continue
      multiplier_2sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else (cluster_shape[0] // 2, cluster_shape[1], cluster_shape[2])
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_2sm[0],
          math_inst.instruction_shape[1]     * multiplier_2sm[1],
          math_inst.instruction_shape[2] * 2 * multiplier_2sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    for kernel_data_type in kernel_data_types:
      # Update input AB type
      kernel_data_type["a_type"] = math_inst.element_a
      kernel_data_type["b_type"] = math_inst.element_b

      # Update layout alignment
      # alignment for d might be different for each kernel_data_type
      layouts_copy = copy.deepcopy(layouts)
      for layout in layouts_copy:
        # alignment for a, 2 for sparsity
        layout[0][1] = get_tma_alignment_elt(kernel_data_type["a_type"]) * ( 2 if layout[0][0] == LayoutType.RowMajor else 1)
        # alignment for b
        layout[1][1] = get_tma_alignment_elt(kernel_data_type["b_type"])
        # alignment for d
        layout[2][1] = get_tma_alignment_elt(kernel_data_type["d_type"])

      CreateSparseGemmUniversal3xOperator(manifest, layouts_copy, tile_descriptions, [kernel_data_type],
        [[KernelScheduleType.SparseTmaWarpSpecialized2SmSm100, EpilogueScheduleType.TmaWarpSpecialized2Sm]],
        tile_schedulers=tile_schedulers)

def GenerateSM100_SparseTensorOp_mixed_8bits_UMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  # layouts for ABC and their alignments.
  layouts = [
    # Alignment requirement will be over-write below
    [[LayoutType.RowMajor, -1], [LayoutType.ColumnMajor, -1], [LayoutType.RowMajor, -1]],
  ]

  thor_sm = ThorSMRenumbering(cuda_version)

  min_cc = 100
  max_cc = thor_sm

  tile_schedulers = [
    TileSchedulerType.Default, TileSchedulerType.StreamK
  ]

  math_instructions_1sm = [
    # Runtime Dtype
    MathInstruction(
      [128, 128, 64],
      DataType.f4, DataType.f4, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [128, 256, 64],
      DataType.f4, DataType.f4, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
  
    MathInstruction(
      [128, 128, 64],
      DataType.f6, DataType.f6, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [128, 256, 64],
      DataType.f6, DataType.f6, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
  ]

  math_instructions_2sm = [
    # Runtime DType
    MathInstruction(
      [256, 128, 64],
      DataType.f4, DataType.f4, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [256, 256, 64],
      DataType.f4, DataType.f4, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
  
    MathInstruction(
      [256, 128, 64],
      DataType.f6, DataType.f6, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [256, 256, 64],
      DataType.f6, DataType.f6, DataType.f32,
      OpcodeClass.SparseTensorOp,
      MathOperation.multiply_add),
  ]

  # 1xSM MMA kernels
  for math_inst in math_instructions_1sm:
    tile_descriptions = []
    for cluster_shape in sm100_cluster_shape_1sm:
      if thor_sm in manifest.compute_capabilities_baseline :
        if cluster_shape == [4,4,1] :
          continue
      multiplier_1sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else cluster_shape
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_1sm[0],
          math_inst.instruction_shape[1]     * multiplier_1sm[1],
          math_inst.instruction_shape[2] * 2 * multiplier_1sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    kernel_data_types = [
      # void_c
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.f16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : DataType.f32,
      },
      # none void_c
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : DataType.f32,
      },
    ]

    for kernel_data_type in kernel_data_types:
      # Update layout alignment
      # alignment for d might be different for each kernel_data_type
      layouts_filtered = []
      for layout in layouts:
        layout_filter = copy.deepcopy(layout)
        # * A_K : Logical TileShape_K % 256 == 0
        # * A_M : TileShape_M % 128 == 0
        # * B_N : TileSize_N % 128 == 0
        # * B_K : TileSize_K % 128 == 0
        if ((layout_filter[0][0] == LayoutType.RowMajor and (math_inst.instruction_shape[2] * 2) % 256 == 0) or \
            (layout_filter[0][0] == LayoutType.ColumnMajor and math_inst.instruction_shape[0] % 128 == 0)) and \
           ((layout_filter[1][0] == LayoutType.RowMajor and math_inst.instruction_shape[1] % 128 == 0) or \
            (layout_filter[1][0] == LayoutType.ColumnMajor and (math_inst.instruction_shape[0] * 2) % 128 == 0)):
          # alignment for a, 2 for sparsity
          layout_filter[0][1] = get_tma_alignment_elt(kernel_data_type["a_type"]) * ( 2 if layout[0][0] == LayoutType.RowMajor else 1)
          # alignment for b
          layout_filter[1][1] = get_tma_alignment_elt(kernel_data_type["b_type"])
          # alignment for d
          layout_filter[2][1] = get_tma_alignment_elt(kernel_data_type["d_type"])
          layouts_filtered.append(layout_filter)

      CreateSparseGemmUniversal3xOperator(manifest, layouts_filtered, tile_descriptions, [kernel_data_type],
        [[KernelScheduleType.SparseTmaWarpSpecialized1SmSm100, EpilogueScheduleType.TmaWarpSpecialized1Sm]],
        tile_schedulers=tile_schedulers)

  # 2xSM MMA kernels
  for math_inst in math_instructions_2sm:
    tile_descriptions = []
    for cluster_shape in sm100_cluster_shape_2sm:
      if thor_sm in manifest.compute_capabilities_baseline :
        if cluster_shape == [4,4,1] :
          continue
      multiplier_2sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else (cluster_shape[0] // 2, cluster_shape[1], cluster_shape[2])
      tile_descriptions.append(
        TileDescription([
          math_inst.instruction_shape[0]     * multiplier_2sm[0],
          math_inst.instruction_shape[1]     * multiplier_2sm[1],
          math_inst.instruction_shape[2] * 2 * multiplier_2sm[2]],
          0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    kernel_data_types = [
      # void_c
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.f16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : DataType.f32,
      },
      # none void_c
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : DataType.f32,
      },
    ]

    for kernel_data_type in kernel_data_types:
      # Update layout alignment
      # alignment for d might be different for each kernel_data_type
      layouts_filtered = []
      for layout in layouts:
        layout_filter = copy.deepcopy(layout)
        # * A_K : Logical TileShape_K % 256 == 0
        # * A_M : TileShape_M % 128 == 0
        # * B_N : TileSize_N % 256 == 0
        # * B_K : TileSize_K % 128 == 0
        if ((layout_filter[0][0] == LayoutType.RowMajor and (math_inst.instruction_shape[2] * 2) % 256 == 0) or \
            (layout_filter[0][0] == LayoutType.ColumnMajor and math_inst.instruction_shape[0] % 128 == 0)) and \
           ((layout_filter[1][0] == LayoutType.RowMajor and math_inst.instruction_shape[1] % 256 == 0) or \
            (layout_filter[1][0] == LayoutType.ColumnMajor and (math_inst.instruction_shape[0] * 2) % 128 == 0)):
          # alignment for a, 2 for sparsity
          layout_filter[0][1] = get_tma_alignment_elt(kernel_data_type["a_type"]) * ( 2 if layout[0][0] == LayoutType.RowMajor else 1)
          # alignment for b
          layout_filter[1][1] = get_tma_alignment_elt(kernel_data_type["b_type"])
          # alignment for d
          layout_filter[2][1] = get_tma_alignment_elt(kernel_data_type["d_type"])
          layouts_filtered.append(layout_filter)

      CreateSparseGemmUniversal3xOperator(manifest, layouts_filtered, tile_descriptions, [kernel_data_type],
        [[KernelScheduleType.SparseTmaWarpSpecialized2SmSm100, EpilogueScheduleType.TmaWarpSpecialized2Sm]],
        tile_schedulers=tile_schedulers)

# Conv Utility functions
def make_dims_and_alignments_triple(dim: int, bit_per_element_A: int, bit_per_element_B: int, bit_per_element_C: int):
  bit_alignment_required_by_tma = 128
  return ((dim, bit_alignment_required_by_tma // bit_per_element_A), # A
          (dim, bit_alignment_required_by_tma // bit_per_element_B), # B
          (dim, bit_alignment_required_by_tma // bit_per_element_C)) # C

def make_math_instruction_w_output(data_types: Tuple[DataType, DataType, DataType, DataType],
                          instruction_shape: Tuple[int, int, int]) -> (MathInstruction, DataType):
  default_opcode = OpcodeClass.TensorOp
  default_math_op = MathOperation.multiply_add
  [A_data_type, B_data_type, Acc_data_type, Out_data_type] = data_types
  return (MathInstruction(
    instruction_shape,
    A_data_type, B_data_type, Acc_data_type,
    default_opcode,
    default_math_op
  ), Out_data_type)

"""
Generate CUTLASS 3 convolution kernel(s) for SM100.

This is meant to be called from GenerateSM100.
"""
def GenerateSM100_TensorOp_16b_UMMA_conv3x(manifest, cuda_version,
                                           log_indent_level: int = 0):
  log_debug_line('GenerateSM100_TensorOp_16b_UMMA_conv3x', log_indent_level)
  log_indent_level = log_indent_level + 1

  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  thor_sm = ThorSMRenumbering(cuda_version)

  minimum_compute_capability = 100
  maximum_compute_capability = thor_sm

  spatial_dims = [2, 3]

  conv_kinds = [
    ConvKind.Fprop,
    ConvKind.Dgrad,
    ConvKind.Wgrad
  ]

  stages = 0 # zero means "deduce the number of stages automatically"

  data_types_and_instruction_shapes_1sm = [
    # ((A,B,Acc,C/D), (InstM,InstN,InstK))
    ((DataType.f16, DataType.f16, DataType.f16, DataType.f16),    (64, 128, 16)),
    ((DataType.f16, DataType.f16, DataType.f16, DataType.f16),    (128, 128, 16)),
    ((DataType.f16, DataType.f16, DataType.f16, DataType.f16),    (128, 256, 16)),
    ((DataType.f16, DataType.f16, DataType.f32, DataType.f16),    (64, 128, 16)),
    ((DataType.f16, DataType.f16, DataType.f32, DataType.f16),    (128, 128, 16)),
    ((DataType.f16, DataType.f16, DataType.f32, DataType.f16),    (128, 256, 16)),
    ((DataType.bf16, DataType.bf16, DataType.f32, DataType.bf16), (64, 128, 16)),
    ((DataType.bf16, DataType.bf16, DataType.f32, DataType.bf16), (128, 128, 16)),
    ((DataType.bf16, DataType.bf16, DataType.f32, DataType.bf16), (128, 256, 16)),
  ]
  math_instructions_w_output_1sm = map(lambda x: make_math_instruction_w_output(*x),
                          data_types_and_instruction_shapes_1sm)

  cluster_shapes_1sm = [[1,1,1], [1,2,1], [1,4,1],[4,4,1]]

  if thor_sm in manifest.compute_capabilities_baseline :
    cluster_shapes_1sm = [[1,1,1], [1,2,1], [1,4,1]]

  # tile_descriptions is a 2-level list.
  # Each inner list is for each cluster shape.
  for math_inst, output_type in math_instructions_w_output_1sm:
    tile_descriptions = []
    for cluster_shape in cluster_shapes_1sm:
      cluster_multiplier = cluster_shape
      # Unlike SM90, SM100 tile shape calculation includes cluster shape.
      tile_shape = [
        math_inst.instruction_shape[0]     * cluster_multiplier[0],
        math_inst.instruction_shape[1]     * cluster_multiplier[1],
        math_inst.instruction_shape[2] * 4 * cluster_multiplier[2]
      ]
      warp_count = [4, 1, 1]
      tile_description = TileDescription(
        tile_shape, stages, warp_count, math_inst,
        minimum_compute_capability, maximum_compute_capability,
        cluster_shape)
      tile_descriptions.append(tile_description)

      # It's typical to get the data types from the math instruction.
      data_type = {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : output_type,
        "d_type"   : output_type,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      }

      dims_and_alignments = [make_dims_and_alignments_triple(dim, DataTypeSize[data_type["a_type"]], DataTypeSize[data_type["b_type"]], DataTypeSize[data_type["d_type"]]) for dim in spatial_dims]

      # Schedules
      mainloop_schedule = KernelScheduleType.ImplicitTmaWarpSpecialized1SmSm100
      epilogue_schedule = EpilogueScheduleType.ScheduleAuto
      schedule_pairs = [
        (mainloop_schedule, epilogue_schedule)
      ]

      for conv_kind in conv_kinds:
        CreateConvOperator3x(manifest,
                            dims_and_alignments = dims_and_alignments,
                            tile_descriptions = tile_descriptions,
                            data_types = data_type,
                            schedule_pairs = schedule_pairs,
                            conv_kind = conv_kind,
                            log_indent_level = log_indent_level)

  data_types_and_instruction_shapes_2sm = [
    # ((A,B,Acc,C/D), (InstM,InstN,InstK))
    ((DataType.f16, DataType.f16, DataType.f16, DataType.f16),    (128, 128, 16)),
    ((DataType.f16, DataType.f16, DataType.f16, DataType.f16),    (128, 256, 16)),
    ((DataType.f16, DataType.f16, DataType.f16, DataType.f16),    (256, 256, 16)),
    ((DataType.f16, DataType.f16, DataType.f32, DataType.f16),    (128, 128, 16)),
    ((DataType.f16, DataType.f16, DataType.f32, DataType.f16),    (128, 256, 16)),
    ((DataType.f16, DataType.f16, DataType.f32, DataType.f16),    (256, 256, 16)),
    ((DataType.bf16, DataType.bf16, DataType.f32, DataType.bf16), (128, 128, 16)),
    ((DataType.bf16, DataType.bf16, DataType.f32, DataType.bf16), (128, 256, 16)),
    ((DataType.bf16, DataType.bf16, DataType.f32, DataType.bf16), (256, 256, 16)),
  ]
  math_instructions_w_output_2sm = map(lambda x: make_math_instruction_w_output(*x),
                          data_types_and_instruction_shapes_2sm)

  cluster_shapes_2sm = [[2,1,1], [2,2,1], [2,4,1], [4,1,1], [4,2,1], [4,4,1]]
  if thor_sm in manifest.compute_capabilities_baseline :
    cluster_shapes_2sm = [[2,1,1], [2,2,1], [2,4,1], [4,1,1], [4,2,1]]

  for math_inst, output_type in math_instructions_w_output_2sm:
    tile_descriptions = []
    for cluster_shape in cluster_shapes_2sm:
      cluster_multiplier = (cluster_shape[0] // 2, cluster_shape[1], cluster_shape[2])
      # Unlike SM90, SM100 tile shape calculation includes cluster shape.
      tile_shape = [
        math_inst.instruction_shape[0]     * cluster_multiplier[0],
        math_inst.instruction_shape[1]     * cluster_multiplier[1],
        math_inst.instruction_shape[2] * 4 * cluster_multiplier[2]
      ]
      warp_count = [4, 1, 1]
      tile_description = TileDescription(
        tile_shape, stages, warp_count, math_inst,
        minimum_compute_capability, maximum_compute_capability,
        cluster_shape)
      tile_descriptions.append(tile_description)

      # It's typical to get the data types from the math instruction.
      data_type = {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : output_type,
        "d_type"   : output_type,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      }

      dims_and_alignments = [make_dims_and_alignments_triple(dim, DataTypeSize[data_type["a_type"]], DataTypeSize[data_type["b_type"]], DataTypeSize[data_type["d_type"]]) for dim in spatial_dims]

      # Schedules
      mainloop_schedule = KernelScheduleType.ImplicitTmaWarpSpecialized2SmSm100
      epilogue_schedule = EpilogueScheduleType.ScheduleAuto
      schedule_pairs = [
        (mainloop_schedule, epilogue_schedule)
      ]

      for conv_kind in conv_kinds:
        CreateConvOperator3x(manifest,
                            dims_and_alignments = dims_and_alignments,
                            tile_descriptions = tile_descriptions,
                            data_types = data_type,
                            schedule_pairs = schedule_pairs,
                            conv_kind = conv_kind,
                            log_indent_level = log_indent_level)

def GenerateSM100_TensorOp_fp8_UMMA_conv3x(manifest, cuda_version,
                                           log_indent_level: int = 0):
  # Instantiate Fp8 Fprop kernels with e4m3 A/B, f32 Acc, e4m3/bf16/f16/f32 C/D
  log_debug_line('GenerateSM100_TensorOp_fp8_UMMA_conv3x', log_indent_level)
  log_indent_level = log_indent_level + 1

  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  thor_sm = ThorSMRenumbering(cuda_version)

  minimum_compute_capability = 100
  maximum_compute_capability = thor_sm

  spatial_dims = [2, 3]
  stages = 0 # zero means "deduce the number of stages automatically"

  data_types_and_instruction_shapes_1sm = [
    # ((A,B,Acc,C/D), (InstM,InstN,InstK))
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.e4m3),   (64, 128, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.e4m3),   (128, 128, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.e4m3),   (128, 256, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.f16),    (64, 128, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.f16),    (128, 128, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.f16),    (128, 256, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.bf16),   (64, 128, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.bf16),   (128, 128, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.bf16),   (128, 256, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.f32),    (64, 128, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.f32),    (128, 128, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.f32),    (128, 256, 32)),
  ]
  math_instructions_w_output_1sm = map(lambda x: make_math_instruction_w_output(*x),
                          data_types_and_instruction_shapes_1sm)

  cluster_shapes_1sm = [[1,1,1], [1,2,1], [1,4,1],[4,4,1]]
  if thor_sm in manifest.compute_capabilities_baseline :
    cluster_shapes_1sm = [[1,1,1], [1,2,1], [1,4,1]]

  for math_inst, output_type in math_instructions_w_output_1sm:
    tile_descriptions = []
    for cluster_shape in cluster_shapes_1sm:
      cluster_multiplier = cluster_shape
      # Unlike SM90, SM100 tile shape calculation includes cluster shape.
      tile_shape = [
        math_inst.instruction_shape[0]     * cluster_multiplier[0],
        math_inst.instruction_shape[1]     * cluster_multiplier[1],
        math_inst.instruction_shape[2] * 4 * cluster_multiplier[2]
      ]
      warp_count = [4, 1, 1]
      tile_description = TileDescription(
        tile_shape, stages, warp_count, math_inst,
        minimum_compute_capability, maximum_compute_capability,
        cluster_shape)
      tile_descriptions.append(tile_description)

      data_type = {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : output_type,
        "d_type"   : output_type,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      }

      dims_and_alignments = [make_dims_and_alignments_triple(dim, DataTypeSize[data_type["a_type"]], DataTypeSize[data_type["b_type"]], DataTypeSize[data_type["d_type"]]) for dim in spatial_dims]

      # Schedules
      mainloop_schedule = KernelScheduleType.ImplicitTmaWarpSpecialized1SmSm100
      epilogue_schedule = EpilogueScheduleType.ScheduleAuto
      schedule_pairs = [
        (mainloop_schedule, epilogue_schedule)
      ]

      CreateConvOperator3x(manifest,
                          dims_and_alignments = dims_and_alignments,
                          tile_descriptions = tile_descriptions,
                          data_types = data_type,
                          schedule_pairs = schedule_pairs,
                          conv_kind = ConvKind.Fprop,
                          log_indent_level = log_indent_level)

  data_types_and_instruction_shapes_2sm = [
    # ((A,B,Acc,C/D), (InstM,InstN,InstK))
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.e4m3),   (128, 128, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.e4m3),   (128, 256, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.e4m3),   (256, 256, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.f16),    (128, 128, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.f16),    (128, 256, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.f16),    (256, 256, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.bf16),   (128, 128, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.bf16),   (128, 256, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.bf16),   (256, 256, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.f32),    (128, 128, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.f32),    (128, 256, 32)),
    ((DataType.e4m3, DataType.e4m3, DataType.f32, DataType.f32),    (256, 256, 32)),
  ]
  math_instructions_w_output_2sm = map(lambda x: make_math_instruction_w_output(*x),
                          data_types_and_instruction_shapes_2sm)

  cluster_shapes_2sm = [[2,1,1], [2,2,1], [2,4,1], [4,1,1], [4,2,1], [4,4,1]]
  if thor_sm in manifest.compute_capabilities_baseline :
    cluster_shapes_2sm = [[2,1,1], [2,2,1], [2,4,1], [4,1,1], [4,2,1]]

  for math_inst, output_type in math_instructions_w_output_2sm:
    tile_descriptions = []
    for cluster_shape in cluster_shapes_2sm:
      cluster_multiplier = (cluster_shape[0] // 2, cluster_shape[1], cluster_shape[2])
      # Unlike SM90, SM100 tile shape calculation includes cluster shape.
      tile_shape = [
        math_inst.instruction_shape[0]     * cluster_multiplier[0],
        math_inst.instruction_shape[1]     * cluster_multiplier[1],
        math_inst.instruction_shape[2] * 4 * cluster_multiplier[2]
      ]
      warp_count = [4, 1, 1]
      tile_description = TileDescription(
        tile_shape, stages, warp_count, math_inst,
        minimum_compute_capability, maximum_compute_capability,
        cluster_shape)
      tile_descriptions.append(tile_description)

      data_type = {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : output_type,
        "d_type"   : output_type,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      }

      dims_and_alignments = [make_dims_and_alignments_triple(dim, DataTypeSize[data_type["a_type"]], DataTypeSize[data_type["b_type"]], DataTypeSize[data_type["d_type"]]) for dim in spatial_dims]

      # Schedules
      mainloop_schedule = KernelScheduleType.ImplicitTmaWarpSpecialized2SmSm100
      epilogue_schedule = EpilogueScheduleType.ScheduleAuto
      schedule_pairs = [
        (mainloop_schedule, epilogue_schedule)
      ]

      CreateConvOperator3x(manifest,
                          dims_and_alignments = dims_and_alignments,
                          tile_descriptions = tile_descriptions,
                          data_types = data_type,
                          schedule_pairs = schedule_pairs,
                          conv_kind = ConvKind.Fprop,
                          log_indent_level = log_indent_level)

def GenerateSM120_TensorOp_mixed_8bits_UMMA_gemm_with_block_scaled(manifest, cuda_version):
  # SM120 MMA with mixed F4/F6/F8 inputs + block scale
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 8):
    return

  layouts = [
    [[LayoutType.RowMajor,    128], [LayoutType.ColumnMajor, 128], [LayoutType.RowMajor,    0]]
  ]

  instruction_sizes = [
    [16, 8, 32]
  ]

  tile_sizes = [
    [128, 128, 128]
  ]

  cluster_shape = [1,1,1]

  ab_types  = [
    DataType.e2m1, 
    DataType.e2m3, 
    DataType.e3m2,
    DataType.e5m2,
    DataType.e4m3,
  ]

  acc_types = [ DataType.f32 ]

  def is_pingpong(kernel_schedule):
    if kernel_schedule == KernelScheduleType.Mxf8f6f4TmaWarpSpecializedPingpongSm120:
      return True
    else:
      return False
    
  def tile_schedulers(sfdtype, kernel_schedule):
    # Pingpong kernel schedule doesn't support stream-K.
    # Only use the stream-K scheduler for non-void SFD to limit kernel count. When SFD is void,
    # the epilogue is the traditional linear combination, for which we already have tests with stream-K
    if is_pingpong(kernel_schedule):
      return [TileSchedulerType.Default]
    elif sfdtype["type"] == DataType.void:
      return [TileSchedulerType.Default]
    else:
      return [TileSchedulerType.Default, TileSchedulerType.StreamK]

  min_cc = 120
  max_cc = 121

  epi_type = DataType.f32
  
  math_instructions = []

  kernel_schedules = [
    KernelScheduleType.Mxf8f6f4TmaWarpSpecializedCooperativeSm120,
    KernelScheduleType.Mxf8f6f4TmaWarpSpecializedPingpongSm120
  ]

  for instr_size, a_type, b_type, acc_type in product(instruction_sizes, ab_types, ab_types, acc_types):
    math_instructions.append(
      MathInstruction(
        instr_size,
        a_type, b_type, acc_type,
        OpcodeClass.BlockScaledTensorOp,
        MathOperation.multiply_add,
        DataType.ue8m0)
    )

  for math_inst in math_instructions:
    tile_descriptions = []
    for tile_size in tile_sizes:
      tile_descriptions.append(
        TileDescription(tile_size, 0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 32, "layout" : LayoutType.RowMajor}
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e3m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : epi_type,
        "sf_type"  : math_inst.element_scale_factor,
        "sfd_type" : {"type": DataType.ue8m0, "vector_size": 32, "layout" : LayoutType.RowMajor}
      }
    ]

    # Set alignment d based on Destination format.
    for layout in layouts:
      layout[2][1] = 128 // DataTypeSize[data_types[0]["d_type"]]

    for data_type, kernel_schedule in product(data_types, kernel_schedules):
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
        [[kernel_schedule, EpilogueScheduleType.ScheduleAuto]], 
        tile_schedulers = tile_schedulers(data_type["sfd_type"], kernel_schedule),
        gemm_kind = GemmKind.BlockScaledUniversal3x
        )

def GenerateSM120_TensorOp_fp4_UMMA_gemm_with_block_scaled(manifest, cuda_version):
  # SM120 MMA with with F4 + block scale
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 8):
    return

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.RowMajor,    32], [LayoutType.ColumnMajor, 32], [LayoutType.RowMajor,    0]]
  ]

  instruction_sizes = [
    [16, 8, 64]
  ]

  tile_sizes_cooperative = [
    [128, 128, 128],
    [128, 128, 256],
    [256, 128, 128]
  ]

  tile_sizes_pingpong = [
    [128, 128, 128],
    [128, 128, 256]
  ]

  cluster_shape = [1,1,1]

  ab_types  = [
    DataType.e2m1
  ]

  sf_types  = [
    DataType.ue4m3,
    DataType.ue8m0
  ]

  acc_types = [ DataType.f32 ]

  def is_pingpong(kernel_schedule):
    if kernel_schedule == KernelScheduleType.Nvf4TmaWarpSpecializedPingpongSm120 or \
       kernel_schedule == KernelScheduleType.Mxf4TmaWarpSpecializedPingpongSm120:
      return True
    else:
      return False
  
  def is_nvf4(kernel_schedule):
    if kernel_schedule == KernelScheduleType.Nvf4TmaWarpSpecializedCooperativeSm120 or \
       kernel_schedule == KernelScheduleType.Nvf4TmaWarpSpecializedPingpongSm120:
      return True
    else:
      return False
    
  def tile_schedulers(sfdtype, kernel_schedule):
    # Pingpong kernel schedule doesn't support stream-K.
    # Only use the stream-K scheduler for non-void SFD to limit kernel count. When SFD is void,
    # the epilogue is the traditional linear combination, for which we already have tests with stream-K
    if is_pingpong(kernel_schedule):
      return [TileSchedulerType.Default]
    elif sfdtype["type"] == DataType.void:
      return [TileSchedulerType.Default]
    else:
      return [TileSchedulerType.Default, TileSchedulerType.StreamK]

  min_cc = 120
  max_cc = 121

  epi_type = DataType.f32
  
  math_instructions = []

  kernel_schedules = [
    KernelScheduleType.Nvf4TmaWarpSpecializedCooperativeSm120,
    KernelScheduleType.Nvf4TmaWarpSpecializedPingpongSm120,
    KernelScheduleType.Mxf4TmaWarpSpecializedCooperativeSm120,
    KernelScheduleType.Mxf4TmaWarpSpecializedPingpongSm120
  ]

  for instr_size, a_type, b_type, acc_type, sf_type in product(instruction_sizes, ab_types, ab_types, acc_types, sf_types):
    math_instructions.append(
      MathInstruction(
        instr_size,
        a_type, b_type, acc_type,
        OpcodeClass.BlockScaledTensorOp,
        MathOperation.multiply_add,
        sf_type)
    )

  for math_inst in math_instructions:
    for kernel_schedule in kernel_schedules:
      tile_descriptions = []
      tile_sizes = tile_sizes_pingpong if is_pingpong(kernel_schedule) else tile_sizes_cooperative
      for tile_size in tile_sizes:
        # nvf4 kernel only supports ue4m3 SF
        # mxf4 kernel only supports ue8m0 SF
        if (math_inst.element_scale_factor == DataType.ue4m3 and is_nvf4(kernel_schedule)) or \
           (math_inst.element_scale_factor == DataType.ue8m0 and not is_nvf4(kernel_schedule)):
          tile_descriptions.append(
            TileDescription(tile_size, 0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

      data_types = [
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : DataType.void,
          "d_type"   : DataType.f32,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : epi_type,
          "sf_type"  : math_inst.element_scale_factor,
          "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
        },
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : DataType.void,
          "d_type"   : DataType.e2m1,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : epi_type,
          "sf_type"  : math_inst.element_scale_factor,
          "sfd_type" : {"type": DataType.ue8m0, "vector_size": 32, "layout" : LayoutType.RowMajor}
        },
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : DataType.void,
          "d_type"   : DataType.e5m2,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : epi_type,
          "sf_type"  : math_inst.element_scale_factor,
          "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
        },
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : DataType.f16,
          "d_type"   : DataType.e5m2,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : epi_type,
          "sf_type"  : math_inst.element_scale_factor,
          "sfd_type" : {"type": DataType.void, "vector_size": None, "layout" : None}
        },
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : DataType.void,
          "d_type"   : DataType.e2m1,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : epi_type,
          "sf_type"  : math_inst.element_scale_factor,
          "sfd_type" : {"type": DataType.ue8m0, "vector_size": 16, "layout" : LayoutType.RowMajor}
        },
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : DataType.f16,
          "d_type"   : DataType.e2m1,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : epi_type,
          "sf_type"  : math_inst.element_scale_factor,
          "sfd_type" : {"type": DataType.ue8m0, "vector_size": 16, "layout" : LayoutType.RowMajor}
        },
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : DataType.f16,
          "d_type"   : DataType.e2m1,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : epi_type,
          "sf_type"  : math_inst.element_scale_factor,
          "sfd_type" : {"type": DataType.ue8m0, "vector_size": 32, "layout" : LayoutType.RowMajor}
        }
      ]

      # Set alignment d based on Destination format.
      for layout in layouts:
        layout[2][1] = 128 // DataTypeSize[data_types[0]["d_type"]]

      for data_type in data_types:
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
          [[kernel_schedule, EpilogueScheduleType.ScheduleAuto]], 
          tile_schedulers = tile_schedulers(data_type["sfd_type"], kernel_schedule),
          gemm_kind = GemmKind.BlockScaledUniversal3x
          ) 

def GenerateSM120_Sparse_TensorOp_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 8):
    return

  layouts = [
    [[LayoutType.RowMajor, 256], [LayoutType.ColumnMajor, 128], [LayoutType.RowMajor, 0]]
  ]

  tile_sizes = [
    [128, 128, 256]
  ]

  cluster_shape = [1,1,1]
  
  warp_count = [4, 2, 1]

  acc_types = [ DataType.f32 ]

  instruction_sizes_mxf8f6f4 = [
    [16, 8, 64]
  ]

  ab_types_mxf8f6f4  = [
    DataType.e2m1, 
    #DataType.e2m3, 
    DataType.e3m2,
    #DataType.e5m2,
    DataType.e4m3,
  ]

  def tile_schedulers(kernel_schedule):
      return [TileSchedulerType.Default]

  min_cc = 120
  max_cc = 121

  kernel_schedules = [
    KernelScheduleType.F8f6f4SparseTmaWarpSpecializedCooperativeSm120,
  ]

  math_instructions_mxf8f6f4 = []

  for instr_size, a_type, b_type, acc_type in product(instruction_sizes_mxf8f6f4, ab_types_mxf8f6f4, ab_types_mxf8f6f4, acc_types):
    math_instructions_mxf8f6f4.append(
      MathInstruction(
        instr_size,
        a_type, b_type, acc_type,
        OpcodeClass.SparseTensorOp,
        MathOperation.multiply_add)
    )

  # Create gemm operator for mxf8f6f4
  for math_inst in math_instructions_mxf8f6f4:
    tile_descriptions_mxf8f6f4 = []
    for tile_size in tile_sizes:
      tile_descriptions_mxf8f6f4.append(
        TileDescription(tile_size, 0, warp_count, math_inst, min_cc, max_cc, cluster_shape))

    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : DataType.f32
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : DataType.f32
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e4m3,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : DataType.f32
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.f16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : DataType.f32
      }
    ]

    for data_type, kernel_schedule in product(data_types, kernel_schedules):
      # Set alignment d based on Destination format
      for layout in layouts:
        layout[2][1] = int(128 // DataTypeSize[data_type["d_type"]])
      # Create gemm operator
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions_mxf8f6f4, data_type,
        [[kernel_schedule, EpilogueScheduleType.ScheduleAuto]], 
        tile_schedulers = tile_schedulers(kernel_schedule),
        gemm_kind = GemmKind.SparseUniversal3x)

def GenerateSM120_TensorOp_fp8_UMMA_gemm_with_blockwise(manifest, cuda_version, gemm_kind=GemmKind.BlockwiseUniversal3x):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 8):
    return

  layouts = [
    [[LayoutType.RowMajor, 128], [LayoutType.ColumnMajor, 128], [LayoutType.RowMajor, 16]],
    [[LayoutType.RowMajor, 128], [LayoutType.ColumnMajor, 128], [LayoutType.ColumnMajor, 16]]
  ]

  cooperative_tile_sizes = [
    [128, 128, 128]
  ]
  pingpong_tile_sizes = [
    [64, 128, 128]
  ]

  def get_tile_sizes(kernel_scheduler):
    if kernel_scheduler == KernelScheduleType.BlockwiseTmaWarpSpecializedPingpongSm120:
      return pingpong_tile_sizes
    return cooperative_tile_sizes

  def get_warp_count(kernel_scheduler):
    if kernel_scheduler == KernelScheduleType.BlockwiseTmaWarpSpecializedPingpongSm120:
      return [2, 2, 1]
    return [4, 2, 1]

  def get_sf_sizes(tile_size):
    sf_sizes = []
    for vec_m in [1, 128]:
      if tile_size[0] % vec_m > 0:
        continue
      for vec_n in [1, 128]:
        if tile_size[1] % vec_m > 0:
          continue
        sf_sizes.append(
          [vec_m, vec_n, 128]
        )
    return sf_sizes

  cluster_shape = [1,1,1]

  acc_types = [ DataType.f32 ]

  instruction_sizes = [
    [16, 8, 32]
  ]

  def tile_schedulers(kernel_schedule):
      return [TileSchedulerType.Default]

  min_cc = 120
  max_cc = 121

  kernel_schedulers = [
    KernelScheduleType.BlockwiseTmaWarpSpecializedCooperativeSm120,
    KernelScheduleType.BlockwiseTmaWarpSpecializedPingpongSm120
  ]

  ab_types = [
    [DataType.e4m3, DataType.e4m3],
    [DataType.e4m3, DataType.e5m2]
  ]

  math_instructions = []

  for instr_size, ab_type, acc_type in product(instruction_sizes, ab_types, acc_types):
    a_type, b_type = ab_type
    math_instructions.append(
      MathInstruction(
        instr_size,
        a_type, b_type, acc_type,
        OpcodeClass.TensorOp,
        MathOperation.multiply_add)
    )

  # Create gemm operator for mxf8f6f4
  for kernel_schedule in kernel_schedulers:
    tile_sizes = get_tile_sizes(kernel_schedule)
    warp_count = get_warp_count(kernel_schedule)
    for math_inst in math_instructions:
      tile_descriptions = []
      for tile_size in tile_sizes:
        sf_sizes = get_sf_sizes(tile_size)
        for sf_size in sf_sizes:
          tile_descriptions.append(
            TileDescription(tile_size, 0, warp_count, math_inst, min_cc, max_cc, cluster_shape,
                            explicit_vector_sizes=sf_size)
          )

      data_types = [
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : DataType.f16,
          "d_type"   : DataType.f16,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : DataType.f32
        },
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : DataType.bf16,
          "d_type"   : DataType.bf16,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : DataType.f32
        },
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : DataType.void,
          "d_type"   : DataType.f16,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : DataType.f32
        },
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : DataType.void,
          "d_type"   : DataType.bf16,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : DataType.f32
        }
      ]

      for data_type in data_types:
        # Set alignment d based on Destination format
        for layout in layouts:
          layout[2][1] = int(128 // DataTypeSize[data_type["d_type"]])
        # Create gemm operator
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
          [[kernel_schedule, EpilogueScheduleType.ScheduleAuto]], 
          tile_schedulers = tile_schedulers(kernel_schedule),
          gemm_kind = gemm_kind)

def GenerateSM100(manifest, cuda_version):
  arch_family_cc = ['100f', '101f', '103a']
  if CudaToolkitVersionSatisfies(cuda_version, 13, 0):
    for old_cc, new_cc in [('101f', '110f')]:
      arch_family_cc = [cc.replace(old_cc, new_cc) for cc in arch_family_cc]

  #
  # Dense Gemm
  #
  GenerateSM100_TensorOp_16b_UMMA_gemm(manifest, cuda_version)

  GenerateSM100_TensorOp_32b_UMMA_gemm(manifest, cuda_version)

  if not bool(set(manifest.compute_capabilities_feature_set).intersection(arch_family_cc)):
    GenerateSM100_TensorOp_int8_UMMA_gemm(manifest, cuda_version)

  GenerateSM100_TensorOp_fp8_UMMA_gemm(manifest, cuda_version)
  # grouped GEMM
  GenerateSM100_TensorOp_fp8_UMMA_gemm(manifest, cuda_version, gemm_kind=GemmKind.GroupedUniversal3x)
  GenerateSM100_TensorOp_16b_UMMA_gemm(manifest, cuda_version, gemm_kind=GemmKind.GroupedUniversal3x)

  # StreamK is included in regular generation
  GenerateSM100_TensorOp_mixed_8bits_UMMA_gemm(manifest, cuda_version)

  # Blockwise kernels
  GenerateSM100_TensorOp_fp8_UMMA_gemm_with_blockwise(manifest, cuda_version)
  GenerateSM100_TensorOp_fp8_UMMA_gemm_with_blockwise(manifest, cuda_version, gemm_kind=GemmKind.GroupedBlockwiseUniversal3x)

  #
  # Sparse Gemm
  #
  GenerateSM100_SparseTensorOp_32b_UMMA_gemm(manifest, cuda_version)
  GenerateSM100_SparseTensorOp_16b_UMMA_gemm(manifest, cuda_version)
  if not bool(set(manifest.compute_capabilities_feature_set).intersection(arch_family_cc)):
    GenerateSM100_SparseTensorOp_int8_UMMA_gemm(manifest, cuda_version)
  GenerateSM100_SparseTensorOp_fp8_UMMA_gemm(manifest, cuda_version)
  GenerateSM100_SparseTensorOp_mixed_8bits_UMMA_gemm(manifest, cuda_version)

  #
  # Block Scaled Gemm
  #
  GenerateSM100_TensorOp_mixed_8bits_UMMA_gemm_with_block_scaled(manifest, cuda_version)
  GenerateSM100_TensorOp_mixed_8bits_UMMA_gemm_with_block_scaled(manifest, cuda_version, gemm_kind=GemmKind.GroupedBlockScaledUniversal3x)
  GenerateSM100_TensorOp_fp4_UMMA_gemm_with_block_scaled(manifest, cuda_version)
  GenerateSM100_TensorOp_fp4_UMMA_gemm_with_block_scaled(manifest, cuda_version,  gemm_kind=GemmKind.GroupedBlockScaledUniversal3x)
  
  GenerateSM103_TensorOp_fp4_ultra_UMMA_gemm_with_block_scaled(manifest, cuda_version)
  GenerateSM103_TensorOp_fp4_ultra_UMMA_gemm_with_block_scaled(manifest, cuda_version, gemm_kind=GemmKind.GroupedBlockScaledUniversal3x)
  #
  # Conv
  #
  GenerateSM100_TensorOp_16b_UMMA_conv3x(manifest, cuda_version)
  GenerateSM100_TensorOp_fp8_UMMA_conv3x(manifest, cuda_version)


def GenerateSM120(manifest, cuda_version):
  # StreamK is included in regular generation #
  #
  # Dense Block Scaled Gemm
  #
  GenerateSM120_TensorOp_mixed_8bits_UMMA_gemm_with_block_scaled(manifest, cuda_version)
  GenerateSM120_TensorOp_fp4_UMMA_gemm_with_block_scaled(manifest, cuda_version)

  #
  # Sparse Gemm
  #
  GenerateSM120_Sparse_TensorOp_gemm(manifest, cuda_version)
  GenerateSM120_TensorOp_fp8_UMMA_gemm_with_blockwise(manifest, cuda_version)
  GenerateSM120_TensorOp_fp8_UMMA_gemm_with_blockwise(manifest, cuda_version, gemm_kind=GemmKind.GroupedBlockwiseUniversal3x)

###################################################################################################

def GenerateSM90_Conv3x(manifest, cuda_version,
                        log_indent_level: int = 0):
  """
  Generate CUTLASS 3 convolution kernel(s) for SM90.

  This is meant to be called from GenerateSM90.
  """
  log_debug_line('GenerateSM90_Conv3x', log_indent_level)
  log_indent_level = log_indent_level + 1

  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  minimum_compute_capability = 90
  maximum_compute_capability = 90

  spatial_dims = (2, 3)

  # MMA shapes (MMA_M, MMA_N, MMA_K):
  #
  # Different hardware MMA instructions may have different MMA shapes.
  # This function may generate kernels with different MMA shapes for
  # different data types, either because the hardware only supports
  # certain shapes for certain types, or for performance reasons
  # (CUTLASS doesn't need to generate all valid kernels for the
  # profiler library, just the best-performing ones).
  #
  # The kernel names refer to tile shapes (TILE_M, TILE_N, TILE_K)
  # instead of MMA shapes.  For SM >= 90 kernels, TILE_K = 4 * MMA_K,
  # where 4, the "number of MMA instructions per tile," is determined
  # through some combination of modeling and experiment.
  #
  # For performance on sm90, generally CUTLASS generates 64x128
  # instead of 128x64.
  mma_64x64x16  = ( 64,  64,  16)
  mma_64x64x8   = ( 64,  64,   8)

  num_mma_per_tile = 4

  # Cluster shapes (1, 1, 1) and (2, 2, 1) are valid,
  # but not included, because they tend not to perform as well.
  cluster_shapes = (
    (2, 1, 1),
    (1, 2, 1),
   )

  fp16 = DataType.f16
  bf16 = DataType.bf16
  fp32 = DataType.f32
  s8   = DataType.s8
  s32  = DataType.s32

  # When generating kernels, the usual way is to specify 4 types,
  # (A, B, Acc, C/D).  Tests instead have 5 types,
  # (ElementAct, ElementFlt, ElementOut, ElementAcc, ElementCompute),
  # where ElementCompute is also called 'epi_type',
  # and corresponds to the type of epilogue activations.
  # This script maps tests' 5 types to 4 types
  # by making ElementCompute the same as ElementOut.

  fp16_fp32_fp16_fp32 = {
    'a_type':   fp16, # ElementAct(ivation)
    'b_type':   fp16, # ElementF(i)lt(er)
    'c_type':   fp32, # ElementAcc
    'd_type':   fp32, # ElementOut (used only by CollectiveEpilogue)
    'acc_type': fp16, # ElementAcc
    'epi_type': fp32, # ElementCompute (used only by CollectiveEpilogue)
    'alignment_A': 8, # tma alignment elements of A
    'alignment_B': 8, # tma alignment elements of B
    'alignment_C': 4, # tma alignment elements of C
  }
  fp16_fp32_fp32_fp32 = {
    'a_type':   fp16,
    'b_type':   fp16,
    'c_type':   fp32,
    'd_type':   fp32,
    'acc_type': fp32,
    'epi_type': fp32,
    'alignment_A': 8,
    'alignment_B': 8,
    'alignment_C': 4,
  }
  fp32_fp32_fp32_fp32 = {
    'a_type':   fp32,
    'b_type':   fp32,
    'c_type':   fp32,
    'd_type':   fp32,
    'acc_type': fp32,
    'epi_type': fp32,
    'alignment_A': 4,
    'alignment_B': 4,
    'alignment_C': 4,
  }
  s8_s32_s32_s32 = {
    'a_type':     s8,
    'b_type':     s8,
    'c_type':    s32,
    'd_type':    s32,
    'acc_type':  s32,
    'epi_type':  s32,
    'alignment_A': 16,
    'alignment_B': 16,
    'alignment_C': 4,
  }

  # Other NVIDIA libraries may have the habit of specifying data types like this.
  bf16bf16_bf16f32_f32 = {
    'a_type':   bf16,
    'b_type':   bf16,
    'c_type':   fp32,
    'd_type':   fp32,
    'acc_type': fp32,
    'epi_type': fp32,
    'alignment_A': 8,
    'alignment_B': 8,
    'alignment_C': 4,
  }
  f16f16_f16f16_f16 = {
    'a_type':   fp16,
    'b_type':   fp16,
    'c_type':   fp16,
    'd_type':   fp16,
    'acc_type': fp16,
    'epi_type': fp16,
    'alignment_A': 8,
    'alignment_B': 8,
    'alignment_C': 8,
  }
  f16f16_f16f32_f32 = {
    'a_type':   fp16,
    'b_type':   fp16,
    'c_type':   fp16,
    'd_type':   fp16,
    'acc_type': fp32,
    'epi_type': fp32,
    'alignment_A': 8,
    'alignment_B': 8,
    'alignment_C': 8,
  }
  f32f32_tf32f32_f32 = fp32_fp32_fp32_fp32

  i8i8_i8i32_f32 = {
    'a_type':     s8,
    'b_type':     s8,
    'c_type':    s32,
    'd_type':    s32,
    'acc_type':  s32,
    'epi_type':  s32,
    'alignment_A': 16,
    'alignment_B': 16,
    'alignment_C': 4,
  }

  # Each element in the outermost iterable is one combination of
  #
  # (ConvKind, spatial_dimension, data_types, byte_alignments, mma_sizes, cluster_sizes)
  #
  # for which to generate a kernel.  spatial_dimension is the spatial
  # dimension of the convolution: either 1, 2, or 3.  byte_alignments
  # is a triple of required minimum byte alignments for A, B, and C.
  #
  # Note that itertools functions produce a single-pass generator.
  # The code doesn't need a multipass iterable, but if one did, one
  # could call `tuple` or `list` on the generator.
  #
  # While this happens to use the same cluster sizes for each element,
  # the code doesn't require that.  Different convolution kinds, data
  # types, or mma sizes might have different optimal cluster sizes.
  combinations_of_parameters = chain(
    # The following are all the kernels exercised in the unit tests.
    # Please try to keep in sync with the unit tests.
    product(
      (
        ConvKind.Fprop,
      ),
      spatial_dims,
      (
        fp16_fp32_fp16_fp32,
        fp16_fp32_fp32_fp32,
        s8_s32_s32_s32,
      ),
      (
        mma_64x64x16,
      ),
      cluster_shapes
    ),
    product(
      (
        ConvKind.Fprop,
      ),
      spatial_dims,
      (
        fp32_fp32_fp32_fp32,
      ),
      (
        mma_64x64x8,
      ),
      cluster_shapes
    ),
    product(
      (
        ConvKind.Dgrad,
        ConvKind.Wgrad
      ),
      spatial_dims,
      (
        fp16_fp32_fp16_fp32,
        fp16_fp32_fp32_fp32,
      ),
      (
        mma_64x64x16,
      ),
      cluster_shapes
    ),
    # Kernels not necessarily in the unit tests, but used elsewhere
    # and thus useful to have generated for profiling.  They may
    # duplicate kernels above.  All of them are 2-D.  In general,
    # CUTLASS prefers 64 x 128 to 128 x 64 on sm90, even if the
    # hardware permits 128 x 64.
    (
      # Fprop
      #
      # bf16bf16_bf16f32_f32
      #
      # cluster shape (2, 1, 1)
      #
      (ConvKind.Fprop, 2, bf16bf16_bf16f32_f32, (128, 256,  8), (2, 1, 1)),
      (ConvKind.Fprop, 2, bf16bf16_bf16f32_f32, (128, 256, 16), (2, 1, 1)),
      (ConvKind.Fprop, 2, bf16bf16_bf16f32_f32, (256, 128,  8), (2, 1, 1)),
      (ConvKind.Fprop, 2, bf16bf16_bf16f32_f32, (256, 128, 16), (2, 1, 1)),
      #
      # f16f16_f16f16_f16
      #
      # cluster shape (1, 1, 1)
      #
      (ConvKind.Fprop, 2,    f16f16_f16f16_f16, ( 64,  64,  8), (1, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f16_f16, ( 64,  64, 16), (1, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f16_f16, ( 64, 128,  8), (1, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f16_f16, ( 64, 128, 16), (1, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f16_f16, ( 64, 256,  8), (1, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f16_f16, ( 64, 256, 16), (1, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f16_f16, (128, 128,  8), (1, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f16_f16, (128, 128, 16), (1, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f16_f16, (128, 256,  8), (1, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f16_f16, (128, 256, 16), (1, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f16_f16, (256,  64,  8), (1, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f16_f16, (256,  64, 16), (1, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f16_f16, (256, 128,  8), (1, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f16_f16, (256, 128, 16), (1, 1, 1)),
      #
      # f16f16_f16f32_f32
      #
      # cluster shape (2, 1, 1)
      #
      (ConvKind.Fprop, 2,    f16f16_f16f32_f32, (128, 192,  8), (2, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f32_f32, (128, 192, 16), (2, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f32_f32, (128, 256,  8), (2, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f32_f32, (128, 256, 16), (2, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f32_f32, (256,  96,  8), (2, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f32_f32, (256,  96, 16), (2, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f32_f32, (256, 128,  8), (2, 1, 1)),
      (ConvKind.Fprop, 2,    f16f16_f16f32_f32, (256, 128, 16), (2, 1, 1)),
      #
      # f32f32_tf32f32_f32
      #
      # cluster shape (2, 1, 1)
      #
      (ConvKind.Fprop, 2,   f32f32_tf32f32_f32, (128, 192,  8), (2, 1, 1)),
      (ConvKind.Fprop, 2,   f32f32_tf32f32_f32, (128, 256,  8), (2, 1, 1)),
      (ConvKind.Fprop, 2,   f32f32_tf32f32_f32, (256, 128,  8), (2, 1, 1)),
      (ConvKind.Fprop, 2,   f32f32_tf32f32_f32, (256,  96,  8), (2, 1, 1)),
      #
      # i8i8_i8i32_f32
      #
      # cluster shape (2, 1, 1)
      #
      (ConvKind.Fprop, 2,       i8i8_i8i32_f32, (128, 256, 16), (2, 1, 1)),
      (ConvKind.Fprop, 2,       i8i8_i8i32_f32, (128, 256, 32), (2, 1, 1)),
      (ConvKind.Fprop, 2,       i8i8_i8i32_f32, (256, 128, 16), (2, 1, 1)),
      (ConvKind.Fprop, 2,       i8i8_i8i32_f32, (256, 128, 32), (2, 1, 1)),
      #
      # Dgrad
      #
      # bf16bf16_bf16f32_f32
      #
      # cluster shape (2, 1, 1)
      #
      (ConvKind.Dgrad, 2, bf16bf16_bf16f32_f32, (128, 256,  8), (2, 1, 1)),
      (ConvKind.Dgrad, 2, bf16bf16_bf16f32_f32, (128, 256, 16), (2, 1, 1)),
      (ConvKind.Dgrad, 2, bf16bf16_bf16f32_f32, (256, 128,  8), (2, 1, 1)),
      (ConvKind.Dgrad, 2, bf16bf16_bf16f32_f32, (256, 128, 16), (2, 1, 1)),
      #
      # f16f16_f16f16_f16
      #
      # cluster shape (1, 1, 1)
      #
      (ConvKind.Dgrad, 2,    f16f16_f16f16_f16, ( 64,  64,  8), (1, 1, 1)),
      (ConvKind.Dgrad, 2,    f16f16_f16f16_f16, ( 64,  64, 16), (1, 1, 1)),
      (ConvKind.Dgrad, 2,    f16f16_f16f16_f16, ( 64, 128,  8), (1, 1, 1)),
      (ConvKind.Dgrad, 2,    f16f16_f16f16_f16, ( 64, 128, 16), (1, 1, 1)),
      (ConvKind.Dgrad, 2,    f16f16_f16f16_f16, ( 64, 256,  8), (1, 1, 1)),
      (ConvKind.Dgrad, 2,    f16f16_f16f16_f16, ( 64, 256, 16), (1, 1, 1)),
      (ConvKind.Dgrad, 2,    f16f16_f16f16_f16, (128, 128,  8), (1, 1, 1)),
      (ConvKind.Dgrad, 2,    f16f16_f16f16_f16, (128, 128, 16), (1, 1, 1)),
      (ConvKind.Dgrad, 2,    f16f16_f16f16_f16, (128, 256,  8), (1, 1, 1)),
      (ConvKind.Dgrad, 2,    f16f16_f16f16_f16, (128, 256, 16), (1, 1, 1)),
      (ConvKind.Dgrad, 2,    f16f16_f16f16_f16, (256,  64,  8), (1, 1, 1)),
      (ConvKind.Dgrad, 2,    f16f16_f16f16_f16, (256,  64, 16), (1, 1, 1)),
      (ConvKind.Dgrad, 2,    f16f16_f16f16_f16, (256, 128,  8), (1, 1, 1)),
      (ConvKind.Dgrad, 2,    f16f16_f16f16_f16, (256, 128, 16), (1, 1, 1)),
      #
      # f16f16_f16f32_f32
      #
      # cluster shape (2, 1, 1)
      #
      (ConvKind.Dgrad, 2,    f16f16_f16f32_f32, (128, 256,  8), (2, 1, 1)),
      (ConvKind.Dgrad, 2,    f16f16_f16f32_f32, (128, 256, 16), (2, 1, 1)),
      (ConvKind.Dgrad, 2,    f16f16_f16f32_f32, (256, 128,  8), (2, 1, 1)),
      (ConvKind.Dgrad, 2,    f16f16_f16f32_f32, (256, 128, 16), (2, 1, 1)),
    ),
  )

  # SM >= 90 kernels don't actually use warp_count, but the
  # TileDescription class needs it.  The 4 in the default
  # warp_count has nothing to do with num_mma_per_tile.
  warp_count = [4, 1, 1]

  stages = 0 # zero means "deduce the number of stages automatically"

  mainloop_schedule = KernelScheduleType.ImplicitTmaWarpSpecializedSm90
  epilogue_schedule = EpilogueScheduleType.TmaWarpSpecialized
  schedule_pairs = (
    (mainloop_schedule, epilogue_schedule),
  )
  tile_schedulers = (
    TileSchedulerType.Default, # -> void
  )

  def make_math_instruction(data_types: Dict[str, DataType],
                            mma_shape: Tuple[int, int, int]) -> MathInstruction:
    default_opcode = OpcodeClass.TensorOp
    default_math_op = MathOperation.multiply_add
    return MathInstruction(
      mma_shape,
      data_types['a_type'], data_types['b_type'], data_types['c_type'],
      default_opcode,
      default_math_op
    )

  for (conv_kind, spatial_dim, data_types, mma_shape, cluster_shape) in combinations_of_parameters:
    math_inst = make_math_instruction(data_types, mma_shape)
    tile_shape = (mma_shape[0], mma_shape[1], num_mma_per_tile * mma_shape[2])
    tile_description = TileDescription(tile_shape, stages, warp_count, math_inst,
      minimum_compute_capability, maximum_compute_capability, cluster_shape)
    assert(isinstance(spatial_dim, int))
    dims_and_alignments = (
      (
        (spatial_dim, data_types['alignment_A']),
        (spatial_dim, data_types['alignment_B']),
        (spatial_dim, data_types['alignment_C']),
      ),
    )
    CreateConvOperator3x(manifest,
                         dims_and_alignments = dims_and_alignments,
                         tile_descriptions = [tile_description],
                         data_types = data_types,
                         schedule_pairs = schedule_pairs,
                         tile_schedulers = tile_schedulers,
                         conv_kind = conv_kind,
                         log_indent_level = log_indent_level)

def GenerateSM90(manifest, cuda_version):
  GenerateSM90_TensorOp_16b_WGMMA_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_16b_WGMMA_alignx_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_tf32_WGMMA_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_tf32_WGMMA_alignx_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_int8_WGMMA_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_int8_WGMMA_alignx_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_fp8_WGMMA_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_fp8_WGMMA_alignx_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_mixed_dtype_WGMMA_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_1684(manifest, cuda_version)
  GenerateSM90_TensorOp_16b_WGMMA_gemm(manifest, cuda_version, gemm_kind=GemmKind.GroupedUniversal3x)
  GenerateSM90_TensorOp_fp8_WGMMA_gemm(manifest, cuda_version, gemm_kind=GemmKind.GroupedUniversal3x)
  GenerateSM90_TensorOp_1684_complex(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_complex_gaussian(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_rank_k(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_rank_k_complex(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_rank_k_complex_gaussian(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_trmm(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_trmm_complex(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_trmm_complex_gaussian(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_symm(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_symm_complex(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_symm_complex_gaussian(manifest, cuda_version)
  GenerateSM90_Conv3x(manifest, cuda_version)
  GenerateSM90_SparseTensorOp_16b_WGMMA_gemm(manifest, cuda_version)
  GenerateSM90_SparseTensorOp_tf32_WGMMA_gemm(manifest, cuda_version)
  GenerateSM90_SparseTensorOp_int8_WGMMA_gemm(manifest, cuda_version)
  GenerateSM90_SparseTensorOp_fp8_WGMMA_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_fp8_WGMMA_gemm_with_blockwise(manifest, cuda_version)
  GenerateSM90_TensorOp_fp8_WGMMA_gemm_with_blockwise(manifest, cuda_version, gemm_kind=GemmKind.GroupedBlockwiseUniversal3x)

###################################################################################################

def numeric_log_level(log_level: str) -> int:
  """
  Converts the string identifier of the log level
  into the numeric identifier used in setting the log level.

  :param x: string representation of log level (e.g., 'INFO', 'DEBUG')
  :type x: str

  :return: numeric representation of log level
  :rtype: int
  """
  numeric_level = getattr(logging, log_level.upper(), None)
  if not isinstance(numeric_level, int):
    raise ValueError(f'Invalid log level: {log_level}')
  return numeric_level

# This function for defining the ArgumentParser is used to make it easy for the CUTLASS Python interface
# to leverage the functionality in this file without running this script via a shell prompt.
def define_parser():
  parser = argparse.ArgumentParser(description="Generates device kernel registration code for CUTLASS Kernels")
  parser.add_argument("--operations", default="all", help="Specifies the operation to generate (gemm, all)")
  parser.add_argument("--build-dir", default=".", required=False, help="CUTLASS top-level build directory")
  parser.add_argument("--curr-build-dir", default=".", help="CUTLASS current build directory. cmake files will be emitted in this directory")
  parser.add_argument("--generator-target", default='library', help="Target of CUTLASS Library Generator.")
  parser.add_argument("--architectures", default='53;60;61;70;75;80;90;100', help="Target compute architectures")
  parser.add_argument("--kernels", default='', help='Comma-delimited list to filter kernels by name.  ' +
                      'Specifying this as \"all\" includes ALL the kernels, ' +
                      'while not specifying this includes only the default set of kernels.')
  parser.add_argument("--ignore-kernels", default='', help='Comma-delimited list of kernels ' +
                      'to exclude from build.  For backwards compatibility reasons, ' +
                      'this option only takes effect if --kernels is set to a nonempty value.')
  parser.add_argument("--exclude-kernels", default='', help='Comma-delimited list of kernels ' +
                      'to exclude from build.  In contrast to --ignore-kernels, ' +
                      'this option always takes effect, ' +
                      'whether or not --kernels is set to a nonempty value.  ' +
                      'It also can exclude kernels from the filter file ' +
                      '(see --kernel-filter-file option below).')
  parser.add_argument("--filter-by-cc", default='True', type=str, help='If enabled, kernels whose compute capability range is not satisfied by the build target are excluded.')
  parser.add_argument("--cuda-version", default="11.0.0", help="Semantic version string of CUDA Toolkit")
  parser.add_argument('--kernel-filter-file',   type=str, default=None, required=False, help='Full path of filter file')
  parser.add_argument('--heuristics-problems-file',   type=str, default=None, required=False, help='Full path of heuristics problem size description file, as a json list')
  parser.add_argument('--heuristics-testlist-file',   type=str, default=None, required=False, help='Full path of heuristics testlist CSV file, to be passed to cutlass_profiler')
  parser.add_argument('--heuristics-gpu',   type=str, default=None, required=False, help='GPU to use for evaluating heuristics offline. None or `auto` to autodetect using cuda', choices=['', 'auto', 'H100_SXM', 'H100_PCIE', 'H100_NVL', 'H200_SXM', 'H20_SXM', 'B200', 'GB200_NVL', 'RTX_5080', 'RTX_5090', 'RTX_PRO_6000'])
  parser.add_argument('--heuristics-configs-per-problem',   type=int, default=10, required=False, help='Number of kernel configs to generate for each problem in the problem list')
  parser.add_argument('--heuristics-restrict-kernels', action='store_true', help='Restrict heuristics mode to use only the default set of kernels emitted by generator.py')
  parser.add_argument('--selected-kernel-list',   type=str, default=None, required=False,
                        help='Specify the output log file containing all enabled kernels in this build')
  parser.add_argument("--interface-dir", default=None, required=False, help="Interface header to kernels")
  parser.add_argument("--disable-full-archs-compilation", action="store_true", required=False, help="Disable compilation for every archs in --architectures")
  parser.add_argument("--log-level", default='info', type=numeric_log_level, required=False,
                      help='Logging level to be used by the generator script')
  parser.add_argument('--instantiation-level', type=str, default="", required=False, help="Instantiation level for SM90 kernels. Set to `max` and make sure `--kernels` is not empty to generate all possible configurations.")
  _add_package_disablement_flag(parser)
  return parser


if __name__ == "__main__":
  parser = define_parser()
  args = parser.parse_args()

  # Set the logging level based on the user-provided `--log-level` command-line option
  logging.basicConfig(level=args.log_level)

  manifest = Manifest(args)

  archs = args.architectures.split(';')

  if args.heuristics_problems_file:
    filter_manifest_and_write_heuristics_file(manifest, args)

  GenerateSM50(manifest, args.cuda_version)
  GenerateSM60(manifest, args.cuda_version)
  GenerateSM61(manifest, args.cuda_version)
  GenerateSM70(manifest, args.cuda_version)
  GenerateSM75(manifest, args.cuda_version)
  GenerateSM80(manifest, args.cuda_version)
  GenerateSM89(manifest, args.cuda_version)
  GenerateSM90(manifest, args.cuda_version)

  blackwell_arch_list = [
    "100a", "100f",
    "101a", "101f",
    "103a", "103f",
    "110a", "110f",
    "120a", "120f",
    "121a", "121f",
  ]
  blackwell_enabled_arch = any(arch in blackwell_arch_list for arch in archs)
  if blackwell_enabled_arch:
    GenerateSM100(manifest, args.cuda_version)
    GenerateSM120(manifest, args.cuda_version)

  if 'library' in args.generator_target.split(','):
    manifest.emit(GeneratorTarget.Library)

  if 'kernel_testlist_l0' in args.generator_target.split(','):
    emit_gemm_kernel_testlist(manifest, args.curr_build_dir, args.architectures, "functional_L0")

  if 'kernel_testlist_l1' in args.generator_target.split(','):
    emit_gemm_kernel_testlist(manifest, args.curr_build_dir, args.architectures, "functional_L1")
  
  if args.selected_kernel_list is not None:
    if len(manifest.selected_kernels) > 0:
      with open(args.selected_kernel_list, 'w') as file_writer:
        for line in manifest.selected_kernels:
          file_writer.write("%s\n" % line)

###################################################################################################

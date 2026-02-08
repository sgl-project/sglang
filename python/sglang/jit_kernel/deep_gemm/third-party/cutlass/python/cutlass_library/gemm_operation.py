
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
Utilities for emitting GEMM kernels
"""

import collections
import enum
import functools
import logging
import operator
import os.path
import shutil

try:
  import builtins
  if hasattr(builtins, "CUTLASS_IGNORE_PACKAGE") and CUTLASS_IGNORE_PACKAGE == True:
    raise ImportError("Disabling attempt to import cutlass_library")
  from cutlass_library.library import *
except ImportError:
  from library import *

_LOGGER = logging.getLogger(__name__)

###################################################################################################
#
# Data structure modeling a GEMM operation
#
###################################################################################################

#
class GemmOperation:
  #
  def __init__(self, gemm_kind, arch, tile_description, A, B, C, element_epilogue, \
      epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity8, D = None,
      kernel_schedule = KernelScheduleType.ScheduleAuto, epilogue_schedule = EpilogueScheduleType.ScheduleAuto,
      tile_scheduler = TileSchedulerType.Default, mixed_input_mode = None, mixed_input_shuffle = False,
      ScaleFactorA = None, ScaleFactorB = None, ScaleFactorD = None, 
      ScaleFactorMVecSize = None, ScaleFactorNVecSize = None, ScaleFactorKVecSize = None):

    kinds_3x = {
      GemmKind.Universal3x,
      GemmKind.SparseUniversal3x,
      GemmKind.BlockScaledUniversal3x, 
      GemmKind.GroupedUniversal3x,
      GemmKind.GroupedBlockScaledUniversal3x,
      GemmKind.BlockwiseUniversal3x,
      GemmKind.GroupedBlockwiseUniversal3x,
    }
    self.is_3x = gemm_kind in kinds_3x
    self.prefix = "3x" if self.is_3x else ""
    self.operation_kind = OperationKind.Gemm
    self.arch = arch
    self.tile_description = tile_description
    self.gemm_kind = gemm_kind
    self.A = A
    self.B = B
    self.C = C
    self.D = D

    if is_block_scaled(gemm_kind):
      self.ScaleFactorA = ScaleFactorA
      self.ScaleFactorB = ScaleFactorB
      self.ScaleFactorD = ScaleFactorD["tensor"]
      self.ScaleFactorVectorSize = ScaleFactorD["vector_size"]

    if is_blockwise(gemm_kind):
      self.ScaleFactorMVecSize = ScaleFactorMVecSize
      self.ScaleFactorNVecSize = ScaleFactorNVecSize
      self.ScaleFactorKVecSize = ScaleFactorKVecSize

    if self.D == None:
      self.D = self.C

    if not self.is_3x:
      assert(kernel_schedule == KernelScheduleType.ScheduleAuto)
      assert(epilogue_schedule == EpilogueScheduleType.ScheduleAuto)
    self.kernel_schedule = kernel_schedule
    self.epilogue_schedule = epilogue_schedule
    self.element_epilogue = element_epilogue
    self.epilogue_functor = epilogue_functor

    if self.is_3x and epilogue_functor == EpilogueFunctor.LinearCombination:
      self.epilogue_functor = EpilogueFunctor3x.LinearCombination

    self.swizzling_functor = swizzling_functor
    self.tile_scheduler = tile_scheduler

    # Only enable mixed input mode and mixed input shuffle for Hopper
    self.mixed_input_mode = None
    if self.is_mixed_input() and self.arch >= 90 and self.arch < 100:
      self.mixed_input_mode = mixed_input_mode
    self.mixed_input_shuffle = (self.mixed_input_mode is not None) and mixed_input_shuffle

  #
  def is_complex(self):
    complex_operators = [
      MathOperation.multiply_add_complex,
      MathOperation.multiply_add_complex_gaussian,
      MathOperation.multiply_add_complex_fast_f32
    ]
    return self.tile_description.math_instruction.math_operation in complex_operators

  #
  def is_mixed_input(self):
    return self.A.element != self.B.element

  #
  def is_planar_complex(self):
    return self.gemm_kind in (GemmKind.PlanarComplex, GemmKind.PlanarComplexArray)

  #
  def accumulator_type(self):
    accum = self.tile_description.math_instruction.element_accumulator

    if self.is_complex():
      return get_complex_from_real(accum)

    return accum

  #
  def short_math_name(self):
    if self.tile_description.math_instruction.math_operation == MathOperation.multiply_add_complex_gaussian:
      return "g%s" % ShortDataTypeNames[self.accumulator_type()]
    return ShortDataTypeNames[self.accumulator_type()]


  #
  def core_name(self):
    ''' The basic operation kind is prefixed with a letter indicating the accumulation type. '''

    inst_shape = ''
    inst_operation = ''
    intermediate_type = ''

    math_operations_map = {
      MathOperation.xor_popc: 'xor',
      MathOperation.and_popc: 'and',
      MathOperation.multiply_add_fast_accum: 'fastaccum',
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

      inst_shape = "{0}{1}{2}".format(*tuple(self.tile_description.math_instruction.instruction_shape)) if not self.is_3x else ""

      inst_shape += math_op_string

      if self.tile_description.math_instruction.element_a != self.A.element and \
        self.tile_description.math_instruction.element_a != self.tile_description.math_instruction.element_accumulator:
        intermediate_type = DataTypeNames[self.tile_description.math_instruction.element_a]

    short_math_name = self.short_math_name() if not self.is_3x else ""

    return "%s%s%s%s" % (short_math_name, inst_shape, intermediate_type, GemmKindNames[self.gemm_kind])

  # Generates a string representing the MMA instruction.
  def extended_name(self):
    ''' Append data types if they differ from compute type. '''
    element_sfa = ""
    element_sfb = ""
    if self.is_complex():
      extended_name = "${core_name}"
    else:
      if self.is_mixed_input():
        extended_name = "${core_name}_${element_a}_${element_b}"
        if self.C.element != self.tile_description.math_instruction.element_accumulator:
          extended_name = "${element_c}_" + extended_name
      elif is_blockwise(self.gemm_kind):
        extended_name = "${core_name}_${element_sfa}x${element_a}_${element_sfb}x${element_b}"
        element_sfa = DataTypeNames[self.accumulator_type()]
        element_sfb = DataTypeNames[self.accumulator_type()]
      else:
        extended_name = "${core_name}"
        if self.C.element != self.tile_description.math_instruction.element_accumulator:
          extended_name = "${element_c}_" + extended_name
        if self.A.element != self.tile_description.math_instruction.element_accumulator:
          extended_name += "_${element_a}"

    extended_name = SubstituteTemplate(extended_name, {
      'element_a': DataTypeNames[self.A.element],
      'element_sfa' : element_sfa,
      'element_b': DataTypeNames[self.B.element],
      'element_sfb' : element_sfb,
      'element_c': DataTypeNames[self.C.element],
      'core_name': self.core_name()
      })

    return extended_name

  #
  def mixed_input_mode_name(self):
    mode_name_mapping = {
      MixedInputMode.ConvertOnly: "_cvt",
      MixedInputMode.ScaleOnly: "_scl",
      MixedInputMode.ScaleWithZeroPoint: "_sclzr"
    }
    mode_name = mode_name_mapping.get(self.mixed_input_mode, "")
    if self.mixed_input_shuffle:
      mode_name = mode_name + "_shfl"
    return mode_name

  def extended_name_3x(self):
    '''Generates a string representing the MMA atom. Assumes accumulator type is C type.'''
    extended_name = "{core_name}_{element_a}_{element_b}_{element_acc}_{element_c}_{element_d}".format(
      element_a = DataTypeNames[self.A.element],
      element_b = DataTypeNames[self.B.element],
      element_acc = DataTypeNames[self.accumulator_type()],
      element_c = DataTypeNames[self.C.element],
      element_d = DataTypeNames[self.D.element],
      core_name = self.core_name())

    if is_block_scaled(self.gemm_kind):
      d_type_names = DataTypeNames[self.D.element]

      if self.ScaleFactorD.element != DataType.void:
        d_type_names = DataTypeNames[self.ScaleFactorD.element] + "x" + d_type_names

      extended_name = "{core_name}_{element_sfa}x{element_a}_{element_sfb}x{element_b}_{element_acc}_{element_c}_{element_d}".format(
        element_sfa = DataTypeNames[self.ScaleFactorA],
        element_a = DataTypeNames[self.A.element],
        element_sfb = DataTypeNames[self.ScaleFactorB],
        element_b = DataTypeNames[self.B.element],
        element_acc = DataTypeNames[self.accumulator_type()],
        element_c = DataTypeNames[self.C.element],
        element_d = d_type_names,
        core_name = self.core_name())

    if is_blockwise(self.gemm_kind):
      d_type_names = DataTypeNames[self.D.element]

      extended_name = "{core_name}_{sfvec_m_size}x{sfvec_k_size}{element_sfa}x{element_a}_{sfvec_n_size}x{sfvec_k_size}{element_sfb}x{element_b}_{element_acc}_{element_c}_{element_d}".format(
        element_sfa = DataTypeNames[self.accumulator_type()],
        element_a = DataTypeNames[self.A.element],
        element_sfb = DataTypeNames[self.accumulator_type()],
        element_b = DataTypeNames[self.B.element],
        element_acc = DataTypeNames[self.accumulator_type()],
        element_c = DataTypeNames[self.C.element],
        element_d = d_type_names,
        sfvec_m_size = self.ScaleFactorMVecSize,
        sfvec_n_size = self.ScaleFactorNVecSize,
        sfvec_k_size = self.ScaleFactorKVecSize,
        core_name = self.core_name())

    if self.mixed_input_mode != None:
      extended_name = extended_name + self.mixed_input_mode_name()
    return extended_name

  def datatype_name_3x(self):
    '''Generates a string representing the MMA atom. Assumes accumulator type is C type.'''
    datatype_name = "{element_a}_{element_b}_{element_acc}_{element_c}_{element_d}".format(
      element_a = DataTypeNames[self.A.element],
      element_b = DataTypeNames[self.B.element],
      element_acc = DataTypeNames[self.accumulator_type()],
      element_c = DataTypeNames[self.C.element],
      element_d = DataTypeNames[self.D.element])
    return datatype_name

  # Generates a short string representing the AB layout tags (e.g. nt or tn)
  def layout_name(self):
    if self.is_complex() or self.is_planar_complex():
      return "%s%s" % (
        ShortComplexLayoutNames[(self.A.layout, self.A.complex_transform)],
        ShortComplexLayoutNames[(self.B.layout, self.B.complex_transform)]
      )
    return "%s%s" % (ShortLayoutTypeNames[self.A.layout], ShortLayoutTypeNames[self.B.layout])

  # Generates a short string representing the ABC layout tags (e.g. ntn or tnn)
  def layout_name_3x(self):
    if self.is_complex() or self.is_planar_complex():
      return "{}{}{}".format(
        ShortComplexLayoutNames[(self.A.layout, self.A.complex_transform)],
        ShortComplexLayoutNames[(self.B.layout, self.B.complex_transform)],
        ShortComplexLayoutNames[(self.C.layout, self.C.complex_transform)])
    else:
      return "{}{}{}".format(
        ShortLayoutTypeNames[self.A.layout],
        ShortLayoutTypeNames[self.B.layout],
        ShortLayoutTypeNames[self.C.layout])

  # Generates a short string representing underlying kernel schedule type
  def kernel_schedule_name_3x(self):
    return KernelScheduleSuffixes[self.kernel_schedule]

  # Generates a short string representing underlying epilogue schedule type
  def epilogue_schedule_name_3x(self):

    if is_block_scaled(self.gemm_kind):
      if self.ScaleFactorD.element != DataType.void:
        return EpilogueScheduleSuffixes[self.epilogue_schedule] + "_epiVs" + str(self.ScaleFactorVectorSize)+ShortLayoutTypeNames[self.ScaleFactorD.layout]
    
    return EpilogueScheduleSuffixes[self.epilogue_schedule]

  # Generate a short string representing the operation class
  def opcode_class_name(self):
    return OpcodeClassNames[self.tile_description.math_instruction.opcode_class]

  def get_collective_tile_shape(self):
    """
    Get the tile shape passed to the collective builder.
    On Blackwell, this is different than the operation.tile_description.tile_shape.
    """
    is_sm100_kernel = (self.arch == 100 or self.arch == 103)
    if not is_sm100_kernel:
      return self.tile_description.tile_shape

    opcode_class_main = self.tile_description.math_instruction.opcode_class
    instruction_shape = self.tile_description.math_instruction.instruction_shape
    tile_shape_m, tile_shape_n, tile_shape_k = self.tile_description.tile_shape
    if opcode_class_main in [OpcodeClass.TensorOp, OpcodeClass.BlockScaledTensorOp, OpcodeClass.SparseTensorOp]:
      tile_shape_m = instruction_shape[0]
      tile_shape_n = instruction_shape[1]
    return (tile_shape_m, tile_shape_n, tile_shape_k)

  # Generates the full kernel function name
  def procedural_name(self):
    return self._procedural_name

  @functools.cached_property
  def _procedural_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
    opcode_class_name = OpcodeClassNames[self.tile_description.math_instruction.opcode_class]
    if self.arch >= 90:
      kernel_name_template = "cutlass{p}_sm{ar}_{op}_{ex}{ct}{cs}_{l}_{s}_align{al}{t}{k}{e}"
      tile_shape = self.get_collective_tile_shape()
      return kernel_name_template.format(
          p = self.prefix,
          ar = self.arch,
          op = opcode_class_name,
          ex = self.extended_name_3x(),
          ct = '_' + 'x'.join([str(i) for i in tile_shape]) if tile_shape[0] > 0 else "",
          cs = '_' + 'x'.join([str(i) for i in self.tile_description.cluster_shape]),
          l = self.tile_description.stages,
          s = self.layout_name_3x(),
          al = str(max(self.A.alignment, self.B.alignment)),
          t = TileSchedulerSuffixes[self.tile_scheduler],
          k = self.kernel_schedule_name_3x(),
          e = self.epilogue_schedule_name_3x())
    else:
      threadblock = self.tile_description.procedural_name()
      return "cutlass{p}_{op}_{ex}_{tb}_{l}_align{a}".format(
          p = self.prefix,
          op = opcode_class_name,
          ex = self.extended_name(),
          tb = threadblock,
          l = self.layout_name(),
          a = str(max(self.A.alignment, self.B.alignment)))

  #
  def configuration_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
    return self.procedural_name()

  def __hash__(self):
    return hash(self.configuration_name())

  def __eq__(self, other):
    return self.configuration_name() == other.configuration_name()

###################################################################################################
#
# Data structure modeling a grouped GEMM operation
#
###################################################################################################

#
class GroupedGemmOperation(GemmOperation):
  #
  def __init__(self, gemm_kind, arch, tile_description, A, B, C, element_epilogue, \
      epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity8, \
      scheduler_mode = GroupScheduleMode.Device):
    super().__init__(gemm_kind, arch, tile_description, A, B, C, element_epilogue, \
                     epilogue_functor, swizzling_functor)

    self.scheduler_mode = scheduler_mode

  #
  def procedural_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
    base = super().procedural_name()
    return SubstituteTemplate(
      base + "_schedule${schedule}",
      {
        'schedule': ShortGroupScheduleModeNames[self.scheduler_mode]
      })


###################################################################################################
#
# Emits single instances of a CUTLASS device-wide operator
#
###################################################################################################

#
class EmitGemmInstance:
  ''' Responsible for emitting a CUTLASS template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = []
    self.gemm_template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::device::Gemm<
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor},
    ${stages},
    ${align_a},
    ${align_b},
    false,
    ${math_operation}
    ${residual}
  >;
"""
    self.gemm_complex_template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::device::GemmComplex<
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor},
    ${stages},
    ${transform_a},
    ${transform_b},
    ${math_operation}
    ${residual}
  >;
"""

  #
  def instance_template(self):
    return """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<Operation_${operation_name}>("${operation_name}"));
${compile_guard_end}
"""

  #
  def emit(self, operation):

    warp_shape = [operation.tile_description.threadblock_shape[idx] // operation.tile_description.warp_count[idx] for idx in range(3)]

    epilogue_vector_length = int(min(operation.C.alignment * DataTypeSize[operation.C.element], 128) / DataTypeSize[operation.C.element])

    residual = ''

    values = {
      'operation_name': operation.procedural_name(),
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[operation.A.layout],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b': LayoutTag[operation.B.layout],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[operation.C.layout],
      'element_accumulator': DataTypeTag[operation.accumulator_type()],
      'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
      'arch': "cutlass::arch::Sm%d" % operation.arch,
      'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
      'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
      'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
      'warp_shape_m': str(warp_shape[0]),
      'warp_shape_n': str(warp_shape[1]),
      'warp_shape_k': str(warp_shape[2]),
      'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]),
      'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]),
      'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]),
      'epilogue_vector_length': str(epilogue_vector_length),
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'residual': residual
    }

    template = self.gemm_complex_template if operation.is_complex() else self.gemm_template

    return SubstituteTemplate(template, values)

###################################################################################################

class EmitSparseGemmInstance:
  ''' Responsible for emitting a CUTLASS template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = []
    self.gemm_template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::device::SparseGemm<
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor},
    ${stages},
    ${align_a},
    ${align_b},
    false,
    ${math_operation}
    ${residual}
  >;
"""

  #
  def instance_template(self):
    return """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<Operation_${operation_name}>("${operation_name}"));
${compile_guard_end}
"""

  #
  def emit(self, operation):

    warp_shape = [operation.tile_description.threadblock_shape[idx] // operation.tile_description.warp_count[idx] for idx in range(3)]

    epilogue_vector_length = int(min(operation.C.alignment * DataTypeSize[operation.C.element], 128) / DataTypeSize[operation.C.element])

    residual = ''

    values = {
      'operation_name': operation.procedural_name(),
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[operation.A.layout],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b': LayoutTag[operation.B.layout],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[operation.C.layout],
      'element_accumulator': DataTypeTag[operation.accumulator_type()],
      'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
      'arch': "cutlass::arch::Sm%d" % operation.arch,
      'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
      'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
      'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
      'warp_shape_m': str(warp_shape[0]),
      'warp_shape_n': str(warp_shape[1]),
      'warp_shape_k': str(warp_shape[2]),
      'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]),
      'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]),
      'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]),
      'epilogue_vector_length': str(epilogue_vector_length),
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'residual': residual
    }

    template = self.gemm_template

    return SubstituteTemplate(template, values)

###################################################################################################


#
class EmitGemmUniversalInstance:
  ''' Responsible for emitting a CUTLASS template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = [
      "cutlass/cutlass.h",
      "cutlass/numeric_types.h",
      "cutlass/arch/arch.h",
      "cutlass/arch/mma.h",
      "cutlass/layout/matrix.h",
      "cutlass/gemm/device/gemm.h",
      "cutlass/gemm/device/gemm_universal_adapter.h",
      "cutlass/gemm/kernel/default_gemm_universal.h",
    ]
    self.builtin_epilogue_functor_template = """
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >
"""
    self.gemm_template = """
// Gemm operator ${operation_name}
using ${operation_name}_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    ${element_b}, ${layout_b}, ${transform_b}, ${align_b},    // transposed B operand
    ${element_a}, ${layout_a}, ${transform_a}, ${align_a},    // transposed A operand
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor},
    ${swizzling_functor},
    ${stages},
    ${math_operation}
>::GemmKernel;

// Define named type
struct ${operation_name}${operation_suffix} :
  public ${operation_name}_base { };
"""
    self.gemm_template_interleaved = """
// Gemm operator ${operation_name}
using ${operation_name}_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    ${element_a}, ${layout_a}, ${transform_a}, ${align_a},
    ${element_b}, ${layout_b}, ${transform_b}, ${align_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor},
    ${swizzling_functor},
    ${stages},
    ${math_operation}
>::GemmKernel;

// Define named type
struct ${operation_name}${operation_suffix} :
  public ${operation_name}_base { };
"""

  #
  def instance_template(self):
    return """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<
      cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>
    >("${operation_name}"));
${compile_guard_end}
"""

  #
  def emit(self, operation):

    threadblock_shape = operation.tile_description.threadblock_shape
    warp_count = operation.tile_description.warp_count

    warp_shape = [threadblock_shape[idx] // warp_count[idx] for idx in range(3)]

    transpose_layouts = {
      LayoutType.ColumnMajor: LayoutType.RowMajor,
      LayoutType.RowMajor: LayoutType.ColumnMajor
    }

    if operation.A.layout in transpose_layouts.keys() and \
      operation.B.layout in transpose_layouts.keys() and \
      operation.C.layout in transpose_layouts.keys():

      instance_layout_A = transpose_layouts[operation.A.layout]
      instance_layout_B = transpose_layouts[operation.B.layout]
      instance_layout_C = transpose_layouts[operation.C.layout]

      gemm_template = self.gemm_template
    else:
      instance_layout_A, instance_layout_B, instance_layout_C = \
        (operation.A.layout, operation.B.layout, operation.C.layout)

      gemm_template = self.gemm_template_interleaved
    #

    # Support built-in epilogue functors or user-defined functions
    if isinstance(operation.epilogue_functor, enum.Enum):

      epilogue_vector_length = \
        min(operation.C.alignment * DataTypeSize[operation.C.element], 128) // DataTypeSize[operation.C.element]

      values = {
        'epilogue_vector_length': str(epilogue_vector_length),
        'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
        'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      }
      epilogue_functor = SubstituteTemplate(self.builtin_epilogue_functor_template, values)
    else:
      epilogue_functor = self.epilogue_functor.emit_declaration()
    #

    values = {
      'operation_name': operation.procedural_name(),
      'operation_suffix': self.operation_suffix,
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[instance_layout_A],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b': LayoutTag[instance_layout_B],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[instance_layout_C],
      'element_accumulator': DataTypeTag[operation.accumulator_type()],
      'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
      'arch': "cutlass::arch::Sm%d" % operation.arch,
      'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
      'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
      'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
      'warp_shape_m': str(warp_shape[0]),
      'warp_shape_n': str(warp_shape[1]),
      'warp_shape_k': str(warp_shape[2]),
      'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]),
      'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]),
      'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]),
      'epilogue_functor': epilogue_functor,
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation]
    }

    return SubstituteTemplate(gemm_template, values)


###################################################################################################

class EmitGemmUniversal3xInstance:
  ''' Responsible for emitting a CUTLASS 3.x template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = [
      "cutlass/cutlass.h",
      "cutlass/gemm/gemm.h",
      "cutlass/numeric_types.h",
      "cutlass/gemm/kernel/gemm_universal.hpp",
      "cutlass/gemm/collective/collective_builder.hpp",
      "cutlass/epilogue/collective/collective_builder.hpp",
      "cutlass/detail/blockwise_scale_layout.hpp",
    ]
    self.builtin_epilogue_functor_template = \
"""${epilogue_functor}<
      ${element_d},
      ${element_epilogue},
      ${element_c},
      ${element_epilogue}
    >"""

    self.gemm_template = """

using ${operation_name}_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    ${arch}, ${opcode_class_epi},
    cute::Shape<cute::_${tile_shape_m}, cute::_${tile_shape_n}, cute::_${tile_shape_k}>,
    cute::Shape<${cluster_shape_m}, ${cluster_shape_n}, ${cluster_shape_k}>,
    ${epi_tile_mn},
    ${element_accumulator}, ${element_epilogue},
    ${element_c}, ${layout_c}, ${align_c},
    ${element_d}, ${layout_d}, ${align_d},
    ${epilogue_schedule},
    ${epilogue_functor}
  >::CollectiveOp;

${mixed_dtype_prepare_code}
${blockwise_prepare_code}

using ${operation_name}_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    ${arch}, ${opcode_class_main},
    ${element_a}, ${layout_a}, ${align_a},
    ${element_b}, ${layout_b}, ${align_b},
    ${element_accumulator},
    cute::Shape<cute::_${tile_shape_m}, cute::_${tile_shape_n}, cute::_${tile_shape_k}>,
    cute::Shape<${cluster_shape_m}, ${cluster_shape_n}, ${cluster_shape_k}>,
    ${stages},
    ${kernel_schedule}
  >::CollectiveOp;

// Gemm operator ${operation_name}
using ${operation_name}_base = cutlass::gemm::kernel::GemmUniversal<
    ${problem_shape},
    ${operation_name}_mainloop,
    ${operation_name}_epilogue,
    ${tile_scheduler}>;

// Define named type
struct ${operation_name} :
  public ${operation_name}_base { };

"""
  #
  def instance_template(self):
    return """
${compile_guard_start}
  {
    using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>;
    manifest.append(
      new ${gemm_kind}<GemmKernel>("${operation_name}"));
  }
${compile_guard_end}
"""

  
  def emit_block_scale_epilogue_functor(self, operation):
    block_scaled_template = """
      ${epilogue_functor}<
        ${epi_vs},
        ${element_d},
        ${element_accumulator},
        ${element_sfd},
        ${layout_sfd},
        ${element_c},
        ${element_scalar}
      >
    """
    block_scaled_values = {
      'epi_vs'  : str(operation.ScaleFactorVectorSize),
      'element_d': str(DataTypeTag[operation.D.element]),
      'element_sfd': str(DataTypeTag[operation.ScaleFactorD.element]),
      'layout_sfd': LayoutTag[operation.ScaleFactorD.layout],
      'epilogue_functor': EpilogueFunctor3xTag[EpilogueFunctor3x.LinearCombinationBlockScaleFactor],
      'element_accumulator': str(DataTypeTag[operation.accumulator_type()]),
      'element_scalar': str(DataTypeTag[operation.accumulator_type()]),
      'element_c': str(DataTypeTag[operation.C.element]),
    }
    return SubstituteTemplate(block_scaled_template, block_scaled_values)
  

  @staticmethod
  def pointerize_if_grouped(operation, layout):
    return layout if not is_grouped(operation.gemm_kind) else layout + "* "

  @staticmethod
  def transform_layout_A_if_blockwise(operation, layout):
    layout_sfa = f"{operation.procedural_name()}_LayoutSFA"
    layout_sfa = layout_sfa if not is_grouped(operation.gemm_kind) else layout_sfa + "* "
    return layout if not is_blockwise(operation.gemm_kind) else f"cute::tuple<{layout}, {layout_sfa}>"

  @staticmethod
  def transform_layout_B_if_blockwise(operation, layout):
    layout_sfb = f"{operation.procedural_name()}_LayoutSFB"
    layout_sfb = layout_sfb if not is_grouped(operation.gemm_kind) else layout_sfb + "* "
    return layout if not is_blockwise(operation.gemm_kind) else f"cute::tuple<{layout}, {layout_sfb}>"

  @staticmethod
  def problem_shape(operation):
    gemm_shape_type = "cute::Shape<int,int,int,int>"
    grouped_gemm_shape_type = "cute::Shape<int,int,int>"
    grouped_gemm_shape_type = "cutlass::gemm::GroupProblemShape<" + grouped_gemm_shape_type + ">"

    return gemm_shape_type if not is_grouped(operation.gemm_kind) else grouped_gemm_shape_type

  def emit(self, operation):
    _LOGGER.debug("*** EmitGemmConfigurationLibrary::emit(operation)")
    _LOGGER.debug("***   operation.procedural_name(): " + operation.procedural_name())
    _LOGGER.debug("***   tile_shape: " + str(operation.tile_description.tile_shape))
    _LOGGER.debug("***   warp_count: " + str(operation.tile_description.warp_count))

    opcode_class_main = operation.tile_description.math_instruction.opcode_class
    opcode_class_epi = opcode_class_main
    
    tile_shape = operation.tile_description.tile_shape
    instruction_shape = operation.tile_description.math_instruction.instruction_shape
    cluster_m = operation.tile_description.cluster_shape[0]
    cluster_n = operation.tile_description.cluster_shape[1]
    cta_n = tile_shape[1] // cluster_n if cluster_n > 0 else tile_shape[1]
    tile_shape_m, tile_shape_n, tile_shape_k = operation.get_collective_tile_shape()
 
    # stage count set to zero indicates builder automatic stage selection
    if operation.tile_description.stages > 0:
      stage_count_string = f"cutlass::gemm::collective::StageCount<{str(operation.tile_description.stages)}>"
    elif opcode_class_main == OpcodeClass.SparseTensorOp and operation.arch == 100:
      stage_count_string = f"cutlass::gemm::collective::StageCountAutoCarveoutEpi<{str(operation.procedural_name())}_epilogue>"
    else:
      stage_count_string = f"cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename {str(operation.procedural_name())}_epilogue::SharedStorage))>"

    epi_tile_mn = "cutlass::epilogue::collective::EpilogueTileAuto"

    instance_layout_A, instance_layout_B, instance_layout_C , instance_layout_D = \
      (operation.A.layout, operation.B.layout, operation.C.layout, operation.D.layout)

    # 3.0 profiler integration only supports trivial epilogues for now
    epilogue_vector_length = 1

    # Support built-in epilogue functors or user-defined functions
    if isinstance(operation.epilogue_functor, enum.Enum):
      values = {
        'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
        'epilogue_functor': EpilogueFunctor3xTag[operation.epilogue_functor],
      }
      epilogue_functor = SubstituteTemplate(self.builtin_epilogue_functor_template, values)
      
      if is_block_scaled(operation.gemm_kind) and operation.ScaleFactorD.element != DataType.void:
        epilogue_functor =  self.emit_block_scale_epilogue_functor(operation)


    else:
      epilogue_functor = self.epilogue_functor.emit_declaration()

      if is_block_scaled(operation.gemm_kind) and operation.ScaleFactorD.element != DataType.void:
        epilogue_functor =  self.emit_block_scale_epilogue_functor(operation)

    #
    # Cutlass3x complex kernels' ElementA(B) is a tuple in collective mainloop builder, e.g. cute::tuple<Element, Transform>, Transform : cute::identity / cute::conjugate.
    element_a = DataTypeTag[operation.A.element] if not operation.is_complex() else f"cute::tuple<{str(DataTypeTag[operation.A.element])},{str(ComplexTransformTag3x[operation.A.complex_transform])}>"
    element_b = DataTypeTag[operation.B.element] if not operation.is_complex() else f"cute::tuple<{str(DataTypeTag[operation.B.element])},{str(ComplexTransformTag3x[operation.B.complex_transform])}>"
    epilogue_schedule_type = EpilogueScheduleTag[operation.epilogue_schedule]
    
    if opcode_class_main == OpcodeClass.BlockScaledTensorOp:
      grouped = is_grouped(operation.gemm_kind)
      if cta_n == 256 and operation.kernel_schedule == to_grouped_schedule(KernelScheduleType.Nvf4TmaWarpSpecialized1SmSm100, grouped):
        epi_tile_mn = "cute::Shape<cute::_128,cute::_64>"
        if is_tma_epilogue(operation.epilogue_schedule):
          epilogue_schedule_type = EpilogueScheduleTag[to_grouped_schedule(EpilogueScheduleType.TmaWarpSpecialized1Sm, grouped)]
      if cta_n == 256 and operation.kernel_schedule == to_grouped_schedule(KernelScheduleType.Nvf4TmaWarpSpecialized2SmSm100, grouped):
        epi_tile_mn = "cute::Shape<cute::_128,cute::_64>"
        if is_tma_epilogue(operation.epilogue_schedule):
          epilogue_schedule_type = EpilogueScheduleTag[to_grouped_schedule(EpilogueScheduleType.TmaWarpSpecialized2Sm, grouped)]
      # SM103 FP4 Ultra
      is_sm103_fp4_ultra_1sm_kernel_schedule = operation.kernel_schedule in [to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized1SmVs32Sm103, grouped),
                                                                             to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized1SmVs16Sm103, grouped),
                                                                             to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized1SmVs32Sm103DisablePrefetch, grouped),
                                                                             to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized1SmVs16Sm103DisablePrefetch, grouped),
                                                                             to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized1SmVs32Sm103TmaPrefetch, grouped),
                                                                             to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized1SmVs16Sm103TmaPrefetch, grouped)
                                                                             ]
      is_sm103_fp4_ultra_2sm_kernel_schedule = operation.kernel_schedule in [to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized2SmVs32Sm103, grouped),
                                                                             to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized2SmVs16Sm103, grouped),
                                                                             to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized2SmVs32Sm103DisablePrefetch, grouped),
                                                                             to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized2SmVs16Sm103DisablePrefetch, grouped),
                                                                             to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized2SmVs32Sm103TmaPrefetch, grouped),
                                                                             to_grouped_schedule(KernelScheduleType.MxNvf4UltraTmaWarpSpecialized2SmVs16Sm103TmaPrefetch, grouped)
                                                                             ]
      if cta_n == 256 and is_sm103_fp4_ultra_1sm_kernel_schedule:
        epi_tile_mn = "cute::Shape<cute::_128,cute::_64>"
        if is_tma_epilogue(operation.epilogue_schedule):
          epilogue_schedule_type = EpilogueScheduleTag[to_grouped_schedule(EpilogueScheduleType.TmaWarpSpecialized1Sm, grouped)]
      if cta_n == 256 and is_sm103_fp4_ultra_2sm_kernel_schedule:
        epi_tile_mn = "cute::Shape<cute::_128,cute::_64>"
        if is_tma_epilogue(operation.epilogue_schedule):
          epilogue_schedule_type = EpilogueScheduleTag[to_grouped_schedule(EpilogueScheduleType.TmaWarpSpecialized2Sm, grouped)]

      element_a = f'cute::tuple<{str(element_a)},{str(DataTypeTag[operation.ScaleFactorA])}>'
      element_b = f'cute::tuple<{str(element_b)},{str(DataTypeTag[operation.ScaleFactorB])}>'

    alignment_c = get_tma_alignment(operation.C.element) \
                  if is_tma_epilogue(operation.epilogue_schedule) and opcode_class_epi != OpcodeClass.Simt \
                  else operation.C.alignment
    alignment_d = get_tma_alignment(operation.D.element) \
                  if is_tma_epilogue(operation.epilogue_schedule) and opcode_class_epi != OpcodeClass.Simt \
                  else operation.D.alignment

    operation_name_str = operation.procedural_name()
    layout_a_str = LayoutTag[instance_layout_A]
    layout_b_str = LayoutTag[instance_layout_B]
    mixed_dtype_prepare_code = ""
    if operation.mixed_input_mode != None:
      A_dtype = operation.A.element
      B_dtype = operation.B.element
      A_dtype_bits = DataTypeSize[A_dtype]
      B_dtype_bits = DataTypeSize[B_dtype]
      is_A_dtype_narrow = A_dtype_bits < B_dtype_bits
      if is_A_dtype_narrow:
        narrow_dtype, wide_dtype = (A_dtype, B_dtype)
        narrow_dtype_bits, wide_dtype_bits = (A_dtype_bits, B_dtype_bits)
      else:
        narrow_dtype, wide_dtype = (B_dtype, A_dtype)
        narrow_dtype_bits, wide_dtype_bits = (B_dtype_bits, A_dtype_bits)

      narrow_tag = DataTypeTag[narrow_dtype]
      wide_tag   = DataTypeTag[wide_dtype]
      scale_tag  = DataTypeTag[wide_dtype]
      zero_tag   = DataTypeTag[wide_dtype]

      do_shuffle = False
      value_shuffle_str = ""
      if narrow_dtype_bits == 4 and wide_dtype_bits == 16:
        value_shuffle_str = "cute::Layout<cute::Shape<cute::_2,cute::_4>, cute::Stride<cute::_4,cute::_1>>"
        do_shuffle = True
      if narrow_dtype_bits == 8 and wide_dtype_bits == 16:
        value_shuffle_str = "cute::Layout<cute::Shape<cute::_2,cute::_2>, cute::Stride<cute::_2,cute::_1>>"
        do_shuffle = True
      do_shuffle = operation.mixed_input_shuffle and do_shuffle

      if do_shuffle:
        if is_A_dtype_narrow:
          stride_narrow_str = f"cutlass::detail::TagToStrideA_t<{layout_a_str}>"
          layout_a_str = f"{operation_name_str}_LayoutNarrowReordered"
        else:
          stride_narrow_str = f"cutlass::detail::TagToStrideB_t<{layout_b_str}>"
          layout_b_str = f"{operation_name_str}_LayoutNarrowReordered"
        # The {operation_name_str}_ prefixs in mixed_dtype_prepare_code and
        # layout_{a, b}_str are to prevent errors in Windows platform unity build
        mixed_dtype_prepare_code = f"""
using {operation_name_str}_StrideNarrow = {stride_narrow_str};
using {operation_name_str}_ValueShuffle = {value_shuffle_str};
static constexpr int {operation_name_str}_NumShuffleAtoms = 1;
using {operation_name_str}_MmaAtomShape = cute::Layout<cute::Shape<cute::_1, cute::Int<{operation_name_str}_NumShuffleAtoms>>>;
using {operation_name_str}_LayoutAtomQuant = decltype(cutlass::compute_memory_reordering_atom<{wide_tag}, {operation_name_str}_MmaAtomShape, {operation_name_str}_ValueShuffle>());
using {operation_name_str}_LayoutNarrowReordered = decltype(cute::tile_to_shape({operation_name_str}_LayoutAtomQuant{{}}, cute::Layout<cute::Shape<int,int,int>, {operation_name_str}_StrideNarrow>{{}}));
        """

      mixed_input_modes_to_element = {
        MixedInputMode.ConvertOnly: narrow_tag,
        MixedInputMode.ScaleOnly: f"cute::tuple<{narrow_tag}, {scale_tag}>",
        MixedInputMode.ScaleWithZeroPoint: f"cute::tuple<{narrow_tag}, {scale_tag}, {zero_tag}>"
      }
      narrow_element = mixed_input_modes_to_element.get(operation.mixed_input_mode, narrow_tag)

      if narrow_dtype == DataType.s4 and (wide_dtype == DataType.e4m3 or wide_dtype == DataType.e5m2):
        narrow_element = f"cute::tuple<{narrow_tag}, cutlass::Array<{scale_tag}, 8>>"

      if is_A_dtype_narrow:
        element_a = narrow_element
      else:
        element_b = narrow_element

    blockwise_prepare_code = ""
    if is_blockwise(operation.gemm_kind):
      sfm_vec_size = operation.ScaleFactorMVecSize
      sfn_vec_size = operation.ScaleFactorNVecSize
      sfk_vec_size = operation.ScaleFactorKVecSize
      blockwise_prepare_code = f"""
using {operation_name_str}_ScaleConfig = cutlass::detail::Sm{operation.arch}BlockwiseScaleConfig<{sfm_vec_size}, {sfn_vec_size}, {sfk_vec_size}>;
using {operation_name_str}_LayoutSFA = decltype({operation_name_str}_ScaleConfig::deduce_layoutSFA());
using {operation_name_str}_LayoutSFB = decltype({operation_name_str}_ScaleConfig::deduce_layoutSFB());
      """

    values = {
      'operation_name': operation_name_str,
      'operation_suffix': self.operation_suffix,
      'problem_shape': self.problem_shape(operation),
      'element_a': element_a,
      'layout_a': self.transform_layout_A_if_blockwise(operation, self.pointerize_if_grouped(operation, layout_a_str)),
      'element_b': element_b,
      'layout_b': self.transform_layout_B_if_blockwise(operation, self.pointerize_if_grouped(operation, layout_b_str)),
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': self.pointerize_if_grouped(operation, LayoutTag[instance_layout_C]),
      'element_d': DataTypeTag[operation.D.element],
      'layout_d': self.pointerize_if_grouped(operation, LayoutTag[instance_layout_D]),
      'element_accumulator': DataTypeTag[operation.accumulator_type()],
      'opcode_class_main': OpcodeClassTag[opcode_class_main],
      'opcode_class_epi': OpcodeClassTag[opcode_class_epi],
      'arch': "cutlass::arch::Sm%d" % operation.arch,
      'tile_shape_m': str(tile_shape_m),
      'tile_shape_n': str(tile_shape_n),
      'tile_shape_k': str(tile_shape_k),
      'cluster_shape_m': 'cute::_' + str(operation.tile_description.cluster_shape[0]) if operation.tile_description.cluster_shape[0] > 0 else "int",
      'cluster_shape_n': 'cute::_' + str(operation.tile_description.cluster_shape[1]) if operation.tile_description.cluster_shape[1] > 0 else "int",
      'cluster_shape_k': 'cute::_' + str(operation.tile_description.cluster_shape[2]) if operation.tile_description.cluster_shape[2] > 0 else "int",
      'instruction_shape_m': str(instruction_shape[0]),
      'instruction_shape_n': str(instruction_shape[1]),
      'instruction_shape_k': str(instruction_shape[2]),
      'kernel_schedule' : str(KernelScheduleTag[operation.kernel_schedule]),
      'epilogue_schedule' : str(epilogue_schedule_type),
      'epi_tile_mn' : epi_tile_mn,
      'epilogue_functor': epilogue_functor,
      'stages': stage_count_string,
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'align_c': str(alignment_c),
      'align_d': str(alignment_d),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'epilogue_vector_length': str(epilogue_vector_length),
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'tile_scheduler': str(TileSchedulerTag[operation.tile_scheduler]),
      'mixed_dtype_prepare_code': mixed_dtype_prepare_code,
      'blockwise_prepare_code' : blockwise_prepare_code
    }

    return SubstituteTemplate(self.gemm_template, values)

###################################################################################################

#
class EmitGemmPlanarComplexInstance:
  ''' Responsible for emitting a CUTLASS template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = []
    self.template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = typename cutlass::gemm::kernel::DefaultGemmPlanarComplexUniversal<
    ${element_a}, ${layout_a}, ${transform_a}, ${alignment_a},
    ${element_b}, ${layout_b}, ${transform_b}, ${alignment_b},
    ${element_c}, cutlass::layout::RowMajor,
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    cutlass::epilogue::thread::LinearCombinationPlanarComplex<
      ${element_c},
      ${alignment_c},
      ${element_accumulator},
      ${element_epilogue}
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    ${stages},
    ${math_operator}
  >::GemmKernel;

  struct ${operation_name} :
    public Operation_${operation_name} { };
"""

  #
  def instance_template(self):
    return """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<
    cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>
  >("${operation_name}"));
${compile_guard_end}
"""

  #
  def emit(self, operation):

    warp_shape = [operation.tile_description.threadblock_shape[idx] // operation.tile_description.warp_count[idx] for idx in range(3)]

    # exchange and transpose A and B types, layouts, and complex transforms since the C layout is row-major
    transposed_layout_A = TransposedLayout[operation.A.layout]
    transposed_layout_B = TransposedLayout[operation.B.layout]

    values = {
      'operation_name': operation.procedural_name(),
      'element_a': DataTypeTag[operation.B.element],
      'layout_a': LayoutTag[transposed_layout_B],
      'transform_a': ComplexTransformTag[operation.B.complex_transform],
      'alignment_a': str(operation.B.alignment),
      'element_b': DataTypeTag[operation.A.element],
      'layout_b': LayoutTag[transposed_layout_A],
      'transform_b': ComplexTransformTag[operation.A.complex_transform],
      'alignment_b': str(operation.A.alignment),
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[operation.C.layout],
      'element_accumulator': DataTypeTag[operation.tile_description.math_instruction.element_accumulator],
      'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
      'arch': "cutlass::arch::Sm%d" % operation.arch,
      'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
      'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
      'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
      'warp_shape_m': str(warp_shape[0]),
      'warp_shape_n': str(warp_shape[1]),
      'warp_shape_k': str(warp_shape[2]),
      'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]),
      'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]),
      'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]),
      'alignment_c': str(operation.C.alignment),
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'stages': str(operation.tile_description.stages),
      'math_operator': 'cutlass::arch::OpMultiplyAdd'
    }

    return SubstituteTemplate(self.template, values)

###################################################################################################

#
class EmitGemmPlanarComplexArrayInstance:
  ''' Responsible for emitting a CUTLASS template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = []
    self.template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = typename cutlass::gemm::kernel::DefaultGemmPlanarComplexUniversal<
    ${element_a}, ${layout_a}, ${transform_a}, ${alignment_a},
    ${element_b}, ${layout_b}, ${transform_b}, ${alignment_b},
    ${element_c}, cutlass::layout::RowMajor,
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    cutlass::epilogue::thread::LinearCombinationPlanarComplex<
      ${element_c},
      ${alignment_c},
      ${element_accumulator},
      ${element_epilogue}
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    ${stages},
    ${math_operator}
  >::GemmArrayKernel;

  struct ${operation_name} : public Operation_${operation_name} { };
"""

  #
  def instance_template(self):
    return """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<
    cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>
  >("${operation_name}"));
${compile_guard_end}
"""

  #
  def emit(self, operation):

    warp_shape = [operation.tile_description.threadblock_shape[idx] // operation.tile_description.warp_count[idx] for idx in range(3)]

    # exchange and transpose A and B types, layouts, and complex transforms since the C layout is row-major
    transposed_layout_A = TransposedLayout[operation.A.layout]
    transposed_layout_B = TransposedLayout[operation.B.layout]

    values = {
      'operation_name': operation.procedural_name(),
      'element_a': DataTypeTag[operation.B.element],
      'layout_a': LayoutTag[transposed_layout_B],
      'transform_a': ComplexTransformTag[operation.B.complex_transform],
      'alignment_a': str(operation.B.alignment),
      'element_b': DataTypeTag[operation.A.element],
      'layout_b': LayoutTag[transposed_layout_A],
      'transform_b': ComplexTransformTag[operation.A.complex_transform],
      'alignment_b': str(operation.A.alignment),
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[operation.C.layout],
      'element_accumulator': DataTypeTag[operation.tile_description.math_instruction.element_accumulator],
      'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
      'arch': "cutlass::arch::Sm%d" % operation.arch,
      'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
      'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
      'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
      'warp_shape_m': str(warp_shape[0]),
      'warp_shape_n': str(warp_shape[1]),
      'warp_shape_k': str(warp_shape[2]),
      'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]),
      'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]),
      'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]),
      'alignment_c': str(operation.C.alignment),
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'stages': str(operation.tile_description.stages),
      'math_operator': 'cutlass::arch::OpMultiplyAdd'
    }

    return SubstituteTemplate(self.template, values)

###################################################################################################

#
class EmitGemmGroupedInstance:
  ''' Responsible for emitting a CUTLASS template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = [
      "cutlass/cutlass.h",
      "cutlass/numeric_types.h",
      "cutlass/arch/arch.h",
      "cutlass/arch/mma.h",
      "cutlass/layout/matrix.h",
      "cutlass/gemm/device/gemm.h",
      "cutlass/gemm/kernel/gemm_grouped.h",
      "cutlass/gemm/kernel/default_gemm_grouped.h",
      "cutlass/gemm/device/gemm_grouped.h"
    ]
    self.builtin_epilogue_functor_template = \
"""${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >"""

    self.gemm_template = """
// Gemm operator ${operation_name}
using ${operation_name}_base =
  typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ${element_a}, ${layout_a}, ${transform_a}, ${align_a},
    ${element_b}, ${layout_b}, ${transform_b}, ${align_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor},
    ${swizzling_functor},
    ${stages},
    ${scheduler_mode},
    ${math_operation}
>::GemmKernel;

// Define named type
struct ${operation_name}${operation_suffix} :
  public ${operation_name}_base { };
"""

  #
  def instance_template(self):
    return """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<
    cutlass::gemm::device::GemmGrouped<${operation_name}>
  >("${operation_name}"));
${compile_guard_end}
"""

  #
  def emit(self, operation):

    threadblock_shape = operation.tile_description.threadblock_shape
    warp_count = operation.tile_description.warp_count

    warp_shape = [threadblock_shape[idx] // warp_count[idx] for idx in range(3)]

    transpose_layouts = {
      LayoutType.ColumnMajor: LayoutType.RowMajor,
      LayoutType.RowMajor: LayoutType.ColumnMajor
    }

    instance_layout_A, instance_layout_B, instance_layout_C = \
      (operation.A.layout, operation.B.layout, operation.C.layout)
    #

    # Support built-in epilogue functors or user-defined functions
    if isinstance(operation.epilogue_functor, enum.Enum):

      epilogue_vector_length = \
        min(operation.C.alignment * DataTypeSize[operation.C.element], 128) // DataTypeSize[operation.C.element]

      values = {
        'epilogue_vector_length': str(epilogue_vector_length),
        'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
        'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      }
      epilogue_functor = SubstituteTemplate(self.builtin_epilogue_functor_template, values)
    else:
      epilogue_functor = self.epilogue_functor.emit_declaration()
    #

    values = {
      'operation_name': operation.procedural_name(),
      'operation_suffix': self.operation_suffix,
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[instance_layout_A],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b': LayoutTag[instance_layout_B],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[instance_layout_C],
      'element_accumulator': DataTypeTag[operation.accumulator_type()],
      'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
      'arch': "cutlass::arch::Sm%d" % operation.arch,
      'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
      'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
      'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
      'warp_shape_m': str(warp_shape[0]),
      'warp_shape_n': str(warp_shape[1]),
      'warp_shape_k': str(warp_shape[2]),
      'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]),
      'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]),
      'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]),
      'epilogue_functor': epilogue_functor,
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'scheduler_mode': GroupScheduleModeTag[operation.scheduler_mode],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation]
    }

    return SubstituteTemplate(self.gemm_template, values)

###################################################################################################
#
# Emitters functions for all targets
#
###################################################################################################

class EmitGemmConfigurationLibrary:
  def __init__(self, operation_path, configuration_name):
    self.configuration_name = configuration_name
    self.configuration_path = os.path.join(operation_path, "%s.cu" % configuration_name).replace('\\', '/')

    self.instance_emitter = {
      GemmKind.Gemm: EmitGemmInstance,
      GemmKind.Sparse: EmitSparseGemmInstance,
      GemmKind.Universal: EmitGemmUniversalInstance,
      GemmKind.Universal3x: EmitGemmUniversal3xInstance,
      GemmKind.SparseUniversal3x: EmitGemmUniversal3xInstance,
      GemmKind.BlockScaledUniversal3x: EmitGemmUniversal3xInstance,  
      GemmKind.PlanarComplex: EmitGemmPlanarComplexInstance,
      GemmKind.PlanarComplexArray: EmitGemmPlanarComplexArrayInstance,
      GemmKind.Grouped: EmitGemmGroupedInstance,
      GemmKind.GroupedUniversal3x: EmitGemmUniversal3xInstance,
      GemmKind.GroupedBlockScaledUniversal3x: EmitGemmUniversal3xInstance,
      GemmKind.BlockwiseUniversal3x: EmitGemmUniversal3xInstance,
      GemmKind.GroupedBlockwiseUniversal3x: EmitGemmUniversal3xInstance,
    }

    self.gemm_kind_wrappers = {
      GemmKind.Gemm: 'GemmOperation',
      GemmKind.Sparse: 'GemmSparseOperation',
      GemmKind.Universal: 'GemmUniversalOperation',
      GemmKind.Universal3x: 'GemmUniversal3xOperation',
      GemmKind.SparseUniversal3x: 'SparseGemmUniversal3xOperation',
      GemmKind.BlockScaledUniversal3x: 'BlockScaledGemmUniversal3xOperation', 
      GemmKind.PlanarComplex: 'GemmPlanarComplexOperation',
      GemmKind.PlanarComplexArray: 'GemmPlanarComplexArrayOperation',
      GemmKind.Grouped: 'GemmGroupedOperation',
      GemmKind.GroupedUniversal3x: 'GroupedGemmUniversal3xOperation',
      GemmKind.GroupedBlockScaledUniversal3x: 'GroupedBlockScaledGemmUniversal3xOperation',
      GemmKind.BlockwiseUniversal3x: 'BlockwiseGemmUniversal3xOperation',
      GemmKind.GroupedBlockwiseUniversal3x: 'GroupedBlockwiseGemmUniversal3xOperation',
    }

    self.wmma_guard_start = "#if defined(CUTLASS_ARCH_WMMA_SM${sm_number}_ENABLED)"

    self.separator = """
///////////////////////////////////////////////////////////////////////////////////////////////////

"""

    self.header_template = """
/*
  Generated by gemm_operation.py - Do not edit.
*/
"""

    self.initialize_function_template = """

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_${configuration_name}(Manifest &manifest) {

"""
    self.epilogue_template = """

}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

"""

  def __enter__(self):
    _LOGGER.debug("*** EmitGemmConfigurationLibrary::__enter__")
    _LOGGER.debug("***   configuration_path (file to write): " +
                  str(self.configuration_path))

    self.configuration_file = open(self.configuration_path, "w")
    self.configuration_file.write(self.header_template)
    self.configuration_file.write(self.separator)

    self.includes = collections.OrderedDict([
      ("cutlass/cutlass.h", None),
      ("cutlass/library/library.h", None),
      ("cutlass/library/manifest.h", None),
      ("library_internal.h", None),
      ("gemm_operation.h", None),
      ("gemm_operation_3x.hpp", None),
      ("grouped_gemm_operation_3x.hpp", None),
      ("sparse_gemm_operation_3x.hpp", None),
      ("block_scaled_gemm_operation_3x.hpp", None),   
      ("blockwise_gemm_operation_3x.hpp", None),   
      ("cutlass/arch/wmma.h", None),
      ("cutlass/numeric_types.h", None)
    ])
    self.instance_definitions = []
    self.instance_wrappers = []

    self.operations = []
    return self

  def emit(self, operation):
    _LOGGER.debug("*** EmitGemmConfigurationLibrary::emit(operation)")
    _LOGGER.debug("***   operation.gemm_kind: " + str(operation.gemm_kind))

    emitter = self.instance_emitter[operation.gemm_kind]()

    for incl in emitter.includes:
      self.includes[incl] = None

    self.operations.append(operation)

    self.instance_definitions.append(emitter.emit(operation))

    self.instance_wrappers.append(SubstituteTemplate(emitter.instance_template(), {
      'configuration_name': self.configuration_name,
      'operation_name': operation.procedural_name(),
      'gemm_kind': self.gemm_kind_wrappers[operation.gemm_kind],
      'compile_guard_start': SubstituteTemplate(self.wmma_guard_start, {'sm_number': str(operation.arch)}) \
        if operation.tile_description.math_instruction.opcode_class == OpcodeClass.WmmaTensorOp else "",
      'compile_guard_end': "#endif" \
        if operation.tile_description.math_instruction.opcode_class == OpcodeClass.WmmaTensorOp else ""
      }))

  def __exit__(self, exception_type, exception_value, traceback):

    # Write includes
    for incl, _ in self.includes.items():
      include_statement = "#include \"%s\"\n" % incl
      self.configuration_file.write(include_statement)

    self.configuration_file.write(self.separator)

    # Write instance definitions in top-level namespace
    for instance_definition in self.instance_definitions:
      self.configuration_file.write(instance_definition)

    # Add wrapper objects within initialize() function
    self.configuration_file.write(SubstituteTemplate(self.initialize_function_template, {
      'configuration_name': self.configuration_name
      }))

    for instance_wrapper in self.instance_wrappers:
      self.configuration_file.write(instance_wrapper)

    self.configuration_file.write(self.epilogue_template)
    self.configuration_file.close()

###################################################################################################
###################################################################################################

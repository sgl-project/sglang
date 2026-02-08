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
Utilities for emitting Conv3d kernels
"""

import enum
import logging
import os.path
import shutil
from string import Template

try:
  import builtins
  if hasattr(builtins, "CUTLASS_IGNORE_PACKAGE") and CUTLASS_IGNORE_PACKAGE == True:
    raise ImportError("Disabling attempt to import cutlass_library")
  from cutlass_library.library import *
  from cutlass_library.conv3x_emitter import EmitConv3xInstance, EmitConv3xIncludes
except ImportError:
  from library import *
  from conv3x_emitter import EmitConv3xInstance, EmitConv3xIncludes

_LOGGER = logging.getLogger(__name__)

###################################################################################################

#
class Conv3dOperation:
  #
  def __init__(self, conv_kind, iterator_algorithm, arch, tile_description, A, B, C, element_epilogue, \
    stride_support, epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4):

    self.operation_kind = OperationKind.Conv3d
    self.arch = arch
    self.tile_description = tile_description
    self.conv_kind = conv_kind
    self.A = A
    self.B = B
    self.C = C
    self.element_epilogue = element_epilogue
    self.epilogue_functor = epilogue_functor
    self.iterator_algorithm = iterator_algorithm
    self.stride_support = stride_support
    self.swizzling_functor = swizzling_functor

  #
  def is_mixed_input(self):
    return self.A.element != self.B.element

  #
  def core_name(self):
    ''' The basic operation kind is prefixed with a letter indicating the accumulation type. '''

    intermediate_type = ''

    if self.tile_description.math_instruction.opcode_class == OpcodeClass.TensorOp:
      inst_shape = "%d%d%d" % tuple(self.tile_description.math_instruction.instruction_shape)
      if self.tile_description.math_instruction.element_a != self.A.element and \
        self.tile_description.math_instruction.element_a != self.tile_description.math_instruction.element_accumulator:
        intermediate_type = DataTypeNames[self.tile_description.math_instruction.element_a]
    else:
      inst_shape = ''

    return "%s%s%s%s3d_%s" % (ShortDataTypeNames[self.tile_description.math_instruction.element_accumulator], \
      inst_shape, intermediate_type, ConvKindNames[self.conv_kind], IteratorAlgorithmNames[self.iterator_algorithm])

  #
  def extended_name(self):
    ''' Append data types if they differ from compute type. '''
    if self.C.element != self.tile_description.math_instruction.element_accumulator and \
      self.A.element != self.tile_description.math_instruction.element_accumulator:
      extended_name = "${element_c}_${core_name}_${element_a}"
    elif self.C.element == self.tile_description.math_instruction.element_accumulator and  \
      self.A.element != self.tile_description.math_instruction.element_accumulator:
      extended_name = "${core_name}_${element_a}"
    else:
      extended_name = "${core_name}"

    extended_name = SubstituteTemplate(extended_name, {
      'element_a': DataTypeNames[self.A.element],
      'element_c': DataTypeNames[self.C.element],
      'core_name': self.core_name()
      })

    return extended_name

  #
  def configuration_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''

    opcode_class_name = OpcodeClassNames[self.tile_description.math_instruction.opcode_class]

    threadblock = "%dx%d_%dx%d" % (
      self.tile_description.threadblock_shape[0],
      self.tile_description.threadblock_shape[1],
      self.tile_description.threadblock_shape[2],
      self.tile_description.stages
    )

    if self.stride_support == StrideSupport.Unity:
      configuration_name = "cutlass_${opcode_class}_${extended_name}_${threadblock}_unity_stride"
    else:
      configuration_name = "cutlass_${opcode_class}_${extended_name}_${threadblock}"

    return SubstituteTemplate(
      configuration_name,
      {
        'opcode_class': opcode_class_name,
        'extended_name': self.extended_name(),
        'threadblock': threadblock,
      }
    )

  #
  def procedural_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
    return self.configuration_name()

###################################################################################################
#
# Emits single instances of a CUTLASS device-wide operator
#
###################################################################################################

class EmitConv3dInstance:
  def __init__(self):
    # Emitter for CUTLASS 3 convolution operations
    self.conv3x_emitter = EmitConv3xInstance()
    self.template = """
  // Conv3d${conv_kind_name} ${iterator_algorithm_name} kernel instance "${operation_name}"
  using ${operation_name}_base =
  typename cutlass::conv::kernel::DefaultConv3d${conv_kind_name}<
    ${element_a},
    cutlass::layout::TensorNDHWC,
    ${element_b},
    cutlass::layout::TensorNDHWC,
    ${element_c},
    cutlass::layout::TensorNDHWC,
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k} >,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor}, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    ${stages},
    cutlass::arch::OpMultiplyAdd,
    ${iterator_algorithm},
    ${stride_support}
  >::Kernel;
"""

  def emit(self, operation):
    _LOGGER.debug("*** EmitConv3dInstance::emit")
    _LOGGER.debug("***   operation: procedural_name()=" + operation.procedural_name())

    if hasattr(operation, 'is_3x') and operation.is_3x:
      _LOGGER.debug("***   CUTLASS 3 operation")
      return self.conv3x_emitter.emit(operation)

    _LOGGER.debug("***   CUTLASS 2 operation")

    warp_shape = [int(operation.tile_description.threadblock_shape[idx] / operation.tile_description.warp_count[idx]) for idx in range(3)]

    epilogue_vector_length = int(min(operation.C.alignment * DataTypeSize[operation.C.element], 128) / DataTypeSize[operation.C.element])

    values = {
      'operation_name': operation.procedural_name(),
      'conv_kind': ConvKindTag[operation.conv_kind],
      'conv_kind_name': ConvKindNames[operation.conv_kind].capitalize(),
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[operation.A.layout],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b': LayoutTag[operation.B.layout],
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
      'epilogue_vector_length': str(epilogue_vector_length),
      'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'iterator_algorithm': IteratorAlgorithmTag[operation.iterator_algorithm],
      'iterator_algorithm_name': IteratorAlgorithmNames[operation.iterator_algorithm].capitalize(),
      'stride_support': StrideSupportTag[operation.stride_support]
    }

    return SubstituteTemplate(self.template, values)

###################################################################################################
#
# Generator functions for all layouts
#
###################################################################################################

#
def GenerateConv3dTensorOp(manifest, tile_descriptions, min_cc, align = 128):

  for tile in tile_descriptions:
    for conv_kind in [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad]:

      if conv_kind == ConvKind.Fprop or (tile.math_instruction.element_accumulator in [DataType.f16, DataType.f32]):

        #
        output_types = [tile.math_instruction.element_a, tile.math_instruction.element_accumulator] \
          if DataTypeSize[tile.math_instruction.element_accumulator] == 32 \
          else [tile.math_instruction.element_accumulator,]

        for output_type in output_types:
          A = TensorDescription(tile.math_instruction.element_a, LayoutType.TensorNDHWC, int(align / DataTypeSize[tile.math_instruction.element_a]))
          B = TensorDescription(tile.math_instruction.element_b, LayoutType.TensorNDHWC, int(align / DataTypeSize[tile.math_instruction.element_b]))
          C = TensorDescription(output_type,  LayoutType.TensorNDHWC, max(1, int(align / DataTypeSize[output_type])))

          manifest.append(Conv3dOperation(conv_kind, min_cc, tile, A, B, C, tile.math_instruction.element_accumulator))

class EmitConv3dIncludes:
  '''Emit includes that are specific to the operation.'''

  def __init__(self):
    self.includes = ['conv3d_operation.h']
    self.emitter_3x = EmitConv3xIncludes()

  def operation_is_3x(self, operation) -> bool:
    """Whether operation is a CUTLASS 3 convolution (as opposed to CUTLASS 2)"""
    return hasattr(operation, 'is_3x') and operation.is_3x

  def emit(self, operation) -> str:
    if self.operation_is_3x(operation):
      return self.emitter_3x.emit(operation)

    return '\n'.join(f"#include \"{incl}\"" for incl in self.includes) + \
      "\n\n///////////////////////////////////////////////////////////////////////////////////////////////////"

###################################################################################################
#
# Emitters functions for all targets
#
###################################################################################################

class EmitConv3dConfigurationLibrary:
  def __init__(self, operation_path, configuration_name):
    self.configuration_name = configuration_name
    self.configuration_path = os.path.join(operation_path, "%s.cu" % configuration_name)

    self.instance_emitter = EmitConv3dInstance()
    self.includes_emitter = EmitConv3dIncludes()

    self.header_template = """
/*
  Generated by conv3d_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
"""

    self.instance_template = """
${stub_begin}
${operation_instance}
// Derived class
struct ${operation_name} :
  public ${operation_name}_base { };
${stub_end}
///////////////////////////////////////////////////////////////////////////////////////////////////

"""

    self.configuration_header = """

namespace cutlass {
namespace library {

// Initialize all instances
void initialize_${configuration_name}(Manifest &manifest) {
"""

    self.configuration_instance = """${stub_begin}
  using Operation_${operation_name} = cutlass::conv::device::${kernel_name}<
    ${operation_name}>;

  manifest.append(new cutlass::library::${operation_wrapper}<
      Operation_${operation_name}
    >(
      "${operation_name}"
    ));
${stub_end}
"""

    self.configuration_epilogue = "}\n"

    self.epilogue_template = """

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

"""

  def operation_is_3x(self, operation):
    """Whether operation is a CUTLASS 3 convolution (as opposed to CUTLASS 2)"""
    return hasattr(operation, 'is_3x') and operation.is_3x

  def __enter__(self):
    """
    Open the configuration_file, and write the "header" C++ code to it.

    The "header" consists of a comment (that this is generated code,
    so it should not be edited), and includes that are common
    to both the CUTLASS 2 and the CUTLASS 3 cases.
    """
    _LOGGER.debug('*** EmitConv3dConfigurationLibrary::__enter__')
    _LOGGER.debug('***   configuration_path (file to write): ' +
                  str(self.configuration_path))
    _LOGGER.debug('***   configuration_name: ' + self.configuration_name)
    self.configuration_file = open(self.configuration_path, "w")

    self.configuration_file.write(SubstituteTemplate(self.header_template, {
      'configuration_name': self.configuration_name
      }))
    self.operations = []
    return self

  def emit(self, operation):
    """
    Write three pieces of C++ code to the configuration_file
    (that was opened by the __enter__ method above):

    1. the header includes that are specific to the operation
       (CUTLASS 2 vs. CUTLASS 3);

    2. the "operation instance" (a "using" declaration ending in "_base"); and

    3. the "operation name" (declaration and definition of a derived class
       of the above operation instance).

    The "using" declaration turns a C++ class name, possibly namespace-qualified,
    possibly also with angle brackets, into a C-style, easily demangled identifier.
    """
    _LOGGER.debug('*** EmitConv3dConfigurationLibrary::emit')
    _LOGGER.debug('***   operation.procedural_name(): ' + operation.procedural_name())
    self.operations.append(operation)

    self.configuration_file.write(self.includes_emitter.emit(operation))

    stub_begin = ''
    stub_end = ''
    # It can be useful to stub (comment) out instantiations for testing.
    # In this case, one need only set is_stub to True.
    is_stub = False
    if is_stub:
      stub_begin = "// STUB for now\n#if 0"
      stub_end = '#endif // 0'

    self.configuration_file.write(Template(self.instance_template).substitute({
      'configuration_name': self.configuration_name,
      'operation_name': operation.procedural_name(),
      'operation_instance': self.instance_emitter.emit(operation),
      'stub_begin': stub_begin,
      'stub_end': stub_end
      }))

  def __exit__(self, exception_type, exception_value, traceback):
    """
    Write the rest of the C++ code to the configuration_file, and close the file.

    The "rest of the C++ code" has the following components.

    1. Configuration header: Open the namespace(s), and open the definition
       of the "initialize_${configuration_name}" registration function
       that registers the operation with the Manifest.
       ("Registration" helps turn C++ compile-time polymorphism
       (via template parameters) into a run-time choice of parameters.)

    2. Configuration instance: In the body of the registration function,
       make a "using" declaration Operation_${operation_name} for the
       operation type (which uses operation_name as its template argument).
       Then, tell the manifest about the operation via a "manifest.append" call.
       The argument of the call is a new instance of
       "SomethingOperation<Operation_${operation_name}>"
       (replace Something with a specific name).

    3. Configuration epilogue: Close the definition of the registration function.

    4. Epilogue template: Close the namespace(s).
    """

    _LOGGER.debug('*** EmitConv3dConfigurationLibrary::__exit__')
    _LOGGER.debug('***   configuration_path (file to write): ' +
                  str(self.configuration_path))
    _LOGGER.debug('***   configuration_name: ' + self.configuration_name)

    self.configuration_file.write(SubstituteTemplate(self.configuration_header, {
      'configuration_name': self.configuration_name
      }))

    for operation in self.operations:
      stub_begin = ''
      stub_end = ''
      # It can be useful to stub (comment) out instantiations for testing.
      # In this case, one need only set is_stub to True.
      is_stub = False
      if is_stub:
        stub_begin = "// STUB for now\n#if 0"
        stub_end = "#endif // 0"

      kernel_name = 'ImplicitGemmConvolution'
      operation_wrapper = 'Conv3dOperation'
      if self.operation_is_3x(operation):
        kernel_name = 'ConvUniversalAdapter'
        operation_wrapper = 'ConvOperation3x'

      self.configuration_file.write(SubstituteTemplate(self.configuration_instance, {
        'configuration_name': self.configuration_name,
        'operation_name': operation.procedural_name(),
        'kernel_name': kernel_name,
        'operation_wrapper': operation_wrapper,
        'stub_begin': stub_begin,
        'stub_end': stub_end
      }))

    self.configuration_file.write(self.configuration_epilogue)
    self.configuration_file.write(self.epilogue_template)
    self.configuration_file.close()


###################################################################################################
###################################################################################################

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
Utilities for emitting CUTLASS >= 3 convolution kernels
"""

import enum
import os.path
import shutil
import logging
from string import Template

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
# Emits single instances of a CUTLASS device-wide operator
#
###################################################################################################

class EmitConv3xInstance:
  def __init__(self):
    _LOGGER.debug("*** EmitConv3xInstance::__init__")

    # Define epilogue type first, so that the mainloop type
    # can use it with StageCountAutoCarveout.
    self.template = """

// CUTLASS >= 3 convolution ${conv_kind_name} kernel instance "${operation_name}"
using ${operation_name}_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    ${arch},
    ${opcode_class_epi},
    ${mma_tile_shape},               // mma tile shape
    ${cluster_shape},                // cluster shape
    ${epi_tile_mn},
    ${element_accumulator},
    ${element_compute},
    ${element_c}, ${layout_c}, 128 / cute::sizeof_bits_v<${element_c}>,
    ${element_d}, ${layout_d}, 128 / cute::sizeof_bits_v<${element_d}>,
    ${epilogue_schedule}
    // , class FusionOpOrCallbacks = cutlass::epilogue::fusion::LinearCombination<ElementD,ElementCompute>
  >::CollectiveOp;

using ${operation_name}_mainloop =
  typename cutlass::conv::collective::CollectiveBuilder<
    ${arch},
    ${opcode_class_main},
    ${conv_kind},         // kFprop, kDgrad, or kWgrad
    ${element_a}, ${layout_a}, 128 / cute::sizeof_bits_v<${element_a}>,
    ${element_b}, ${layout_b}, 128 / cute::sizeof_bits_v<${element_b}>,
    ${element_accumulator},
    ${mma_tile_shape},        // mma tile shape
    ${cluster_shape},         // cluster shape
    ${stages},
    ${kernel_schedule}
  >::CollectiveOp;

using ${operation_name}_problem_shape = cutlass::conv::ConvProblemShape<${conv_kind}, ${operation_name}_mainloop::NumSpatialDimensions>;

// Unit tests call this "ConvKernel".
// Conv operator ${operation_name}
using ${operation_name}_base = cutlass::conv::kernel::ConvUniversal<
    ${operation_name}_problem_shape,
    ${operation_name}_mainloop,
    ${operation_name}_epilogue,
    ${tile_scheduler}
  >;
"""

  def arch_number_to_type(self, arch: int) -> str:
    return f"cutlass::arch::Sm{arch}"

  def mma_tile_shape(self, operation, cta_m, cta_n, cta_k) -> str:
    mma_m = cta_m
    mma_n = cta_n
    mma_k = cta_k

    if operation.arch >= 100:
      # MmaTileShape (mma_m, mma_n, mma_k) is passed to kernel mainloop where
      # mma_m = cta_m for 1sm version and mma_m = cta_m * 2 for 2sm version.
      # If schedule is auto and cluster size is static and cta_m % 64 == 0 and cluster_m % 2 == 0, 2sm kernel version is allocated,
      # otherwise 1sm kernel is allocated.
      cta_m_per_mma_instruction = 1
      if "2sm" in operation.procedural_name() :
        cta_m_per_mma_instruction = 2
      elif "1sm" in operation.procedural_name() :
        cta_m_per_mma_instruction = 1
      elif operation.tile_description.cluster_shape[0] > 0 and operation.tile_description.cluster_shape[0] % 2 == 0 and cta_m % 64 == 0 :
        cta_m_per_mma_instruction = 2
      mma_m = cta_m * cta_m_per_mma_instruction

    # For all three kinds of convolutions, the tile shape's K mode
    # differs from GEMM in that needs to be wrapped in a Shape.
    # For Wgrad convolutions specifically,
    # the N tile shape also needs to be wrapped in a Shape.
    m_template = 'cute::_${mma_m}'
    if operation.conv_kind == ConvKind.Wgrad:
      n_template = 'cute::Shape<cute::_${mma_n}>'
    else:
      n_template = 'cute::_${mma_n}'
    k_template = 'cute::Shape<cute::_${mma_k}>'

    mma_tile_shape_template = f'cute::Shape<{m_template}, {n_template}, {k_template}>'
    values = {
      'mma_m': mma_m,
      'mma_n': mma_n,
      'mma_k': mma_k
    }
    return Template(mma_tile_shape_template).substitute(values)

  def cluster_shape(self, operation) -> str:
    m_template = 'cute::_${cluster_shape_m}' if operation.tile_description.cluster_shape[0] > 0 else 'int(0)'
    n_template = 'cute::_${cluster_shape_n}' if operation.tile_description.cluster_shape[1] > 0 else 'int(0)'
    k_template = 'cute::_${cluster_shape_k}' if operation.tile_description.cluster_shape[2] > 0 else 'int(0)'
    cluster_shape_template = f'cute::Shape<{m_template}, {n_template}, {k_template}>'
    values = {
      'cluster_shape_m': operation.tile_description.cluster_shape[0],
      'cluster_shape_n': operation.tile_description.cluster_shape[1],
      'cluster_shape_k': operation.tile_description.cluster_shape[2],
    }
    return Template(cluster_shape_template).substitute(values)

  def stage_count(self, operation) -> str:
    # stages == 0 tells builder to pick the number of stages automatically
    namespace_prefix = 'cutlass::conv::collective::'
    if operation.tile_description.stages > 0:
      return f"{namespace_prefix}StageCount<{str(operation.tile_description.stages)}>"
    else:
      return f"{namespace_prefix}StageCountAutoCarveout<sizeof(typename {operation.procedural_name()}_epilogue::SharedStorage)>"

  def emit(self, operation) -> str:
    _LOGGER.debug("*** EmitConv3xInstance::emit")
    _LOGGER.debug("***   operation: procedural_name()=" + operation.procedural_name())

    # Identify the operation as CUTLASS 3 by its is_3x field
    if (not hasattr(operation, 'is_3x')) or (not operation.is_3x):
      raise RuntimeError("operation must be a CUTLASS 3 operation")

    epi_tile_mn = "cutlass::epilogue::collective::EpilogueTileAuto"
    opcode_class_main = OpcodeClassTag[operation.tile_description.math_instruction.opcode_class]
    opcode_class_epi = opcode_class_main

    tile_shape = operation.tile_description.tile_shape
    cluster_m = operation.tile_description.cluster_shape[0]
    cluster_n = operation.tile_description.cluster_shape[1]

    cta_m, cta_n, cta_k = tile_shape
    # account for static/dynamic cluster shapes
    if operation.arch >= 100:
      cta_m = cta_m // cluster_m if cluster_m > 0 else cta_m
      cta_n = cta_n // cluster_n if cluster_n > 0 else cta_n

    warp_count = operation.tile_description.warp_count
    epilogue_schedule = EpilogueScheduleTag[operation.epilogue_schedule]

    # KernelScheduleTag and TileSchedulerTag both hard-code the
    # namespace qualification of KernelScheduleAuto as
    # "cutlass::gemm::collective::" (unless the tag is 'void').
    #
    # For TileSchedulerTag, this namespace is fine, since CUTLASS 3
    # convolutions use the same tile schedulers (from the same
    # cutlass::gemm::collective namespace) as GEMMs.
    kernel_schedule = KernelScheduleTag[operation.kernel_schedule].replace('gemm::', 'conv::')
    tile_scheduler = TileSchedulerTag[operation.tile_scheduler]
    opcode_class = OpcodeClassTag[operation.tile_description.math_instruction.opcode_class]

    values = {
      'operation_name': operation.procedural_name(),
      'conv_kind':      ConvKindTag[operation.conv_kind],
      'conv_kind_name': ConvKindNames[operation.conv_kind].capitalize(),
      'element_a':      DataTypeTag[operation.A.element],
      'layout_a':       LayoutTag[operation.A.layout],
      'align_a':        int(operation.A.alignment),
      'element_b':      DataTypeTag[operation.B.element],
      'layout_b':       LayoutTag[operation.B.layout],
      'align_b':        int(operation.B.alignment),
      'element_c':      DataTypeTag[operation.C.element],
      'layout_c':       LayoutTag[operation.C.layout],
      'align_c':        int(operation.C.alignment),
      'element_d':      DataTypeTag[operation.D.element],
      'layout_d':       LayoutTag[operation.D.layout],
      'align_d':        int(operation.D.alignment),
      'element_accumulator':   DataTypeTag[operation.accumulator_type()],
      'opcode_class':          opcode_class,
      'arch':                  self.arch_number_to_type(operation.arch),
      'mma_tile_shape':        self.mma_tile_shape(operation, cta_m, cta_n, cta_k),
      'cluster_shape':         self.cluster_shape(operation),
      'opcode_class_epi':      opcode_class_epi,
      'opcode_class_main':     opcode_class_main,
      'epi_tile_mn':           epi_tile_mn,
      'stages':                self.stage_count(operation),
      'kernel_schedule':       kernel_schedule,
      'epilogue_schedule':     epilogue_schedule,
      'tile_scheduler':        tile_scheduler,
      'element_compute':       DataTypeTag[operation.element_compute]
    }
    return Template(self.template).substitute(values)

class EmitConv3xIncludes:
  def __init__(self):
    _LOGGER.debug("*** EmitConv3xIncludes::__init__")
    self.includes = ['conv_operation_3x.hpp',
                     'cutlass/conv/device/conv_universal_adapter.hpp',
                     'cutlass/conv/kernel/conv_universal.hpp',
                     'cutlass/conv/collective/collective_builder.hpp',
                     'cutlass/epilogue/collective/collective_builder.hpp']

  def emit(self, operation) -> str:
    _LOGGER.debug("*** EmitConv3xIncludes::emit")
    return '\n'.join(f"#include \"{incl}\"" for incl in self.includes) + \
      "\n\n///////////////////////////////////////////////////////////////////////////////////////////////////"

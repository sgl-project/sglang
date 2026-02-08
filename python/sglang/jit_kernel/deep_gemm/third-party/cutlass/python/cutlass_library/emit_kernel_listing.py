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

#
#
# \brief Generates the CUTLASS kernel listing with kernel filtering
#

#

###############################################################################
# Example usage:
# generator.py --operations all --generator-target kernel_listing \
# --architectures "70;75;80" --kernels "*" --disable-cutlass-package-imports
###############################################################################

import collections
import csv
import json
import math
import os

try:
  import builtins
  if hasattr(builtins, "CUTLASS_IGNORE_PACKAGE") and CUTLASS_IGNORE_PACKAGE == True:
    raise ImportError("Disabling attempt to import cutlass_library")
  from cutlass_library.library import *
except ImportError:
  from library import *

audit_csv_fields = [
  "KernelType", "KernelName", "Type_A", "Type_B", "Type_C", "Type_Acc", "Type_EpilogueScale", "Type_D", "Type_SFA", "Type_SFD",
  "Layout_A", "Layout_B", "Layout_C", "Layout_D", 
  "Alignment_A", "Alignment_B", "Alignment_C", "Alignment_D",  
  "1SM/2SM", 
  "StreamK Enabled", "Support Runtime_Cluster_Shape", "Support Runtime_Input_Types",
  "Test Counts"
]

audit_csv_runtime_fields = [
  "KerneIndex", "KernelName", 
  "Inst_M", "Inst_N", "Inst_K", "Tile_M", "Tile_N", "Tile_K",
  "Cluster_M", "Cluster_N", "Cluster_K", "Preferred_Cluster_M", "Preferred_Cluster_N", "Preferred_Cluster_K", "Fallback_Cluster_M", "Fallback_Cluster_N", "Fallback_Cluster_K",
  "M", "N", "K", "L", "Alpha_val", "Beta_val",
  "Runtime_Input_Types Enabled", "Runtime_Cluster_Shape Enabled"
]

def hash_cutlass_string(input_string):
  mma_cluster_shape_pattern = r"_\d+x\d+x\d+"         # Matches MMA and Cluster shapes (e.g., '_128x128x256', '_0x0x1')

  # Remove MMA and Cluster shapes (e.g., '_128x128x256', '_0x0x1')
  output = re.sub(mma_cluster_shape_pattern, "", input_string)

  return output

def transform_hashed_string(hashed_kernel_name, runtime_datatype_a, runtime_datatype_b):
  # Define a dictionary mapping the detected types to runtime values
  datatype_map = {
    'f4_f4': runtime_datatype_a + '_' + runtime_datatype_b,
    'f4_f6': runtime_datatype_a + '_' + runtime_datatype_b,
    'f4_f8': runtime_datatype_a + '_' + runtime_datatype_b,
    'f6_f4': runtime_datatype_a + '_' + runtime_datatype_b,
    'f6_f6': runtime_datatype_a + '_' + runtime_datatype_b,
    'f6_f8': runtime_datatype_a + '_' + runtime_datatype_b,
    'f8_f4': runtime_datatype_a + '_' + runtime_datatype_b,
    'f8_f6': runtime_datatype_a + '_' + runtime_datatype_b,
    'f8_f8': runtime_datatype_a + '_' + runtime_datatype_b,
    'ue8m0xf4_ue8m0xf4': 'ue8m0x' + runtime_datatype_a + '_ue8m0x' + runtime_datatype_b,
    'ue4m3xf4_ue4m3xf4': 'ue4m3x' + runtime_datatype_a + '_ue4m3x' + runtime_datatype_b,
    'ue8m0xf4_ue8m0xf6': 'ue8m0x' + runtime_datatype_a + '_ue8m0x' + runtime_datatype_b,
    'ue8m0xf4_ue8m0xf8': 'ue8m0x' + runtime_datatype_a + '_ue8m0x' + runtime_datatype_b,
    'ue8m0xf6_ue8m0xf4': 'ue8m0x' + runtime_datatype_a + '_ue8m0x' + runtime_datatype_b,
    'ue8m0xf6_ue8m0xf6': 'ue8m0x' + runtime_datatype_a + '_ue8m0x' + runtime_datatype_b,
    'ue8m0xf8_ue8m0xf4': 'ue8m0x' + runtime_datatype_a + '_ue8m0x' + runtime_datatype_b,
    'ue8m0xf8_ue8m0xf6': 'ue8m0x' + runtime_datatype_a + '_ue8m0x' + runtime_datatype_b,
    'ue8m0xf8_ue8m0xf8': 'ue8m0x' + runtime_datatype_a + '_ue8m0x' + runtime_datatype_b,
  }

  # Regular expression to detect all the keys in datatype_map
  pattern = re.compile(r'(' + '|'.join(map(re.escape, datatype_map.keys())) + r')')

  # Replace detected patterns using the dictionary
  updated_kernel_name = pattern.sub(lambda match: datatype_map[match.group(0)], hashed_kernel_name)

  return updated_kernel_name

# This helper function reports foundational kernel features: datatypes, layouts, alignment and stream-k.
def get_kernel_features(operation, kernel_name,
              dynamic_datatype, runtime_input_datatype):
  numcta_inst = "2sm" if "2sm" in kernel_name else "1sm"
  math_inst = operation.tile_description.math_instruction

  if dynamic_datatype:
      dtype_name_A = runtime_input_datatype[0]
      dtype_name_B = runtime_input_datatype[1]
  else:
      dtype_name_A = DataTypeNames[operation.A.element]
      dtype_name_B = DataTypeNames[operation.B.element]

  layout_name_A = ShortLayoutTypeNames[operation.A.layout]
  layout_name_B = ShortLayoutTypeNames[operation.B.layout]
  layout_name_C = ShortLayoutTypeNames[operation.C.layout]
  layout_name_D = ShortLayoutTypeNames[operation.D.layout]

  scale_factor_D_type = operation.ScaleFactorD.element if hasattr(operation, "ScaleFactorD") else DataType.void
  scale_factor_A_type = getattr(operation, "ScaleFactorA", DataType.void)
  audit_vals = [
          "BlockScaledGEMM" if math_inst.opcode_class == OpcodeClass.BlockScaledTensorOp else "GEMM",
          kernel_name,
          dtype_name_A,
          dtype_name_B,
          DataTypeNames[operation.C.element],
          DataTypeNames[operation.tile_description.math_instruction.element_accumulator],
          DataTypeNames[operation.element_epilogue],
          DataTypeNames[operation.D.element],
          DataTypeNames[scale_factor_D_type],
          DataTypeNames[scale_factor_A_type],
          layout_name_A,
          layout_name_B,
          layout_name_C,
          layout_name_D,
          str(operation.A.alignment),
          str(operation.B.alignment),
          str(operation.C.alignment),
          str(operation.D.alignment),
          numcta_inst,
          "Y" if 'stream_k' in kernel_name else "N",
  ]
  return audit_vals

# This helper function reports other performance-related kernel parameters and those can be specified at runtime: cluster_shape, instruction shap, m/n/k and alpha/beta.
def get_kernel_params(operation, kernel_name, cluster_shape, fallback_cluster_shape, problem_shape, alpha, beta, dynamic_datatype, dynamic_cluster):
  math_inst = operation.tile_description.math_instruction
  audit_vals = [
          str(math_inst.instruction_shape[0]),
          str(math_inst.instruction_shape[1]),
          str(math_inst.instruction_shape[2]),
          str(operation.tile_description.threadblock_shape[0]),
          str(operation.tile_description.threadblock_shape[1]),
          str(operation.tile_description.threadblock_shape[2]),
          str(operation.tile_description.cluster_shape[0]),
          str(operation.tile_description.cluster_shape[1]),
          str(operation.tile_description.cluster_shape[2]),
          str(cluster_shape[0]),
          str(cluster_shape[1]),
          str(cluster_shape[2]),
          str(fallback_cluster_shape[0]),
          str(fallback_cluster_shape[1]),
          str(fallback_cluster_shape[2]),
          str(problem_shape[0]),
          str(problem_shape[1]),
          str(problem_shape[2]),
          str(problem_shape[3]),
          str(alpha),
          str(beta),
          "Y" if dynamic_datatype else "N",
          "Y" if dynamic_cluster else "N",
  ]
  return audit_vals


def _getSubOperationType(kernel):

  if kernel.operation_kind == OperationKind.Gemm:
      return GemmKindNames[kernel.gemm_kind]
  elif kernel.operation_kind == OperationKind.Conv2d:
    return "conv_" + ConvKindNames[kernel.conv_kind]
  elif kernel.operation_kind == OperationKind.Syrk:
    return "syrk_" + SyrkKindNames[kernel.syrk_kind]
  elif kernel.operation_kind == OperationKind.Trmm:
    return "trmm_" + TrmmKindNames[kernel.trmm_kind]
  elif kernel.operation_kind == OperationKind.Symm:
    return "symm_" + SymmKindNames[kernel.symm_kind]
  else:
    raise Exception("Unsupported kernel type")

def _get_inst_shape(math_instruction):
  return "".join(str(x) for x in math_instruction.instruction_shape)

def _is_simt_inst(math_instruction):
  return _get_inst_shape(math_instruction) in ["111","114"]

def _getInstType(input_precision, accumulate_precision, math_instruction):

  # inst_shape
  inst_shape = _get_inst_shape(math_instruction)

  # input precision
  if input_precision == "fp32" and inst_shape != "111":
    inp = "tf32"
  else:
    inp = input_precision

  # Handle SIMT op types first
  if _is_simt_inst(math_instruction):

    simt_input_precision_to_inst = {
      "fp32": "FFMA",
      "fp64": "DFMA",
      "fp16": "HFMA",
      "int8": "IDP4A",
    }
    inst = simt_input_precision_to_inst[input_precision]

  else: # Tensor op instructions

    if accumulate_precision == "cf64":
      fp64_acc_map = {
        MathOperation.multiply_add_complex_gaussian : "gz",
        MathOperation.multiply_add_complex          : "z",
      }
      acc = fp64_acc_map[math_instruction.math_operation]
    else:
      tensor_op_acc_map = {
        "fp32" : "s",
        "cf32" : "s",
        "fp16" : "h",
        "int32": "i",
        "fp64" : "d",
      }
      acc = tensor_op_acc_map[accumulate_precision]

    inst = "{}{}{}".format(acc, inst_shape, inp)

  return inst
# TODO: Computes FLOps/Bytes for GEMM - revisit for conv
def _computeFlopsPerByte(operation, m, n, k, batch_count=1, beta=0.0, num_groups=1):
  assert not (batch_count > 1 and num_groups > 1)

  # TODO: adjust for sparsity
  gmem_bytes = (
    (DataTypeSize[operation.A.element] * m // 8) * k +
    (DataTypeSize[operation.B.element] * n // 8) * k +
    (DataTypeSize[operation.C.element] * m // 8) * n
  )

  # TODO: complex-valued support
  flops = 2 * (m * n * k)

  if bool(beta):
    gmem_bytes += (DataTypeSize[operation.C.element] * m // 8) * n
    flops += 2 * m * n

  multiplier = max(batch_count, num_groups)
  gmem_bytes *= multiplier
  flops *= multiplier

  return flops / gmem_bytes

def emit_gemm_kernel_testlist(manifest, curr_build_dir, arch, mode
                              ):
  # For functional testing, we prefer to run reference computing on device if any
  reference_device_archs = ["100a", "103a"]
  run_reference_on_device = True if arch in reference_device_archs and mode in ["functional_L0", "functional_L1"] else False
  profiler_flags_for_verification = "device" if run_reference_on_device else "host"

  # beta values for L0 and L1
  # TODO: randomize beta values for wider coverage
  beta_values = [0.5]

  is_supported_arch = (arch in ["100a", "100f", "101a", "101f", "103a", "110a", "110f", "120a", "120f", "121a", "121f"])

  is_runtime_datatype_enabled = mode == "functional_L0" and is_supported_arch

  if (mode == "functional_L0") and is_supported_arch:
    problem_waves = [0.5, 1.25, 2.5]

    #
    # Dense Gemm
    #

    sm100_mma_data_type_general = [
      'gemm_f16_f16_f16_f16_f16',
      'gemm_f16_f16_f16_void_f16',
      #'gemm_f16_f16_f32_f16_f16',
      'tf32gemm_f32_f32_f32_f32_f32',
      'bf16gemm_f32_f32_f32_f32_f32',
    ]

    exclude_archs = arch not in ("103a")
    if exclude_archs:
      sm100_mma_data_type_general.append('gemm_s8_s8_s32_s8_s8')

    sm100_mma_data_type_runtime_dtype = [
      'gemm.*f4_f4_f32_f32_f32',
      'gemm.*f6_f6_f32_f32_f32',
      'gemm.*f8_f8_f32_f32_f32',
    ]

    sm100_mma_cluster_size = [
      '8x1x1',
      '4x4x1', '2x1x1',
      '0x0x1' # dynamic cluster
    ]

    # Restrict to two layouts to reduce L0 build and test time.
    sm100_mma_layouts = [ 
      'tnt', 
      'ntn' 
    ]

    # regex list must be in kernel procedural name order
    sm100_mma_filter_regex_1sm = "cutlass3x_sm100_tensorop.*(" + ").*(".join([ "|".join(x) for x in [sm100_mma_data_type_general, sm100_mma_cluster_size, sm100_mma_layouts]]) + ").*1sm.*"
    sm100_mma_filter_regex_2sm = "cutlass3x_sm100_tensorop.*(" + ").*(".join([ "|".join(x) for x in [sm100_mma_data_type_general, sm100_mma_cluster_size, sm100_mma_layouts]]) + ").*2sm.*"

    sm100_mma_filter_regex_1sm_runtime = "cutlass3x_sm100_tensorop.*(" + ").*(".join([ "|".join(x) for x in [sm100_mma_data_type_runtime_dtype, sm100_mma_cluster_size, sm100_mma_layouts]]) + ").*1sm.*"
    sm100_mma_filter_regex_2sm_runtime = "cutlass3x_sm100_tensorop.*(" + ").*(".join([ "|".join(x) for x in [sm100_mma_data_type_runtime_dtype, sm100_mma_cluster_size, sm100_mma_layouts]]) + ").*2sm.*"

    #
    # Block Scale Gemm
    #

    block_scaled_data_type = [
      # runtime datatypes
      'gemm.*ue8m0xf4_ue8m0xf4_f32_f16_e5m2',
      'gemm.*ue4m3xf4_ue4m3xf4_f32_f16_e5m2',
      'gemm.*ue8m0xf4_ue8m0xf6_f32_f16_e5m2',
      #'gemm.*ue8m0xf4_ue8m0xf4_f32_f16_ue8m0xe2m1',
      'gemm.*ue8m0xf6_ue8m0xf6_f32_f16_ue8m0xe3m2',
    ]

    block_scaled_tile_k = ['x128_', 'x256_']

    sm103_block_scaled_data_type = [
      'gemm.*ue8m0xf4_ue8m0xf4_f32_f16_e5m2',
      'gemm.*ue8m0xf4_ue8m0xf4_f32_f16_ue8m0xe2m1',
    ]

    sm103_block_scaled_tile_k = ['x768_']

    block_scaled_cluster_size = [
      '4x4x1', '2x1x1',
      '0x0x1' # dynamic cluster
    ]

    block_scaled_layouts = ['tnt']
    # regex list must be in kernel procedural name order
    block_scaled_filter_regex_1sm = "cutlass3x_sm100_bstensorop.*(" + ").*(".join([ "|".join(x) for x in [block_scaled_data_type, block_scaled_tile_k, block_scaled_cluster_size, block_scaled_layouts]]) + ").*1sm.*"
    block_scaled_filter_regex_2sm = "cutlass3x_sm100_bstensorop.*(" + ").*(".join([ "|".join(x) for x in [block_scaled_data_type, block_scaled_tile_k, block_scaled_cluster_size, block_scaled_layouts]]) + ").*2sm.*"
    
    sm103_block_scaled_prefetch_policy = ['tmapf']
    sm103_block_scaled_filter_regex_1sm = "cutlass3x_sm103_bstensorop.*(" + ").*(".join([ "|".join(x) for x in [sm103_block_scaled_data_type, sm103_block_scaled_tile_k, block_scaled_cluster_size, block_scaled_layouts]]) + ").*1sm.*(" + "|".join(sm103_block_scaled_prefetch_policy) + ").*"
    sm103_block_scaled_filter_regex_2sm = "cutlass3x_sm103_bstensorop.*(" + ").*(".join([ "|".join(x) for x in [sm103_block_scaled_data_type, sm103_block_scaled_tile_k, block_scaled_cluster_size, block_scaled_layouts]]) + ").*2sm.*(" + "|".join(sm103_block_scaled_prefetch_policy) + ").*"

    if arch in ["100a", "100f"]:
      kernel_filter = f"({sm100_mma_filter_regex_1sm})|" \
                      f"({sm100_mma_filter_regex_2sm})|" \
                      f"({sm100_mma_filter_regex_1sm_runtime})|" \
                      f"({sm100_mma_filter_regex_2sm_runtime})|" \
                      f"({block_scaled_filter_regex_1sm})|" \
                      f"({block_scaled_filter_regex_2sm})"
    elif arch in ["101a", "101f", "110a", "110f"]:
      kernel_filter = f"({sm100_mma_filter_regex_1sm})|" \
                      f"({sm100_mma_filter_regex_2sm})|" \
                      f"({sm100_mma_filter_regex_1sm_runtime})|" \
                      f"({sm100_mma_filter_regex_2sm_runtime})|" \
                      f"({block_scaled_filter_regex_1sm})|" \
                      f"({block_scaled_filter_regex_2sm})"
    elif arch in ["103a"]:
      kernel_filter = f"({sm100_mma_filter_regex_1sm})|" \
                      f"({sm100_mma_filter_regex_2sm})|" \
                      f"({sm100_mma_filter_regex_1sm_runtime})|" \
                      f"({sm100_mma_filter_regex_2sm_runtime})|" \
                      f"({block_scaled_filter_regex_1sm})|" \
                      f"({block_scaled_filter_regex_2sm})|" \
                      f"({sm103_block_scaled_filter_regex_1sm})|" \
                      f"({sm103_block_scaled_filter_regex_2sm})"
    elif arch in ["120a", "120f", "121a", "121f"]:

      # blockscaled sm120_mma kernels
      blockscaled_sm120_mma_kernel_cta_tiles = [
        [ '128x128' ]
      ]

      # Restrict to two layouts to reduce L0 build and test time.
      blockscaled_sm120_mma_layouts = [ 'tn' ]
      filter_regex_blockscaled_sm120_mma = "cutlass3x_sm120_bstensorop.*(" + ").*(".join([ "|".join(x) for x in [blockscaled_sm120_mma_kernel_cta_tiles[0], blockscaled_sm120_mma_layouts]]) + ").*"
      
      problem_waves = [0.5, 1.25, 2.5]

      kernel_filter = f"({filter_regex_blockscaled_sm120_mma})"
    else:
      error_message = "unsupported arch, only support sm100a, sm100f, sm101a, sm101f, sm110a, sm110f, sm103a, sm120a, sm120f, sm121a, sm121f"
      raise Exception(error_message)

  elif mode == "functional_L1":
    sm100_mma_cluster_size = [
                    '0x0x1' # dynamic cluster
                     ]
    # Restrict to two layouts to reduce L1 build and test time.
    sm100_mma_layouts = ['tnt', 'ntn']
    sm100_mma_filter_regex_1sm = "cutlass3x_sm100_tensorop.*(" + ").*(".join([ "|".join(x) for x in [sm100_mma_cluster_size, sm100_mma_layouts]]) + ").*1sm.*"
    sm100_mma_filter_regex_2sm = "cutlass3x_sm100_tensorop.*(" + ").*(".join([ "|".join(x) for x in [sm100_mma_cluster_size, sm100_mma_layouts]]) + ").*2sm.*"
    block_scaled_data_type = [
      'ue8m0xe2m1_ue8m0xe2m1_f32_f16_e5m2',
      'ue8m0xe2m1_ue8m0xe2m3_f32_f16_e5m2',
      'ue8m0xmx8s26_ue8m0xmx8s26_f32_f16_e5m2',
      'ue8m0xe2m1_ue8m0xe2m1_f32_f16_ue8m0xe2m1',
      'ue8m0xe2m3_ue8m0xe2m3_f32_f16_ue8m0xe3m2',
    ]

    sm103_block_scaled_data_type = [
      'ue8m0xe2m1_ue8m0xe2m1_f32_f16_e5m2',
      'ue8m0xe2m1_ue8m0xe2m1_f32_f16_ue8m0xe2m1',
    ]

    block_scaled_cluster_size = ['0x0x1']
    block_scaled_layouts = ['tnt']

    # regex list must be in kernel procedural name order
    block_scaled_filter_regex_1sm = "cutlass3x_sm100_bstensorop.*(" + ").*(".join([ "|".join(x) for x in [block_scaled_data_type, block_scaled_cluster_size, block_scaled_layouts]]) + ").*1sm.*"
    block_scaled_filter_regex_2sm = "cutlass3x_sm100_bstensorop.*(" + ").*(".join([ "|".join(x) for x in [block_scaled_data_type, block_scaled_cluster_size, block_scaled_layouts]]) + ").*2sm.*"

    sm103_block_scaled_filter_regex_1sm = "cutlass3x_sm103_bstensorop.*(" + ").*(".join([ "|".join(x) for x in [sm103_block_scaled_data_type, block_scaled_cluster_size, block_scaled_layouts]]) + ").*1sm.*"
    sm103_block_scaled_filter_regex_2sm = "cutlass3x_sm103_bstensorop.*(" + ").*(".join([ "|".join(x) for x in [sm103_block_scaled_data_type, block_scaled_cluster_size, block_scaled_layouts]]) + ").*2sm.*"

    filter_regex_sm100_mma = f"({sm100_mma_filter_regex_1sm})|" \
                          f"({sm100_mma_filter_regex_2sm})|" \
                          f"({block_scaled_filter_regex_1sm})|" \
                          f"({block_scaled_filter_regex_2sm})" \
                          f"({sm103_block_scaled_filter_regex_1sm})|" \
                          f"({sm103_block_scaled_filter_regex_2sm})"
    # CTA tiles for sm120 MMA - only run one tile size to reduce build/test times
    sm120_mma_kernel_cta_tiles = [
      # h1688, s1688, i16832, i8816
      [ '256x128' ],
      # d884, c1688,
      [ '128x128' ],
      # c1688, z884
      [ '128x64' ],
      # gz884
      [ '64x64' ]
    ]

    # sm120 MMA instruction shapes, planar complex type excluded as they are not required
    sm120_mma_instruction_shapes = [
      [ 'h1688gemm_(?!planar_complex)',
        's1688gemm_f16',
        's1688gemm_bf16',
        's1688gemm_tf32',
        'i16832gemm',
        'i8816gemm' ],
      [ 'd884gemm', 'c1688tf32gemm' ] ,
      [ 'c1688gemm',
        'z884gemm'  ],
      [ 'gz884gemm']
    ]

    # It's not pretty, but not sure why different instructions support different tile sizes.
    filter_regex_sm120_mma_0 = "cutlass_tensorop.*(" + ").*(".join([ "|".join(x) for x in [sm120_mma_instruction_shapes[0], sm120_mma_kernel_cta_tiles[0]]]) + ").*"
    filter_regex_sm120_mma_1 = "cutlass_tensorop.*(" + ").*(".join([ "|".join(x) for x in [sm120_mma_instruction_shapes[1], sm120_mma_kernel_cta_tiles[1]]]) + ").*"
    filter_regex_sm120_mma_2 = "cutlass_tensorop.*(" + ").*(".join([ "|".join(x) for x in [sm120_mma_instruction_shapes[2], sm120_mma_kernel_cta_tiles[2]]]) + ").*"
    filter_regex_sm120_mma_3 = "cutlass_tensorop.*(" + ").*(".join([ "|".join(x) for x in [sm120_mma_instruction_shapes[3], sm120_mma_kernel_cta_tiles[3]]]) + ").*"

    filter_regex_sm120_mma = f"({filter_regex_sm120_mma_0})|({filter_regex_sm120_mma_1})|({filter_regex_sm120_mma_2})|({filter_regex_sm120_mma_3})"

    problem_waves = [0.5, 1.25, 2.5]

    if arch in ["120a", "120f", "121a", "121f"]:
      kernel_filter = f"({filter_regex_sm120_mma})"
    else:
      kernel_filter = f"({filter_regex_sm100_mma})"
  else:
    raise ValueError()

  outfile_name    = os.path.join(curr_build_dir, f"FK_{mode}_testlist_SM{arch}_cutlass3x_gemm.csv")

  audit_file_name = os.path.join(curr_build_dir, f"FK_{mode}_audit_SM{arch}_cutlass3x_gemm.csv")

  audit_file_params_name = os.path.join(curr_build_dir, f"FK_{mode}_audit_params_SM{arch}_cutlass3x_gemm.csv")

  kernel_filter_re = re.compile(kernel_filter)
  testcase_counter = 0
  kernels_emitted = 0
  kernels_total = 0

  perf_json_list = []
  kernel_name_set = set()

  testlist_csv_fields = ["testcase", "metadata"]
  testlist_csv_rows = []
  auditlist_csv_map = {}
  auditlist_csv_params_map = {}

  kernel_features = {}

  for cc in manifest.operations[OperationKind.Gemm].keys():
    for kernel_name, operation_l in manifest.operations[OperationKind.Gemm][cc].items():
      assert(len(operation_l) == 1)
      kernels_total += 1
      if len(kernel_filter_re.findall(kernel_name)) == 0:
          continue
      # Only test f16 I/O void C kernels in void C kernel set
      # Exception: Use void C kernels for more accurate perf testing
      if '_void_' in kernel_name and  'perf_' not in mode:
        if 'f16_f16_f16_void_f16' not in kernel_name :
          continue

      kernels_emitted += 1
      kernel_name_set.add(kernel_name)
      hashed_kernel_name = hash_cutlass_string(kernel_name)
      operation = operation_l[0]

      dynamic_cluster = (operation.tile_description.cluster_shape[0] == 0
                          or operation.tile_description.cluster_shape[1] == 0)

      dynamic_datatype = "f8" in kernel_name or "f6" in kernel_name or "f4" in kernel_name

      runtime_input_datatypes = [None]

      if dynamic_datatype:
        if "f4_f4" in kernel_name:
          runtime_input_datatypes = [['e2m1','e2m1']]
        elif "f4_f6" in kernel_name:
          runtime_input_datatypes = [['e2m1','e3m2']]
        elif "f4_f8" in kernel_name:
          runtime_input_datatypes = [['e2m1','e4m3']]

        elif "f6_f4" in kernel_name:
          runtime_input_datatypes = [['e3m2','e2m1']]
        elif "f6_f6" in kernel_name:
          runtime_input_datatypes = [['e3m2','e3m2']]
        elif "f6_f8" in kernel_name:
          runtime_input_datatypes = [['e3m2','e4m3']]

        elif "f8_f4" in kernel_name:
          runtime_input_datatypes = [['e4m3','e2m1']]
        elif "f8_f6" in kernel_name:
          runtime_input_datatypes = [['e4m3','e3m2']]
        elif "f8_f8" in kernel_name:
          runtime_input_datatypes = [
                                    # mask out those not covered in statically encoded test cases
                                    #  ['e5m2','e4m3'],
                                    #  ['e4m3','e5m2'],
                                      ['e4m3','e4m3']
                                    ]

        # block scaled kernels
        elif "ue8m0xf4_ue8m0xf4" in kernel_name:
          runtime_input_datatypes = [['e2m1','e2m1']]
        elif "ue4m3xf4_ue4m3xf4" in kernel_name:
          runtime_input_datatypes = [['e2m1','e2m1']]
        elif "ue8m0xf4_ue8m0xf6" in kernel_name:
          runtime_input_datatypes = [['e2m1','e2m3']]
        elif "ue8m0xf4_ue8m0xf8" in kernel_name:
          runtime_input_datatypes = [['e2m1','e4m3']]

        elif "ue8m0xf6_ue8m0xf4" in kernel_name:
          runtime_input_datatypes = [['e2m3','e2m1']]
        elif "ue8m0xf6_ue8m0xf6" in kernel_name:
          runtime_input_datatypes = [['e2m3','e2m3']]
        elif "ue8m0xf8_ue8m0xf4" in kernel_name:
          runtime_input_datatypes = [['e4m3','e2m1']]

        elif "ue8m0xf8_ue8m0xf4" in kernel_name:
          runtime_input_datatypes = [['e4m3','e2m1']]
        elif "ue8m0xf8_ue8m0xf6" in kernel_name:
          runtime_input_datatypes = [['e4m3','e2m3']]
        elif "ue8m0xf8_ue8m0xf8" in kernel_name:
          runtime_input_datatypes = [['e4m3','e4m3']]

      if "bstensorop" in kernel_name or is_blockwise(manifest.operations_by_name[kernel_name].gemm_kind):
        profiler_flags_for_verification = "host"

      # reduce L1 test runtime if reference kernel is not running on device.
      if mode == "functional_L1" and profiler_flags_for_verification == "host" :
        problem_waves = [0.5, 2.5]
      

      if dynamic_cluster:
        if mode == "functional_L0":
          runtime_cluster_shapes = [[1,1,1],                   [2,2,1]]
        else:
          runtime_cluster_shapes = [[1,1,1], [1,2,1], [2,1,1], [2,2,1], [1,4,1], [4,1,1], [2,4,1], [4,2,1], [4,4,1]]
          # reduce L1 test runtime if reference kernel is not running on device.
          if profiler_flags_for_verification == "host":
            runtime_cluster_shapes = [[1,1,1], [1,2,1], [2,1,1], [2,2,1], [1,4,1], [4,1,1]]
        cta_tile_shape_m, cta_tile_shape_n, cta_tile_shape_k = operation.tile_description.threadblock_shape
      else:
        runtime_cluster_shapes = [operation.tile_description.cluster_shape]
        cta_tile_shape_m = int(operation.tile_description.threadblock_shape[0] / operation.tile_description.cluster_shape[0])
        cta_tile_shape_n = int(operation.tile_description.threadblock_shape[1] / operation.tile_description.cluster_shape[1])
        cta_tile_shape_k = int(operation.tile_description.threadblock_shape[2] / operation.tile_description.cluster_shape[2])

      alignment_a = operation.A.alignment
      alignment_b = operation.B.alignment
      alignment_c = operation.C.alignment
      alignment_ab_max = max(alignment_a, alignment_b)

      layout3x = operation.layout_name_3x()
      data_types = operation.datatype_name_3x()

      ctas_per_mma_instruction = 1
      if '_2sm' in kernel_name:
        ctas_per_mma_instruction = 2
        valid_cluster_shapes = []

        # Remove any cluster shapes that have cluster_m that is not divisible by 2
        for cs in runtime_cluster_shapes:
          if cs[0] % 2 == 0:
            valid_cluster_shapes.append(cs)
        runtime_cluster_shapes = valid_cluster_shapes

      kernel_problem_waves = problem_waves
      if mode == "functional_L0" or mode == "functional_L1":
        # for functional testing, we want to perturb just a little from even shapes
        # large K = 8 is chosen such that some kernels will warp around their smem buffers, and some will not
        # -16 ensures that we are TMA aligned even for FP8/Int8
        min_k = alignment_ab_max if cta_tile_shape_k == alignment_ab_max else cta_tile_shape_k - alignment_ab_max
        max_k = (cta_tile_shape_k*8) - alignment_ab_max
        problem_shapes_k = [min_k, max_k]
        sm_count = 16
        swizzle_sizes = [0]
        # Larger k and less than half wave trigger streamk +separate reduction case to be generated
        if 'stream_k' in kernel_name:
          problem_shapes_k = [max_k, cta_tile_shape_k*32]
          kernel_problem_waves = [0.125, 1.25, 2.5]
      else:
        raise ValueError

      if "void" in kernel_name:
        beta_values = [0]

      alignment_shift_m = max(alignment_c, alignment_a)
      alignment_shift_n = max(alignment_c, alignment_b)

      is_first_line = True
      for index_waves, waves in enumerate(kernel_problem_waves):
        for index_k, k in enumerate(problem_shapes_k):
          for beta in beta_values:
            for cluster_shape in runtime_cluster_shapes:
              for runtime_input_datatype in runtime_input_datatypes:
                for swizzle_size in swizzle_sizes:
                  grid_size = waves * sm_count
                  cluster_shape_m, cluster_shape_n, cluster_shape_k = tuple(cluster_shape)
                  if cluster_shape_m >= cluster_shape_n:
                    grid_m = cluster_shape_m
                    grid_n = grid_size / grid_m
                    grid_n = max( int((grid_n + cluster_shape_n - 1) / cluster_shape_n) * cluster_shape_n, 1)
                  else:
                    grid_n = cluster_shape_n
                    grid_m = grid_size / grid_n
                    grid_m = max( int((grid_m + cluster_shape_m - 1) / cluster_shape_m) * cluster_shape_m, 1)

                  verification_required = False
                  if mode == "functional_L0" or mode == "functional_L1":
                    if '_void_' not in kernel_name:
                      verification_required = True

                    m = max(int(grid_m * cta_tile_shape_m), alignment_ab_max)
                    n = max(int(grid_n * cta_tile_shape_n), alignment_ab_max)
                    k = int(k)

                    # For functional testing, we want to perturb just a little from even shapes.
                    # Only do this if the perturbation does not cause one of the dimensions of the
                    # problem size to go to zero. This can occur for blockscaling kernels for which
                    # the alignment requirements for A and B can be quite large (e.g., 256).
                    if m > alignment_shift_m:
                      m -= alignment_shift_m
                    if n > alignment_shift_n:
                      n -= alignment_shift_n

                    if '_n32t32_' in kernel_name:
                      continue
                  batch_count = 1
                  if mode == "functional_L0" or mode == "functional_L1" :
                    if index_waves == 0 and index_k == 0 :
                      batch_count = 3 if mode == "functional_L0" else 5
                  gemm_op = "gemm"

                  grouped = is_grouped(manifest.operations_by_name[kernel_name].gemm_kind)
                  num_groups = 1
                  if grouped:
                    gemm_op = "grouped_gemm"
                    num_groups = 3 # small to limit test time in host block-scaled reference kernels
                    batch_count = 1
                  elif "bstensorop" in kernel_name:
                    gemm_op = "block_scaled_gemm"
                  elif is_blockwise(manifest.operations_by_name[kernel_name].gemm_kind):
                    gemm_op = "blockwise_gemm"

                  problem_size_category = ['smallK','largeK'][index_k] + '_' + ['beta==0','beta!=0'][bool(beta)]

                  assert m > 0 and n > 0 and k > 0

                  # Emit per-testcase metadata for perf testing usage, eventually in perf database
                  metadata_dict = {
                    "input_params": {
                      'problem_size_category' : problem_size_category,
                      'operation' : _getSubOperationType(operation),
                      'datatype' : data_types,
                      'layout' : layout3x,
                      'm' : m,
                      'n' : n,
                      'k' : k,
                      'beta' : beta,
                      'flops_per_byte' : _computeFlopsPerByte(operation, m, n, k, batch_count, beta, num_groups)
                    },
                    "runtime_params": {
                      'ctas_per_mma_instruction' : ctas_per_mma_instruction,
                      'tilesize_m' : cta_tile_shape_m,
                      'tilesize_n' : cta_tile_shape_n,
                      'tilesize_k' : cta_tile_shape_k,
                      'cluster_shape_m' : cluster_shape_m,
                      'cluster_shape_n' : cluster_shape_n,
                    }
                  }

                  cluster_m_fallback = ctas_per_mma_instruction if dynamic_cluster else cluster_shape_m
                  cluster_n_fallback = 1 if dynamic_cluster else cluster_shape_n
                  cluster_k_fallback = 1 if dynamic_cluster else cluster_shape_k


                  if dynamic_datatype:
                    runtime_datatype_a, runtime_datatype_b = tuple(runtime_input_datatype)
                    metadata_dict["runtime_params"]["runtime_datatype_a"] = runtime_datatype_a
                    metadata_dict["runtime_params"]["runtime_datatype_b"] = runtime_datatype_b

                  testcase_metadata = [
                    f"cutlass_profiler --operation={gemm_op}" +
                    (f" --verification-providers=device --providers=cutlass" if profiler_flags_for_verification == "device" else " --mode=trace") +
                    f" --error-on-no-match --error-if-nothing-is-profiled" +
                    f" --kernels={kernel_name}" +
                    f" --m={str(m)}" +
                    f" --n={str(n)}" +
                    f" --k={str(k)}" +
                    (f" --num_groups={str(num_groups)}" if grouped else "") +
                    f" --cluster_m={str(cluster_shape_m)}" +
                    f" --cluster_n={str(cluster_shape_n)}" +
                    f" --cluster_k={str(cluster_shape_k)}" +
                    f" --cluster_m_fallback={str(cluster_m_fallback)}" +
                    f" --cluster_n_fallback={str(cluster_n_fallback)}" +
                    f" --cluster_k_fallback={str(cluster_k_fallback)}" +
                    f" --beta={str(beta)}" +
                    ("" if grouped else f" --batch_count={str(batch_count)}") +
                    f" --swizzle_size={str(swizzle_size)}" +
                    f" --verification-required={str(verification_required).lower()}"
                  ] \

                  output_dynamic_datatype = dynamic_datatype
                  if output_dynamic_datatype:
                    testcase_metadata[0] += (f" --runtime_input_datatype_a={runtime_datatype_a}" +
                                              f" --runtime_input_datatype_b={runtime_datatype_b}")

                  testcase_metadata.append(json.dumps(metadata_dict))
                  testlist_csv_rows.append(testcase_metadata)
                  testcase_counter += 1

                  alpha = 1.0

                  if dynamic_datatype:
                    hashed_kernel_name = transform_hashed_string(hashed_kernel_name, runtime_datatype_a, runtime_datatype_b)

                  # If kernel_name is new, initialize its feature set with defaults
                  if hashed_kernel_name not in kernel_features:
                    kernel_features[hashed_kernel_name] = {
                      "is_support_dynamic_cluster": False,
                      "is_support_dynamic_datatype": False,
                    }

                  # Update features for the hashed kernel name
                  kernel_features[hashed_kernel_name]["is_support_dynamic_cluster"] |= dynamic_cluster
                  kernel_features[hashed_kernel_name]["is_support_dynamic_datatype"] |= dynamic_datatype

                  if hashed_kernel_name not in auditlist_csv_params_map:
                    auditlist_csv_params_map[hashed_kernel_name] = []

                  audit_row_params = get_kernel_params(
                    operation,
                    hashed_kernel_name,
                    (cluster_shape_m, cluster_shape_n, cluster_shape_k),
                    (cluster_m_fallback, cluster_n_fallback, cluster_k_fallback),
                    (m, n, k, batch_count),
                    alpha, beta,
                    dynamic_datatype, dynamic_cluster
                  )

                  auditlist_csv_params_map[hashed_kernel_name].append(audit_row_params)

                  if hashed_kernel_name not in auditlist_csv_map:
                    audit_row = get_kernel_features(operation, hashed_kernel_name, dynamic_datatype, runtime_input_datatype)
                    auditlist_csv_map[hashed_kernel_name] = audit_row

  with open(outfile_name, 'w') as testlist_csv:
    csv_writer = csv.writer(testlist_csv, delimiter=',')
    csv_writer.writerow(testlist_csv_fields)
    csv_writer.writerows(testlist_csv_rows)

  with open(audit_file_name, 'w') as auditlist_csv:
    csv_writer = csv.writer(auditlist_csv, delimiter=',')
    csv_writer.writerow(audit_csv_fields)
    for hashed_kernel_name, row in auditlist_csv_map.items():
      # Append the dynamic features as "Y" or "N"
      dynamic_cluster_flag = "Y" if kernel_features[hashed_kernel_name]["is_support_dynamic_cluster"] else "N"
      dynamic_datatype_flag = "Y" if kernel_features[hashed_kernel_name]["is_support_dynamic_datatype"] else "N"
      test_count = len(auditlist_csv_params_map[hashed_kernel_name])
      csv_writer.writerow(row + [dynamic_cluster_flag, dynamic_datatype_flag, test_count])

  with open(audit_file_params_name, 'w') as auditlist_csv:
    csv_writer = csv.writer(auditlist_csv, delimiter=',')
    csv_writer.writerow(audit_csv_runtime_fields)
    for kernel_index, (hashed_kernel_name, rows) in enumerate(auditlist_csv_params_map.items(), start=1):
      for i, row in enumerate(rows):
        if i == 0:
          csv_writer.writerow([kernel_index, hashed_kernel_name] + row)
        else:
          csv_writer.writerow(["", ""] + row)

  print(f"Generated a total of {testcase_counter} test cases for {kernels_emitted} kernels out of {kernels_total} total.")

  # Generate a newline separated list of kernel filters
  assert(len(kernel_name_set) == kernels_emitted)
  output_filter_enabled = True
  if output_filter_enabled:
    kernel_filter_outfile_name = os.path.join(curr_build_dir, f"FK_{mode}_testlist_SM{arch}_cutlass3x_gemm_kernel_filter.list")
  with open(kernel_filter_outfile_name, "w") as file:
      kernel_name_set = set(map(lambda x: x.replace("_epi_tma", ""), kernel_name_set))
      for kernel_name in kernel_name_set:
          file.write(kernel_name + "\n")

  # Sort L0 and L1 kernel list and csv file to avoid mixing cutlass3.x kernels and sm120_mma kernels in cutlass2.x generated together.
  if mode == "functional_L0" or mode == "functional_L1":
    # Sort the .csv file
    outfile_name = os.path.join(curr_build_dir, f"FK_{mode}_testlist_SM{arch}_cutlass3x_gemm.csv")
    with open(outfile_name) as file:
      data = file.readlines()
      data.sort()
    with open(outfile_name, 'w') as file:
      for i in range(len(data)):
        file.write(data[i])
    # Sort the kernel list
    kernel_filter_outfile_name = os.path.join(curr_build_dir, f"FK_{mode}_testlist_SM{arch}_cutlass3x_gemm_kernel_filter.list")
    with open(kernel_filter_outfile_name) as file:
      data = file.readlines()
      data.sort()
    with open(kernel_filter_outfile_name, 'w') as file:
      for i in range(len(data)):
        file.write(data[i])


#################################################################################################
#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Utilities for selecting CUTLASS library kernels based on problem description
"""
import json
import csv

try:
  import builtins
  if hasattr(builtins, "CUTLASS_IGNORE_PACKAGE") and CUTLASS_IGNORE_PACKAGE == True:
    raise ImportError("Disabling attempt to import cutlass_library")
  from cutlass_library.library import *
  from cutlass_library.generator import *
  from cutlass_library.heuristics_provider import *
except ImportError:
  from library import *
  from generator import *
  from heuristics_provider import *

try:
  from .sm90_utils import (
    get_valid_schedules,
    generate_data_types_from_math_instruction,
    fix_alignments,
  )
except ImportError:
  from sm90_utils import (
    get_valid_schedules,
    generate_data_types_from_math_instruction,
    fix_alignments,
  )

_LOGGER = logging.getLogger(__name__)

dtype_map = {v: k for k, v in DataTypeNames.items()}

def serialize_heuristics_results_to_json(problems_with_configs, outfile_path):
  """
  Utilitiy function to write heuristics results to a json file for debug

  args:
    problems_with_configs: List of problems provided to the heuristic, with a list of operations added to each problem dict
    outfile_path: Outfile path
      
  returns:
    None
  """
  pc_copy = problems_with_configs.copy()
  for p in pc_copy:
    for k, v in p.items():
      if isinstance(v, DataType):
        p[k] = DataTypeNames[v]
      elif isinstance(v, LayoutType):
        p[k] = ShortLayoutTypeNames[v]
    configs = p['configs']
    for c in configs:
      for k, v in c.items():
        if isinstance(v, DataType):
          c[k] = DataTypeNames[v]
        elif isinstance(v, LayoutType):
          c[k] = ShortLayoutTypeNames[v]
  with open(outfile_path, 'w') as f:
    json.dump(pc_copy, f, indent=2)

def get_single_gemm_config(m, n, k, batch_count, layouts, dtypes, alignment_a, alignment_b, voidC=False, use_fast_acc=True, count=1, provider=None):
  """
  Get heuristic-suggested GEMM kernel configurations for a single GEMM problem.

  args:
    m, n, k: GEMM dimensions
    batch_count: batch count
    layouts: tuple of layouts of type LayoutType
    use_fast_acc: Use fast accumulation for FP8. Ignored for other precisions
    count: Number of configs to return
    provider: Heuristics provider to use

  returns:
    A list of dictionaries containing the suggested kernel configurations and additional info from the input required to define a Cutlass GemmOperation, with the following keys:
      - 'cta_tile_m', 'cta_tile_m', 'cta_tile_k': CTA tile size
      - 'instr_tile_m', 'instr_tile_n', 'instr_tile_k': Instruction tile size
      - 'stages': kernel pipeline stage count
      - 'cluster_m', 'cluster_n', 'cluster_k': cluster size
      - 'layout_a', 'layout_b': input tensor layouts of type LayoutType
      - 'alignment_a', 'alignment_b': input tensor alignments, in count of elements
      - 'dtype_a', 'dtype_b', 'dtype_acc': dtypes of a, b, and accumulator, of type DataType
      - 'swizzle_size' : suggested threadblock swizzle 
      - 'split_k_slices': number of partitions of the k dimension for splitK
      - 'raster_order': raster order for CTAs over output tiles ('along_m' or 'along_n')
  """
  if provider is None:
    provider = MatmulHeuristics()
  return provider.get_configs(m, n, k, batch_count, dtypes, layouts, alignment_a, alignment_b, voidC=voidC, use_fast_acc=use_fast_acc, count=count)

def get_gemm_configs(problems, provider=None, count=1):
  """
  Get heuristic-suggested GEMM kernel configurations for a set of GEMM problems.

  args:
    problems: List of dictionaries describing GEMM problems with the following keys:
      - 'm', 'n', 'k': Matrix dimensions (required)
      - 'dtype_a': Data type of matrix A (required)
      - 'dtype_b': Data type of matrix B (required)
      - 'dtype_c': Data type of matrix C (default: None)
      - 'dtype_d': Data type of matrix D (required)
      - 'dtype_acc': Compute data type (default 'f32')
      - 'layout': Operation layout (e.g. 'tnt')
      - 'alignment_a': Memory access granularity of A, in units of elements (default: 16 bytes equivalent elements)
      - 'alignment_b': Memory access granularity of B, in units of elements (default: 16 bytes equivalent elements)
      - 'alpha': Scalar multiplier for A*B (default: 1.0)
      - 'beta': Scalar multiplier for C (default: 0.0)
      - 'batch_count': Number of GEMM operations in batch (default: 1)
      - 'use_fast_acc': Enable fast accumulation for FP8 on Hopper (default: True)
    provider: Heuristics provider to use
    count: Number of configurations to return per problem (defualt: 1)
      
  returns:
    A copy of the input dictionary, with key `configs` added containing the selected gemm configs
  """
  ret = []

  for problem in problems:
    problem = problem.copy()

    try:
      m = problem['m']
      n = problem['n']
      k = problem['k']
      dtype_a = problem['dtype_a']
      dtype_b = problem['dtype_b']
      dtype_d = problem['dtype_d']
      layout = problem['layout']
    except KeyError as e:
      _LOGGER.error(f"Missing required parameter {e} for problem {problem}")
      raise

    operation = problem.get('operation', 'gemm')
    batch_count = problem.get('batch_count', 1)
    dtype_acc = problem.get('dtype_acc', 'f32')
    dtype_c = problem.get('dtype_c', None)
    alpha = problem.get('alpha', 1.0)
    beta = problem.get('beta', 0.0)
    use_fast_acc = problem.get('use_fast_acc', True)

    if operation != OperationKindNames[OperationKind.Gemm]:
      raise ValueError(f"Unsupported operation {operation}")
    if not (len(layout) == 3 and all(c in "nt" for c in layout)):
      raise ValueError(f"layout must be a 3-character string containing only 'n' or 't', got {layout}")
    layouts = tuple(LayoutType.RowMajor if l == 't' else LayoutType.ColumnMajor for l in layout)

    try:
      dtype_list = [dtype_a.lower(), dtype_b.lower(), dtype_acc.lower(), dtype_c.lower() if dtype_c is not None else dtype_d.lower(), dtype_d.lower()]
      dtypes = tuple(dtype_map[dt] for dt in dtype_list)
    except KeyError as dt:
      _LOGGER.error(f"Unsupported data type: {dt}")
      raise

    alignment_a = problem.get('alignment_a', 128 // DataTypeSize[dtypes[0]])
    alignment_b = problem.get('alignment_b', 128 // DataTypeSize[dtypes[1]])

    configs = get_single_gemm_config(m, n, k, batch_count, layouts, dtypes, alignment_a, alignment_b, beta==0.0, use_fast_acc, count, provider)
    problem['configs'] = configs

    ret.append(problem)

  return ret


def generate_sm100_from_heuristics_configs(manifest, cuda_version, kernel_configs):
  """
  Generate CUTLASS operations based on the list of configs provided by the heuristic provider

  args:
    manifest: manifest argument to which to add operations, or None to just return the operations without a manifest (for pruning an existing manifest)
    cuda_version: Cuda compiler version for generating cutlass operations
    kernel_configs: list of configs generated by the heuristic
      
  returns:
    (configs, operations): a list of heuristic-provided kernel configs along with a one-to-one corresponding list of the generated operations
  """
  min_cc = 100
  max_cc = 101
  if manifest is None:
    # Use a dummy manifest so we can use existing CreateGemmOperator functions
    manifest = Manifest()

  configs = []
  operations = []
  for config in kernel_configs:
    layout = ([config['layout_a'], config['alignment_a']], [config['layout_b'], config['alignment_b']], [config['layout_d'], 128 // DataTypeSize[config['dtype_d']]])
    element_a, element_b, element_accumulator, element_c, element_d = config['dtype_a'], config['dtype_b'], config['dtype_acc'], config['dtype_c'], config['dtype_d']

    # nvMMH assumes 2sm instruction for !(cluster_m % 2)
    is_2sm = config['cluster_m'] % 2 == 0
    instruction_shape = [(2 * config['cta_tile_m']) if is_2sm else config['cta_tile_m'], config['cta_tile_n'], config['cta_tile_k'] // 4]
    math_instruction = MathInstruction(
      instruction_shape,
      element_a, element_b, element_accumulator,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add
    )

    data_types = [
      {
        "a_type"   : math_instruction.element_a,
        "b_type"   : math_instruction.element_b,
        "c_type"   : DataType.void if config['voidC'] else math_instruction.element_accumulator,
        "d_type"   : element_d,
        "acc_type" : math_instruction.element_accumulator,
        "epi_type" : math_instruction.element_accumulator,
      }
    ]

    tile_multiplier = (config['cluster_m'] // (2 if is_2sm else 1), config['cluster_n'], config['cluster_k'])
    tile_description = TileDescription(
      [instruction_shape[0] * tile_multiplier[0],
       instruction_shape[1] * tile_multiplier[1],
       instruction_shape[2] * 4 * tile_multiplier[2]],
      0,
      [4,1,1],
      math_instruction,
      min_cc,
      max_cc,
      cluster_shape=(config['cluster_m'], config['cluster_n'], config['cluster_k'])
    )

    schedules = []
    if is_2sm:
      schedules.append([KernelScheduleType.TmaWarpSpecialized2SmSm100, EpilogueScheduleType.TmaWarpSpecialized2Sm])
    else:
      schedules.append([KernelScheduleType.TmaWarpSpecialized1SmSm100, EpilogueScheduleType.TmaWarpSpecialized1Sm])

    for o in CreateGemmUniversal3xOperator(manifest, [layout], [tile_description], data_types, schedules, tile_schedulers=[TileSchedulerType.Default, TileSchedulerType.StreamK], gemm_kind=GemmKind.Universal3x):
      configs.append(config)
      operations.append(o)

 
  return configs, operations


def generate_sm90_from_heuristics_configs(manifest, cuda_version, kernel_configs):
  """
  Generate CUTLASS operations based on the list of configs provided by the heuristic provider

  args:
    manifest: manifest argument to which to add operations, or None to just return the operations without a manifest (for pruning an existing manifest)
    cuda_version: Cuda compiler version for generating cutlass operations
    kernel_configs: list of configs generated by the heuristic
      
  returns:
    (configs, operations): a list of heuristic-provided kernel configs along with a one-to-one corresponding list of the generated operations
  """
  min_cc, max_cc = 90, 90

  if manifest is None:
    # Use a dummy manifest so we can use existing CreateGemmOperator functions
    manifest = Manifest()

  configs = []
  operations = []
  for config in kernel_configs:

    is_aligned = (config['alignment_a'] * DataTypeSize[config['dtype_a']] >= 128) and (config['alignment_b'] * DataTypeSize[config['dtype_b']] >= 128)
    layout = ([config['layout_a'], config['alignment_a']], [config['layout_b'], config['alignment_b']], [LayoutType.ColumnMajor, 1])
    element_a, element_b, element_accumulator, element_c, element_d = config['dtype_a'], config['dtype_b'], config['dtype_acc'], config['dtype_c'], config['dtype_d']

    # instr shape and warp config are unused for emitting 3x collective builder code
    dummy_instr_shape = [0, 0, 0]
    math_instruction = MathInstruction(
      dummy_instr_shape,
      element_a, element_b, element_accumulator,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add
    )

    data_types = generate_data_types_from_math_instruction(math_instruction, element_source=element_c, element_dest=element_d)
    if is_aligned:
      layout = fix_alignments(data_types, layout, alignment_bits=128)

    # instr shape and warp config are unused for emitting 3x collective builder code
    dummy_warp_count = [0, 0, 0]
    tile_description = TileDescription(
      [config['cta_tile_m'], config['cta_tile_n'], config['cta_tile_k']],
      0,
      dummy_warp_count,
      math_instruction,
      min_cc,
      max_cc,
      cluster_shape=(config['cluster_m'], config['cluster_n'], config['cluster_k'])
    )

    schedules, stream_k_schedules = get_valid_schedules(
      tile_description=tile_description,
      cuda_version=cuda_version,
      is_aligned=is_aligned,
      data_types=data_types,
      instantiation_level=9000, # don't prune schedules: we didn't get any schedule suggestion from the heuristic
      layout=layout,
      gemm_kind=GemmKind.Universal3x,
      enable_fp8_fast_acc=config['use_fast_acc']
    )

    if len(schedules):
      for o in CreateGemmUniversal3xOperator(manifest, [layout], [tile_description], data_types, schedules, gemm_kind=GemmKind.Universal3x):
        configs.append(config)
        operations.append(o)

    if len(stream_k_schedules):
      for o in CreateGemmUniversal3xOperator(manifest, [layout], [tile_description], data_types,
                                    stream_k_schedules,
                                    tile_schedulers=[TileSchedulerType.StreamK]):
        configs.append(config)
        operations.append(o)


  return configs, operations

def filter_manifest_and_write_heuristics_file(manifest, args):
  """
  Prune a manifest according to heuristics suggestions from the problems file

  args:
    manifest: Cutlass manifest to prune
    args: generator.py args, requires:
      - args.heuristics_problems_file
      - args.heuristics_gpu
      - args.heuristics_testlist_file
      
  returns:
    A list of dictionaries, each of which has information about an operation and a problem from the input problems
  """
  heuristics_problems = []
  with open(args.heuristics_problems_file, 'r') as f:
    heuristics_problems = json.load(f)
  gpu = None if (args.heuristics_gpu == "auto" or args.heuristics_gpu == "") else args.heuristics_gpu
  mmh = MatmulHeuristics(gpu=gpu)
  if any(('100' in arch) for arch in args.architectures.split(';')):
    mmh.set_cta_div_n(64)
  problems_with_configs = get_gemm_configs(heuristics_problems, provider=mmh, count=args.heuristics_configs_per_problem)

  all_configs_and_operations = []
  operations = []
  for problem in problems_with_configs:
    if any('90' in arch for arch in args.architectures.split(';')):
        problem_configs, problem_operations = generate_sm90_from_heuristics_configs(None if args.heuristics_restrict_kernels else manifest, args.cuda_version, problem['configs'])
    if any(('100' in arch) or ('101' in arch) for arch in args.architectures.split(';')):
        problem_configs, problem_operations = generate_sm100_from_heuristics_configs(None if args.heuristics_restrict_kernels else manifest, args.cuda_version, problem['configs'])
        
    operations += problem_operations
    problem_without_configs = {k: v for k, v in problem.items() if k != 'configs'}
    with_problem_size = [{'operation_name': o.procedural_name(), **problem_without_configs, **c} for c, o in zip(problem_configs, problem_operations)]
    all_configs_and_operations += with_problem_size

  for operation in operations:
    manifest.add_kernel_filter(f"^{operation.procedural_name()}$")
  if not all_configs_and_operations:
    raise Exception("No valid configurations generated")
  write_profiler_testlist_to_csv(all_configs_and_operations, args.heuristics_testlist_file)
  return all_configs_and_operations

def write_profiler_testlist_to_csv(configs_list, outfile_path):
  """
  Write a list of configs to a testlist to be consumed by cutlass_profiler

  args:
    configs_list: List of kernel configs along with runtime arguments and any other columns to include in the CSV, expressed as a list of dictionaries
    outfile_path: Outfile path
      
  returns:
    None
  """
  profiler_testlist = configs_list.copy()
  for c in profiler_testlist:
    for k, v in c.items():
      if isinstance(v, DataType):
        c[k] = DataTypeNames[v]
      elif isinstance(v, LayoutType):
        c[k] = ShortLayoutTypeNames[v]

  with open(outfile_path, mode='w', newline='') as ofile:
    k_names = profiler_testlist[0].keys()

    writer = csv.DictWriter(ofile, fieldnames=k_names)
    writer.writeheader()
    writer.writerows(profiler_testlist)

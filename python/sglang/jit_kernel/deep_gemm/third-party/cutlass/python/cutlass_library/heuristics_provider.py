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
Providers for kernel selection heuristics
"""

import sys
import os
import glob
import logging
import ctypes
import functools


try:
  import builtins
  if hasattr(builtins, "CUTLASS_IGNORE_PACKAGE") and CUTLASS_IGNORE_PACKAGE == True:
    raise ImportError("Disabling attempt to import cutlass_library")
  from cutlass_library.library import DataType, LayoutType
except ImportError:
  from library import DataType, LayoutType

class MatmulHeuristics:

  def __init__(self, gpu = None):
    import nvMatmulHeuristics
    self.mmh_lib = nvMatmulHeuristics
    self.gpu = gpu

    if 'CUTLASS_NVMMH_SO_PATH' in os.environ:
      nvmmhInterfaceEx = functools.partial(self.mmh_lib.NvMatmulHeuristicsInterfaceEx, path=os.environ['CUTLASS_NVMMH_SO_PATH'])
    else:
      nvmmhInterfaceEx = self.mmh_lib.NvMatmulHeuristicsInterfaceEx

    self.lh = nvmmhInterfaceEx(
      backend=self.mmh_lib.NvMatmulHeuristicsTarget["CUTLASS3"],
      flags=self.mmh_lib.NvMatmulHeuristicsFlags.PERF_MODEL_BASED_AUTO_TUNING,
      load_discovery_implicitly=True,
      gpu=self.mmh_lib.NvMatmulHeuristicsNvidiaGpu[self.gpu] if self.gpu else None
    )
    self.backend = self.lh.createBackend(self.mmh_lib.NvMatmulHeuristicsTarget["CUTLASS3"])

  def _layout_from_cutlass(self, layouts):
    assert(len(layouts)==3)
    full_layout_str = ''.join('t' if l == LayoutType.RowMajor else 'n' for l in layouts)
    input_layouts = full_layout_str[:2].upper() 
    lh_layout = input_layouts + '_' + str("ROW_MAJOR" if full_layout_str[-1]=='t' else "COL_MAJOR")
    return self.mmh_lib.NvMatmulHeuristicsMatmulLayout[lh_layout]

  def _precision_from_cutlass_dtypes(self, dtypes):
    dtype_to_cublas = {
      DataType.f64: 'D',
      DataType.f32: 'S',
      DataType.f16: 'H',
      DataType.bf16: 'T',
      DataType.e4m3: 'Q',
      DataType.e5m2: 'R',
      DataType.s32: 'I',
      DataType.s8: 'B',
    }

    dtype_a, dtype_b, dtype_compute, dtype_c, dtype_d = dtypes

    a_c = dtype_to_cublas[dtype_a]

    if a_c.lower() != 'q':
      return a_c + dtype_to_cublas[dtype_compute] + dtype_to_cublas[dtype_d]
    else:
      return a_c + dtype_to_cublas[dtype_b] + dtype_to_cublas[dtype_c] + dtype_to_cublas[dtype_compute] + dtype_to_cublas[dtype_d]

  def set_cta_div_n(self, div_n):
    cta_n_div_requirement = ctypes.c_int(div_n) 
    self.lh.setBackendValueProperty(
      self.backend,
      self.mmh_lib.NvMatmulHeuristicsBackendProperty.CTA_TILE_N_DIV_REQUIREMENT,
      ctypes.byref(cta_n_div_requirement),
      ctypes.sizeof(cta_n_div_requirement)
    )

  def set_cta_div_m(self, div_m):
    cta_m_div_requirement = ctypes.c_int(div_m) 
    self.lh.setBackendValueProperty(
      self.backend,
      self.mmh_lib.NvMatmulHeuristicsBackendProperty.CTA_TILE_M_DIV_REQUIREMENT,
      ctypes.byref(cta_m_div_requirement),
      ctypes.sizeof(cta_m_div_requirement)
    )

  def get_configs(self, m, n, k, batch_count, dtypes, layouts, align_a, align_b, voidC=False, use_fast_acc=True, count=1):
    if use_fast_acc:
      disable_fast_acc_for_fp8 = ctypes.c_int(0)
    else:   
      disable_fast_acc_for_fp8 = ctypes.c_int(1)
    self.lh.setBackendValueProperty(
      self.backend,
      self.mmh_lib.NvMatmulHeuristicsBackendProperty.DISABLE_FAST_ACC_FOR_FP8,
      ctypes.byref(disable_fast_acc_for_fp8),
      ctypes.sizeof(disable_fast_acc_for_fp8)
    )

    precision = self._precision_from_cutlass_dtypes(dtypes)
    layout = self._layout_from_cutlass(layouts)

    matmul_problem = self.lh.makeNvMatmulHeuristicsProblem(m, n, k, layout, batch_count)
    configs = self.lh.getEx(matmul_problem, count, self.backend, precision=precision)

    ret = []
    for c in configs:
      kernel = c['kernel']
      problem = c['problem']

      r = {}
      r['estimated_runtime'] = c['runtime']
      r['cta_tile_m'] = kernel.cta_tile_m
      r['cta_tile_n'] = kernel.cta_tile_n
      r['cta_tile_k'] = kernel.cta_tile_k
      r['instr_tile_m'] = kernel.instr_tile_m
      r['instr_tile_n'] = kernel.instr_tile_n
      r['instr_tile_k'] = kernel.instr_tile_k
      r['warp_tile_m'] = kernel.warp_tile_m
      r['warp_tile_n'] = kernel.warp_tile_n
      r['warp_tile_k'] = kernel.warp_tile_k
      r['cluster_m'] = kernel.cluster_m
      r['cluster_n'] = kernel.cluster_n
      r['cluster_k'] = 1
      r['layout_a'] = layouts[0]
      r['layout_b'] = layouts[1]
      r['layout_d'] = layouts[2]
      r['dtype_a'] = dtypes[0]
      r['dtype_b'] = dtypes[1]
      r['dtype_acc'] = dtypes[2]
      r['dtype_c'] = dtypes[3]
      r['dtype_d'] = dtypes[4]
      r['alignment_a'] = align_a
      r['alignment_b'] = align_b
      r['swizzle_size'] = kernel.swizzle_factor
      r['raster_order'] = 'along_m' if kernel.cta_order==0 else 'along_n'
      r['split_k_slices'] = kernel.split_k
      r['use_fast_acc'] = use_fast_acc
      r['voidC'] = voidC

      ret.append(r)

    return ret


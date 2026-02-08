#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Low-level functionality tests for GEMM with F16 operands on SM90
"""

from functools import partial
import logging
import unittest

import cutlass_cppgen
from cutlass_cppgen.backend.utils.device import device_cc

from utils import LayoutCombination, add_test_gemm


cutlass_cppgen.set_log_level(logging.WARNING)
cc = 90
dtype = cutlass_cppgen.DataType.f16

@unittest.skipIf(device_cc() < cc, 'Device compute capability is insufficient for SM90 tests.')
@unittest.skipIf(cutlass_cppgen.utils.datatypes.torch_type(dtype) is None, f'Version of torch installed does not contain a datatype match for {dtype}')
class GemmF16Sm90(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass


add_test_specialized = partial(add_test_gemm, cls=GemmF16Sm90, element=dtype,
                               warp_count=None, compilation_modes=['nvcc'])

add_test_tensorop = partial(add_test_specialized, opclass=cutlass_cppgen.OpcodeClass.TensorOp)

# Tests with 1x1x1 clusters
add_test_unit_cluster = partial(add_test_tensorop, cluster_shape=[1, 1, 1])
add_test_unit_cluster(layouts=LayoutCombination.NNN, alignments=[8, 8, 8], element_output=cutlass_cppgen.DataType.f16,
                      element_accumulator=cutlass_cppgen.DataType.f32, threadblock_shape=[128, 128, 32], stages=3)
add_test_unit_cluster(layouts=LayoutCombination.NNT, alignments=[8, 8, 8], element_output=cutlass_cppgen.DataType.f16,
                      element_accumulator=cutlass_cppgen.DataType.f32, threadblock_shape=[128, 128, 32], stages=None)
add_test_unit_cluster(layouts=LayoutCombination.NTN, alignments=[8, 8, 8], element_output=cutlass_cppgen.DataType.f16,
                      element_accumulator=cutlass_cppgen.DataType.f32, threadblock_shape=[128, 128, 32], stages=None)
add_test_unit_cluster(layouts=LayoutCombination.NTT, alignments=[8, 8, 8], element_output=cutlass_cppgen.DataType.f16,
                      element_accumulator=cutlass_cppgen.DataType.f32, threadblock_shape=[128, 128, 32], stages=None)
add_test_unit_cluster(layouts=LayoutCombination.TNN, alignments=[8, 8, 8], element_output=cutlass_cppgen.DataType.f16,
                      element_accumulator=cutlass_cppgen.DataType.f32, threadblock_shape=[128, 128, 32], stages=None)
add_test_unit_cluster(layouts=LayoutCombination.TNT, alignments=[4, 4, 8], element_output=cutlass_cppgen.DataType.f16,
                      element_accumulator=cutlass_cppgen.DataType.f32, threadblock_shape=[128, 128, 32], stages=None)
add_test_unit_cluster(layouts=LayoutCombination.TNT, alignments=[4, 4, 8], element_output=cutlass_cppgen.DataType.f16,
                      element_accumulator=cutlass_cppgen.DataType.f16, threadblock_shape=[128, 128, 32], stages=None)
add_test_unit_cluster(layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass_cppgen.DataType.f16,
                      element_accumulator=cutlass_cppgen.DataType.f16, threadblock_shape=[128, 128, 32], stages=None)
add_test_unit_cluster(layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass_cppgen.DataType.f16,
                      element_accumulator=cutlass_cppgen.DataType.f32, threadblock_shape=[ 64,  64, 64], stages=5)
add_test_unit_cluster(layouts=LayoutCombination.TNT, alignments=[2, 2, 2], element_output=cutlass_cppgen.DataType.f16,
                      element_accumulator=cutlass_cppgen.DataType.f16, threadblock_shape=[128, 128, 32], stages=None)

# Tests with different cluster shapes
add_test_cluster_shape = partial(add_test_tensorop, threadblock_shape=[64, 128, 64], stages=None)
add_test_cluster_shape(layouts=LayoutCombination.TTN, alignments=[8, 8, 8], element_output=cutlass_cppgen.DataType.f16,
                       element_accumulator=cutlass_cppgen.DataType.f16, cluster_shape=[2, 2, 1])
add_test_cluster_shape(layouts=LayoutCombination.TNN, alignments=[8, 8, 4], element_output=cutlass_cppgen.DataType.f32,
                       element_accumulator=cutlass_cppgen.DataType.f32, cluster_shape=[2, 2, 1])
add_test_cluster_shape(layouts=LayoutCombination.NTN, alignments=[8, 8, 4], element_output=cutlass_cppgen.DataType.f32,
                       element_accumulator=cutlass_cppgen.DataType.f32, cluster_shape=[2, 2, 1])
add_test_cluster_shape(layouts=LayoutCombination.NNN, alignments=[8, 8, 4], element_output=cutlass_cppgen.DataType.f32,
                       element_accumulator=cutlass_cppgen.DataType.f32, cluster_shape=[2, 2, 1])
add_test_cluster_shape(layouts=LayoutCombination.TTN, alignments=[8, 8, 4], element_output=cutlass_cppgen.DataType.f32,
                       element_accumulator=cutlass_cppgen.DataType.f32, cluster_shape=[1, 4, 1])
add_test_cluster_shape(layouts=LayoutCombination.TTN, alignments=[8, 8, 4], element_output=cutlass_cppgen.DataType.f32,
                       element_accumulator=cutlass_cppgen.DataType.f32, cluster_shape=[2, 4, 1])
add_test_cluster_shape(layouts=LayoutCombination.TTN, alignments=[8, 8, 4], element_output=cutlass_cppgen.DataType.f32,
                       element_accumulator=cutlass_cppgen.DataType.f32, cluster_shape=[4, 1, 1])
add_test_cluster_shape(layouts=LayoutCombination.TTN, alignments=[8, 8, 4], element_output=cutlass_cppgen.DataType.f32,
                       element_accumulator=cutlass_cppgen.DataType.f32, cluster_shape=[4, 2, 1])

# Tests for different schedule modes
add_test_schedule = partial(add_test_specialized, layouts=LayoutCombination.TTN, alignments=[8, 8, 4],
                            element_output=cutlass_cppgen.DataType.f32, element_accumulator=cutlass_cppgen.DataType.f32,
                            opclass=cutlass_cppgen.OpcodeClass.TensorOp, threadblock_shape=[128, 128, 64], stages=None)
add_test_schedule(
    cluster_shape=[1, 1, 1],
    kernel_schedule=cutlass_cppgen.KernelScheduleType.TmaWarpSpecializedPingpong,
    epilogue_schedule=cutlass_cppgen.EpilogueScheduleType.TmaWarpSpecialized
)
add_test_schedule(
    cluster_shape=[1, 1, 1],
    kernel_schedule=cutlass_cppgen.KernelScheduleType.TmaWarpSpecializedCooperative,
    epilogue_schedule=cutlass_cppgen.EpilogueScheduleType.TmaWarpSpecializedCooperative
)
add_test_schedule(
    cluster_shape=[2, 1, 1],
    kernel_schedule=cutlass_cppgen.KernelScheduleType.TmaWarpSpecializedPingpong,
    epilogue_schedule=cutlass_cppgen.EpilogueScheduleType.TmaWarpSpecialized
)
add_test_schedule(
    cluster_shape=[2, 1, 1],
    kernel_schedule=cutlass_cppgen.KernelScheduleType.TmaWarpSpecializedCooperative,
    epilogue_schedule=cutlass_cppgen.EpilogueScheduleType.TmaWarpSpecializedCooperative
)

# Tests using SIMT
add_test_simt = partial(add_test_specialized, opclass=cutlass_cppgen.OpcodeClass.Simt, alignments=[1, 1, 1], cluster_shape=[1, 1, 1], stages=2)
add_test_simt(layouts=LayoutCombination.NNN, element_output=cutlass_cppgen.DataType.f16, element_accumulator=cutlass_cppgen.DataType.f32, threadblock_shape=[128, 128, 8])
add_test_simt(layouts=LayoutCombination.TNN, element_output=cutlass_cppgen.DataType.f16, element_accumulator=cutlass_cppgen.DataType.f32, threadblock_shape=[ 64, 128, 8])
add_test_simt(layouts=LayoutCombination.NTN, element_output=cutlass_cppgen.DataType.f16, element_accumulator=cutlass_cppgen.DataType.f32, threadblock_shape=[128,  64, 8])
add_test_simt(layouts=LayoutCombination.TTN, element_output=cutlass_cppgen.DataType.f16, element_accumulator=cutlass_cppgen.DataType.f32, threadblock_shape=[ 64,  64, 8])
add_test_simt(layouts=LayoutCombination.NNT, element_output=cutlass_cppgen.DataType.f16, element_accumulator=cutlass_cppgen.DataType.f16, threadblock_shape=[128, 128, 8])

# Tests with void-C kernels
add_test_cluster_shape(layouts=LayoutCombination.NNT, alignments=[8, 8, 8], element_output=cutlass_cppgen.DataType.f16,
                       element_accumulator=cutlass_cppgen.DataType.f32, threadblock_shape=[128, 128, 32], stages=None,
                       cluster_shape=[2, 1, 1], element_C=cutlass_cppgen.DataType.void)

if __name__ == '__main__':
    unittest.main()

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
Low-level functionality tests for GEMM with S8 operands on SM90
"""

from functools import partial
import logging
import unittest

import cutlass_cppgen
from cutlass_cppgen.backend.utils.device import device_cc

from utils import LayoutCombination, add_test_gemm


cutlass_cppgen.set_log_level(logging.WARNING)
cc = 90
dtype = cutlass_cppgen.DataType.s8


@unittest.skipIf(device_cc() < cc, 'Device compute capability is insufficient for SM90 tests.')
@unittest.skipIf(cutlass_cppgen.utils.datatypes.torch_type(dtype) is None, f'Version of torch installed does not contain a datatype match for {dtype}')
class GemmS8Sm90(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass


add_test_specialized = partial(add_test_gemm, cls=GemmS8Sm90, element=dtype, compilation_modes=['nvcc'])

add_test_tensorop = partial(add_test_specialized, opclass=cutlass_cppgen.OpcodeClass.TensorOp)

# Tests with 1x1x1 clusters
add_test_tensorop(layouts=LayoutCombination.TNN, alignments=[16, 16, 16], element_output=cutlass_cppgen.DataType.s8,
                  element_accumulator=cutlass_cppgen.DataType.s32, cluster_shape=[1, 1, 1], threadblock_shape=[128, 128, 128], stages=3)
add_test_tensorop(layouts=LayoutCombination.TNT, alignments=[16, 16, 16], element_output=cutlass_cppgen.DataType.s8,
                  element_accumulator=cutlass_cppgen.DataType.s32, cluster_shape=[1, 1, 1], threadblock_shape=[128, 128, 128], stages=None)
add_test_tensorop(layouts=LayoutCombination.TNT, alignments=[16, 16,  8], element_output=cutlass_cppgen.DataType.s8,
                  element_accumulator=cutlass_cppgen.DataType.s32, cluster_shape=[1, 1, 1], threadblock_shape=[128, 128, 128], stages=None)
add_test_tensorop(layouts=LayoutCombination.TNT, alignments=[16, 16, 16], element_output=cutlass_cppgen.DataType.s8,
                  element_accumulator=cutlass_cppgen.DataType.s32, cluster_shape=[1, 1, 1], threadblock_shape=[64,  128, 128], stages=None)
add_test_tensorop(layouts=LayoutCombination.TNT, alignments=[16, 16, 16], element_output=cutlass_cppgen.DataType.s8,
                  element_accumulator=cutlass_cppgen.DataType.s32, cluster_shape=[1, 1, 1], threadblock_shape=[128,  64,  32], stages=None)
add_test_tensorop(layouts=LayoutCombination.TNT, alignments=[ 4,  4, 16], element_output=cutlass_cppgen.DataType.s8,
                  element_accumulator=cutlass_cppgen.DataType.s32, cluster_shape=[1, 1, 1], threadblock_shape=[128, 128, 128], stages=None)

# Tests with different cluster shapes
add_test_tensorop(layouts=LayoutCombination.TNT, alignments=[16, 16, 16], element_output=cutlass_cppgen.DataType.s8,
                  element_accumulator=cutlass_cppgen.DataType.s32, cluster_shape=[2, 2, 1], threadblock_shape=[128, 128, 128], stages=None)
add_test_tensorop(layouts=LayoutCombination.TNT, alignments=[16, 16, 16], element_output=cutlass_cppgen.DataType.s8,
                  element_accumulator=cutlass_cppgen.DataType.s32, cluster_shape=[1, 4, 1], threadblock_shape=[128, 128, 128], stages=None)

# Tests with warp-specialized ping-pong schedule
add_test_tensorop(layouts=LayoutCombination.TNT, alignments=[16, 16, 16], element_output=cutlass_cppgen.DataType.s8,
                  element_accumulator=cutlass_cppgen.DataType.s32, cluster_shape=[2, 1, 1], threadblock_shape=[128, 128, 128], stages=None,
                  kernel_schedule=cutlass_cppgen.KernelScheduleType.TmaWarpSpecializedPingpong,
                  epilogue_schedule=cutlass_cppgen.EpilogueScheduleType.TmaWarpSpecialized)

# Tests for SIMT
add_test_simt = partial(add_test_specialized, opclass=cutlass_cppgen.OpcodeClass.Simt)
add_test_simt(layouts=LayoutCombination.TNN, alignments=[1, 1, 1], element_output=cutlass_cppgen.DataType.s8,
              element_accumulator=cutlass_cppgen.DataType.s32, cluster_shape=[1, 1, 1], threadblock_shape=[64, 32, 8], stages=2)


if __name__ == '__main__':
    unittest.main()

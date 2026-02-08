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
Low-level functionality tests for GEMM with S8 operands on SM80
"""

from functools import partial
import logging
import unittest

import cutlass_cppgen
from cutlass_cppgen.backend.utils.device import device_cc

from utils import LayoutCombination, add_test_gemm


cutlass_cppgen.set_log_level(logging.WARNING)
cc = 80
dtype = cutlass_cppgen.DataType.s8


@unittest.skipIf(device_cc() < cc, 'Device compute capability is insufficient for SM80 tests.')
@unittest.skipIf(cutlass_cppgen.utils.datatypes.torch_type(dtype) is None, f'Version of torch installed does not contain a datatype match for {dtype}')
class GemmS8Sm80(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass


@unittest.skipIf(device_cc() < cc, 'Device compute capability is insufficient for SM80 tests.')
@unittest.skipIf(cutlass_cppgen.utils.datatypes.torch_type(dtype) is None, f'Version of torch installed does not contain a datatype match for {dtype}')
class GemmS8Sm80StreamK(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass


add_test_specialized = partial(add_test_gemm, element=dtype, cc=cc, cluster_shape=[1, 1, 1])

# Tests using TensorOp
add_test_tensorop = partial(add_test_specialized, opclass=cutlass_cppgen.OpcodeClass.TensorOp)

add_test_tensorop(cls=GemmS8Sm80, layouts=LayoutCombination.TNN, alignments=[16, 16, 16],  element_output=cutlass_cppgen.DataType.s8, element_C=cutlass_cppgen.DataType.s8,
                  element_accumulator=cutlass_cppgen.DataType.s32, threadblock_shape=[256, 128, 64], warp_count=[4, 2, 1], stages=3)
add_test_tensorop(cls=GemmS8Sm80, layouts=LayoutCombination.TNT, alignments=[16, 16, 16],  element_output=cutlass_cppgen.DataType.s8, element_C=cutlass_cppgen.DataType.s8,
                  element_accumulator=cutlass_cppgen.DataType.s32, threadblock_shape=[128, 256, 64], warp_count=[2, 4, 1], stages=3)
add_test_tensorop(cls=GemmS8Sm80, layouts=LayoutCombination.TNN, alignments=[16, 16,  4], element_output=cutlass_cppgen.DataType.s32, element_C=cutlass_cppgen.DataType.s32,
                  element_accumulator=cutlass_cppgen.DataType.s32, threadblock_shape=[ 64,  64, 64], warp_count=[1, 1, 1], stages=4)

# Tests using SIMT
add_test_simt = partial(add_test_specialized, opclass=cutlass_cppgen.OpcodeClass.Simt)

add_test_simt(cls=GemmS8Sm80, layouts=LayoutCombination.NNN, alignments=[1, 1, 1],  element_output=cutlass_cppgen.DataType.s8, element_C=cutlass_cppgen.DataType.s8,
              element_accumulator=cutlass_cppgen.DataType.s32, threadblock_shape=[128, 128, 8], warp_count=[2, 2, 1], stages=2)
add_test_simt(cls=GemmS8Sm80, layouts=LayoutCombination.TNN, alignments=[1, 1, 1],  element_output=cutlass_cppgen.DataType.s8, element_C=cutlass_cppgen.DataType.s8,
              element_accumulator=cutlass_cppgen.DataType.s32, threadblock_shape=[ 64, 128, 8], warp_count=[1, 2, 1], stages=2)
add_test_simt(cls=GemmS8Sm80, layouts=LayoutCombination.NTN, alignments=[1, 1, 1],  element_output=cutlass_cppgen.DataType.s8, element_C=cutlass_cppgen.DataType.s8,
              element_accumulator=cutlass_cppgen.DataType.s32, threadblock_shape=[128,  64, 8], warp_count=[2, 1, 1], stages=2)
add_test_simt(cls=GemmS8Sm80, layouts=LayoutCombination.TTN, alignments=[1, 1, 1], element_output=cutlass_cppgen.DataType.s32, element_C=cutlass_cppgen.DataType.s32,
              element_accumulator=cutlass_cppgen.DataType.s32, threadblock_shape=[ 64,  64, 8], warp_count=[1, 1, 1], stages=2)
add_test_simt(cls=GemmS8Sm80, layouts=LayoutCombination.NNT, alignments=[1, 1, 1], element_output=cutlass_cppgen.DataType.s32, element_C=cutlass_cppgen.DataType.s32,
              element_accumulator=cutlass_cppgen.DataType.s32, threadblock_shape=[128, 128, 8], warp_count=[2, 2, 1], stages=2)

# Stream K tests
add_test_streamk = partial(add_test_specialized, opclass=cutlass_cppgen.OpcodeClass.TensorOp, swizzle=cutlass_cppgen.swizzle.ThreadblockSwizzleStreamK)
add_test_streamk(cls=GemmS8Sm80StreamK, layouts=LayoutCombination.TNT, alignments=[16, 16, 16], element_output=cutlass_cppgen.DataType.s8, element_C=cutlass_cppgen.DataType.s8,
                 element_accumulator=cutlass_cppgen.DataType.s32, threadblock_shape=[128, 256, 64], warp_count=[2, 4, 1], stages=3)


if __name__ == '__main__':
    unittest.main()

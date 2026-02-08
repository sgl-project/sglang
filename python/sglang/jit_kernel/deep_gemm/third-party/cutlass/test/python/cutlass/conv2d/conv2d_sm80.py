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
Low-level functionality tests for Conv2d opreations on SM80
"""

import logging
import unittest

import cutlass_cppgen
from cutlass_cppgen.backend.utils.device import device_cc

from conv2d_test_utils import *


cutlass_cppgen.set_log_level(logging.WARNING)
cc = 80


@unittest.skipIf(device_cc() < cc, 'Device compute capability is invalid for SM80 tests.')
class Conv2dSm80(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass


conv_problems = get_conv_problems()


# Tests for optimized & analytic
for conv_kind in ["fprop", "wgrad", "dgrad"]:
    # F16, simt
    add_test(
        Conv2dSm80, cc, conv_kind, conv_problems, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f16,
        opclass="simt", threadblock_shape=[128, 128, 8],
        warp_count=[4, 2, 1], stages=2, instruction_shape=[1, 1, 1])
    # F16, tensor op
    add_test(
        Conv2dSm80, cc, conv_kind, conv_problems, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f16,
        opclass="tensor_op", threadblock_shape=[128, 128, 64],
        warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 16])
    # F16, tensor op, analytic iterator
    add_test(
        Conv2dSm80, cc, conv_kind, conv_problems, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f16,
        opclass="tensor_op", threadblock_shape=[128, 128, 64],
        warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 16], iterator_algorithm="analytic")
    # F16, tensor op, f32 output
    add_test(
        Conv2dSm80, cc, conv_kind, conv_problems, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f32,
        opclass="tensor_op", threadblock_shape=[128, 128, 64],
        warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 16])
    # F16, tensor op, different tile description
    add_test(
        Conv2dSm80, cc, conv_kind, conv_problems, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f16,
        opclass="tensor_op", threadblock_shape=[128, 64, 32],
        warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 8])
    # F32, simt
    add_test(
        Conv2dSm80, cc, conv_kind, conv_problems, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f32,
        opclass="simt", threadblock_shape=[128, 128, 8],
        warp_count=[4, 2, 1], stages=4, instruction_shape=[1, 1, 1])
    # Tf32, tensorop
    add_test(
        Conv2dSm80, cc, conv_kind, conv_problems, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f32,
        opclass="tensor_op", threadblock_shape=[128, 128, 16],
        warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 8]
    )
    # Split-K
    add_test(
        Conv2dSm80, cc, conv_kind, conv_problems, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f16,
        opclass="tensor_op", threadblock_shape=[128, 128, 64],
        warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 16], split_k_mode="serial",
        split_k_slices=2)
    add_test(
        Conv2dSm80, cc, conv_kind, conv_problems, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f16,
        opclass="tensor_op", threadblock_shape=[128, 128, 64],
        warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 16], split_k_mode="parallel",
        split_k_slices=5)
    # Swizzling functor
    add_test(
        Conv2dSm80, cc, conv_kind, conv_problems, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f16,
        opclass="tensor_op", threadblock_shape=[128, 64, 32],
        warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 8], swizzle=4)

# Tests for few channels and fixed channels
# F16, tensor op, few channels
for c, tb, stage, inst in zip([2, 1],
                                [[128, 128, 64], [128, 128, 32]],
                                [3, 2],
                                [[16, 8, 16], [16, 8, 8]]):
    add_test(
        Conv2dSm80, cc, "fprop", conv2d_few_channel_problemsizes(c), cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f16,
        opclass="tensor_op", threadblock_shape=tb,
        warp_count=[2, 2, 1], stages=stage, instruction_shape=inst, iterator_algorithm="few_channels"
    )
# F16, tensor op, fixed channels
for c in [8, 4, 2]:
    add_test(
        Conv2dSm80, cc, "fprop", conv2d_few_channel_problemsizes(c), cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f16,
        opclass="tensor_op", threadblock_shape=[128, 128, 64],
        warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 16], iterator_algorithm="fixed_channels"
    )

# Test activations
for activation in ["relu", "leaky_relu"]:
    for split_k_mode, split_k_slices in zip(["parallel", "serial", "parallel"], [1, 7, 5]):
        add_test(
            Conv2dSm80, cc, "fprop", conv_problems, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f16,
            opclass="tensor_op", threadblock_shape=[128, 128, 64],
            warp_count=[2, 2, 1], stages=3, instruction_shape=[16, 8, 16], split_k_mode=split_k_mode,
            split_k_slices=split_k_slices, activation=activation)


if __name__ == '__main__':
    unittest.main()

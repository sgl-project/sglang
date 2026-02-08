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
Tests the high-level Conv2d interface
"""

from math import ceil
import unittest

import cutlass_cppgen
import cutlass_cppgen.utils.datatypes as datatypes
from cutlass_cppgen.backend.utils.device import device_cc
from utils import ExpectException
import os


class Conv2dEquivalence:
    """
    Helper class for testing the equivalence of different constructions of the Conv2d interface
    """
    def __init__(self, conv_kind, element_A, element_B, element_C, element_D, element_accumulator,
                 alignment_A, alignment_B, alignment_C):

        self.element_A = element_A
        self.element_B = element_B
        self.element_C = element_C
        self.element_D = element_D
        self.element_accumulator = element_accumulator
        self.alignment_A = alignment_A
        self.alignment_B = alignment_B
        self.alignment_C = alignment_C

        self.conv_kind = conv_kind

        self.plan = cutlass_cppgen.op.Conv2d(
            kind=self.conv_kind, element_A=element_A, element_B=element_B, element_C=element_C,
            element_D=element_D, element_accumulator=element_accumulator)

        self.op = self.plan.construct(
            alignment_A=self.alignment_A, alignment_B=self.alignment_B,
            alignment_C=self.alignment_C)

    def _plans_equal(self, other_plan) -> bool:
        """
        Compares whether two plans are equal

        :param other_plan: plan to compare against the default Conv2d
        :type other_plan: cutlass_cppgen.op.Conv2d

        :return: whether `other_plan` is equivalent to `self.plan`
        :rtype: bool
        """
        other_op = other_plan.construct(
            alignment_A=self.alignment_A, alignment_B=self.alignment_B,
            alignment_C=self.alignment_C)

        return self.op.rt_module.emit() == other_op.rt_module.emit()

    def generic_test(self):
        """
        Tests the equivalence of various constructions of the Conv2d interface when using CUTLASS data types
        and layouts for constructing the Conv2d interface
        """
        if not datatypes.is_numpy_available():
            return

        # Test when specifying all parameters
        plan_other = cutlass_cppgen.op.Conv2d(
            kind=self.conv_kind,
            element_A=self.element_A, element_B=self.element_B, element_C=self.element_C,
            element_D=self.element_D, element_accumulator=self.element_accumulator)
        assert self._plans_equal(plan_other)

        # Test when specifying all parameters but A
        plan_other = cutlass_cppgen.op.Conv2d(
            kind=self.conv_kind,
            element_B=self.element_B, element_C=self.element_C,
            element_D=self.element_D, element_accumulator=self.element_accumulator,
            element=self.element_A)
        assert self._plans_equal(plan_other)

        # Test when specifying all parameters but A and B as tensors using generic element and output
        plan_other = cutlass_cppgen.op.Conv2d(
            kind=self.conv_kind,
            element_C=self.element_C,
            element_D=self.element_D, element_accumulator=self.element_accumulator,
            element=self.element_A)
        assert self._plans_equal(plan_other)

        # Test without explicit accumulator. Only run if the type of C and the accumulator are equal
        if self.element_C == self.element_accumulator:
            plan_other = cutlass_cppgen.op.Conv2d(
                kind=self.conv_kind,
                element_C=self.element_C,
                element_D=self.element_D,
                element=self.element_A)
            assert self._plans_equal(plan_other)

        # Test with only the generic types. Only rune if the types of A, B, C, and D are the same
        if (self.element_A == self.element_B and self.element_A == self.element_C and self.element_A == self.element_D
            and self.element_A == self.element_accumulator):
            plan_other = cutlass_cppgen.op.Conv2d(kind=self.conv_kind, element=self.element_A)
            assert self._plans_equal(plan_other)

    def numpy_test(self):
        """
        Tests the equivalence of various constructions of the Conv2d interface when using numpy as a frontend
        """
        if not datatypes.is_numpy_available():
            return

        import numpy as np
        type_A = datatypes.numpy_type(self.element_A)
        type_B = datatypes.numpy_type(self.element_B)
        type_C = datatypes.numpy_type(self.element_C)
        type_D = datatypes.numpy_type(self.element_D)
        type_accum = datatypes.numpy_type(self.element_accumulator)

        size = (2, 2)
        A = np.zeros(size, dtype=type_A)
        B = np.zeros(size, dtype=type_B)
        C = np.zeros(size, dtype=type_C)
        D = np.zeros(size, dtype=type_D)

        return self.tensor_test(type_A, type_B, type_C, type_D, type_accum, A, B, C, D)

    def torch_test(self):
        """
        Tests the equivalence of various constructions of the Conv2d interface when using torch as a frontend
        """
        if not datatypes.is_torch_available():
            return

        import torch
        type_A = datatypes.torch_type(self.element_A)
        type_B = datatypes.torch_type(self.element_B)
        type_C = datatypes.torch_type(self.element_C)
        type_D = datatypes.torch_type(self.element_D)
        type_accum = datatypes.torch_type(self.element_accumulator)

        size = (2, 2)

        A = torch.empty(size, dtype=type_A)
        B = torch.empty(size, dtype=type_B)
        C = torch.empty(size, dtype=type_C)
        D = torch.empty(size, dtype=type_D)

        return self.tensor_test(type_A, type_B, type_C, type_D, type_accum, A, B, C, D)

    def tensor_test(self, type_A, type_B, type_C, type_D, type_accum, A, B, C, D):
        # Test when specifying all parameters via tensors
        plan_np = cutlass_cppgen.op.Conv2d(kind=self.conv_kind, A=A, B=B, C=C, D=D, element_accumulator=type_accum)
        assert self._plans_equal(plan_np)

        # Test when specifying all parameters but A as tensors
        plan_np = cutlass_cppgen.op.Conv2d(kind=self.conv_kind, B=B, C=C, D=D, element_accumulator=type_accum, element_A=type_A)
        assert self._plans_equal(plan_np)

        # Test when specifying all parameters but A and B as tensors and using generic element and output
        if type_A == type_B:
            plan_np = cutlass_cppgen.op.Conv2d(kind=self.conv_kind, C=C, D=D, element_accumulator=type_accum, element=type_A)
            assert self._plans_equal(plan_np)

        # Test without explicit accumulator. Only run if the type of C and the accumulator.
        if type_C == type_accum:
            plan_np = cutlass_cppgen.op.Conv2d(kind=self.conv_kind, A=A, B=B, C=C, D=D)
            assert self._plans_equal(plan_np)

        # Test with only the generic types and layouts. Only run if types and layouts of A, B, C, and D are the same.
        if (type_A == type_B and type_A == type_C and type_A == type_D and type_A == type_accum):
            plan_np = cutlass_cppgen.op.Conv2d(kind=self.conv_kind, element=type_A)
            assert self._plans_equal(plan_np)

    def test_all(self):
        """
        Runs all tests on the Gemm interface
        """
        self.generic_test()
        self.numpy_test()
        self.torch_test()


@unittest.skipIf(device_cc() <= 80, 'Device compute capability is insufficient for SM80 tests.')
class ConvEquivalenceTest(unittest.TestCase):
    """
    Tests the equivalence of different constructions of the Conv2d interface
    """
    pass

type2alignment = {
    cutlass_cppgen.DataType.f16: 8,
    cutlass_cppgen.DataType.f32: 4
}

def add_test(conv_kind, element_A, element_B, element_C, element_D, element_accumulator):

    test_name = f"test_conv2d_{conv_kind}_{element_A}_{element_B}_{element_C}_{element_D}_{element_accumulator}"

    def run(self):
        conv2d_eq = Conv2dEquivalence(
            conv_kind=conv_kind,
            element_A=element_A, element_B=element_B,
            element_C=element_C, element_D=element_D,
            element_accumulator=element_accumulator,
            alignment_A=type2alignment[element_A], alignment_B=type2alignment[element_B],
            alignment_C=type2alignment[element_C]
        )
        conv2d_eq.test_all()

    setattr(ConvEquivalenceTest, test_name, run)

for conv_kind in ["fprop", "wgrad", "dgrad"]:
    for types in [
        [cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f16],
        [cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f32],
        [cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f16],
        [cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f32],
        [cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f32, cutlass_cppgen.DataType.f32]
    ]:
        add_test(conv_kind, types[0], types[1], types[2], types[3], types[4])


@unittest.skipIf(device_cc() <= 80, 'Device compute capability is insufficient for SM80 tests.')
class Conv2dErrorTests(unittest.TestCase):
    """
    Tests various error scenarios that arise with the high-level Gemm interface
    """

    def test_alignment(self):
        """
        Tests case in which the alignment specified is unsupported
        """
        plan = cutlass_cppgen.op.Conv2d(kind="fprop", element=cutlass_cppgen.DataType.f16)

        with ExpectException(True, 'Alignment 3 is not supported for F16. The construction should fail.'):
            op = plan.construct(alignment_A=3, alignment_B=3, alignment_C=3)

    def test_invalid_tile_description(self):
        """
        Tests scenarios in which an invalid tile description is provided for a given CC
        """
        plan = cutlass_cppgen.op.Conv2d(kind="fprop", element=cutlass_cppgen.DataType.f16)

        td = plan.tile_descriptions()[0]
        td.threadblock_shape=[17, 32, 5]

        plan.tile_description = td
        with ExpectException(True, 'The threadblock shape is invalid. The compilation should fail.'):
            plan.compile()
        # Clean up the error message
        os.remove("./cutlass_python_compilation_device_error.txt")

if __name__ == '__main__':
    unittest.main()

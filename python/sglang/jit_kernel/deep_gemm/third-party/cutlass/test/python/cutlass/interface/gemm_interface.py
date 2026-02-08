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
Tests the high-level GEMM interface
"""

from math import ceil
import unittest

import cutlass_cppgen
import cutlass_cppgen.utils.datatypes as datatypes
from cutlass_cppgen.backend.utils.device import device_cc
from utils import ExpectException


class GemmEquivalence:
    """
    Helper class for testing the equivalence of different constructions of the Gemm interface
    """
    def __init__(self, element_A, element_B, element_C, element_D, element_accumulator,
                 layout_A, layout_B, layout_C, alignment_A, alignment_B, alignment_C):
        self.element_A = element_A
        self.element_B = element_B
        self.element_C = element_C
        self.element_D = element_D
        self.element_accumulator = element_accumulator
        self.layout_A = layout_A
        self.layout_B = layout_B
        self.layout_C = layout_C
        self.alignment_A = alignment_A
        self.alignment_B = alignment_B
        self.alignment_C = alignment_C
        self.plan = cutlass_cppgen.op.Gemm(element_A=element_A, element_B=element_B, element_C=element_C,
                                    element_D=element_D, element_accumulator=element_accumulator,
                                    layout_A=layout_A, layout_B=layout_B, layout_C=layout_C)
        self.op = self.plan.construct(alignment_A=alignment_A, alignment_B=alignment_B, alignment_C=alignment_C)

    def _plans_equal(self, other_plan) -> bool:
        """
        Compares whether two plans are equal

        :param other_plan: plan to compare against the default GEMM
        :type other_plan: cutlass_cppgen.op.Gemm

        :return: whether `other_plan` is equivalent to `self.plan`
        :rtype: bool
        """
        other_op = other_plan.construct(alignment_A=self.alignment_A, alignment_B=self.alignment_B, alignment_C=self.alignment_C)

        # Compare whether the operations are equal by comparing the C++ code that would be emitted for them
        return self.op.rt_module.emit() == other_op.rt_module.emit()

    def generic_test(self):
        """
        Tests the equivalence of various constructions of the Gemm interface when using CUTLASS data types
        and layouts for constructing the Gemm interface
        """
        if not datatypes.is_numpy_available():
            return

        # Test when specifying all parameters
        plan_other = cutlass_cppgen.op.Gemm(element_A=self.element_A, element_B=self.element_B, element_C=self.element_C,
                                  element_D=self.element_D, element_accumulator=self.element_accumulator,
                                  layout_A=self.layout_A, layout_B=self.layout_B, layout_C=self.layout_C)
        assert self._plans_equal(plan_other)

        # Test when specifying all parameters but A
        plan_other = cutlass_cppgen.op.Gemm(element_B=self.element_B, element_C=self.element_C,
                                  element_D=self.element_D, element_accumulator=self.element_accumulator,
                                  layout_B=self.layout_B, layout_C=self.layout_C,
                                  element=self.element_A, layout=self.layout_A)
        assert self._plans_equal(plan_other)

        # Test when specifying all parameters but A and B as tensors and using generic element and output
        # Only run this test if the layouts and types for A and B are equal.
        if self.element_A == self.element_B and self.layout_A == self.layout_B:
            plan_other = cutlass_cppgen.op.Gemm(element_C=self.element_C, element_D=self.element_D, element_accumulator=self.element_accumulator,
                                      layout_C=self.layout_C, element=self.element_A, layout=self.layout_A)
            assert self._plans_equal(plan_other)

        # Test without explicit accumulator. Only run if the type of C and the accumulator.
        if self.element_C == self.element_accumulator:
            plan_other = cutlass_cppgen.op.Gemm(element_A=self.element_A, element_B=self.element_B, element_C=self.element_C,
                                      element_D=self.element_D, layout_A=self.layout_A, layout_B=self.layout_B,
                                      layout_C=self.layout_C)
            assert self._plans_equal(plan_other)

        # Test with only the generic types and layouts. Only run if types and layouts of A, B, C, and D are the same.
        if (self.element_A == self.element_B and self.element_A == self.element_C and self.element_A == self.element_D
            and self.element_A == self.element_accumulator and
            self.layout_A == self.layout_B and self.layout_A == self.layout_C):
            plan_other = cutlass_cppgen.op.Gemm(element=self.element_A, layout=self.layout_A)
            assert self._plans_equal(plan_other)

    def numpy_test(self):
        """
        Tests the equivalence of various constructions of the Gemm interface when using numpy as a frontend
        """
        if not datatypes.is_numpy_available():
            return

        import numpy as np
        type_A = datatypes.numpy_type(self.element_A)
        type_B = datatypes.numpy_type(self.element_B)
        type_C = datatypes.numpy_type(self.element_C)
        type_D = datatypes.numpy_type(self.element_D)
        type_accum = datatypes.numpy_type(self.element_accumulator)

        layout_to_order = {
            cutlass_cppgen.LayoutType.RowMajor: 'C',
            cutlass_cppgen.LayoutType.ColumnMajor: 'F'
        }
        size = (2, 2)
        A = np.zeros(size, order=layout_to_order[self.layout_A], dtype=type_A)
        B = np.zeros(size, order=layout_to_order[self.layout_B], dtype=type_B)
        C = np.zeros(size, order=layout_to_order[self.layout_C], dtype=type_C)
        D = np.zeros(size, order=layout_to_order[self.layout_C], dtype=type_D)

        # Test when specifying all parameters via tensors
        plan_np = cutlass_cppgen.op.Gemm(A=A, B=B, C=C, D=D, element_accumulator=type_accum)
        assert self._plans_equal(plan_np)

        # Test when specifying all parameters but A as tensors
        plan_np = cutlass_cppgen.op.Gemm(B=B, C=C, D=D, element_accumulator=type_accum, element_A=type_A, layout_A=self.layout_A)
        assert self._plans_equal(plan_np)

        # Test when specifying all parameters but A and B as tensors and using generic element and output
        # Only run this test if the layouts and types for A and B are equal.
        if type_A == type_B and self.layout_A == self.layout_B:
            plan_np = cutlass_cppgen.op.Gemm(C=C, D=D, element_accumulator=type_accum, element=type_A, layout=self.layout_A)
            assert self._plans_equal(plan_np)

        # Test without explicit accumulator. Only run if the type of C and the accumulator.
        if type_C == type_accum:
            plan_np = cutlass_cppgen.op.Gemm(A=A, B=B, C=C, D=D)
            assert self._plans_equal(plan_np)

        # Test with only the generic types and layouts. Only run if types and layouts of A, B, C, and D are the same.
        if (type_A == type_B and type_A == type_C and type_A == type_D and type_A == type_accum and
            self.layout_A == self.layout_B and self.layout_A == self.layout_C):
            plan_np = cutlass_cppgen.op.Gemm(element=type_A, layout=self.layout_A)
            assert self._plans_equal(plan_np)

    def test_all(self):
        """
        Runs all tests on the Gemm interface
        """
        self.generic_test()
        self.numpy_test()


class GemmEquivalenceTest(unittest.TestCase):
    """
    Tests the equivalence of different constructions of the Gemm interface
    """
    @unittest.skipIf(device_cc() < 70, "Device compute capability is insufficient for FP16 Tensor Core tests.")
    def test_gemm_equivalence_f16_f16_f16_f16_f16_ttt_8_8_8(self):
        gemm_eq = GemmEquivalence(
                element_A=cutlass_cppgen.DataType.f16, element_B=cutlass_cppgen.DataType.f16, element_C=cutlass_cppgen.DataType.f16,
                element_D=cutlass_cppgen.DataType.f16, element_accumulator=cutlass_cppgen.DataType.f16,
                layout_A=cutlass_cppgen.LayoutType.RowMajor, layout_B=cutlass_cppgen.LayoutType.RowMajor, layout_C=cutlass_cppgen.LayoutType.RowMajor,
                alignment_A=8, alignment_B=8, alignment_C=8)
        gemm_eq.test_all()

    @unittest.skipIf(device_cc() < 70, "Device compute capability is insufficient for FP16 Tensor Core tests.")
    def test_gemm_equivalence_f16_f16_f16_f16_f32_ntn_8_8_8(self):
        gemm_eq = GemmEquivalence(
                element_A=cutlass_cppgen.DataType.f16, element_B=cutlass_cppgen.DataType.f16, element_C=cutlass_cppgen.DataType.f16,
                element_D=cutlass_cppgen.DataType.f16, element_accumulator=cutlass_cppgen.DataType.f32,
                layout_A=cutlass_cppgen.LayoutType.ColumnMajor, layout_B=cutlass_cppgen.LayoutType.RowMajor, layout_C=cutlass_cppgen.LayoutType.ColumnMajor,
                alignment_A=8, alignment_B=8, alignment_C=8)
        gemm_eq.test_all()

    @unittest.skipIf(device_cc() < 70, "Device compute capability is insufficient for FP16 Tensor Core tests.")
    def test_gemm_equivalence_f16_f16_f16_f16_f16_ttt_4_4_4(self):
        gemm_eq = GemmEquivalence(
                element_A=cutlass_cppgen.DataType.f16, element_B=cutlass_cppgen.DataType.f16, element_C=cutlass_cppgen.DataType.f16,
                element_D=cutlass_cppgen.DataType.f16, element_accumulator=cutlass_cppgen.DataType.f16,
                layout_A=cutlass_cppgen.LayoutType.RowMajor, layout_B=cutlass_cppgen.LayoutType.RowMajor, layout_C=cutlass_cppgen.LayoutType.RowMajor,
                alignment_A=8, alignment_B=8, alignment_C=8)
        gemm_eq.test_all()

    @unittest.skipIf(device_cc() < 80, "Device compute capability is insufficient for F64 Tensor Core tests.")
    def test_gemm_equivalence_f64_f64_f64_f64_f64_tnt_1_1_1(self):
        gemm_eq = GemmEquivalence(
                element_A=cutlass_cppgen.DataType.f64, element_B=cutlass_cppgen.DataType.f64, element_C=cutlass_cppgen.DataType.f64,
                element_D=cutlass_cppgen.DataType.f64, element_accumulator=cutlass_cppgen.DataType.f64,
                layout_A=cutlass_cppgen.LayoutType.RowMajor, layout_B=cutlass_cppgen.LayoutType.ColumnMajor, layout_C=cutlass_cppgen.LayoutType.RowMajor,
                alignment_A=1, alignment_B=1, alignment_C=1)
        gemm_eq.test_all()


class GemmErrorTests(unittest.TestCase):
    """
    Tests various error scenarios that arise with the high-level Gemm interface
    """

    def test_alignment(self):
        """
        Tests case in which the alignment specified is unsupported
        """
        plan = cutlass_cppgen.op.Gemm(element=cutlass_cppgen.DataType.f16, layout=cutlass_cppgen.LayoutType.RowMajor)

        with ExpectException(True, 'Alignment 16 is not supported for F16. The construction should fail.'):
            op = plan.construct(alignment_A=16, alignment_B=16, alignment_C=16)

    def test_tensorop_availability(self):
        """
        Tests case in which only SIMT operations are available but TensorOp is requested
        """
        cc = device_cc()

        # F64 Tensor Core operations are only avaiable on certain devices
        supports_tensorop_f64 = cc in [80, 89, 90]
        plan = cutlass_cppgen.op.Gemm(cc=cc, element=cutlass_cppgen.DataType.f64, layout=cutlass_cppgen.LayoutType.RowMajor)

        error_msg = f'Incorrectly raised an exception for availability of TensorOp with F64 operands on SM{cc}'
        with ExpectException(not supports_tensorop_f64, error_msg):
            plan.opclass = cutlass_cppgen.OpcodeClass.TensorOp

        expected_opclass = cutlass_cppgen.OpcodeClass.TensorOp if supports_tensorop_f64 else cutlass_cppgen.OpcodeClass.Simt
        assert plan.opclass == expected_opclass, f'Expected opclass to be {expected_opclass}, but received {plan.opclass} for SM{cc}'

    @unittest.skipIf(device_cc() < 70, "Device compute capability is insufficient for F16 Tensor Core tests.")
    def test_opclass_switch(self):
        """
        Tests cases in which the opcode class in question is switched (e.g., from TensorOp to SIMT)
        """
        plan = cutlass_cppgen.op.Gemm( element=cutlass_cppgen.DataType.f16, layout=cutlass_cppgen.LayoutType.RowMajor)
        assert plan.opclass == cutlass_cppgen.OpcodeClass.TensorOp

        # Ensure that all tile descriptions have opclass of TensorOp
        for td in plan.tile_descriptions():
            assert td.math_instruction.opcode_class == cutlass_cppgen.OpcodeClass.TensorOp

        plan.opclass = cutlass_cppgen.OpcodeClass.Simt

        # Ensure that all tile descriptions have opclass of Simt
        for td in plan.tile_descriptions():
            assert td.math_instruction.opcode_class == cutlass_cppgen.OpcodeClass.Simt

    def test_invalid_tile_description(self):
        """
        Tests scenarios in which an invalid tile description is provided for a given CC
        """
        cc = device_cc()
        plan = cutlass_cppgen.op.Gemm(cc=cc, element=cutlass_cppgen.DataType.f16, layout=cutlass_cppgen.LayoutType.RowMajor)
        td = plan.tile_descriptions()[0]
        stages = td.stages

        # Zero stage count is valid for SM90+, as this is used to indicate that the builder's auto stage
        # count should be used
        with ExpectException(cc < 90, f'Requested zero stages'):
            td.stages = 0
            plan.construct(td)

        if cc < 90:
            with ExpectException(cc < 80, f'Requested more than 2 stages on SM{cc}'):
                td.stages = 3
                plan.construct(td)
        elif cc == 90:
            original_kschedule = td.kernel_schedule
            original_eschedule = td.epilogue_schedule
            with ExpectException(False, f'Incorrectly flagged an error for insufficient shared memory'):
                td.kernel_schedule = cutlass_cppgen.KernelScheduleType.TmaWarpSpecializedPingpong
                td.epilogue_schedule = cutlass_cppgen.EpilogueScheduleType.NoSmemWarpSpecialized
                td.stages = 3
                plan.construct(td)
            # Reset schedules
            td.kernel_schedule = original_kschedule
            td.epilogue_schedule = original_eschedule
        elif cc in [100, 101, 103]:
            with ExpectException(False, f'Incorrectly flagged an error for insufficient shared memory'):
                td.stages = 3
                plan.construct(td)

        with ExpectException(True, f'Requested too many stages'):
            td.stages = 100
            plan.construct(td)

        # Reset stage count
        td.stages = stages

        cluster_shape = td.cluster_shape
        with ExpectException(cc < 90, f'Requested non-unit cluster shape on SM{cc}'):
            td.cluster_shape = [2, 1, 1]
            plan.construct(td)

        # Reset cluster shape
        td.cluster_shape = cluster_shape

        with ExpectException(cc < 90, f'Requested a non-auto schedule on SM{cc}'):
            td.kernel_schedule = cutlass_cppgen.KernelScheduleType.TmaWarpSpecializedPingpong
            td.epilogue_schedule = cutlass_cppgen.EpilogueScheduleType.TmaWarpSpecialized
            plan.construct(td)

        with ExpectException(cc == 90, f'Requested a non-auto kernel schedule with an auto epilogue schedule'):
            td.kernel_schedule = cutlass_cppgen.KernelScheduleType.TmaWarpSpecializedPingpong
            td.epilogue_schedule = cutlass_cppgen.EpilogueScheduleType.ScheduleAuto
            plan.construct(td)

        with ExpectException(cc == 90, f'Requested an auto kernel schedule with a non-auto epilogue schedule'):
            td.kernel_schedule = cutlass_cppgen.KernelScheduleType.ScheduleAuto
            td.epilogue_schedule = cutlass_cppgen.EpilogueScheduleType.TmaWarpSpecialized
            plan.construct(td)

        with ExpectException(cc < 90, f'Requested a tile scheduler on SM{cc}'):
            td.kernel_schedule = cutlass_cppgen.KernelScheduleType.TmaWarpSpecializedCooperative
            td.epilogue_schedule = cutlass_cppgen.EpilogueScheduleType.TmaWarpSpecializedCooperative
            td.tile_scheduler = cutlass_cppgen.TileSchedulerType.StreamK
            plan.construct(td)

        # Ensure that all returned tile descriptions are unique
        ops = {}
        for i, td in enumerate(plan.tile_descriptions()):
            op = plan.construct(td)
            code_str = op.rt_module.emit()
            if code_str in ops:
                conflicting_td = ops[code_str]
                assert False, f'Multiple tile descriptions emitted {code_str}\nTile descriptions are:\n{td}\n{conflicting_td}'


if __name__ == '__main__':
    unittest.main()

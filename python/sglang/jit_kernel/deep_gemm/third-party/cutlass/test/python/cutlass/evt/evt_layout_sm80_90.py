################################################################################
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
################################################################################

"""
Unit test for store nodes in SM90
"""

import logging
import unittest

import cutlass_cppgen
from cutlass_cppgen.backend import *
from cutlass_cppgen.epilogue import *

from utils.evt_testbed import EVTTestBed, EVTTestCaseBase

cutlass_cppgen.set_log_level(logging.WARNING)


@unittest.skipIf(device_cc() not in [80, 86, 89, 90], "This unittest is only supported on CC [80, 86, 89, 90]")
class TestEVTLayout(EVTTestCaseBase):

    def test_permute_1(self):
        """
        Returning a tensor with shape [m, n]
        """
        def evt_permute(accum, alpha, C):
            F = alpha * accum
            F_permute = permute(F, indices=(0, 2, 1))
            D_permute = F_permute + permute(C, indices=(0, 2, 1))
            D = permute(D_permute, indices=(0, 2, 1))
            return D, F

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "alpha": 0.5,
                "C": self.fake_tensor(self.element, (l, m, n)),
                "F": self.fake_tensor(self.element, (l, m, n)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_permute, example_inputs)
            input_keys = ["C", "alpha"]
            result_keys = ["D", "F"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    @unittest.skipIf(device_cc() != 90, "This unittest is for cc = Sm90 only")
    def test_permute_2(self):
        """
        Returning a tensor with shape [m, n]
        """
        def evt_permute(accum, alpha, C):
            F = alpha * accum
            F_permute = permute(F, indices=(0, 2, 1))
            D = F_permute + C
            return D, F

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "alpha": 0.5,
                "C": self.fake_tensor(self.element, (l, n, m)),
                "F": self.fake_tensor(self.element, (l, m, n)),
                "D": self.fake_tensor(self.element, (l, n, m)),
            }

            launcher = EVTTestBed(self.element, evt_permute, example_inputs)
            input_keys = ["C", "alpha"]
            result_keys = ["D", "F"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    @unittest.skipIf(device_cc() != 90, "This unittest is for cc = Sm90 only")
    def test_permute_3(self):
        """
        Returning a tensor with shape [m, n]
        """
        def evt_permute(accum, alpha, C):
            F = alpha * accum
            F_permute = permute(F, indices=(1, 0, 2))
            D = F_permute + C
            return D, F

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "alpha": 0.5,
                "C": self.fake_tensor(self.element, (m, l, n)),
                "F": self.fake_tensor(self.element, (l, m, n)),
                "D": self.fake_tensor(self.element, (m, l, n)),
            }

            launcher = EVTTestBed(self.element, evt_permute, example_inputs)
            input_keys = ["C", "alpha"]
            result_keys = ["D", "F"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_reshape(self):
        """
        Test reshape
        """
        def evt_reshape(accum, alpha, TensorE):
            F = alpha * accum
            E_reshape = reshape(TensorE, new_shape=(512, 1))
            D = F + E_reshape
            return D

        example_inputs = {
            "accum": self.fake_tensor(self.element, (self.l, self.m, self.n)),
            "alpha": 0.5,
            "TensorE": self.fake_tensor(self.element, (16, 32)),
            "D": self.fake_tensor(self.element, (self.l, self.m, self.n)),
        }

        launcher = EVTTestBed(self.element, evt_reshape, example_inputs)
        input_keys = ["alpha", "TensorE"]
        result_keys = ["D"]
        launcher.verify(self.problem_size, input_keys, result_keys, self.l)

    def test_reshape2(self):
        """
        Test reshape
        """
        def evt_reshape(accum, alpha, TensorE):
            F = alpha * accum
            F_reshape = reshape(F, new_shape=(2, 3, 512, 256))
            D = F_reshape + TensorE
            return D

        example_inputs = {
            "accum": self.fake_tensor(self.element, (self.l, self.m, self.n)),
            "alpha": 0.5,
            "TensorE": self.fake_tensor(self.element, (2, 3, 1, self.n)),
            "D": self.fake_tensor(self.element, (2, 3, self.m, self.n)),
        }

        launcher = EVTTestBed(self.element, evt_reshape, example_inputs)
        input_keys = ["alpha", "TensorE"]
        result_keys = ["D"]
        launcher.verify(self.problem_size, input_keys, result_keys, self.l)


if __name__ == '__main__':
    unittest.main()

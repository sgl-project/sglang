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
class TestEVTStore(EVTTestCaseBase):

    @unittest.skipIf(device_cc() != 90, "This test is only for CC 90")
    def test_invalid_store(self):
        """
        Test invalid store
        """
        def evt_invalid_store(accum):
            D = accum
            F = D + 1 # D has users, which is not allowed on SM90 or higher
            return D, F
        
        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "D": self.fake_tensor(self.element, (l, m, n)),
                "F": self.fake_tensor(self.element, (l, m, n))
            }
            with self.assertRaisesRegex(
                    RuntimeError, 
                    r"On SM90 or higher, D is expected to be a output node with 0 users " 
                    r"to enable smem reuse between C and D, but got 1"
                ):
                launcher = EVTTestBed(self.element, evt_invalid_store, example_inputs)
            
            break  # Only need to test once

    def test_aux_store(self):
        """
        Returning a tensor with shape [m, n]
        """
        def evt_aux_store(accum, alpha, C):
            F = alpha * accum
            D = F + C
            return D, F

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "alpha": 0.5,
                "C": self.fake_tensor(self.element, (l, m, n)),
                "F": self.fake_tensor(self.element, (l, m, n)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_aux_store, example_inputs)
            input_keys = ["C", "alpha"]
            result_keys = ["D", "F"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_col_reduce(self):
        """
        Reduction [m, n] -> [m, 1]
        """
        def evt_row_reduce(accum, alpha, C):
            acc_row_max = max(accum, dim=[2,])
            F = alpha * accum
            F_row_max = max(F, dim=[0, 2])
            D = F + C
            return D, F_row_max, acc_row_max

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "alpha": 2.0,
                "C": self.fake_tensor(self.element, (l, m, n)),
                "F_row_max": self.fake_tensor(np.float32, (m, 1)),
                "acc_row_max": self.fake_tensor(np.float32, (l, m, 1)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_row_reduce, example_inputs)
            input_keys = ["C", "alpha"]
            result_keys = ["D", "F_row_max", "acc_row_max"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_row_reduce(self):
        """
        Reduction [m, n] -> [n]
        """
        def evt_col_reduce(accum, alpha, C):
            acc_col_max = max(accum, dim=[1,])
            F = alpha * accum
            F_col_max = max(F, dim=[0, 1])
            D = F + C
            return D, F_col_max, acc_col_max

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "alpha": 2.0,
                "C": self.fake_tensor(self.element, (l, m, n)),
                "F_col_max": self.fake_tensor(np.float32, (n,)),
                "acc_col_max": self.fake_tensor(np.float32, (l, 1, n)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_col_reduce, example_inputs)
            input_keys = ["C", "alpha"]
            result_keys = ["D", "F_col_max", "acc_col_max"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_scalar_reduce(self):
        """
        Reduction [m, n] -> [1,]
        """
        def evt_scalar_reduce(accum, alpha, C):
            acc_max = max(accum, dim=[1, 2])
            F = alpha * accum
            F_max = max(F, dim=[0, 1, 2])
            D = F + C
            return D, F_max, acc_max

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "alpha": 2.0,
                "C": self.fake_tensor(self.element, (l, m, n)),
                "acc_max": self.fake_tensor(np.float32, (l, 1, 1)),
                "F_max": self.fake_tensor(np.float32, (1,)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_scalar_reduce, example_inputs)
            input_keys = ["C", "alpha"]
            result_keys = ["D", "F_max", "acc_max"]
            launcher.verify((m, n, k), input_keys, result_keys, l)


if __name__ == '__main__':
    unittest.main()

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
Unit test for load nodes in SM90
"""

import logging
import unittest

import cutlass_cppgen
from cutlass_cppgen.backend import *
from cutlass_cppgen.epilogue import *

from utils.evt_testbed import EVTTestBed, EVTTestCaseBase

cutlass_cppgen.set_log_level(logging.WARNING)


@unittest.skipIf(device_cc() not in [80, 86, 89, 90], "This unittest is only supported on CC [80, 86, 89, 90]")
class TestEVTLoad(EVTTestCaseBase):

    def test_tensor_load(self):
        """
        Load extra tensor with shape [m, n]
        """
        def evt_tensor_load(accum, C, aux, aux_batch):
            D = accum + C + aux + aux_batch
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "aux": self.fake_tensor(self.element, (m, n)),
                "aux_batch": self.fake_tensor(np.float32, (l, m, n)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_tensor_load, example_inputs)
            input_keys = ["C", "aux", "aux_batch"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_row_broadcast(self):
        """
        Load extra tensor with shape [1, n]
        """
        def evt_row_broadcast(accum, C, bias, bias_batch):
            D = accum + C + bias + bias_batch
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "bias": self.fake_tensor(self.element, (n,)),
                "bias_batch": self.fake_tensor(np.float32, (l, 1, n)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_row_broadcast, example_inputs)
            input_keys = ["C", "bias", "bias_batch"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_column_broadcast(self):
        """
        Load extra tensor with shape [m, 1]
        """
        def evt_column_broadcast(accum, C, bias, bias_batch):
            D = accum + C + bias + bias_batch
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "bias": self.fake_tensor(self.element, (m, 1)),
                "bias_batch": self.fake_tensor(np.float32, (l, m, 1)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_column_broadcast, example_inputs)
            input_keys = ["C", "bias", "bias_batch"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_scalar_broadcast(self):
        """
        Load extra tensor with shape [1, 1]
        """
        def evt_scalar_broadcast(accum, C, alpha, alpha_batch):
            D = accum + C + alpha + alpha_batch
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "alpha": 0.5,
                "alpha_batch": self.fake_tensor(np.float32, (l, 1, 1)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_scalar_broadcast, example_inputs)
            input_keys = ["C", "alpha", "alpha_batch"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)


if __name__ == '__main__':
    unittest.main()

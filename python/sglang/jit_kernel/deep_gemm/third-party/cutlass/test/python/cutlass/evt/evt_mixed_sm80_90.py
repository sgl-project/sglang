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
Unittest for mixed types of nodes in SM90
"""

import logging
import unittest

import cutlass_cppgen
from cutlass_cppgen.backend import *
from cutlass_cppgen.epilogue import *
from cutlass_cppgen.swizzle import ThreadblockSwizzleStreamK

from utils.evt_testbed import EVTTestBed, EVTTestCaseBase

cutlass_cppgen.set_log_level(logging.WARNING)


@unittest.skipIf(device_cc() not in [80, 86, 89, 90], "This unittest is only supported on CC [80, 86, 89, 90]")
class TestEVTMixed(EVTTestCaseBase):

    def test_same_variable_used_multiple_times(self):
        """
        The same variable z0 is used multiple times
        """
        def evt_aux_store(accum):
            z0 = relu(accum)
            D = z0 + z0
            return z0, D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "D": self.fake_tensor(self.element, (l, m, n)),
                "z0": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_aux_store, example_inputs)
            input_keys = ["accum"]
            result_keys = ["z0", "D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_no_lca(self):
        """
        The same variable z0 is used multiple times
        """
        def evt_no_lca(accum, bias):
            E = relu(accum)
            F = E + bias
            tmp_2 = E + 2
            D = tmp_2 + E
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "D": self.fake_tensor(self.element, (l, m, n)),
                "bias": self.fake_tensor(self.element, (m,1), stride=(1,0)),
            }

            launcher = EVTTestBed(self.element, evt_no_lca, example_inputs)
            input_keys = ["accum", "bias"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_mixed_dag(self):
        def evt_mixed_dag(accum, alpha, C, beta, aux, cbias, rbias):
            F = alpha * accum + (beta * C + aux)
            F_row_max = max(F, dim=[0, 1])
            E = relu(F + 1) + cbias + rbias
            E_col_max = max(E, dim=[0, 2])
            D = E + F
            return D, F, F_row_max, E_col_max

        if device_cc() == 80:
            alignments = [2, 4, 8]
        else:
            # Sm90 EVT currently only supports 128-bit alignment
            alignments = [8,]
        for align in alignments:
            for m, n, k, l in self.get_problem_sizes(align):
                example_inputs = {
                    "accum": self.fake_tensor(self.element, (l, m, n)),
                    "alpha": 1.0,
                    "C": self.fake_tensor(self.element, (l, m, n)),
                    "beta": 1.0,
                    "aux": self.fake_tensor(self.element, (l, m, n)),
                    "cbias": self.fake_tensor(self.element, (m, 1)),
                    "rbias": self.fake_tensor(self.element, (n,)),
                    "D": self.fake_tensor(self.element, (l, m, n)),
                    "F": self.fake_tensor(self.element, (l, m, n)),
                    "F_row_max": self.fake_tensor(DataType.f32, (n,)),
                    "E_col_max": self.fake_tensor(DataType.f32, (m, 1))
                }

                launcher = EVTTestBed(self.element, evt_mixed_dag, example_inputs)
                input_keys = ["alpha", "C", "beta", "aux", "cbias", "rbias"]
                result_keys = ["D", "F", "F_row_max", "E_col_max"]
                launcher.verify((m, n, k), input_keys, result_keys, l)

    @unittest.skipIf(device_cc() not in [80, 89], "This unittest is for cc 80 and 89 only")
    def test_mixed_dag_float(self):
        def evt_mixed_dag(accum, alpha, C, beta, aux, cbias, rbias):
            F = alpha * accum + (beta * C + aux)
            F_row_max = max(F, dim=[0, 1])
            E = relu(F + 1) + cbias + rbias
            E_col_max = max(E, dim=[0, 2])
            D = E + F
            return D, F, F_row_max, E_col_max

        for align in [3, 2, 4]:
            for m, n, k, l in self.get_problem_sizes(align):
                example_inputs = {
                    "accum": self.fake_tensor(np.float32, (l, m, n)),
                    "alpha": 1.0,
                    "C": self.fake_tensor(np.float32, (l, m, n)),
                    "beta": 1.0,
                    "aux": self.fake_tensor(np.float32, (l, m, n)),
                    "cbias": self.fake_tensor(np.float32, (m, 1)),
                    "rbias": self.fake_tensor(np.float32, (n,)),
                    "D": self.fake_tensor(np.float32, (l, m, n)),
                    "F": self.fake_tensor(np.float32, (l, m, n)),
                    "F_row_max": self.fake_tensor(np.float32, (n,)),
                    "E_col_max": self.fake_tensor(np.float32, (m, 1))
                }
                launcher = EVTTestBed(DataType.f32, evt_mixed_dag, example_inputs)
                input_keys = ["alpha", "C", "beta", "aux", "cbias", "rbias"]
                result_keys = ["D", "F", "F_row_max", "E_col_max"]
                launcher.verify((m, n, k), input_keys, result_keys, l)

    @unittest.skipIf(device_cc() not in [80, 89], "This unittest is for cc 80 and 89 only")
    def test_mixed_dag_stage2(self):
        def evt_mixed_dag(accum, alpha, C, beta, aux, cbias, rbias):
            F = alpha * accum + (beta * C + aux)
            F_row_max = max(F, dim=[0, 1])
            E = relu(F + 1) + cbias + rbias
            E_col_max = max(E, dim=[0, 2])
            D = E + F
            return D, F, F_row_max, E_col_max

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "alpha": 1.0,
                "C": self.fake_tensor(self.element, (l, m, n)),
                "beta": 1.0,
                "aux": self.fake_tensor(self.element, (l, m, n)),
                "cbias": self.fake_tensor(self.element, (m, 1)),
                "rbias": self.fake_tensor(self.element, (n,)),
                "D": self.fake_tensor(self.element, (l, m, n)),
                "F": self.fake_tensor(self.element, (l, m, n)),
                "F_row_max": self.fake_tensor(DataType.f32, (n,)),
                "E_col_max": self.fake_tensor(DataType.f32, (m, 1))
            }

            launcher = EVTTestBed(self.element, evt_mixed_dag, example_inputs, epilogue_stages=2)
            input_keys = ["alpha", "C", "beta", "aux", "cbias", "rbias"]
            result_keys = ["D", "F", "F_row_max", "E_col_max"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    @unittest.skipIf(device_cc() not in [80, 89], "This unittest is for cc 80 and 89 only")
    def test_mixed_dag_partition_k(self):
        def evt_mixed_dag(accum, alpha, C, beta, aux, cbias, rbias):
            F = alpha * accum + (beta * C + aux)
            F_row_max = max(F, dim=[0, 1])
            E = relu(F + 1) + cbias + rbias
            E_col_max = max(E, dim=[0, 2])
            D = E + F
            return D, F, F_row_max, E_col_max

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "alpha": 1.0,
                "C": self.fake_tensor(self.element, (l, m, n)),
                "beta": 1.0,
                "aux": self.fake_tensor(self.element, (l, m, n)),
                "cbias": self.fake_tensor(self.element, (m, 1)),
                "rbias": self.fake_tensor(self.element, (n,)),
                "D": self.fake_tensor(self.element, (l, m, n)),
                "F": self.fake_tensor(self.element, (l, m, n)),
                "F_row_max": self.fake_tensor(DataType.f32, (n,)),
                "E_col_max": self.fake_tensor(DataType.f32, (m, 1))
            }

            tile_description = {
                "threadblock_shape": [128, 128, 64],
                "warp_count": [2, 2, 2]
            }

            launcher = EVTTestBed(self.element, evt_mixed_dag, example_inputs, tile_description=tile_description, epilogue_stages=2)
            input_keys = ["alpha", "C", "beta", "aux", "cbias", "rbias"]
            result_keys = ["D", "F", "F_row_max", "E_col_max"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    @unittest.skipIf(device_cc() not in [80, 89], "This unittest is for cc 80 and 89 only")
    def test_mixed_dag_stream_k(self):
        def evt_mixed_dag(accum, alpha, C, beta, aux, cbias, rbias):
            F = alpha * accum + (beta * C + aux)
            F_row_max = max(F, dim=[0, 1])
            E = relu(F + 1) + cbias + rbias
            E_col_max = max(E, dim=[0, 2])
            D = E + F
            return D, F, F_row_max, E_col_max

        # High per-sm occupancy tile_description
        tile_description = {
            "threadblock_shape": [128, 128, 32],
            "warp_count": [2, 2, 1],
            "stages": 3
        }
        tds = [None, tile_description]
        for td in tds:
            for m, n, k, l in self.get_problem_sizes(8, k=960, batch_count=[1, 3]):
                if l == 1:
                    example_inputs = {
                        "accum": self.fake_tensor(self.element, (m, n)),
                        "alpha": 1.0,
                        "C": self.fake_tensor(self.element, (m, n)),
                        "beta": 1.0,
                        "aux": self.fake_tensor(self.element, (m, n)),
                        "cbias": self.fake_tensor(self.element, (m, 1)),
                        "rbias": self.fake_tensor(self.element, (n,)),
                        "D": self.fake_tensor(self.element, (m, n)),
                        "F": self.fake_tensor(self.element, (m, n)),
                        "F_row_max": self.fake_tensor(DataType.f32, (n,)),
                        "E_col_max": self.fake_tensor(DataType.f32, (m, 1))
                    }
                else:
                    example_inputs = {
                        "accum": self.fake_tensor(self.element, (l, m, n)),
                        "alpha": 1.0,
                        "C": self.fake_tensor(self.element, (l, m, n)),
                        "beta": 1.0,
                        "aux": self.fake_tensor(self.element, (l, m, n)),
                        "cbias": self.fake_tensor(self.element, (m, 1)),
                        "rbias": self.fake_tensor(self.element, (n,)),
                        "D": self.fake_tensor(self.element, (l, m, n)),
                        "F": self.fake_tensor(self.element, (l, m, n)),
                        "F_row_max": self.fake_tensor(DataType.f32, (n,)),
                        "E_col_max": self.fake_tensor(DataType.f32, (m, 1))
                    }

                if td is not None:
                    launcher = EVTTestBed(
                        self.element, evt_mixed_dag, example_inputs,
                        tile_description=td,
                        swizzling_functor=ThreadblockSwizzleStreamK, backend="torch")
                else:
                    launcher = EVTTestBed(
                        self.element, evt_mixed_dag, example_inputs,
                        swizzling_functor=ThreadblockSwizzleStreamK, backend="torch")

                input_keys = ["alpha", "C", "beta", "aux", "cbias", "rbias"]
                result_keys = ["D", "F", "F_row_max", "E_col_max"]
                launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_mixed_dag_no_batch(self):
        def evt_mixed_dag_no_batch(accum, alpha, C, beta, aux, cbias, rbias):
            F = alpha * accum + (beta * C + aux)
            F_row_max = max(F, dim=[0, 1])
            E = relu(F + 1) + cbias + rbias
            E_col_max = max(E, dim=[0, 2])
            D = E + F
            return D, F, F_row_max, E_col_max

        for m, n, k, _ in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (m, n)),
                "alpha": 1.0,
                "C": self.fake_tensor(self.element, (m, n)),
                "beta": 1.0,
                "aux": self.fake_tensor(self.element, (m, n)),
                "cbias": self.fake_tensor(self.element, (m, 1)),
                "rbias": self.fake_tensor(self.element, (n,)),
                "D": self.fake_tensor(self.element, (m, n)),
                "F": self.fake_tensor(self.element, (m, n)),
                "F_row_max": self.fake_tensor(DataType.f32, (n,)),
                "E_col_max": self.fake_tensor(DataType.f32, (m, 1))
            }

            launcher = EVTTestBed(self.element, evt_mixed_dag_no_batch, example_inputs)
            input_keys = ["alpha", "C", "beta", "aux", "cbias", "rbias"]
            result_keys = ["D", "F", "F_row_max", "E_col_max"]
            launcher.verify((m, n, k), input_keys, result_keys, 1)

if __name__ == '__main__':
    unittest.main()

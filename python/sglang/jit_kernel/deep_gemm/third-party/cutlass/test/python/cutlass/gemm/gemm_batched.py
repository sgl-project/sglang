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
High-level tests for running batched GEMMs
"""

from functools import partial
import logging
from math import prod
import unittest

import cutlass_cppgen
from cutlass_cppgen.backend.utils.device import device_cc
import torch

from utils import LayoutCombination

cutlass_cppgen.set_log_level(logging.WARNING)

torch.manual_seed(2023)


def pytorch_reference(A, B, C, alpha, beta):
    # Get the batch count. Assume that any of A, B, and C
    # with a batch dimension ahve matching batch count. Thus,
    # we break out of the loop once we have found the first
    # tensor containing a batch dimension.
    batch_count = (1,)
    for tensor in [A, B, C]:
        if len(tensor.shape) > 2:
            batch_count = tensor.shape[:-2]
            break

    int_batch_count = prod(batch_count)

    def add_batch(tensor):
        if len(tensor.shape) == 2:
            return tensor.unsqueeze(0).repeat(int_batch_count, 1, 1)
        else:
            return tensor.reshape(-1, tensor.size(-2), tensor.size(-1))

    # Reshape tensors to have batch dimension
    A = add_batch(A)
    B = add_batch(B)
    C = add_batch(C)

    ret = (torch.bmm(A, B) * alpha) + (C * beta)
    reshape_vals = batch_count + C.shape[-2:]
    return ret.reshape(*reshape_vals)


def initialize(rows, cols, batch):
    tensor = torch.randint(-3, 3, size=(rows*cols*prod(batch),), device='cuda').half()
    if len(batch) > 0 and prod(batch) > 1:
        reshape_vals = batch + (rows, cols)
        return tensor.reshape(*reshape_vals)
    else:
        return tensor.reshape(rows, cols)


class GemmF16Batched(unittest.TestCase):
    def run_batched(self, batch_count: tuple, batch_A: bool, batch_B: bool, batch_C: bool):
        M = 512
        N = 256
        K = 128
        alpha = 1.
        beta = 2.

        A = initialize(M, K, batch_count if batch_A else (1,))
        B = initialize(K, N, batch_count if batch_B else (1,))
        C = initialize(M, N, batch_count if batch_C else (1,))
        D = initialize(M, N, batch_count)

        plan = cutlass_cppgen.op.Gemm(A=A, B=B, C=C, D=D, element_accumulator=cutlass_cppgen.DataType.f32)
        plan.run(A, B, C, D, alpha, beta)
        reference = pytorch_reference(A, B, C, alpha, beta)
        assert reference.equal(D)

    def test_batched_ABC(self):
        self.run_batched((3,), True, True, True)
        self.run_batched((2, 3), True, True, True)

    def test_batched_AB(self):
        self.run_batched((3,), True, True, False)
        self.run_batched((2, 3), True, True, False)

    def test_batched_AC(self):
        self.run_batched((3,), True, False, True)
        self.run_batched((2, 3), True, False, True)

    def test_batched_BC(self):
        self.run_batched((3,), False, True, True)
        self.run_batched((2, 3), False, True, True)

    def test_batched_A(self):
        self.run_batched((3,), True, False, False)
        self.run_batched((2, 3), True, False, False)

    def test_batched_B(self):
        self.run_batched((3,), False, True, False)
        self.run_batched((2, 3), False, True, False)

if __name__ == '__main__':
    unittest.main()

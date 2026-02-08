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
Tests emitting a CUTLASS kernel to a PyTorch CUDA extension
"""

import random
import tempfile
import unittest

from cutlass_library import ConvMode

import cutlass_cppgen

if cutlass_cppgen.utils.datatypes.is_torch_available():
    import torch


def _initialize(dtype, M: int, N: int, K: int):
    """
    Utility function to initialize A, B, C, and D matrices corresponding to dimensions M, N, and K

    :param dtype: data type of tensors
    :param M: M dimension of GEMM problem
    :type M: int
    :param N: N dimension of GEMM problem
    :type N: int
    :param K: N dimension of GEMM problem
    :type K: int

    :return: initialized tensors A, B, C, and D
    :rtype: list
    """
    sizes = [(M, K), (K, N), (M, N), (M, N)]
    return [torch.randint(-3, 3, size, device='cuda').to(dtype) for size in sizes]


def _generate_problems(dtype, num):
    """
    Utility function to generate `num` GEMMs of random sizes

    :param dtype: data type of tensors
    :param num: number of GEMMs to generate
    :type num: int

    :return: lists of A, B, C, and D tensors
    :rtype: list
    """
    valid_sizes = [128, 256, 512, 1024]
    As, Bs, Cs, Ds = [], [], [], []
    for _ in range(num):
        M, N, K = [random.choice(valid_sizes) for _ in range(3)]
        A, B, C, D = _initialize(dtype, M, N, K)
        As.append(A)
        Bs.append(B)
        Cs.append(C)
        Ds.append(D)
    return As, Bs, Cs, Ds

def _generate_conv2d_problem(conv_kind, dtype, ps):
    """
    Utility function to generate conv2d inputs

    :param conv_kind: kind of convolution
    :type conv_kind: str
    :param dtype: data type of tensors
    :param problem_size: the conv2d problem size
    :type problem_size: cutlass_cppgen.shape.Conv2DProblemSize

    :return: initialized tensors A, B, C, and D
    :rtype: list
    """
    if conv_kind == "fprop":
        tensor_A_size = (ps.N, ps.C, ps.H, ps.W)
        tensor_B_size = (ps.K, ps.C, ps.R, ps.S)
        tensor_C_size = (ps.N, ps.K, ps.P, ps.Q)
    elif conv_kind == "dgrad":
        tensor_A_size = (ps.N, ps.K, ps.P, ps.Q)
        tensor_B_size = (ps.K, ps.C, ps.R, ps.S)
        tensor_C_size = (ps.N, ps.C, ps.H, ps.W)
    else:
        tensor_A_size = (ps.N, ps.K, ps.P, ps.Q)
        tensor_B_size = (ps.N, ps.C, ps.H, ps.W)
        tensor_C_size = (ps.K, ps.C, ps.R, ps.S)
    sizes = [tensor_A_size, tensor_B_size, tensor_C_size]
    return [torch.ceil(torch.empty(size, dtype=dtype, device='cuda').uniform_(-4.5, 3.5)).to(memory_format=torch.channels_last) for size in sizes]


@unittest.skipIf(not cutlass_cppgen.utils.datatypes.is_torch_available(), 'PyTorch must be available to run PyTorch extension tests')
class PyTorchExtensionTest(unittest.TestCase):

    def test_gemm(self):
        random.seed(2023)

        dtype = torch.float16
        plan = cutlass_cppgen.op.Gemm(element=dtype, layout=cutlass_cppgen.LayoutType.RowMajor)
        op = plan.construct()

        with tempfile.TemporaryDirectory() as tmpdir:
            mod = cutlass_cppgen.emit.pytorch(op, name='gemm_mod', cc=plan.cc, sourcedir=tmpdir, jit=True)

        A, B, C, _ = _initialize(dtype, 1024, 256, 512)

        D_ref = A @ B
        D = mod.run(A, B)
        assert torch.allclose(D, D_ref)

        D = mod.run(A, B, C)
        assert torch.allclose(D, D_ref)

        D = mod.run(A, B, C, 1.0)
        assert torch.allclose(D, D_ref)

        D = mod.run(A, B, C, 1.0, 0.0)
        assert torch.allclose(D, D_ref)

        alpha = 2.0
        beta = -1.0
        D_ref = (A @ B) * alpha + (beta * C)
        D = mod.run(A, B, C, alpha, beta)
        assert torch.allclose(D, D_ref)

    def test_grouped_gemm(self):
        random.seed(2023)

        dtype = torch.float16
        plan = cutlass_cppgen.op.GroupedGemm(element=dtype, layout=cutlass_cppgen.LayoutType.RowMajor)
        op = plan.construct()

        with tempfile.TemporaryDirectory() as tmpdir:
            mod = cutlass_cppgen.emit.pytorch(op, name='grouped_gemm_mod', cc=plan.cc, sourcedir=tmpdir, jit=True)

        As, Bs, Cs, _ = _generate_problems(dtype, 50)

        def check_all(X, Y):
            for x, y in zip(X, Y):
                assert torch.allclose(x, y)

        Ds_ref = [a @ b for a, b in zip(As, Bs)]
        Ds = mod.run(As, Bs)
        check_all(Ds, Ds_ref)

        Ds = mod.run(As, Bs, Cs)
        check_all(Ds, Ds_ref)

        Ds = mod.run(As, Bs, Cs, 1.0)
        check_all(Ds, Ds_ref)

        Ds = mod.run(As, Bs, Cs, 1.0, 0.0)
        check_all(Ds, Ds_ref)

        alpha = 2.0
        beta = -1.0
        Ds_ref = [(a @ b) * alpha + (beta * c) for a, b, c in zip(As, Bs, Cs)]
        Ds = mod.run(As, Bs, Cs, alpha, beta)
        check_all(Ds, Ds_ref)

    def test_conv2d_fprop(self):
        torch.manual_seed(2023)

        dtype = torch.float16
        plan = cutlass_cppgen.op.Conv2d(kind="fprop", element=dtype, element_accumulator=torch.float32)
        plan.activation = "relu"

        op = plan.construct()
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = cutlass_cppgen.emit.pytorch(op, name="conv2d_mod", cc=plan.cc, sourcedir=tmpdir, jit=True)

        problem_size = cutlass_cppgen.shape.Conv2DProblemSize(
            1, 4, 4, 16,
            8, 3, 3, 16,
            0, 0,
            3, 3,
            1, 1
        )

        A, B, C = _generate_conv2d_problem("fprop", dtype, problem_size)
        stride = (problem_size.stride_h, problem_size.stride_w)
        padding = (problem_size.pad_h, problem_size.pad_w)

        alpha = 1.0
        beta = 0.5

        D_ref = alpha * torch.ops.aten.conv2d(
            A, B, stride=stride, padding=padding
        ) + beta * C
        D_ref = torch.nn.functional.relu(D_ref)
        D = mod.run(A, B, C, stride, padding, alpha=alpha, beta=beta)

        assert torch.allclose(D, D_ref)

        # Test serial split-K
        D_serial_split_k = mod.run(A, B, C, stride, padding, alpha=alpha, beta=beta, split_k_mode="serial", split_k_slices=3)
        assert torch.allclose(D, D_serial_split_k)

        # Test parallel split-K
        D_parallel_split_k = mod.run(A, B, C, stride, padding, alpha=alpha, beta=beta, split_k_mode="parallel", split_k_slices=7)
        assert torch.allclose(D, D_parallel_split_k)


    def test_conv2d_dgrad(self):
        torch.manual_seed(2023)
        dtype = torch.float16
        plan = cutlass_cppgen.op.Conv2d(kind="dgrad", element=dtype, element_accumulator=torch.float32)

        op = plan.construct()
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = cutlass_cppgen.emit.pytorch(op, name="conv2d_dgrad_mod", cc=plan.cc, sourcedir=tmpdir, jit=True)

        problem_size = cutlass_cppgen.shape.Conv2DProblemSize(
            1, 4, 4, 16,
            8, 3, 3, 16,
            0, 0,
            3, 3,
            1, 1,
            ConvMode.CrossCorrelation,
            1, 1
        )

        A, B, C = _generate_conv2d_problem("dgrad", dtype, problem_size)
        stride = (problem_size.stride_h, problem_size.stride_w)
        padding = (problem_size.pad_h, problem_size.pad_w)

        alpha = 1.0
        beta = 0.5
        input_size = (problem_size.N, problem_size.C, problem_size.H, problem_size.W)
        D_ref = alpha * torch.nn.grad.conv2d_input(
            input_size, B, A,
            stride=stride, padding=padding
        ) + beta * C
        D = mod.run(input_size, A, B, C, stride, padding, alpha=alpha, beta=beta, )

        assert torch.allclose(D, D_ref)

    def test_conv2d_wgrad(self):
        torch.manual_seed(2023)
        dtype = torch.float16
        plan = cutlass_cppgen.op.Conv2d(kind="wgrad", element=dtype, element_accumulator=torch.float32)

        op = plan.construct()
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = cutlass_cppgen.emit.pytorch(op, name="conv2d_wgrad_mod", cc=plan.cc, sourcedir=tmpdir, jit=True)

        problem_size = cutlass_cppgen.shape.Conv2DProblemSize(
            1, 4, 4, 16,
            8, 3, 3, 16,
            0, 0,
            3, 3,
            1, 1,
            ConvMode.CrossCorrelation,
            1, 1
        )

        A, B, C = _generate_conv2d_problem("wgrad", dtype, problem_size)
        stride = (problem_size.stride_h, problem_size.stride_w)
        padding = (problem_size.pad_h, problem_size.pad_w)

        alpha = 1.0
        beta = 0.5
        weight_size = (problem_size.K, problem_size.C, problem_size.R, problem_size.S)
        D_ref = alpha * torch.nn.grad.conv2d_weight(
            B, weight_size, A,
            stride=stride, padding=padding
        ) + beta * C
        D = mod.run(weight_size, A, B, C, stride, padding, alpha=alpha, beta=beta)

        assert torch.allclose(D, D_ref)

        # Test serial split-K
        D_serial_split_k = mod.run(weight_size, A, B, C, stride, padding, alpha=alpha, beta=beta, split_k_mode="serial", split_k_slices=3)
        assert torch.allclose(D, D_serial_split_k)

        # Test parallel split-K
        D_parallel_split_k = mod.run(weight_size, A, B, C, stride, padding, alpha=alpha, beta=beta, split_k_mode="parallel", split_k_slices=7)
        assert torch.allclose(D, D_parallel_split_k)


if __name__ == '__main__':
    unittest.main()

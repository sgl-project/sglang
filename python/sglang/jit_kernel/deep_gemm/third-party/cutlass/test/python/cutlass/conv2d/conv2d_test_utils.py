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
Utility functions for Conv2d tests.
"""

from cutlass_library import SubstituteTemplate
import torch

import cutlass_cppgen
from cutlass_library import (
    ConvKind,
    ConvMode,
    DataType,
    DataTypeNames,
    EpilogueScheduleSuffixes,
    KernelScheduleSuffixes,
    LayoutType,
    OpcodeClassNames,
    ShortDataTypeNames,
    ShortLayoutTypeNames,
    SplitKMode,
)
from cutlass_cppgen.shape import Conv2DProblemSize
from cutlass_cppgen.utils.datatypes import numpy_type, torch_type

from conv2d_problem_sizes import TestbedConv2dProblemSizes


def get_name_conv2d(
    arch,
    conv_kind,
    element,
    element_accumulator,
    element_output,
    opclass,
    threadblock_shape,
    warp_count,
    instruction_shape,
    stages,
    iterator_algorithm,
    swizzle,
    split_k_mode,
    split_k_slices,
    activation
):
    """
    Generates a procedural name for a test case for conv2d

    :param arch: compute capability of kernel being generated
    :type arch: int
    :param conv_kind: the convolution type (i.e. fprop, dgrad, wgrad)
    :type conv_kind: str
    :param iterator_algorithm: the iterator algorithm applied
    :type iterator_algorithm: cutlass_library.library.IteratorAlgorithm
    :param element_a: data type of operand A
    :param element_b: data type of operand B
    :param element_c: data type of operand C
    :param element_accumulator: data type used in accumulation
    :param opclass: class of operation being performed (e.g., SIMT, Tensor Core)
    :type opclass: cutlass_cppgen.OpcodeClass
    :param threadblock_shape: indexable container of dimensions of threadblock tiles
    :param stages: number of pipeline stages to use in the kernel
    :type stages: int
    :param stride_support: stride support of dgrad
    :param alignment: int
    :type alignment: int

    :return: str
    """
    if iterator_algorithm is None:
        iterator_algorithm = "AUTO"
    if swizzle is None:
        swizzle = 1
    name_format = "test_SM${arch}_Device_Conv2d_${conv_kind}_${iter_alg}_ImplicitGemm_${eA}nhwc_${eB}nhwc_${eC}nhwc_${opclass}_${acc}_${tbM}x${tbN}x${tbK}_${wM}x${wN}x${wK}_${IM}${IN}${IK}_stage${stages}_swizzle${swizzle}_${split_k_mode}${split_k_slices}_${activation}"

    return SubstituteTemplate(
        name_format,
        {
            "arch": str(arch),
            "conv_kind": conv_kind,
            "iter_alg": iterator_algorithm,
            "eA": DataTypeNames[element],
            "eB": DataTypeNames[element],
            "eC": DataTypeNames[element_output],
            "opclass": opclass,
            "acc": DataTypeNames[element_accumulator],
            "tbM": str(threadblock_shape[0]),
            "tbN": str(threadblock_shape[1]),
            "tbK": str(threadblock_shape[2]),
            "wM": str(threadblock_shape[0] // warp_count[0]),
            "wN": str(threadblock_shape[1] // warp_count[1]),
            "wK": str(threadblock_shape[2] // warp_count[2]),
            "IM": str(instruction_shape[0]),
            "IN": str(instruction_shape[1]),
            "IK": str(instruction_shape[2]),
            "stages": str(stages),
            "swizzle": str(swizzle),
            "split_k_mode": split_k_mode,
            "split_k_slices": str(split_k_slices),
            "activation": activation
        }
    )


def conv2d_few_channel_problemsizes(channels):
    problem_sizes = [
        Conv2DProblemSize(
            1, 8, 8, channels,
            16, 3, 3, channels,
            1, 1,
            2, 2,
            1, 1,
            ConvMode.CrossCorrelation,
            1, 1
        ),
        Conv2DProblemSize(
            1, 16, 16, channels,
            16, 3, 3, channels,
            1, 1,
            2, 2,
            1, 1,
            ConvMode.CrossCorrelation,
            1, 1
        ),
        Conv2DProblemSize(
            1, 16, 16, channels,
            16, 7, 7, channels,
            1, 1,
            1, 1,
            1, 1,
            ConvMode.CrossCorrelation,
            1, 1
        ),
        Conv2DProblemSize(
            1, 224, 224, channels,
            32, 7, 7, channels,
            1, 1,
            1, 1,
            1, 1,
            ConvMode.CrossCorrelation,
            1, 1
        ),
        Conv2DProblemSize(
            1, 224, 224, channels,
            64, 7, 7, channels,
            1, 1,
            2, 2,
            1, 1,
            ConvMode.CrossCorrelation,
            1, 1
        ),
        Conv2DProblemSize(
            1, 224, 224, channels,
            64, 5, 5, channels,
            1, 1,
            1, 1,
            1, 1,
            ConvMode.CrossCorrelation,
            1, 1
        ),
        Conv2DProblemSize(
            1, 224, 224, channels,
            64, 5, 5, channels,
            1, 1,
            2, 2,
            1, 1,
            ConvMode.CrossCorrelation,
            1, 1
        ),
    ]

    return problem_sizes


def validate_problem_size(ps, conv_kind, split_k_slices):
    P = (ps.H + 2 * ps.pad_h - ps.dilation_h * (ps.R - 1) - 1) // ps.stride_h + 1
    Q = (ps.W + 2 * ps.pad_w - ps.dilation_w * (ps.S - 1) - 1) // ps.stride_w + 1
    if P != ps.P or Q != ps.Q:
        return False

    # Split-K (serial or parallel) is not supported for strided dgrad
    if conv_kind == "dgrad" and split_k_slices > 1 and (ps.stride_h > 1 or ps.stride_w > 1):
        return False
    return True


class Conv2dLauncherFrontend:
    def __init__(self, plan: cutlass_cppgen.Conv2d, seed: int = 80, backend="numpy"):
        self.operation = plan
        self.conv_kind = plan.conv_kind
        self.seed = seed
        self.backend = backend

        self.dtype_A = plan._element_a
        self.dtype_B = plan._element_b
        self.dtype_C = plan._element_c
        self.dtype_acc = plan._element_accumulator
        self.layout_A = LayoutType.TensorNHWC
        self.layout_B = LayoutType.TensorNHWC
        self.layout_C = LayoutType.TensorNHWC
        self.layout_D = LayoutType.TensorNHWC

        self.element_compute = DataType.f32

        if self.dtype_A in [cutlass_cppgen.DataType.f16, cutlass_cppgen.DataType.bf16]:
            self.rand_max = 1
        else:
            self.rand_max = 4
        self.activation = plan.activation

    def uniform_init(self, size, dtype):
        tensor = torch.ceil(
            torch.empty(size=size, dtype=torch_type(dtype), device="cuda").uniform_(-self.rand_max - 0.5, self.rand_max - 0.5)
        ).to(memory_format=torch.channels_last)
        return tensor

    def reference(self, ps, A, B, C, alpha, beta, activation):
        if self.conv_kind == ConvKind.Fprop:
            torch_result = alpha * torch.ops.aten.conv2d(
                A,
                B,
                stride=(ps.stride_h, ps.stride_w),
                padding=(ps.pad_h, ps.pad_w),
                dilation=(ps.dilation_h, ps.dilation_w)
            ) + beta * C
        elif self.conv_kind == ConvKind.Dgrad:
            torch_result = alpha * torch.nn.grad.conv2d_input(
                (ps.N, ps.C, ps.H, ps.W),
                B,
                A,
                padding=(ps.pad_h, ps.pad_w),
                stride=(ps.stride_h, ps.stride_w)
            ) + beta * C
        elif self.conv_kind == ConvKind.Wgrad:
            torch_result = alpha * torch.nn.grad.conv2d_weight(
                B,
                (ps.K, ps.C, ps.R, ps.S),
                A,
                padding=(ps.pad_h, ps.pad_w),
                stride=(ps.stride_h, ps.stride_w)
            ) + beta * C
        else:
            raise Exception(f"Conv kind {self.conv_kind} is currently unsupported.")

        if activation == cutlass_cppgen.backend.epilogue.relu:
            torch_result = torch.nn.functional.relu(torch_result)
        elif activation == cutlass_cppgen.backend.epilogue.leaky_relu:
            torch_result = torch.nn.functional.leaky_relu(torch_result, 0.5)
        return torch_result

    def run(self, ps, split_k_mode=SplitKMode.Serial, split_k_slices=1, alpha=1.0, beta=0.0):
        if self.conv_kind == ConvKind.Fprop:
            tensor_A_size = (ps.N, ps.C, ps.H, ps.W)
            tensor_B_size = (ps.K, ps.C, ps.R, ps.S)
            tensor_C_size = (ps.N, ps.K, ps.P, ps.Q)
        elif self.conv_kind == ConvKind.Dgrad:
            tensor_A_size = (ps.N, ps.K, ps.P, ps.Q)
            tensor_B_size = (ps.K, ps.C, ps.R, ps.S)
            tensor_C_size = (ps.N, ps.C, ps.H, ps.W)
        elif self.conv_kind == ConvKind.Wgrad:
            tensor_A_size = (ps.N, ps.K, ps.P, ps.Q)
            tensor_B_size = (ps.N, ps.C, ps.H, ps.W)
            tensor_C_size = (ps.K, ps.C, ps.R, ps.S)
        else:
            raise Exception(f"Conv kind {self.conv_kind} is not supported")

        torch.manual_seed(self.seed)

        tensor_A = self.uniform_init(size=tensor_A_size, dtype=self.dtype_A)
        tensor_B = self.uniform_init(size=tensor_B_size, dtype=self.dtype_B)
        tensor_C = self.uniform_init(size=tensor_C_size, dtype=self.dtype_C)
        tensor_D = torch.zeros_like(tensor_C).to(memory_format=torch.channels_last)
        args = self.operation.run(tensor_A, tensor_B, tensor_C, tensor_D,
            stride=(ps.stride_h, ps.stride_w),
            padding=(ps.pad_h, ps.pad_w),
            dilation=(ps.dilation_h, ps.dilation_w),
            alpha=alpha, beta=beta,
            split_k=(split_k_mode, split_k_slices))

        args.sync()

        tensor_D_ref = self.reference(ps, tensor_A, tensor_B, tensor_C, alpha, beta, self.activation)

        torch.cuda.synchronize()
        passed = torch.allclose(tensor_D, tensor_D_ref, atol=2e-06)

        return passed


def add_test(
    cls,
    cc,
    conv_kind,
    problem_sizes,
    element,
    element_accumulator,
    element_output,
    opclass,
    threadblock_shape,
    warp_count,
    instruction_shape,
    stages,
    iterator_algorithm=None,
    swizzle=None,
    split_k_mode="serial",
    split_k_slices=1,
    activation = "identity"
):
    """Create a test-running function with the given specification"""
    test_name = get_name_conv2d(
        cc, conv_kind, element, element_accumulator,
        element_output, opclass, threadblock_shape, warp_count, instruction_shape, stages,
        iterator_algorithm, swizzle, split_k_mode, split_k_slices, activation)

    def run(self):
        # Create the plan
        plan = cutlass_cppgen.Conv2d(
            kind=conv_kind,
            element=element,
            element_accumulator=element_accumulator,
            element_C=element_output,
            element_D=element_output
        )

        # Set the opclass
        plan.opclass = opclass
        # Set the tile description
        td = {
            "threadblock_shape": threadblock_shape,
            "warp_count": warp_count,
            "stages": stages,
            "instruction_shape": instruction_shape,
        }

        plan.tile_description = td
        # Set iterator algorithm
        if iterator_algorithm is not None:
            plan.iterator_algorithm = iterator_algorithm
        # Set swizzling functor
        if swizzle is not None:
            plan.swizzling_stride = swizzle

        if activation != "identity":
            if activation == "leaky_relu":
                plan.activation = (cutlass_cppgen.epilogue.leaky_relu, 0.5)
            else:
                plan.activation = getattr(cutlass_cppgen.epilogue, activation)

        conv2d_launcher = Conv2dLauncherFrontend(plan, 80, backend="torch")

        for ps in problem_sizes:
            if not validate_problem_size(ps, conv_kind, split_k_slices):
                continue

            self.assertTrue(conv2d_launcher.run(ps, split_k_mode, split_k_slices, 1.0, 2.0))

    setattr(cls, test_name, run)

    return run


def get_conv_problems():
    # 64: minimum channel size
    conv_problems = TestbedConv2dProblemSizes(64).all

    # Insert alignment 4 & 2 tests
    conv_problems += [
        Conv2DProblemSize(
            1, 4, 4, 12,
            8, 3, 3, 12,
            0, 0,
            3, 3,
            1, 1,
            ConvMode.CrossCorrelation,
            1, 1
        ),
        Conv2DProblemSize(
            1, 4, 4, 14,
            8, 3, 3, 14,
            0, 0,
            3, 3,
            1, 1,
            ConvMode.CrossCorrelation,
            1, 1
        ),
        Conv2DProblemSize(
            1, 23, 56, 98,
            128, 3, 3, 98,
            4, 5,
            3, 3,
            1, 1,
            ConvMode.CrossCorrelation,
            1, 1
        ),
    ]

    return conv_problems

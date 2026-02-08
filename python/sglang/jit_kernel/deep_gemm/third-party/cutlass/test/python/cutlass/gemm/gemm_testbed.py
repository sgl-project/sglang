#################################################################################################
#
# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from math import prod
import os
import re
import subprocess

import torch

from cutlass_library import (
    DataType,
    DataTypeSize,
    GemmUniversalMode,
    LayoutType,
    OpcodeClass,
    ShortDataTypeNames,
    SwizzlingFunctor
)

from cutlass_cppgen.backend import compiler
from cutlass_cppgen.backend.gemm_operation import GemmArguments, GemmOperationUniversal
from cutlass_cppgen.backend.reduction_operation import ReductionArguments, ReductionOperation
from cutlass_cppgen.shape import GemmCoord, MatrixCoord
from cutlass_cppgen.utils.datatypes import torch_type


class GemmUniversalLauncher:
    def __init__(
        self,
        operation,
        seed=2080,
        verification=True,
        iterations=500,
        compiler_mode= "nvcc",
        **kwargs,
    ) -> None:
        self.math_operation = operation.tile_description.math_instruction.math_operation
        self.verification = verification

        if compiler_mode == "nvcc":
            compiler.nvcc()
        elif compiler_mode == "nvrtc":
            compiler.nvrtc()
        else:
            raise Exception(f"Unexpected compiler string {compiler_mode}")

        op_list = [operation]
        if operation.arch < 90:
            # Split K via Python is currently only supported for pre-SM90 kernels
            self.reduction_operation: ReductionOperation = ReductionOperation(
                shape=MatrixCoord(4, 32 * operation.C.alignment),
                C=operation.C,
                element_accumulator=operation.tile_description.math_instruction.element_accumulator,
                element_compute=operation.epilogue_functor.element_epilogue,
                epilogue_functor=operation.epilogue_functor,
                count=operation.C.alignment,
            )
            op_list.append(self.reduction_operation)

        compiler.add_module(op_list, bypass_cache=False)

        self.operation = operation

        self.dtype_A = torch_type(operation.A.element if not self.operation.switched else self.operation.B.element)
        self.dtype_B = torch_type(operation.B.element if not self.operation.switched else self.operation.A.element)
        self.dtype_C = torch_type(operation.C.element)
        self.dtype_D = torch_type(operation.epilogue_functor.element_output)

        element_size = min(DataTypeSize[operation.A.element], DataTypeSize[operation.B.element])

        if element_size == 1:
            self.rand_max = 1
            self.rand_min = 0
        elif element_size <= 8:
            self.rand_max = 1
            self.rand_min = -1
        elif element_size == 16:
            self.rand_max = 4
            self.rand_min = -4
        else:
            self.rand_max = 8
            self.rand_min = -8

        self.seed = seed

        self.compute_type = operation.epilogue_functor.element_epilogue
        self.accumulator_type = operation.tile_description.math_instruction.element_accumulator

    def print_problem_size(self, p, mode, batch_count):
        if mode == GemmUniversalMode.Gemm:
            mode = "Gemm"
        elif mode == GemmUniversalMode.Batched:
            mode = "GemmBatched"
        elif mode == GemmUniversalMode.GemmSplitKParallel:
            mode = "GemmSplitKParallel"
        print(f"problem: {p.m}, {p.n}, {p.k}\n batch_count: {batch_count}\n mode: {mode}")

    def uniform_init(self, shape, dtype, layout):
        size = prod(shape)
        if dtype.is_floating_point:
            # Initialize data in FP32 and call convert to the data type we desire.
            # This is a workaround for the following error that occurs when attempting to
            # call uniform_ on a tensor with torch.float8_e4m3fn data:
            # RuntimeError: "check_uniform_bounds" not implemented for 'Float8_e4m3fn'
            data = torch.ceil(
                torch.empty(size=(size,), dtype=torch.float32, device="cuda").uniform_(
                    self.rand_min - 0.5, self.rand_max - 0.5)
                ).to(dtype)
        else:
            # PyTorch does not currently support integer-typed matrix multiplications on GPU.
            # Fall back to CPU for integer type references.
            data = torch.empty(size=(size,), dtype=dtype, device="cpu").random_(self.rand_min, self.rand_max + 1)

        is_fp8 = dtype == getattr(torch, "float8_e4m3fn", -1) or dtype == dtype == getattr(torch, "float8_e5m2", -1)

        if dtype == torch.float64 or dtype == torch.float32 or is_fp8:
            data = data.to("cpu")

        data_ref = data.reshape(shape)

        if layout == LayoutType.RowMajor:
            data_cutlass = data_ref
        else:
            data_cutlass = data_ref.transpose(-1, -2).contiguous()

        data_cutlass = data_cutlass.to("cuda")

        # As of this writing, few operations in PyTorch are supported with FP8 data.
        # Thus, we perform computation in FP32 for FP8 reference checks.
        if is_fp8:
            data_ref = data_ref.to(torch.float32)

        return data_cutlass, data_ref

    def reference(self, problem_size, tensor_A, tensor_B, tensor_C, alpha, beta):
        # If any tensor is on CPU, place all tensors on CPU unless only
        # tensor C is on CPU
        # Handle mixed-input cases by casting to the larger data type and overriding
        # to whatever the data type of the larger type is
        if self.dtype_A != self.dtype_B:
            if DataTypeSize[self.operation.A.element] < DataTypeSize[self.operation.B.element]:
                tensor_A = tensor_A.to(self.dtype_B).to(tensor_B.device)
            else:
                tensor_B = tensor_B.to(self.dtype_A).to(tensor_A.device)

        devices = [x.device.type for x in [tensor_A, tensor_B]]
        if tensor_C is not None:
            devices.append(tensor_C.device.type)

        if "cpu" in devices and devices != ["cuda", "cuda", "cpu"]:
            device = torch.device("cpu")
        else:
            device = tensor_A.device

        tensor_A = tensor_A.to(device)
        tensor_B = tensor_B.to(device)
        if tensor_C is not None:
            tensor_C = tensor_C.to(device)

        dtype = torch_type(self.compute_type)
        alpha_torch = torch.tensor([alpha], device=device).to(dtype)
        beta_torch = torch.tensor([beta], device=device).to(dtype)

        tmp = tensor_A @ tensor_B
        tensor_D_ref = (alpha_torch * tmp)
        if tensor_C is not None:
            tensor_D_ref += (tensor_C * beta_torch)
        return tensor_D_ref.to(self.dtype_D)

    def run(self, mode, problem_size, batch_count=1, split_k_slices=1, alpha=1.0, beta=0.0):
        torch.random.manual_seed(self.seed)

        # Assign an actual batch count in cases where we are not running in batched mode.
        # This is to differentiate between the number of split K slices and the batch count,
        # which are overloaded within the single `batch_count` variable.
        if mode == GemmUniversalMode.Batched:
            true_batch_count = batch_count
        else:
            true_batch_count = 1

        def transpose(layout):
            if layout == LayoutType.RowMajor:
                return LayoutType.ColumnMajor
            else:
                return LayoutType.RowMajor

        tensor_A, tensor_A_ref = self.uniform_init(
            (true_batch_count, problem_size.m, problem_size.k),
            self.dtype_A,
            self.operation.A.layout if not self.operation.switched else transpose(self.operation.B.layout),
        )
        tensor_B, tensor_B_ref = self.uniform_init(
            (true_batch_count, problem_size.k, problem_size.n),
            self.dtype_B,
            self.operation.B.layout if not self.operation.switched else transpose(self.operation.A.layout),
        )
        if self.dtype_C is not None:
            tensor_C, tensor_C_ref = self.uniform_init(
                (true_batch_count, problem_size.m, problem_size.n),
                self.dtype_C,
                self.operation.C.layout if not self.operation.switched else transpose(self.operation.C.layout),
            )
        else:
            tensor_C = None
            tensor_C_ref = None

        tensor_D, _ = self.uniform_init(
            (true_batch_count, problem_size.m, problem_size.n),
            self.dtype_D,
            self.operation.C.layout if not self.operation.switched else transpose(self.operation.C.layout),
        )
        tensor_D = torch.zeros_like(tensor_D)

        if self.compute_type in [DataType.s8, DataType.s32, DataType.u8, DataType.u32]:
            alpha = int(alpha)
            beta = int(beta)

        #
        # Launch kernel
        #

        arguments = GemmArguments(
            operation=self.operation,
            problem_size=problem_size,
            A=tensor_A,
            B=tensor_B,
            C=tensor_C,
            D=tensor_D,
            output_op=self.operation.epilogue_type(alpha, beta),
            gemm_mode=mode,
            split_k_slices=split_k_slices,
            batch=batch_count,
        )

        if mode == GemmUniversalMode.GemmSplitKParallel:
            reduction_arguments = ReductionArguments(
                self.reduction_operation,
                problem_size=[problem_size.m, problem_size.n],
                partitions=split_k_slices,
                workspace=arguments.ptr_D,
                destination=tensor_D,
                source=tensor_C,
                output_op=self.reduction_operation.epilogue_type(alpha, beta),
            )

        self.operation.run(arguments)

        if mode == GemmUniversalMode.GemmSplitKParallel:
            self.reduction_operation.run(reduction_arguments)

        passed = True

        if self.verification:
            if mode == GemmUniversalMode.GemmSplitKParallel:
                reduction_arguments.sync()

                # Free memory allocated by args because we are not
                # calling `arguments.sync()` in this case (which will free memory)
                arguments.free()
            else:
                arguments.sync()
            tensor_D_ref = self.reference(
                problem_size,
                tensor_A_ref,
                tensor_B_ref,
                tensor_C_ref,
                alpha,
                beta,
            )

            tensor_D_ref = tensor_D_ref.to('cuda')

            if self.operation.switched or self.operation.C.layout == LayoutType.ColumnMajor:
                tensor_D = tensor_D.transpose(-1, -2).contiguous()

            passed = tensor_D.equal(tensor_D_ref)

            try:
                assert passed
            except AssertionError:
                self.print_problem_size(problem_size, mode, batch_count)
        del arguments
        if mode == GemmUniversalMode.GemmSplitKParallel:
            del reduction_arguments

        return passed


def test_all_gemm(operation: "GemmOperationUniversal", testcase="universal", compilation_mode="nvcc"):
    passed = True

    minimum_operand_element_size = min(
        DataTypeSize[operation.A.element], DataTypeSize[operation.B.element]
    )
    opcode_class = operation.tile_description.math_instruction.opcode_class

    if opcode_class == OpcodeClass.Simt:
        alignment = 1
    else:
        alignment = 128 // minimum_operand_element_size

    alignment_m = alignment
    alignment_n = alignment
    alignment_k = alignment

    # INT8 alignment constraints
    if opcode_class == OpcodeClass.Simt:
        A_is_s8 = operation.A.element == DataType.s8
        B_is_s8 = operation.B.element == DataType.s8

        if A_is_s8 and operation.A.layout == LayoutType.ColumnMajor:
            alignment_m = 4
        if B_is_s8 == DataType.s8 and operation.A.layout == LayoutType.RowMajor:
            alignment_n = 4
        if A_is_s8 and B_is_s8 and (operation.A.layout == LayoutType.RowMajor or operation.B.layout == LayoutType.ColumnMajor):
            alignment_k = 4

    threadblock_k = operation.tile_description.threadblock_shape[2]

    assert testcase != "interleaved"

    supports_split_k = operation.arch < 90 and not operation.swizzling_functor == SwizzlingFunctor.StreamK

    if testcase == "multistage":
        modes = [GemmUniversalMode.Gemm]
        problem_size_m = [16, 528]
        problem_size_n = [16, 528]
        problem_size_k = [
            threadblock_k,
            threadblock_k * operation.tile_description.stages
            + operation.tile_description.math_instruction.instruction_shape[2],
        ]
        problem_alpha = [1.0]
        problem_beta = [0.0]
        batch_counts = [1]
    else:
        modes = [GemmUniversalMode.Gemm]
        batch_counts = [1, 2, 3, 5, 7]
        if supports_split_k:
            modes.append(GemmUniversalMode.GemmSplitKParallel)

        problem_size_m = [alignment_m, 512 - 3 * alignment_m]
        problem_size_n = [alignment_n, 512 - 2 * alignment_n]
        if operation.tile_description.stages is None:
            stages_for_k_calc = 7
        else:
            stages_for_k_calc = operation.tile_description.stages
        problem_size_k = [
            alignment_k,
            threadblock_k * stages_for_k_calc - alignment_k,
            threadblock_k * stages_for_k_calc * 3 - alignment_k,
        ]
        problem_alpha = [1.0]
        problem_beta = [2.0]

    testbed = GemmUniversalLauncher(operation, compiler_mode=compilation_mode)

    for mode in modes:
        for m in problem_size_m:
            for n in problem_size_n:
                for k in problem_size_k:
                    for batch_count in batch_counts:
                        for alpha in problem_alpha:
                            for beta in problem_beta:
                                # skip very small K problems
                                if testcase == "universal":
                                    if k // batch_count < 2 * threadblock_k:
                                        continue

                                problem_size = GemmCoord(m, n, k)

                                if supports_split_k:
                                    split_k_slices = batch_count
                                else:
                                    split_k_slices = 1

                                overridden_mode = mode
                                if mode == GemmUniversalMode.Gemm and batch_count > 1:
                                    overridden_mode = GemmUniversalMode.Batched

                                passed = testbed.run(
                                    overridden_mode,
                                    problem_size,
                                    batch_count,
                                    split_k_slices,
                                    alpha,
                                    beta,
                                )

                                if not passed:
                                    return False

    return passed

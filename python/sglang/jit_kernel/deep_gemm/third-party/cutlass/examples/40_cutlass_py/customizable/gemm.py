################################################################################
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
################################################################################

import sys
print("This example is deprecated. Please see examples/python for examples of using "
      "the CUTLASS Python interface.")
sys.exit(0)

import numpy as np
import cutlass.backend as pycutlass
from cutlass.backend import *
from cutlass.backend.utils.device import device_cc
import cutlass_bindings
from bfloat16 import bfloat16

import argparse


# parse the arguments
parser = argparse.ArgumentParser(description="Launch CUTLASS GEMM kernels from Python: 'D = alpha * A * B + beta * C'")

# Operation description
# math instruction description
parser.add_argument("-i", "--instruction_shape",
                    default=[1, 1, 1], nargs=3, type=int, 
                    help="This option describes the size of MMA op")
parser.add_argument("-ta", "--element_a", default="float32", type=str,
                    choices=['float64', 'float32', 'float16', 'bfloat16', 'int32', 'int8'], 
                    help='Data type of elements in input tensor A')
parser.add_argument("-tb", "--element_b", default="float32", type=str,
                    choices=['float64', 'float32', 'float16', 'bfloat16', 'int32', 'int8'], 
                    help='Data type of elements in input tensor B')
parser.add_argument("-tc", "--element_c", default="float32", type=str,
                    choices=['float64', 'float32', 'float16', 'bfloat16', 'int32', 'int8'], 
                    help='Data type of elements in input tensor C and output tensor D')
parser.add_argument("-tacc", "--element_acc", default="float32", type=str,
                    choices=['float64', 'float32', 'float16', 'bfloat16', 'int32', 'int8'], 
                    help='Data type of accumulator')
parser.add_argument('-m', "--math", default="multiply_add",
                    type=str, choices=["multiply_add", "multiply_add_fast_bf16", "multiply_add_fast_f32"], help="math instruction")
parser.add_argument('-op', "--opcode", default="Simt", type=str,
                    choices=["Simt", 'TensorOp'], 
                    help="This option describes whether you want to use tensor \
                        cores (TensorOp) or regular SIMT cores (Simt) on GPU SM")
# tile description
parser.add_argument("-b", "--threadblock_shape",
                    default=[128, 128, 8], nargs=3, type=int, 
                    help="This option describes the tile size a thread block with compute")
parser.add_argument("-s", "--stages", default=4,
                    type=int, help="Number of pipelines you want to use")
parser.add_argument("-w", "--warp_count", default=[4, 2, 1], nargs=3, type=int, 
                    help="This option describes the number of warps along M, N, and K of the threadblock")
parser.add_argument("-cc", "--compute_capability", default=80,
                    type=int, help="This option describes CUDA SM architecture number")
# A
parser.add_argument('-la', "--layout_a", default="RowMajor", type=str, choices=[
                    "RowMajor", "ColumnMajor", "RowMajorInterleaved32", "ColumnMajorInterleaved32"], 
                    help="Memory layout of input tensor A")
parser.add_argument('-aa', '--alignment_a', default=1,
                    type=int, help="Memory alignment of input tensor A")
# B
parser.add_argument('-lb', "--layout_b", default="RowMajor", type=str, choices=[
                    "RowMajor", "ColumnMajor", "RowMajorInterleaved32", "ColumnMajorInterleaved32"], 
                    help="Memory layout of input tensor B")
parser.add_argument('-ab', '--alignment_b', default=1,
                    type=int, help="Memory alignment of input tensor B")
# C
parser.add_argument('-lc', "--layout_c", default="RowMajor", type=str, choices=[
                    "RowMajor", "ColumnMajor", "RowMajorInterleaved32", "ColumnMajorInterleaved32"], 
                    help="Memory layout of input tensor C and output tensor D")
parser.add_argument('-ac', '--alignment_c', default=1,
                    type=int, help="Memory alignment of input tensor C and output tensor D")
# epilogue
parser.add_argument("-te", "--element_epilogue", default="float32", type=str,
                    choices=['float64', 'float32', 'float16', 'bfloat16'], help='Epilogue datatype')
parser.add_argument("-ep", "--epilogue_functor", default="LinearCombination",
                    type=str, choices=['LinearCombination', 'FastLinearCombinationClamp', 'LinearCombinationClamp'], 
                    help="This option describes the epilogue part of the kernel")
# swizzling
parser.add_argument("-sw", "--swizzling_functor", default="IdentitySwizzle1", type=str, choices=[
                    "IdentitySwizzle1", "IdentitySwizzle2", "IdentitySwizzle4", "IdentitySwizzle8", "HorizontalSwizzle", "BatchedIdentitySwizzle"],
                    help="This option describes how thread blocks are scheduled on GPU")

# Argument
parser.add_argument("-p", "--problem_size",
                    default=[128, 128, 128], nargs=3, type=int, 
                    help="GEMM problem size M, N, K")
parser.add_argument("-alpha", "--alpha", default=1.0, type=float, 
                    help="Scaling factor of A * B")
parser.add_argument("-beta", "--beta", default=0.0, type=float, 
                    help="Scaling factor of C")
parser.add_argument("-gm", "--gemm_mode", default="Gemm", type=str,
                    choices=["Gemm", "GemmSplitKParallel", "Batched", "Array"], 
                    help="GEMM mode. Gemm is used for non-splitK or serial-splitK. \
                        GemmSplitKParallel is used for parallel splitK")
parser.add_argument('-k', '--split_k_slices', default=1,
                    type=int, help="Number of split-k partitions. (default 1)")
parser.add_argument('-bias', '--bias', action='store_true', help="C is bias vector")
parser.add_argument('-batch', '--batch', default=1, type=int, help="batch size for batched GEMM")

# Activation function
parser.add_argument("-activ", "--activation_function", default="identity",
    choices=["identity", "relu", "leaky_relu", "tanh", "sigmoid", "silu", "hardswish", "gelu"], help="activation function")
parser.add_argument("-activ_arg", "--activation_args", default=[], nargs="+", type=float,
    help="addition arguments for activation")
parser.add_argument('--print_cuda', action="store_true",
                    help="print the underlying CUDA kernel")

try:
    args = parser.parse_args()
except:
    sys.exit(0)

cc = device_cc()
if args.compute_capability != cc:
    raise Exception(("Parameter --compute-capability of {} "
                    "does not match that of the device of {}.").format(args.compute_capability, cc))

pycutlass.get_memory_pool(init_pool_size=2**30, max_pool_size=2**32)
pycutlass.compiler.nvcc()

np.random.seed(0)

element_a = getattr(cutlass_bindings, args.element_a)
element_b = getattr(cutlass_bindings, args.element_b)
element_c = getattr(cutlass_bindings, args.element_c)
element_acc = getattr(cutlass_bindings, args.element_acc)
math_operation = getattr(MathOperation, args.math)
opclass = getattr(cutlass_bindings.OpClass, args.opcode)

math_inst = MathInstruction(
    args.instruction_shape, element_a, element_b,
    element_acc, opclass, math_operation
)

tile_description = TileDescription(
    args.threadblock_shape, args.stages, args.warp_count,
    math_inst
)

layout_a = getattr(cutlass_bindings, args.layout_a)
layout_b = getattr(cutlass_bindings, args.layout_b)
layout_c = getattr(cutlass_bindings, args.layout_c)

A = TensorDescription(
    element_a, layout_a, args.alignment_a
)

B = TensorDescription(
    element_b, layout_b, args.alignment_b
)

C = TensorDescription(
    element_c, layout_c, args.alignment_c
)

element_epilogue = getattr(cutlass_bindings, args.element_epilogue)
if (args.activation_function == "identity" 
    or (args.gemm_mode == "GemmSplitKParallel" and args.split_k_slices > 1)):
    #
    epilogue_functor = getattr(pycutlass, args.epilogue_functor)(
        C.element, C.alignment, math_inst.element_accumulator, element_epilogue)
else:
    epilogue_functor = getattr(pycutlass, "LinearCombinationGeneric")(
        getattr(pycutlass, args.activation_function)(element_epilogue),
        C.element, C.alignment, math_inst.element_accumulator, element_epilogue)

swizzling_functor = getattr(cutlass_bindings, args.swizzling_functor)

operation = GemmOperationUniversal(
    arch=args.compute_capability, tile_description=tile_description,
    A=A, B=B, C=C,
    epilogue_functor=epilogue_functor, swizzling_functor=swizzling_functor
)

if args.print_cuda:
    print(operation.rt_module.emit())

operations = [operation, ]

if args.gemm_mode == "GemmSplitKParallel":
    if (args.activation_function == "identity"):
        epilogue_functor_reduction = getattr(pycutlass, args.epilogue_functor)(
            C.element, C.alignment, math_inst.element_accumulator, element_epilogue)
    else:
        epilogue_functor_reduction = getattr(pycutlass, "LinearCombinationGeneric")(
            getattr(pycutlass, args.activation_function)(element_epilogue),
            C.element, C.alignment, math_inst.element_accumulator, element_epilogue)

    reduction_operation = ReductionOperation(
        shape=cutlass_bindings.MatrixCoord(4, 32 * C.alignment),
        C=C, element_accumulator=element_acc,
        element_compute=element_epilogue, 
        epilogue_functor=epilogue_functor_reduction,
        count=C.alignment
    )
    operations.append(reduction_operation)

pycutlass.compiler.add_module(operations)

# User-provide inputs

problem_size = cutlass_bindings.gemm.GemmCoord(
    args.problem_size[0], args.problem_size[1], args.problem_size[2])

tensor_a_size = args.batch * problem_size.m() * problem_size.k()
if args.element_a != "int8":
    if args.element_a == "bfloat16":
        tensor_A = np.ceil(
            np.random.uniform(low=-8.5, high=7.5, size=(tensor_a_size,))
            ).astype(bfloat16)
    else:
        tensor_A = np.ceil(
            np.random.uniform(low=-8.5, high=7.5, size=(tensor_a_size,))
            ).astype(getattr(np, args.element_a))
else:
    tensor_A = np.random.uniform(
        low=-2, high=2,size=(tensor_a_size,)
        ).astype(getattr(np, args.element_a))

tensor_b_size = args.batch * problem_size.k() * problem_size.n()
if args.element_b != "int8":
    if args.element_b == "bfloat16":
        tensor_B = np.ceil(
            np.random.uniform(low=-8.5, high=7.5, size=(tensor_b_size,))
            ).astype(bfloat16)
    else:
        tensor_B = np.ceil(
            np.random.uniform(low=-8.5, high=7.5, size=(tensor_b_size,))
            ).astype(getattr(np, args.element_b))
else:
    tensor_B = np.random.uniform(
        low=-2, high=2, size=(tensor_b_size,)
        ).astype(getattr(np, args.element_b))

if args.element_c != "int8":
    if args.bias:
        if args.layout_c == "RowMajor":
            tensor_c_size = args.batch * problem_size.n()
        elif args.layout_c == "ColumnMajor":
            tensor_c_size = args.batch * problem_size.m()
        else:
            raise ValueError(args.layout_c)
    else:
        tensor_c_size = args.batch * problem_size.m() * problem_size.n()
    if args.element_c == "bfloat16":
        tensor_C = np.ceil(
            np.random.uniform(low=-8.5, high=7.5, size=(tensor_c_size,))
        ).astype(bfloat16)
    else:
        tensor_C = np.ceil(
            np.random.uniform(low=-8.5, high=7.5, size=(tensor_c_size,))
        ).astype(getattr(np, args.element_c))
else:
    tensor_C = np.random.uniform(
        low=-2, high=2, size=(args.batch * problem_size.m() * problem_size.n(),)
    ).astype(getattr(np, args.element_c))

tensor_D = np.zeros(
    shape=(args.batch * problem_size.m() * problem_size.n(),)
).astype(getattr(np, args.element_c))

output_op = operation.epilogue_type(*([args.alpha, args.beta] + args.activation_args))

arguments = GemmArguments(
    operation=operation, problem_size=problem_size,
    A=tensor_A, B=tensor_B, C=tensor_C, D=tensor_D,
    output_op=output_op,
    gemm_mode=getattr(cutlass_bindings.gemm.Mode, args.gemm_mode),
    split_k_slices=args.split_k_slices, batch=args.batch
)

if args.gemm_mode == "GemmSplitKParallel":
    reduction_arguments = ReductionArguments(
        operation=reduction_operation,
        problem_size=[problem_size.m(), problem_size.n()],
        partitions=args.split_k_slices, workspace=arguments.ptr_D,
        destination=tensor_D, source=tensor_C,
        output_op=reduction_operation.epilogue_type(*([args.alpha, args.beta] + args.activation_args)),
        bias = arguments.bias
    )

operation.run(arguments)

if args.gemm_mode == "GemmSplitKParallel":
    reduction_operation.run(reduction_arguments)
    reduction_arguments.sync()
else:
    arguments.sync()

# run the host reference module
reference = ReferenceModule(A, B, C)
tensor_D_ref = reference.run(
    tensor_A, tensor_B, tensor_C, problem_size, args.alpha, args.beta, args.bias, args.batch)

tensor_D_ref = getattr(pycutlass, args.activation_function).numpy(*([tensor_D_ref,] + args.activation_args))

try:
    assert np.array_equal(tensor_D, tensor_D_ref)
except:
    assert np.allclose(tensor_D, tensor_D_ref, atol=1e-5)
print("Passed.")

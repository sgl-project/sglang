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
import csv

import argparse

# parse the arguments
parser = argparse.ArgumentParser(
    description="Launch CUTLASS GEMM Grouped kernels from Python")

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
                    choices=["Simt", 'TensorOp'], help='This option describes whether you want to use tensor \
                        cores (TensorOp) or regular SIMT cores (Simt) on GPU SM')
# tile description
parser.add_argument("-b", "--threadblock_shape",
                    default=[128, 128, 8], nargs=3, type=int, 
                    help="This option describes the tile size a thread block with compute")
parser.add_argument("-s", "--stages", default=4,
                    type=int, help="Number of pipelines you want to use")
parser.add_argument("-w", "--warp_count", default=[
                    4, 2, 1], nargs=3, type=int, 
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
                    "IdentitySwizzle1", "IdentitySwizzle2", "IdentitySwizzle4", "IdentitySwizzle8", "HorizontalSwizzle"],
                    help="This option describes how thread blocks are scheduled on GPU. \
                         NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels. \
                         This parameter is passed in at present to match the APIs of other kernels. The parameter \
                         is unused within the kernel")
# precompute mode
parser.add_argument("-pm", "--precompute_mode",
                    default="Device", type=str, choices=["Host", "Device"],
                    help="Grouped Gemm Scheduing on device only (Device) or using host precompute (Host)")
# arguments
parser.add_argument("-p", "--problem_size_dir", type=str, default="grouped_gemm_problem_size.csv",
                    help="path to the csv file contains the problem sizes")
parser.add_argument("-alpha", "--alpha", default=1.0, type=float, help="alpha")
parser.add_argument("-beta", "--beta", default=0.0, type=float, help="beta")
parser.add_argument('-bias', '--bias', action='store_true', help="C is bias vector")

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
if args.activation_function == "identity":
    epilogue_functor = getattr(pycutlass, args.epilogue_functor)(
        C.element, C.alignment, math_inst.element_accumulator, element_epilogue)
else:
    epilogue_functor = getattr(pycutlass, "LinearCombinationGeneric")(
        getattr(pycutlass, args.activation_function)(element_epilogue),
        C.element, C.alignment, math_inst.element_accumulator, element_epilogue)
swizzling_functor = getattr(cutlass_bindings, args.swizzling_functor)
precompute_mode = getattr(SchedulerMode, args.precompute_mode)

operation = GemmOperationGrouped(
    arch=args.compute_capability, tile_description=tile_description,
    A=A, B=B, C=C,
    epilogue_functor=epilogue_functor, swizzling_functor=swizzling_functor,
    precompute_mode=precompute_mode
)

if args.print_cuda:
    print(operation.rt_module.emit())

pycutlass.compiler.add_module([operation, ])

reference_module = ReferenceModule(A, B, C)

# get problems
problem_sizes = []
with open(args.problem_size_dir) as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        problem_sizes.append(
            cutlass_bindings.gemm.GemmCoord(int(row[0]), int(row[1]), int(row[2]))
        )

problem_count = len(problem_sizes)

tensor_As = []
tensor_Bs = []
tensor_Cs = []
tensor_Ds = []
problem_sizes_coord = []
tensor_D_refs = []

for problem_size in problem_sizes:
    if args.element_a != "int8":
        if args.element_a == "bfloat16":
            tensor_A = np.ceil(np.random.uniform(low=-8.5, high=7.5, size=(problem_size.m()
                                                                           * problem_size.k(),))).astype(bfloat16)
        else:
            tensor_A = np.ceil(np.random.uniform(low=-8.5, high=7.5, size=(problem_size.m()
                                                                           * problem_size.k(),))).astype(getattr(np, args.element_a))
    else:
        tensor_A = np.random.uniform(low=-2, high=2, size=(problem_size.m()
                                                           * problem_size.k(),)).astype(getattr(np, args.element_a))

    if args.element_b != "int8":
        if args.element_b == "bfloat16":
            tensor_B = np.ceil(np.random.uniform(low=-8.5, high=7.5, size=(problem_size.k()
                                                                           * problem_size.n(),))).astype(bfloat16)
        else:
            tensor_B = np.ceil(np.random.uniform(low=-8.5, high=7.5, size=(problem_size.k()
                                                                           * problem_size.n(),))).astype(getattr(np, args.element_b))
    else:
        tensor_B = np.random.uniform(low=-2, high=2, size=(problem_size.k()
                                                           * problem_size.n(),)).astype(getattr(np, args.element_b))

    if args.element_c != "int8":
        if args.bias:
            if args.layout_c == "RowMajor":
                c_size = problem_size.n()
            elif args.layout_c == "ColumnMajor":
                c_size = problem_size.m()
            else:
                raise ValueError(args.layout_c)
        else:
            c_size = problem_size.m() * problem_size.n()
        if args.element_c == "bfloat16":
            tensor_C = np.ceil(
                np.random.uniform(low=-8.5, high=7.5, size=(c_size,))
            ).astype(bfloat16)
        else:
            tensor_C = np.ceil(
                np.random.uniform(low=-8.5, high=7.5, size=(c_size,))
            ).astype(getattr(np, args.element_c))
    else:
        tensor_C = np.random.uniform(
            low=-2, high=2, size=(problem_size.m() * problem_size.n(),)
        ).astype(getattr(np, args.element_c))
    tensor_D = np.zeros(
        shape=(problem_size.m() * problem_size.n(),)
    ).astype(getattr(np, args.element_c))

    tensor_As.append(tensor_A)
    tensor_Bs.append(tensor_B)
    tensor_Cs.append(tensor_C)
    tensor_Ds.append(tensor_D)
    tensor_D_ref = reference_module.run(
        tensor_A, tensor_B, tensor_C, problem_size, 
        args.alpha, args.beta, args.bias)
    tensor_D_ref = getattr(pycutlass, args.activation_function).numpy(*([tensor_D_ref,] + args.activation_args))
    tensor_D_refs.append(tensor_D_ref)
    problem_sizes_coord.append(problem_size)

arguments = GemmGroupedArguments(
    operation, problem_sizes_coord, tensor_As, tensor_Bs, tensor_Cs, tensor_Ds,
    output_op=operation.epilogue_type(*([args.alpha, args.beta] + args.activation_args))
)

operation.run(arguments)

arguments.sync()

for tensor_d, tensor_d_ref in zip(tensor_Ds, tensor_D_refs):
    try:
        assert np.array_equal(tensor_d, tensor_d_ref)
    except:
        assert np.allclose(tensor_d, tensor_d_ref, rtol=1e-5)

print("Passed.")

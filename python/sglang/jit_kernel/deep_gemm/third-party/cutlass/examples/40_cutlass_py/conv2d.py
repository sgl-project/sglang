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
"""
Basic example of using the CUTLASS Python interface to run a 2d convolution
"""

import sys
print("This example is deprecated. Please see examples/python for examples of using "
      "the CUTLASS Python interface.")
sys.exit(0)

import argparse
import numpy as np
import torch

import cutlass_bindings
import cutlass.backend as pycutlass
from cutlass.backend import *
from cutlass.backend.utils.reference_model import Conv2dReferenceModule
from cutlass.backend.utils.device import device_cc


parser = argparse.ArgumentParser(
    description=("Launch a 2d convolution kernel from Python. "
                 "See https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#convo-intro for notation."))
parser.add_argument("--n", default=1, type=int,  help="N dimension of the convolution")
parser.add_argument("--c", default=64, type=int, help="C dimension of the convolution")
parser.add_argument("--h", default=32, type=int, help="H dimension of the convolution")
parser.add_argument("--w", default=32, type=int, help="W dimension of the convolution")
parser.add_argument("--k", default=32, type=int,  help="N dimension of the convolution")
parser.add_argument("--r", default=3, type=int, help="R dimension of the convolution")
parser.add_argument("--s", default=3, type=int, help="S dimension of the convolution")
parser.add_argument('--print_cuda', action="store_true", help="Print the underlying CUDA kernel")

try:
    args = parser.parse_args()
except:
    sys.exit(0)

# Check that the device is of a sufficient compute capability
cc = device_cc()
assert cc >= 70, "The CUTLASS Python Conv2d example requires compute capability greater than or equal to 70."

alignment = 1

np.random.seed(0)

# Allocate a pool of device memory to be used by the kernel
pycutlass.get_memory_pool(init_pool_size=2**30, max_pool_size=2**32)

# Set the compiler to use to NVCC
pycutlass.compiler.nvcc()

# Set up A, B, C and accumulator
A = TensorDescription(cutlass_bindings.float16, cutlass_bindings.TensorNHWC, alignment)
B = TensorDescription(cutlass_bindings.float16, cutlass_bindings.TensorNHWC, alignment)
C = TensorDescription(cutlass_bindings.float32, cutlass_bindings.TensorNHWC, alignment)
element_acc = cutlass_bindings.float32
element_epilogue = cutlass_bindings.float32

# Select instruction shape based on the Tensor Core instructions supported
# by the device on which we are running
if cc == 70:
    instruction_shape = [8, 8, 4]
elif cc == 75:
    instruction_shape = [16, 8, 8]
else:
    # Use CUTLASS kernels for CC 80 by default (e.g., for cases in which SM86 is used)
    cc = 80
    instruction_shape = [16, 8, 16]

math_inst = MathInstruction(
    instruction_shape,
    A.element, B.element, element_acc,
    cutlass_bindings.OpClass.TensorOp,
    MathOperation.multiply_add
)

tile_description = TileDescription(
    [128, 128, 32],   # Threadblock shape
    2,                # Number of stages
    [2, 2, 1],        # Number of warps within each dimension of the threadblock shape
    math_inst
)

epilogue_functor = pycutlass.LinearCombination(C.element, C.alignment, element_acc, element_epilogue)

operation = Conv2dOperation(
    conv_kind=cutlass_bindings.conv.Operator.fprop,
    iterator_algorithm=cutlass_bindings.conv.IteratorAlgorithm.optimized,
    arch=cc, tile_description=tile_description,
    A=A, B=B, C=C, stride_support=StrideSupport.Unity,
    epilogue_functor=epilogue_functor
)

if args.print_cuda:
    print(operation.rt_module.emit())

operations = [operation, ]

# Compile the operation
pycutlass.compiler.add_module(operations)

# Randomly initialize tensors

problem_size = cutlass_bindings.conv.Conv2dProblemSize(
    cutlass_bindings.Tensor4DCoord(args.n, args.h, args.c, args.w),
    cutlass_bindings.Tensor4DCoord(args.k, args.r, args.s, args.c),
    cutlass_bindings.Tensor4DCoord(0, 0, 0, 0),      # Padding
    cutlass_bindings.MatrixCoord(1, 1),              # Strides
    cutlass_bindings.MatrixCoord(1, 1),              # Dilation
    cutlass_bindings.conv.Mode.cross_correlation, 
    1,                                      # Split k slices
    1                                       # Groups
)

tensor_A_size = cutlass_bindings.conv.implicit_gemm_tensor_a_size(operation.conv_kind, problem_size)
tensor_B_size = cutlass_bindings.conv.implicit_gemm_tensor_b_size(operation.conv_kind, problem_size)
tensor_C_size = cutlass_bindings.conv.implicit_gemm_tensor_c_size(operation.conv_kind, problem_size)

tensor_A = torch.ceil(torch.empty(size=(tensor_A_size,), dtype=torch.float16, device="cuda").uniform_(-8.5, 7.5))
tensor_B = torch.ceil(torch.empty(size=(tensor_B_size,), dtype=torch.float16, device="cuda").uniform_(-8.5, 7.5))
tensor_C = torch.ceil(torch.empty(size=(tensor_C_size,), dtype=torch.float32, device="cuda").uniform_(-8.5, 7.5))
tensor_D = torch.ones(size=(tensor_C_size,), dtype=torch.float32, device="cuda")

alpha = 1.
beta = 0.

arguments = Conv2dArguments(
    operation=operation, problem_size=problem_size,
    A=tensor_A, B=tensor_B, C=tensor_C, D=tensor_D,
    output_op=operation.epilogue_type(alpha, beta)
)

# Run the operation
operation.run(arguments)
arguments.sync()

# Run the host reference module and compare to the CUTLASS result
reference = Conv2dReferenceModule(A, B, C, operation.conv_kind)
tensor_D_ref = reference.run(tensor_A, tensor_B, tensor_C, problem_size, alpha, beta)

try:
    assert torch.equal(tensor_D, tensor_D_ref)
except:
    assert torch.allclose(tensor_D, tensor_D_ref, rtol=1e-2)

print("Passed.")

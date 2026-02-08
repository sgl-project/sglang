# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

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

"""
Demonstrating JIT GEMM Implementation with Static Shape Wrapper

This example illustrates how to invoke a JIT-compiled GEMM implementation through a wrapper function
with static shapes. It showcases the integration between PyTorch and CuTe tensors in a JIT context.

Key features demonstrated:
1. Seamless conversion between PyTorch and CuTe tensors using the JitArgument protocol
2. Integration of static shape GEMM operations within a JIT-compiled wrapper function

Core components:
- BufferWithLayout: Handles memory buffer management with configurable stride ordering
- tensor_op_gemm_wrapper: JIT-compiled entry point that orchestrates the GEMM operation

Usage:

.. code-block:: bash

    python examples/ampere/call_from_jit.py

Default configuration:
- Batch dimension (L): 16
- Matrix dimensions: M=512, N=256, K=128
- Precision: Float16 inputs with Float32 accumulation

Requirements:
- CUDA-capable GPU
- PyTorch with CUDA support
"""

import os
import sys
from typing import Type, Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass.torch import dtype as torch_dtype
from cutlass.cute.runtime import make_ptr


# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tensorop_gemm import TensorOpGemm


class BufferWithLayout:
    def __init__(self, ptr: cute.Pointer, stride_order: tuple[int, int, int]):
        self.ptr = ptr

        # static properties
        self.stride_order = stride_order

    def to_tensor(
        self, shape: tuple[int, int, int], *, loc=None, ip=None
    ) -> cute.Tensor:
        assert len(shape) == len(self.stride_order), (
            f"Shape {shape} and stride_order {self.stride_order} must have the "
            "same rank."
        )
        layout = cute.make_ordered_layout(shape, self.stride_order)
        # permute (l, mn, k) -> (mn, k, l)
        res = cute.make_tensor(self.ptr, cute.select(layout, mode=[1, 2, 0]))
        return res

    # Implement JitArgument Protocol and DynamicExpression Protocol

    def __c_pointers__(self):
        """Get the C pointers for the underlying pointer.

        This method is part of the JitArgument Protocol and returns the C pointers
        from the underlying pointer object.

        This is required for user to define a custom data type which can pass to JIT function.
        When JIT compiled function is called, JIT executor will call this method to get raw pointers
        to underlying data object.

        Following condition must be satisfied:

        len(__c_pointers__()) == len(__get_mlir_types__()) == len(__extract_mlir_values__())

        :return: The C pointers from the underlying pointer object
        :rtype: Any
        """
        return self.ptr.__c_pointers__()

    def __get_mlir_types__(self):
        """Get the MLIR types for the underlying pointer.

        This method is part of the JitArgument Protocol and returns the MLIR types
        used for compiler to generate code. It must match the type of the underlying pointers
        returned by __c_pointers__().

        :return: The MLIR types from the underlying pointer object
        :rtype: Any
        """
        return self.ptr.__get_mlir_types__()

    def __extract_mlir_values__(self):
        """Extract MLIR values from the underlying pointer.

        This method is part of the DynamicExpression Protocol and extracts MLIR values
        from the underlying pointer object.

        It is used by compiler to generate function call in MLIR to another JIT function.
        It must match the types returned by __get_mlir_types__().

        :return: The MLIR values extracted from the underlying pointer object
        :rtype: Any
        """
        return self.ptr.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values):
        """Create a new BufferWithLayout instance from MLIR values.

        This method is part of the JitArgument & DynamicExpression Protocol and creates a new
        BufferWithLayout instance with pointer initialized from the given MLIR values.

        It is used by compiler to generate function body in MLIR called by JIT function.
        It must match the types returned by __c_pointers__() and __get_mlir_types__().
        code generator takes function arguments and reconstructs python object which is legal
        inside function body.

        :param values: MLIR values to initialize the underlying pointer
        :type values: Any
        :return: A new BufferWithLayout instance with pointer initialized from values
        :rtype: BufferWithLayout
        """
        return BufferWithLayout(
            self.ptr.__new_from_mlir_values__(values), self.stride_order
        )


@cute.jit
def tensor_op_gemm_wrapper(
    buffer_a: BufferWithLayout,
    buffer_b: BufferWithLayout,
    buffer_c: BufferWithLayout,
    mnkl: cutlass.Constexpr[tuple[int, int, int, int]],
    acc_dtype: Type[cutlass.Numeric],
    atom_layout_mnk: cutlass.Constexpr[tuple[int, int, int]],
):
    print(f"\n[DSL INFO] Input Parameters:")
    print(f"[DSL INFO]   mnkl: {mnkl}")
    print(f"[DSL INFO]   buffer_a: {buffer_a}")
    print(f"[DSL INFO]   buffer_b: {buffer_b}")
    print(f"[DSL INFO]   buffer_c: {buffer_c}")
    print(f"[DSL INFO]   acc_dtype: {acc_dtype}")
    print(f"[DSL INFO]   atom_layout_mnk: {atom_layout_mnk}")

    mA = buffer_a.to_tensor(cute.select(mnkl, mode=[3, 0, 2]))
    mB = buffer_b.to_tensor(cute.select(mnkl, mode=[3, 1, 2]))
    mC = buffer_c.to_tensor(cute.select(mnkl, mode=[3, 0, 1]))

    print(f"\n[DSL INFO] Created Tensors:")
    print(f"[DSL INFO]   mA = {mA}")
    print(f"[DSL INFO]   mB = {mB}")
    print(f"[DSL INFO]   mC = {mC}")

    tensor_op_gemm = TensorOpGemm(
        buffer_a.ptr.value_type,
        buffer_c.ptr.value_type,
        acc_dtype,
        atom_layout_mnk,
    )
    print(f"\n[DSL INFO] Created TensorOpGemm instance")
    print(f"[DSL INFO]   Input dtype: {buffer_a.ptr.value_type}")
    print(f"[DSL INFO]   Output dtype: {buffer_c.ptr.value_type}")
    print(f"[DSL INFO]   Accumulation dtype: {acc_dtype}")
    print(f"[DSL INFO]   Atom layout: {atom_layout_mnk}")

    # No need to compile inside jit function
    tensor_op_gemm(mA, mB, mC)
    print(f"\n[DSL INFO] Executed TensorOpGemm")


def run_tensor_op_gemm_wrapper(mnkl: Tuple[int, int, int, int]):
    print(f"\nRunning TensorOpGemm test with:")
    print(f"Tensor dimensions: {mnkl}")

    ab_dtype = cutlass.Float16
    c_dtype = cutlass.Float16

    a = torch.randn(
        mnkl[3], mnkl[0], mnkl[2], dtype=torch_dtype(ab_dtype), device="cuda"
    )
    b = torch.randn(
        mnkl[3], mnkl[1], mnkl[2], dtype=torch_dtype(ab_dtype), device="cuda"
    )
    c = torch.randn(
        mnkl[3], mnkl[0], mnkl[1], dtype=torch_dtype(c_dtype), device="cuda"
    )

    print(f"Input tensor shapes:")
    print(f"a: {a.shape}, dtype: {a.dtype}")
    print(f"b: {b.shape}, dtype: {b.dtype}")
    print(f"c: {c.shape}, dtype: {c.dtype}\n")

    buffer_a = BufferWithLayout(
        make_ptr(ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
        (2, 1, 0),
    )
    buffer_b = BufferWithLayout(
        make_ptr(ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
        (2, 1, 0),
    )
    buffer_c = BufferWithLayout(
        make_ptr(c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
        (2, 1, 0),
    )

    tensor_op_gemm_wrapper(
        buffer_a,
        buffer_b,
        buffer_c,
        mnkl,  # pass shape as static value
        # no stride passing
        cutlass.Float32,
        (2, 2, 1),
    )
    torch.cuda.synchronize()

    ref = torch.einsum("lmk,lnk->lmn", a, b)
    torch.testing.assert_close(c, ref, atol=1e-05, rtol=1e-05)
    print(f"\n[DSL INFO] Results verified successfully!")
    print(f"First few elements of result: \n{c[:3, :3, :3]}")


if __name__ == "__main__":
    run_tensor_op_gemm_wrapper((512, 256, 128, 16))

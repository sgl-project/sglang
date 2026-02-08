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

import sys
import os
from typing import Tuple
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr


"""
An Example demonstrating how to call off-the-shelf kernel by-passing dlpack protocol

The example shows how to directly pass pointers from PyTorch tensors to off-the-shelf kernels
written by CuTe DSL with a thin customized wrapper jit function. The jit function will be
compiled with inline without introducing overhead.

To run this example:

.. code-block:: bash

    python examples/ampere/call_bypass_dlpack.py


It's worth to mention that by-passing dlpack protocol can resolve the issue that dlpack doesn't handle shape-1
mode correctly. For example, the following code will fail, because dlpack will convert the shape-1 mode
with stride-1 which propagate alignment incorrectly.

.. code-block:: python

    @cute.kernel
    def fails_kernel(gX: cute.Tensor):
        bidx, _, _ = cute.arch.block_idx()
        mX = gX[None, bidx, None]  # We wish to retain alignment
        # assert mX.iterator.alignment == 16


    @cute.jit
    def fails(gX_: cute.Tensor):
        gX = gX_
        fails_kernel(gX).launch(grid=(1, 1, 1), block=(128, 1, 1))


    gX_torch = torch.rand((128, 1, 128), device="cuda", dtype=torch.bfloat16)
    fails(from_dlpack(gX_torch, assumed_align=16))

"""

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tensorop_gemm import TensorOpGemm


@cute.jit
def tensor_op_gemm_wrapper(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    m: cutlass.Int32,
    n: cutlass.Int32,
    k: cutlass.Int32,
    l: cutlass.Int32,
):
    print(f"\n[DSL INFO] Input Parameters:")
    print(f"[DSL INFO]   mnkl: {(m, n, k, l)}")

    # Assume alignment of shape to call tensorop_gemm example
    m = cute.assume(m, divby=8)
    n = cute.assume(n, divby=8)

    # Torch is row major
    a_layout = cute.make_ordered_layout((m, k, l), order=(0, 1, 2))
    b_layout = cute.make_ordered_layout((n, k, l), order=(0, 1, 2))
    c_layout = cute.make_ordered_layout((m, n, l), order=(1, 0, 2))
    mA = cute.make_tensor(a_ptr, layout=a_layout)
    mB = cute.make_tensor(b_ptr, layout=b_layout)
    mC = cute.make_tensor(c_ptr, layout=c_layout)

    print(f"[DSL INFO]   mA: {mA}")
    print(f"[DSL INFO]   mB: {mB}")
    print(f"[DSL INFO]   mC: {mC}")

    tensor_op_gemm = TensorOpGemm(
        a_ptr.value_type, c_ptr.value_type, cutlass.Float32, (2, 2, 1)
    )
    print(f"\n[DSL INFO] Created TensorOpGemm instance")
    print(f"[DSL INFO]   Input dtype: {a_ptr.value_type}")
    print(f"[DSL INFO]   Output dtype: {c_ptr.value_type}")
    print(f"[DSL INFO]   Accumulation dtype: {cutlass.Float32}")
    print(f"[DSL INFO]   Atom layout: {(2, 2, 1)}")

    # No need to compile inside jit function
    tensor_op_gemm(mA, mB, mC)
    print(f"\n[DSL INFO] Executed TensorOpGemm")


def run_tensor_op_gemm_wrapper(mnkl: Tuple[int, int, int, int]):
    print(f"\nRunning TensorOpGemm test with:")
    print(f"Tensor dimensions: {mnkl}")

    # (M,K,L)
    a = torch.randn(
        mnkl[3], mnkl[2], mnkl[0], dtype=torch.float16, device="cuda"
    ).permute(2, 1, 0)
    # (N,K,L)
    b = torch.randn(
        mnkl[3], mnkl[2], mnkl[1], dtype=torch.float16, device="cuda"
    ).permute(2, 1, 0)
    # (N,M,L)
    c = torch.randn(
        mnkl[3], mnkl[0], mnkl[1], dtype=torch.float16, device="cuda"
    ).permute(1, 2, 0)

    print(f"Input tensor shapes:")
    print(f"a: {a.shape}, dtype: {a.dtype}")
    print(f"b: {b.shape}, dtype: {b.dtype}")
    print(f"c: {c.shape}, dtype: {c.dtype}\n")

    a_ptr = make_ptr(
        cutlass.Float16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    b_ptr = make_ptr(
        cutlass.Float16, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    c_ptr = make_ptr(
        cutlass.Float16, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    tensor_op_gemm_wrapper(a_ptr, b_ptr, c_ptr, *mnkl)
    torch.cuda.synchronize()

    ref = torch.einsum("mkl,nkl->mnl", a, b)
    torch.testing.assert_close(c, ref, atol=1e-05, rtol=1e-05)
    print(f"\n[DSL INFO] Results verified successfully!")
    print(f"First few elements of result: \n{c[:3, :3, :3]}")


if __name__ == "__main__":
    run_tensor_op_gemm_wrapper((512, 256, 128, 16))

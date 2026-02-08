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

import cutlass.cute as cute
import cutlass
import torch
import numpy as np
from cutlass.cute.runtime import from_dlpack

"""
A Shared Memory Allocator Example on NVIDIA Ampere architecture using CuTe DSL.

This example demonstrates how to allocate and manage shared memory in JIT kernels by using the SmemAllocator in CuTe DSL.
It shows various ways to allocate different data structures in shared memory:

1. Struct allocation with natural and strict alignment
2. Raw memory block allocation with custom alignment
3. Array allocation with automatic alignment
4. Tensor allocation with layout specification

The example includes:
- Shared storage struct with mixed alignment requirements
- Memory allocation patterns for different data types
- Tensor operations on allocated memory

To run this example:

.. code-block:: bash

    python examples/ampere/smem_allocator.py

The example will allocate shared memory, perform tensor operations, and verify the results.
"""


@cute.struct
class complex:
    real: cutlass.Float32
    imag: cutlass.Float32


# SharedStorage size is 512, alignment is 128
@cute.struct
class SharedStorage:
    # struct elements with natural alignment
    a: cute.struct.MemRange[cutlass.Float32, 32]  # array
    b: cutlass.Int64  # saclar
    c: complex  # nested struct
    # struct elements with strict alignment
    x: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 32],
        128,
    ]
    y: cute.struct.Align[cutlass.Int32, 8]
    z: cute.struct.Align[complex, 16]


@cute.kernel
def kernel(
    const_a: cutlass.Constexpr,
    dst_a: cute.Tensor,
    const_b: cutlass.Constexpr,
    dst_b: cute.Tensor,
    const_c: cutlass.Constexpr,
    dst_c: cute.Tensor,
):
    # Note: SMEM_SIZE bytes (specified in kernel().launch(smem=...)) can be reserved for developer to utilize
    # Note: alignment of inital allocator base ptr is 1024
    allocator = cutlass.utils.SmemAllocator()
    # base ptr of allocator points at: SMEM_ADDR_START (the starting address of available shared memory)

    # -- Allocate a struct --
    # Note: when specified alignment, max(alignment, alignof(struct)) will be applied
    # reserves the section of struct in smem, elements in the struct can be accessed by ptr
    struct_in_smem = allocator.allocate(SharedStorage)
    # base ptr of allocator now points at: SMEM_ADDR_AFTER_STRUCT = SMEM_ADDR_START + aligned_size(struct)

    # -- Allocate a block of memory --
    # reserves a section of 64 bytes in smem, align to 128 bytes, returns the section base ptr
    section_in_smem = allocator.allocate(64, byte_alignment=128)
    # base ptr of allocator now points at: SMEM_ADDR_AFTER_SECTION = SMEM_ADDR_AFTER_STRUCT + aligned_size(section)

    # -- Allocate an array --
    # reserves an int64 array of size 14 in smem, returns the array base ptr
    array_in_smem = allocator.allocate_array(element_type=cutlass.Int64, num_elems=14)
    # base ptr of allocator now points at: SMEM_ADDR_AFTER_ARRAY = SMEM_ADDR_AFTER_SECTION + aligned_size(array)

    # -- Allocate a tensor --
    # Note: use cute.ComposedLayout or cute.Layout to specify layout of tensor
    # Note: iterator swizzle with swizzle layout is currently not supported
    layout = cute.make_layout((16, 2))
    tensor_in_smem = allocator.allocate_tensor(
        element_type=cutlass.Float32, layout=layout, byte_alignment=32, swizzle=None
    )
    # base ptr of allocator now points at: SMEM_ADDR_AFTER_TENSOR = SMEM_ADDR_AFTER_ARRAY + aligned_size(tensor)

    # ptr<f16, smem, align<1024>>
    # ptr<i64, smem, align<128>>
    # ptr<f32, smem, align<8>>
    print(struct_in_smem.a.data_ptr())
    print(struct_in_smem.b)
    print(struct_in_smem.c.real)
    # ptr<i8, smem, align<512>>
    print(section_in_smem)
    # ptr<i64, smem, align<64>>
    print(array_in_smem)
    # tensor<ptr<f16, smem, align<32>> o (16,4):(1,16)>
    print(tensor_in_smem)

    # fill MemRange tensor in struct and copy to dst
    a_tensor = struct_in_smem.a.get_tensor(cute.make_layout((8, 4)))
    a_tensor.fill(const_a)
    cute.printf("cute.struct.MemRange: {}", a_tensor)
    dst_a.store(a_tensor.load())

    # convert block of smem to fill tensor and copy to dst
    layout = cute.make_layout((8, 2))
    sec_ptr = cute.recast_ptr(section_in_smem, dtype=cutlass.Float32)
    sec_tensor = cute.make_tensor(sec_ptr, layout)
    sec_tensor.fill(const_b)
    cute.printf("block of memory: {}", sec_tensor)
    dst_b.store(sec_tensor.load())

    # fill allocated tensor in smem and copy to dst
    tensor_in_smem.fill(const_c)
    cute.printf("tensor in smem: {}", tensor_in_smem)
    dst_c.store(tensor_in_smem.load())


@cute.jit
def run_allocation_kernel(
    const_a: cutlass.Constexpr,
    dst_a: cute.Tensor,
    const_b: cutlass.Constexpr,
    dst_b: cute.Tensor,
    const_c: cutlass.Constexpr,
    dst_c: cute.Tensor,
):
    # additional size for the example, 64(section) + 112(array) + 128(tensor) < 384
    addtional_bytes = 384
    # Note: launch shared memory size is: SMEM_SIZE = 512 + 384 = 896 bytes
    kernel(const_a, dst_a, const_b, dst_b, const_c, dst_c).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
        smem=SharedStorage.size_in_bytes() + addtional_bytes,
    )


def veify_allocation_kernel(const_a, const_b, const_c):
    dst_a = torch.zeros((8, 4), dtype=torch.float32, device="cuda")
    dst_b = torch.zeros((8, 2), dtype=torch.float32, device="cuda")
    dst_c = torch.zeros((16, 2), dtype=torch.float32, device="cuda")

    run_allocation_kernel(
        const_a,
        from_dlpack(dst_a),
        const_b,
        from_dlpack(dst_b),
        const_c,
        from_dlpack(dst_c),
    )

    np.testing.assert_equal(const_a, dst_a.detach().cpu().numpy()[0])
    np.testing.assert_equal(const_b, dst_b.detach().cpu().numpy()[0])
    np.testing.assert_equal(const_c, dst_c.detach().cpu().numpy()[0])


if __name__ == "__main__":
    # prepare cuda context
    cutlass.cuda.initialize_cuda_context()
    # An example for shared memory allocation
    const_a = 0.5
    const_b = 1.0
    const_c = 2.0
    veify_allocation_kernel(const_a, const_b, const_c)

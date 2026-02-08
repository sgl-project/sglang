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

"""
Example of automatic shared memory size computation for configuring kernel launch

This example demonstrates how to let the DSL automatically set shared memory
  size for a kernel launch rather explicitly configuring it at launch time,
  provided that developers are using `SmemAllocator` for all allocations.

Usage:
    python dynamic_smem_size.py         # Show auto inference
"""


@cute.struct
class SharedData:
    """A struct to demonstrate shared memory allocation."""

    values: cute.struct.MemRange[cutlass.Float32, 64]  # 256 bytes
    counter: cutlass.Int32  # 4 bytes
    flag: cutlass.Int8  # 1 byte


@cute.kernel
def kernel():
    """
    Example kernel that allocates shared memory.
    The total allocation will be automatically calculated when smem=None.
    """
    allocator = cutlass.utils.SmemAllocator()

    # Allocate various types of shared memory
    shared_data = allocator.allocate(SharedData)
    raw_buffer = allocator.allocate(512, byte_alignment=64)
    int_array = allocator.allocate_array(element_type=cutlass.Int32, num_elems=128)
    tensor_smem = allocator.allocate_tensor(
        element_type=cutlass.Float16,
        layout=cute.make_layout((32, 16)),
        byte_alignment=16,
        swizzle=None,
    )
    return


@cute.kernel
def kernel_no_smem():
    """
    Example kernel that does not allocates shared memory.
    The total allocation will be automatically calculated as 0 when smem=None.
    """
    tidx, _, _ = cute.arch.block_idx()
    if tidx == 0:
        cute.printf("Hello world")
    return


if __name__ == "__main__":
    # Initialize CUDA context
    cutlass.cuda.initialize_cuda_context()

    print("Launching kernel with auto smem size. (launch config `smem=None`)")

    # Compile the example
    @cute.jit
    def launch_kernel1():
        k = kernel()
        k.launch(
            grid=(1, 1, 1),
            block=(1, 1, 1),
        )
        print(f"Kernel recorded internal smem usage: {k.smem_usage()}")

    @cute.jit
    def launch_kernel2():
        k = kernel_no_smem()
        k.launch(
            grid=(1, 1, 1),
            block=(1, 1, 1),
        )
        print(f"Kernel recorded internal smem usage: {k.smem_usage()}")

    cute.compile(launch_kernel1)
    cute.compile(launch_kernel2)

    print("PASS")

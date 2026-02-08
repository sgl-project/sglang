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


import argparse
import time
from typing import Type

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

"""
An Elementwise Addition Example using CuTe DSL.

This example kernel copies data from global memory to register memory (rmem), performs the elementwise
addition operation, and stores the result back to global memory.

Primary goals of this example are to demonstrate how basic global memory copies can be expressed in
CuTe DSL and illustrate canonical partitioning patterns in CuTe. It also implements canonical
predication for tensors whose shape is not multiple of tile size to guard OOB reads.

Thread-value (or TV) layouts are central to canonical partitioning patterns in CuTe. They provide a
mapping from thread and a thread's value to the set of coordinates within a tile that we have sliced
out from a data tensor.

The input tensors are row-major layout, that leading dimension is the right most dimension. In order
to efficiently copy data from global memory, we must map threads contiguously on row dimension.

Thread ID mapping to 2D coordinates with layout `(4,32):(32,1)`:

    +----+----+----+----+-----+----+
    |    | 0  | 1  | 2  | ... | 31 |
    +----+----+----+----+-----+----+
    | 0  | T0 | T1 | T2 | ... | T31|
    +----+----+----+----+-----+----+
    | 1  |T32 |T33 |T34 | ... |T63 |
    +----+----+----+----+-----+----+
    | 2  |T64 |T65 |T66 | ... |T95 |
    +----+----+----+----+-----+----+
    | 3  |T96 |T97 |T98 | ... |T127|
    +----+----+----+----+-----+----+

As Ampere GPU supports a maximum of 128bit per load/store instruction and each element is 32bit, we
can load 4 elements per instruction. Having additional contiguous values allows for vectorization
across threads (coalesced accesses) and is required for saturating the memory bandwidth.

We use `(4,4):(4,1)` as the val layout in this example. Notice that the major mode is the same as
the major mode of the input tensor - without which vectorization would not be possible.

If you already know the TV layout you want to use for your tiled copy, CuTe DSL provides utility
`cute.make_layout_tv` to build the tiled copy type around it and the atom of your choice.

.. code-block:: python

    thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((4, 4), stride=(4, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    # Tile input tensor to thread blocks: ((TileM,TileN),(RestM,RestN))
    gA = cute.zipped_divide(mA, tiler_mn)

Then we can build tiled copy for input and output tensors with `cute.make_tiled_copy_tv` utility, which
infers the tiler and tv layout for the tiled copy automatically, where `tiler` is the tile size per thread
block and `tv_layout` is the TV layout which maps thread index and inter-thread index of data array per
thread to logical coordinates of elements in input and output tensors.

.. code-block:: python

    blkA = gA[((None, None), bidx)]  # (TileM,TileN)

    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)

    # get slice of tiled_copy_A for current thread
    thr_copy_A = tiled_copy_A.get_slice(tidx)

    # partition per thread block tensor as source of tiled copy
    thrA = thr_copy_A.partition_S(blkA)

    # allocate fragment for gmem->rmem
    frgA = cute.make_fragment_like(thrA)

    # copy data from global memory to register memory
    cute.copy(copy_atom_load, thrA, frgA)


To run this example:

.. code-block:: bash

    python examples/ampere/elementwise_add.py --M 3 --N 12
    python examples/ampere/elementwise_add.py --M 1024 --N 512
    python examples/ampere/elementwise_add.py --M 1024 --N 1024 --benchmark --warmup_iterations 2 --iterations 1000

To collect performance with NCU profiler:

.. code-block:: bash

    # Don't iterate too many times when profiling with ncu
    ncu python examples/ampere/elementwise_add.py --M 2048 --N 2048 --benchmark --iterations 10 --skip_ref_check
"""


@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,  # coordinate tensor
    shape: cute.Shape,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # slice for CTAs
    # logical id -> address
    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]  # (TileM,TileN)
    blkB = gB[blk_coord]  # (TileM,TileN)
    blkC = gC[blk_coord]  # (TileM,TileN)
    blkCrd = cC[blk_coord]  # (TileM, TileN)

    # Note: these prints only run at compile/jit time
    print(f"[DSL INFO] Sliced Tensors per thread block:")
    print(f"[DSL INFO]   blkA = {blkA.type}")
    print(f"[DSL INFO]   blkB = {blkB.type}")
    print(f"[DSL INFO]   blkC = {blkC.type}")
    print(f"[DSL INFO]   blkCrd = {blkCrd.type}")

    # # declare the atoms which will be used later for memory copy
    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)

    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_B = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_C = cute.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    thr_copy_C = tiled_copy_C.get_slice(tidx)

    thrA = thr_copy_A.partition_S(blkA)
    thrB = thr_copy_B.partition_S(blkB)
    thrC = thr_copy_C.partition_S(blkC)

    # allocate fragments for gmem->rmem
    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)

    thrCrd = thr_copy_C.partition_S(blkCrd)
    frgPred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)

    print(f"[DSL INFO] Sliced Tensors per thread:")
    print(f"[DSL INFO]   thrA = {thrA.type}")
    print(f"[DSL INFO]   thrB = {thrB.type}")
    print(f"[DSL INFO]   thrC = {thrC.type}")
    print(f"[DSL INFO]   thrCrd = {thrCrd.type}")

    for i in range(0, cute.size(frgPred), 1):
        val = cute.elem_less(thrCrd[i], shape)
        frgPred[i] = val

    # Print per thread predicate mask
    # if tidx == 0 and bidx == 0:
    #     cute.printf("block_dim = {}", cute.arch.grid_dim())
    #     cute.printf("shape = {}", shape)
    #     cute.print_tensor(thrA)
    #     cute.print_tensor(thrB)
    #     cute.print_tensor(frgPred)

    ##########################################################
    # Move data to reg address space
    ##########################################################

    cute.copy(copy_atom_load, thrA, frgA, pred=frgPred)
    cute.copy(copy_atom_load, thrB, frgB, pred=frgPred)

    # if tidx == 0 and bidx == 0:
    #     cute.print_tensor(frgA)
    #     cute.print_tensor(frgB)

    # Load data before use. The compiler will optimize the copy and load
    # operations to convert some memory ld/st into register uses.
    result = frgA.load() + frgB.load()

    # Save the results back to registers. Here we reuse b's registers.
    frgC.store(result)

    # Copy the results back to c
    cute.copy(copy_atom_store, frgC, thrC, pred=frgPred)


@cute.jit
def elementwise_add(mA, mB, mC, copy_bits: cutlass.Constexpr = 128):
    dtype = mA.element_type
    vector_size = copy_bits // dtype.width

    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    print(f"[DSL INFO] Input Tensors:")
    print(f"[DSL INFO]   mA = {mA.type}")
    print(f"[DSL INFO]   mB = {mB.type}")

    print(f"[DSL INFO] Tiling Parameters:")
    print(f"[DSL INFO]   tiler_mn = {tiler_mn} per thread block")
    print(f"[DSL INFO]   tv_layout = {tv_layout}")

    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
    gB = cute.zipped_divide(mB, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
    gC = cute.zipped_divide(mC, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
    print(f"[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA.type}")
    print(f"[DSL INFO]   gB = {gB.type}")
    print(f"[DSL INFO]   gC = {gC.type}")

    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    print(f"[DSL INFO]   coord tensor = {cC.type}")

    elementwise_add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def run_elementwise_add(
    M,
    N,
    dtype: Type[cutlass.Numeric],
    is_a_dynamic_layout=False,
    is_b_dynamic_layout=False,
    is_result_dynamic_layout=False,
    skip_ref_check=False,
    benchmark=True,
    warmup_iterations=2,
    iterations=200,
):
    print(f"\nRunning Elementwise Add test with:")
    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input and Output Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)
    if dtype.is_integer:
        a = torch.randint(0, 10, (M, N), device=torch.device("cuda"), dtype=torch_dtype)
        b = torch.randint(0, 10, (M, N), device=torch.device("cuda"), dtype=torch_dtype)
    else:
        a = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)
        b = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)

    c = torch.zeros_like(a)

    print(f"Input tensor shapes:")
    print(f"a: {a.shape}, dtype: {a.dtype}")
    print(f"b: {b.shape}, dtype: {b.dtype}")
    print(f"c: {c.shape}, dtype: {c.dtype}\n")

    if not is_a_dynamic_layout:
        a_tensor = from_dlpack(a).mark_layout_dynamic()
    else:
        a_tensor = a

    if not is_b_dynamic_layout:
        b_tensor = from_dlpack(b).mark_layout_dynamic()
    else:
        b_tensor = b

    if not is_result_dynamic_layout:
        c_tensor = from_dlpack(c).mark_layout_dynamic()
    else:
        c_tensor = c

    print("Compiling kernel with cute.compile ...")
    start_time = time.time()
    compiled_func = cute.compile(elementwise_add, a_tensor, b_tensor, c_tensor)
    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time:.4f} seconds")

    print("Executing vector add kernel...")

    # Get current CUstream from torch
    current_stream = cutlass_torch.current_stream()

    if not skip_ref_check:
        compiled_func(a_tensor, b_tensor, c_tensor)
        print("Verifying results...")
        torch.testing.assert_close(a + b, c)
        print("Results verified successfully!")

    if not benchmark:
        return

    def generate_tensors():
        if dtype.is_integer:
            a = torch.randint(
                0, 10, (M, N), device=torch.device("cuda"), dtype=torch_dtype
            )
            b = torch.randint(
                0, 10, (M, N), device=torch.device("cuda"), dtype=torch_dtype
            )
        else:
            a = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)
            b = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)

        c = torch.zeros_like(a)

        if not is_a_dynamic_layout:
            a_tensor = from_dlpack(a).mark_layout_dynamic()
        else:
            a_tensor = a

        if not is_b_dynamic_layout:
            b_tensor = from_dlpack(b).mark_layout_dynamic()
        else:
            b_tensor = b

        if not is_result_dynamic_layout:
            c_tensor = from_dlpack(c).mark_layout_dynamic()
        else:
            c_tensor = c

        return testing.JitArguments(a_tensor, b_tensor, c_tensor)

    avg_time_us = testing.benchmark(
        compiled_func,
        workspace_generator=generate_tensors,
        workspace_count=10,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    # Print execution results
    print(f"Kernel execution time: {avg_time_us / 1e3:.4f} ms")
    print(
        f"Achieved memory throughput: {(3 * a.numel() * dtype.width // 8) / (avg_time_us / 1e6) / 1e9:.2f} GB/s"
    )
    print(f"First few elements of result: \n{c[:3, :3]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of elementwise add to demonstrate the numpy/pytorch as input for kernels"
    )
    parser.add_argument("--M", default=1024, type=int)
    parser.add_argument("--N", default=1024, type=int)
    parser.add_argument("--warmup_iterations", default=2, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(f"Ampere GPU is required to run this example!")

    run_elementwise_add(
        args.M,
        args.N,
        dtype=cutlass.Float32,
        is_a_dynamic_layout=True,
        is_b_dynamic_layout=True,
        is_result_dynamic_layout=True,
        skip_ref_check=args.skip_ref_check,
        benchmark=args.benchmark,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )
    print("\nPASS")

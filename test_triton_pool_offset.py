#!/usr/bin/env python3
"""Test: Pool offset with Triton kernels (closer to SGLang's actual use case).

Validates that the single-offset property holds for Triton kernel graphs
which are the primary kernel type used in SGLang's decode path.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import struct
import ctypes
import torch
from cuda.bindings import driver as cu

import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x + 1.0
    tl.store(y_ptr + offsets, y, mask=mask)


def read_dev_ptrs(kernel_params_ptr, max_params=20):
    """Read device pointer params safely."""
    ptr_size = ctypes.sizeof(ctypes.c_void_p)
    ptrs = []
    try:
        mem_fd = open("/proc/self/mem", "rb")
    except OSError:
        return ptrs
    try:
        for i in range(max_params):
            try:
                mem_fd.seek(kernel_params_ptr + i * ptr_size)
                ptr_bytes = mem_fd.read(ptr_size)
                if len(ptr_bytes) < ptr_size:
                    break
                param_ptr = struct.unpack('<Q', ptr_bytes)[0]
                if param_ptr == 0:
                    break
                mem_fd.seek(param_ptr)
                val_bytes = mem_fd.read(8)
                if len(val_bytes) < 8:
                    break
                value = struct.unpack('<Q', val_bytes)[0]
                try:
                    attr = cu.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE
                    r = cu.cuPointerGetAttribute(attr, value)
                    if r[0] == cu.CUresult.CUDA_SUCCESS and r[1] in (1, 2, 3):
                        ptrs.append(value)
                except Exception:
                    break
            except (OSError, OverflowError, ValueError):
                break
    finally:
        mem_fd.close()
    return ptrs


def get_graph_dev_ptrs_and_bases(graph):
    """Get all device pointers and their allocation bases from kernel nodes."""
    gh = graph.raw_cuda_graph()
    cg = cu.CUgraph(gh)
    result = cu.cuGraphGetNodes(cg, 0)
    num_nodes = result[2] if len(result) > 2 else 0
    if num_nodes == 0:
        nodes_list = result[1] if len(result) > 1 else []
        node_array = [n for n in nodes_list if n is not None]
    else:
        result = cu.cuGraphGetNodes(cg, num_nodes)
        nodes_list = result[1]
        node_array = [n for n in nodes_list if n is not None]

    all_ptrs = {}  # ptr -> (base, size)
    for node in node_array:
        t = cu.cuGraphNodeGetType(node)
        if int(t[1]) == 0:  # KERNEL
            p = cu.cuGraphKernelNodeGetParams(node)
            params = p[1]
            kp = params.kernelParams
            if kp is not None:
                for ptr in read_dev_ptrs(kp):
                    if ptr not in all_ptrs:
                        try:
                            ar = cu.cuMemGetAddressRange(ptr)
                            if ar[0] == cu.CUresult.CUDA_SUCCESS:
                                all_ptrs[ptr] = (int(ar[1]), int(ar[2]))
                            else:
                                all_ptrs[ptr] = (0, 0)
                        except Exception:
                            all_ptrs[ptr] = (0, 0)
        elif int(t[1]) == 1:  # MEMCPY
            p = cu.cuGraphMemcpyNodeGetParams(node)
            params = p[1]
            src = int(params.srcDevice)
            dst = int(params.dstDevice)
            for ptr in [src, dst]:
                if ptr not in all_ptrs:
                    try:
                        ar = cu.cuMemGetAddressRange(ptr)
                        if ar[0] == cu.CUresult.CUDA_SUCCESS:
                            all_ptrs[ptr] = (int(ar[1]), int(ar[2]))
                        else:
                            all_ptrs[ptr] = (0, 0)
                    except Exception:
                        all_ptrs[ptr] = (0, 0)
    return all_ptrs


def test_triton_pool_offset():
    """Test pool offset with Triton kernel graphs."""
    print("=" * 70)
    print("Test: Pool offset with Triton kernel")
    print("=" * 70)

    n = 1024
    x = torch.randn(n, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)

    # Warmup (triggers Triton JIT compilation)
    for _ in range(3):
        add_kernel[(n + 256 - 1) // 256,](x, y, n, BLOCK_SIZE=256)
    torch.cuda.synchronize()

    x_ptr = x.data_ptr()
    y_ptr = y.data_ptr()

    # Capture with Pool A
    pool_a = torch.cuda.graph_pool_handle()
    g_a = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g_a, pool=pool_a):
        add_kernel[(n + 256 - 1) // 256,](x, y, n, BLOCK_SIZE=256)

    ptrs_a = get_graph_dev_ptrs_and_bases(g_a)
    intermediate_a = {p: info for p, info in ptrs_a.items()
                      if p != x_ptr and p != y_ptr and info[0] != 0}
    pool_base_a = min(info[0] for info in intermediate_a.values()) if intermediate_a else 0

    print(f"\nPool A:")
    print(f"  Pool base: 0x{pool_base_a:x}")
    print(f"  Total pointers: {len(ptrs_a)}, Intermediates: {len(intermediate_a)}")

    # Capture with Pool B
    pool_b = torch.cuda.graph_pool_handle()
    g_b = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g_b, pool=pool_b):
        add_kernel[(n + 256 - 1) // 256,](x, y, n, BLOCK_SIZE=256)

    ptrs_b = get_graph_dev_ptrs_and_bases(g_b)
    intermediate_b = {p: info for p, info in ptrs_b.items()
                      if p != x_ptr and p != y_ptr and info[0] != 0}
    pool_base_b = min(info[0] for info in intermediate_b.values()) if intermediate_b else 0

    print(f"\nPool B:")
    print(f"  Pool base: 0x{pool_base_b:x}")
    print(f"  Total pointers: {len(ptrs_b)}, Intermediates: {len(intermediate_b)}")

    # Verify single-offset property
    a_sorted = sorted(ptrs_a.keys())
    b_sorted = sorted(ptrs_b.keys())

    if len(a_sorted) != len(b_sorted):
        print(f"\n  FAIL: Different number of pointers: {len(a_sorted)} vs {len(b_sorted)}")
        return False

    offsets = []
    for i in range(len(a_sorted)):
        a_p = a_sorted[i]
        b_p = b_sorted[i]
        is_known = (a_p == x_ptr or a_p == y_ptr)
        if is_known:
            if a_p != b_p:
                print(f"  FAIL: Known buffer address changed: 0x{a_p:x} -> 0x{b_p:x}")
                return False
        else:
            offsets.append(b_p - a_p)

    if offsets:
        unique_offsets = set(offsets)
        pool_offset = offsets[0]
        pool_base_offset = pool_base_b - pool_base_a
        print(f"\n  Intermediate pointer offsets: {[hex(o) for o in offsets]}")
        if len(unique_offsets) == 1:
            print(f"  PASS: All offsets identical! Offset = 0x{pool_offset:x}")
            if pool_offset != pool_base_offset:
                print(f"  WARNING: Pointer offset 0x{pool_offset:x} != pool base offset 0x{pool_base_offset:x}")
        else:
            print(f"  FAIL: Offsets are NOT identical: {unique_offsets}")
            return False

    # Verify correctness
    x_test = torch.randn(n, device='cuda', dtype=torch.float32)
    expected = x_test + 1.0

    x.copy_(x_test)
    g_b.replay()
    y_result = y.clone()

    match = torch.allclose(y_result, expected, rtol=1e-5, atol=1e-5)
    print(f"\n  Graph B output correct: {match}")

    del g_a, g_b
    torch.cuda.synchronize()
    return match


def test_triton_multiple_captures_same_pool():
    """Test that multiple captures with the same pool produce identical addresses."""
    print("\n" + "=" * 70)
    print("Test: Multiple Triton captures with same pool")
    print("=" * 70)

    n = 1024
    x = torch.randn(n, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)

    # Warmup
    for _ in range(3):
        add_kernel[(n + 256 - 1) // 256,](x, y, n, BLOCK_SIZE=256)
    torch.cuda.synchronize()

    # Capture with same pool
    pool = torch.cuda.graph_pool_handle()

    g1 = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g1, pool=pool):
        add_kernel[(n + 256 - 1) // 256,](x, y, n, BLOCK_SIZE=256)

    g1.replay()
    torch.cuda.synchronize()

    g2 = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g2, pool=pool):
        add_kernel[(n + 256 - 1) // 256,](x, y, n, BLOCK_SIZE=256)

    ptrs1 = get_graph_dev_ptrs_and_bases(g1)
    ptrs2 = get_graph_dev_ptrs_and_bases(g2)

    match = set(ptrs1.keys()) == set(ptrs2.keys())
    print(f"\n  Graph 1: {len(ptrs1)} pointers")
    print(f"  Graph 2: {len(ptrs2)} pointers")
    print(f"  Addresses match: {match}")

    del g1, g2
    torch.cuda.synchronize()
    return match


if __name__ == "__main__":
    results = []

    results.append(("triton_pool_offset", test_triton_pool_offset()))
    results.append(("triton_same_pool", test_triton_multiple_captures_same_pool()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
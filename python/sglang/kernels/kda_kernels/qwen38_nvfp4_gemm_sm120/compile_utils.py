# SPDX-License-Identifier: Apache-2.0

# KDA provenance: this kernel was automatically optimized by the Humanize2
# workflow (https://github.com/PolyArch/humanize) and Kernel Design Agents
# (https://github.com/mit-han-lab/kernel-design-agents).
# Source: https://github.com/BBuf/KDA-Pilot/pull/195 @
# 516c976cee824a236679adf6eb525275a0a9a120.
"""Small CuTe-DSL compile shim owned by the candidate."""

from __future__ import annotations

import torch

_HARDWARE_INFO = None


def _max_active_clusters(cluster_size: int) -> int:
    global _HARDWARE_INFO
    try:
        import cutlass

        if _HARDWARE_INFO is None:
            _HARDWARE_INFO = cutlass.utils.HardwareInfo()
        return _HARDWARE_INFO.get_max_active_clusters(cluster_size)
    except Exception:
        return torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count


def compile_block_scaled_gemm(
    cache,
    cache_key,
    make_gemm_kernel,
    ab_cutlass_dtype,
    sf_dtype,
    c_cutlass_dtype,
    ab_assumed_align,
    cluster_shape_mn,
    swap_ab,
    sf_m,
    sf_n,
    sf_k,
    batch_size,
    cluster_shape_k=1,
    cache_module_name=None,
    device_index=None,
):
    """Compile and memoize one candidate-owned block-scaled GEMM."""
    del cache_module_name
    if device_index is None:
        device_index = torch.cuda.current_device()
    mem_key = (device_index, cache_key)
    if mem_key in cache:
        return cache[mem_key]

    gemm = make_gemm_kernel()
    cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1] * cluster_shape_k
    max_active_clusters = _max_active_clusters(cluster_size)
    compiled = _make_compile_fn(
        gemm,
        ab_cutlass_dtype,
        sf_dtype,
        c_cutlass_dtype,
        ab_assumed_align,
        swap_ab,
        sf_m,
        sf_n,
        sf_k,
        batch_size,
        max_active_clusters,
    )()
    result = (compiled, max_active_clusters)
    cache[mem_key] = result
    return result


def _make_compile_fn(
    gemm,
    ab_cutlass_dtype,
    sf_dtype,
    c_cutlass_dtype,
    ab_assumed_align,
    swap_ab,
    sf_m,
    sf_n,
    sf_k,
    batch_size,
    max_active_clusters,
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    def compile_kernel():
        sym_m = cute.sym_int()
        sym_k = cute.sym_int()
        sym_n = cute.sym_int()
        a_fake = cute.runtime.make_fake_compact_tensor(
            ab_cutlass_dtype,
            (sym_m, sym_k),
            stride_order=(1, 0),
            assumed_align=ab_assumed_align,
        )
        b_fake = cute.runtime.make_fake_compact_tensor(
            ab_cutlass_dtype,
            (sym_n, sym_k),
            stride_order=(1, 0),
            assumed_align=ab_assumed_align,
        )
        if swap_ab:
            c_fake = cute.runtime.make_fake_compact_tensor(
                c_cutlass_dtype,
                (sym_n, sym_m),
                stride_order=(0, 1),
                assumed_align=16,
            )
        else:
            c_fake = cute.runtime.make_fake_compact_tensor(
                c_cutlass_dtype,
                (sym_m, sym_n),
                stride_order=(1, 0),
                assumed_align=16,
            )

        a_sf_ptr = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, 16)
        b_sf_ptr = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, 16)
        alpha_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (1,), assumed_align=4
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            gemm.wrapper,
            a_fake,
            b_fake,
            c_fake,
            sf_m,
            sf_n,
            sf_k,
            batch_size,
            a_sf_ptr,
            b_sf_ptr,
            alpha_fake,
            max_active_clusters,
            stream_fake,
            swap_ab,
            options="--opt-level 2 --enable-tvm-ffi",
        )

    return compile_kernel

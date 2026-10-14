# SPDX-License-Identifier: Apache-2.0
"""Shape-polymorphic compilation for the native Step5 decode entry."""

import functools

import cutlass
import torch
from cutlass import Float32, Int32, cute

from .compile_cache import cached_compile_function
from .decode_sm103 import SparseDecodeSm103


@functools.cache
def device_sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@cached_compile_function
def compile_decode(
    region_capacity: int,
    dynamic_splits: bool,
    mtp_q_len: int,
    device_key: tuple[str, int, tuple[int, int]],
    sm_count: int,
    logical_q_heads: int = 8,
    output_fp32: bool = False,
    kv_fp8: bool = False,
):
    device_type, _, capability = device_key
    if device_type != "cuda" or capability[0] != 10:
        raise ValueError("Step5 CuTe sparse decode requires Blackwell tcgen05")
    # Dynamic planning embeds this value; architecture alone cannot identify
    # a device configuration. The original merge also keys on its SM count.
    kernel = SparseDecodeSm103(
        dtype=cutlass.Float16 if kv_fp8 else cutlass.BFloat16,
        logical_q_heads=logical_q_heads,
        head_dim=192,
        head_dim_v=192,
        block_n=64,
        num_threads=256,
        topk_windows=region_capacity,
        mtp_q_len=mtp_q_len,
        split_capacity=16 if dynamic_splits else 1,
        sm_count=sm_count,
        kv_fp8=kv_fp8,
    )
    kv_dtype = cutlass.Float8E4M3FN if kv_fp8 else cutlass.BFloat16

    def fake(dtype, shape, stride, alignment):
        return cute.runtime.make_fake_tensor(
            dtype, shape, stride, assumed_align=alignment
        )

    # Row count, physical page count and request count are runtime dimensions.
    # Fake descriptors allocate no CUDA storage and hold no request tensors.
    return cute.compile(
        kernel,
        fake(
            cutlass.BFloat16,
            (cute.sym_int(), logical_q_heads, 192),
            (logical_q_heads * 192, 192, 1),
            16,
        ),
        fake(kv_dtype, (cute.sym_int(), 16, 1, 192), (3072, 192, 192, 1), 16),
        fake(kv_dtype, (cute.sym_int(), 16, 1, 192), (3072, 192, 192, 1), 16),
        fake(Float32, (cute.sym_int(),), (1,), 4),
        fake(Float32, (cute.sym_int(),), (1,), 4),
        fake(cutlass.Int64, (cute.sym_int(), region_capacity), (region_capacity, 1), 8),
        fake(Int32, (cute.sym_int(),), (1,), 4),
        fake(Int32, (cute.sym_int(),), (1,), 4),
        fake(Int32, (cute.sym_int(),), (1,), 4),
        fake(Int32, (1,), (1,), 4),
        fake(
            Float32 if output_fp32 else cutlass.BFloat16,
            (cute.sym_int(), logical_q_heads, 192),
            (logical_q_heads * 192, 192, 1),
            16,
        ),
        fake(Float32, (cute.sym_int(), logical_q_heads), (logical_q_heads, 1), 4),
        Float32(1),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 2",
    )


@cached_compile_function
def compile_decode_sm90(
    region_capacity: int,
    dynamic_splits: bool,
    mtp_q_len: int,
    device_key: tuple[str, int, tuple[int, int]],
    sm_count: int,
    logical_q_heads: int = 8,
    output_fp32: bool = False,
    kv_fp8: bool = False,
):
    device_type, _, capability = device_key
    if device_type != "cuda" or capability[0] != 9:
        raise ValueError("Step5 SM90 sparse decode requires Hopper")
    from .sm90.decode_sm90_gqa import _TokenWiseSparseDecodeWGMMA64GQA

    kernel = _TokenWiseSparseDecodeWGMMA64GQA(
        dtype=cutlass.Float16 if kv_fp8 else cutlass.BFloat16,
        logical_q_heads=logical_q_heads,
        head_dim=192,
        head_dim_v=192,
        block_n=64,
        num_threads=256,
        topk_windows=region_capacity,
        mtp_q_len=mtp_q_len,
        split_capacity=16 if dynamic_splits else 1,
        sm_count=sm_count,
        kv_fp8=kv_fp8,
    )
    kv_dtype = cutlass.Float8E4M3FN if kv_fp8 else cutlass.BFloat16

    def fake(dtype, shape, stride, alignment):
        return cute.runtime.make_fake_tensor(
            dtype, shape, stride, assumed_align=alignment
        )

    return cute.compile(
        kernel,
        fake(
            cutlass.BFloat16,
            (cute.sym_int(), logical_q_heads, 192),
            (logical_q_heads * 192, 192, 1),
            16,
        ),
        fake(kv_dtype, (cute.sym_int(), 16, 1, 192), (3072, 192, 192, 1), 16),
        fake(kv_dtype, (cute.sym_int(), 16, 1, 192), (3072, 192, 192, 1), 16),
        fake(Float32, (cute.sym_int(),), (1,), 4) if kv_fp8 else None,
        fake(Float32, (cute.sym_int(),), (1,), 4) if kv_fp8 else None,
        fake(cutlass.Int64, (cute.sym_int(), region_capacity), (region_capacity, 1), 8),
        fake(Int32, (cute.sym_int(),), (1,), 4),
        fake(Int32, (cute.sym_int(),), (1,), 4),
        fake(Int32, (cute.sym_int(),), (1,), 4),
        fake(Int32, (1,), (1,), 4),
        fake(
            Float32 if output_fp32 else cutlass.BFloat16,
            (cute.sym_int(), logical_q_heads, 192),
            (logical_q_heads * 192, 192, 1),
            16,
        ),
        fake(Float32, (cute.sym_int(), logical_q_heads), (logical_q_heads, 1), 4),
        Float32(1),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 2",
    )

# SPDX-License-Identifier: Apache-2.0
"""Shape-polymorphic compilation for native Step5 grouped-union prefill."""

import cutlass
from cutlass import Float32, Int32, cute

from .compile_cache import cached_compile_function
from .prefill_sm103 import SparsePrefillSm103


@cached_compile_function
def compile_prefill(
    union_capacity: int,
    device_key: tuple[str, int, tuple[int, int]],
    logical_q_heads: int = 8,
):
    device_type, _, capability = device_key
    if device_type != "cuda" or capability[0] != 10:
        raise ValueError("Step5 CuTe sparse prefill requires Blackwell tcgen05")

    def fake(dtype, shape, stride, alignment):
        return cute.runtime.make_fake_tensor(
            dtype, shape, stride, assumed_align=alignment
        )

    def union():
        return fake(Int32, (cute.sym_int(), union_capacity), (union_capacity, 1), 4)

    def vector():
        return fake(Int32, (cute.sym_int(),), (1,), 4)

    # Query, group and native KV capacities are runtime dimensions. Only the
    # union row stride is specialized; fake descriptors allocate no CUDA data.
    return cute.compile(
        SparsePrefillSm103(union_capacity, logical_q_heads=logical_q_heads),
        fake(
            cutlass.BFloat16,
            (cute.sym_int(), logical_q_heads, 192),
            (logical_q_heads * 192, 192, 1),
            16,
        ),
        fake(cutlass.BFloat16, (cute.sym_int(), 1, 192), (192, 192, 1), 16),
        fake(cutlass.BFloat16, (cute.sym_int(), 1, 192), (192, 192, 1), 16),
        union(),
        union(),
        vector(),
        union(),
        vector(),
        vector(),
        vector(),
        fake(
            cutlass.BFloat16,
            (cute.sym_int(), logical_q_heads, 192),
            (logical_q_heads * 192, 192, 1),
            16,
        ),
        fake(Float32, (cute.sym_int(), logical_q_heads), (logical_q_heads, 1), 4),
        Float32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 2",
    )


@cached_compile_function
def compile_prefill_sm90(
    union_capacity: int,
    device_key: tuple[str, int, tuple[int, int]],
    logical_q_heads: int = 8,
):
    device_type, _, capability = device_key
    if device_type != "cuda" or capability[0] != 9:
        raise ValueError("Step5 SM90 sparse prefill requires Hopper")
    from .sm90.prefill_union_paged_sm90_gqa import _TokenWiseSparsePagedPrefillUnionGQA

    kernel = _TokenWiseSparsePagedPrefillUnionGQA(
        dtype=cutlass.BFloat16,
        logical_q_heads=logical_q_heads,
        head_dim=192,
        head_dim_v=192,
        block_n=64,
        num_threads=512,
        q_group=128 // logical_q_heads,
        max_union_windows=union_capacity,
        per_query_positions=True,
    )

    def fake(dtype, shape, stride, alignment):
        return cute.runtime.make_fake_tensor(
            dtype, shape, stride, assumed_align=alignment
        )

    def union():
        return fake(Int32, (cute.sym_int(), union_capacity), (union_capacity, 1), 4)

    def vector():
        return fake(Int32, (cute.sym_int(),), (1,), 4)

    def rows():
        return fake(
            cutlass.BFloat16,
            (cute.sym_int(), logical_q_heads, 192),
            (logical_q_heads * 192, 192, 1),
            16,
        )

    kv = (cute.sym_int(), 16, 1, 192), (3072, 192, 192, 1)
    return cute.compile(
        kernel,
        rows(),
        fake(cutlass.BFloat16, *kv, 16),
        fake(cutlass.BFloat16, *kv, 16),
        union(),
        union(),
        vector(),
        union(),
        vector(),
        vector(),
        vector(),
        None,
        rows(),
        fake(Float32, (cute.sym_int(), logical_q_heads), (logical_q_heads, 1), 4),
        Float32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 2",
    )

import functools

import tilelang
import tilelang.language as T
import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.hardware_backend.musa.jit_kernel.tilelang.utils import (
    MUSA_COMMON_PASS_CONFIGS,
    MUSA_COMPILE_FLAGS,
    tilelang_dtype,
)
from sglang.srt.utils.custom_op import register_custom_op

__all__ = ["fused_qkvzba_split_reshape_cat_contiguous"]

_PASS_CONFIGS = dict(MUSA_COMMON_PASS_CONFIGS)
if hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False
elif hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
for _key, _value in (
    ("TL_DISABLE_SAFE_COPY_PREDICATION", True),
    ("TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION", True),
    ("TL_CONFIG_INDEX_BITWIDTH", 32),
):
    if hasattr(tilelang.PassConfigKey, _key):
        _PASS_CONFIGS[getattr(tilelang.PassConfigKey, _key)] = _value


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    target="musa",
    pass_configs=_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _fused_qkvzba_split_reshape_cat_contiguous_kernel(
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
    input_dtype: str,
    ba_dtype: str,
):
    m = T.dynamic("m")
    v_per_group = num_heads_v // num_heads_qk
    total_q = num_heads_qk * head_qk
    total_k = total_q
    total_v = num_heads_v * head_v
    qkv_dim = total_q + total_k + total_v
    total_qkvz = qkv_dim + total_v
    total_ba = num_heads_v * 2
    v_group_dim = v_per_group * head_v

    @T.prim_func
    def sglang_musa_fused_qkvzba_split_reshape_cat_contiguous(
        mixed_qkv: T.Tensor((m, qkv_dim), input_dtype),
        z: T.Tensor((m, num_heads_v, head_v), input_dtype),
        b: T.Tensor((m, num_heads_v), ba_dtype),
        a: T.Tensor((m, num_heads_v), ba_dtype),
        mixed_qkvz: T.Tensor((m, total_qkvz), input_dtype),
        mixed_ba: T.Tensor((m, total_ba), ba_dtype),
    ):
        with T.Kernel(m, num_heads_qk, threads=128) as (row, hq):
            for d in T.Parallel(head_qk):
                mixed_qkv[row, hq * head_qk + d] = mixed_qkvz[row, hq * head_qk + d]
                mixed_qkv[row, total_q + hq * head_qk + d] = mixed_qkvz[
                    row, total_q + hq * head_qk + d
                ]

            for d in T.Parallel(v_group_dim):
                v_offset = hq * v_group_dim + d
                mixed_qkv[row, total_q + total_k + v_offset] = mixed_qkvz[
                    row, total_q + total_k + v_offset
                ]
                z[row, hq * v_per_group + d // head_v, d % head_v] = mixed_qkvz[
                    row, qkv_dim + v_offset
                ]

            for d in T.Parallel(v_per_group):
                v_head = hq * v_per_group + d
                b[row, v_head] = mixed_ba[row, v_head]
                a[row, v_head] = mixed_ba[row, num_heads_v + v_head]

    return sglang_musa_fused_qkvzba_split_reshape_cat_contiguous


_fused_qkvzba_split_reshape_cat_contiguous_kernel.mode = "lazy"


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    target="musa",
    pass_configs=_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _fused_qkvzba_split_reshape_cat_contiguous_row_kernel(
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
    input_dtype: str,
    ba_dtype: str,
    block_elems: int,
):
    m = T.dynamic("m")
    total_q = num_heads_qk * head_qk
    total_k = total_q
    total_v = num_heads_v * head_v
    qkv_dim = total_q + total_k + total_v
    total_qkvz = qkv_dim + total_v
    total_ba = num_heads_v * 2
    qkv_blocks = T.ceildiv(qkv_dim, block_elems)
    v_blocks = T.ceildiv(total_v, block_elems)
    ba_blocks = T.ceildiv(total_ba, block_elems)

    @T.prim_func
    def sglang_musa_fused_qkvzba_split_reshape_cat_contiguous_row(
        mixed_qkv: T.Tensor((m, qkv_dim), input_dtype),
        z: T.Tensor((m, num_heads_v, head_v), input_dtype),
        b: T.Tensor((m, num_heads_v), ba_dtype),
        a: T.Tensor((m, num_heads_v), ba_dtype),
        mixed_qkvz: T.Tensor((m, total_qkvz), input_dtype),
        mixed_ba: T.Tensor((m, total_ba), ba_dtype),
    ):
        with T.Kernel(m, threads=block_elems) as row:
            for block in T.serial(qkv_blocks):
                for i in T.Parallel(block_elems):
                    offset = block * block_elems + i
                    if offset < qkv_dim:
                        mixed_qkv[row, offset] = mixed_qkvz[row, offset]

            for block in T.serial(v_blocks):
                for i in T.Parallel(block_elems):
                    offset = block * block_elems + i
                    if offset < total_v:
                        z[row, offset // head_v, offset % head_v] = mixed_qkvz[
                            row, qkv_dim + offset
                        ]

            for block in T.serial(ba_blocks):
                for i in T.Parallel(block_elems):
                    offset = block * block_elems + i
                    if offset < num_heads_v:
                        b[row, offset] = mixed_ba[row, offset]
                    ba_offset = offset + num_heads_v
                    if ba_offset < total_ba:
                        a[row, offset] = mixed_ba[row, ba_offset]

    return sglang_musa_fused_qkvzba_split_reshape_cat_contiguous_row


_fused_qkvzba_split_reshape_cat_contiguous_row_kernel.mode = "lazy"


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    target="musa",
    pass_configs=_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _fused_qkvzba_split_reshape_cat_contiguous_vec_kernel(
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
    input_dtype: str,
    ba_dtype: str,
    block_threads: int,
    vec_elems: int,
):
    m = T.dynamic("m")
    total_q = num_heads_qk * head_qk
    total_k = total_q
    total_v = num_heads_v * head_v
    qkv_dim = total_q + total_k + total_v
    total_qkvz = qkv_dim + total_v
    total_ba = num_heads_v * 2
    elems_per_block = block_threads * vec_elems
    qkv_blocks = T.ceildiv(qkv_dim, elems_per_block)
    v_blocks = T.ceildiv(total_v, elems_per_block)

    @T.prim_func
    def sglang_musa_fused_qkvzba_split_reshape_cat_contiguous_vec(
        mixed_qkv: T.Tensor((m, qkv_dim), input_dtype),
        z: T.Tensor((m, num_heads_v, head_v), input_dtype),
        b: T.Tensor((m, num_heads_v), ba_dtype),
        a: T.Tensor((m, num_heads_v), ba_dtype),
        mixed_qkvz: T.Tensor((m, total_qkvz), input_dtype),
        mixed_ba: T.Tensor((m, total_ba), ba_dtype),
    ):
        with T.Kernel(m, threads=block_threads) as row:
            tid = T.get_thread_binding()

            for block in T.serial(qkv_blocks):
                base = block * elems_per_block + tid * vec_elems
                for v in T.vectorized(vec_elems):
                    offset = base + v
                    if offset < qkv_dim:
                        mixed_qkv[row, offset] = mixed_qkvz[row, offset]

            for block in T.serial(v_blocks):
                base = block * elems_per_block + tid * vec_elems
                for v in T.vectorized(vec_elems):
                    offset = base + v
                    if offset < total_v:
                        z[row, offset // head_v, offset % head_v] = mixed_qkvz[
                            row, qkv_dim + offset
                        ]

            for v in T.vectorized(vec_elems):
                offset = tid * vec_elems + v
                if offset < num_heads_v:
                    b[row, offset] = mixed_ba[row, offset]
                    a[row, offset] = mixed_ba[row, num_heads_v + offset]

    return sglang_musa_fused_qkvzba_split_reshape_cat_contiguous_vec


_fused_qkvzba_split_reshape_cat_contiguous_vec_kernel.mode = "lazy"


def _check_fused_qkvzba_inputs(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
) -> tuple[int, int, int]:
    if mixed_qkvz.dim() != 2 or mixed_ba.dim() != 2:
        raise RuntimeError("fused_qkvzba expects 2D mixed_qkvz and mixed_ba.")
    if mixed_qkvz.shape[0] != mixed_ba.shape[0]:
        raise RuntimeError("mixed_qkvz and mixed_ba must have the same batch size.")
    if mixed_qkvz.device != mixed_ba.device:
        raise RuntimeError("mixed_qkvz and mixed_ba must be on the same device.")
    if mixed_qkvz.stride(-1) != 1 or mixed_ba.stride(-1) != 1:
        raise RuntimeError("mixed_qkvz and mixed_ba must be last-dim contiguous.")
    if min(num_heads_qk, num_heads_v, head_qk, head_v) <= 0:
        raise RuntimeError("head counts and head dims must be positive.")
    if num_heads_v % num_heads_qk != 0:
        raise RuntimeError("num_heads_v must be divisible by num_heads_qk.")

    tilelang_dtype(mixed_qkvz.dtype)
    tilelang_dtype(mixed_ba.dtype)
    qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    qkvz_dim = qkv_dim + num_heads_v * head_v
    ba_dim = num_heads_v * 2
    if mixed_qkvz.shape[1] != qkvz_dim:
        raise RuntimeError(
            f"mixed_qkvz shape mismatch: got {mixed_qkvz.shape[1]}, "
            f"expected {qkvz_dim}."
        )
    if mixed_ba.shape[1] != ba_dim:
        raise RuntimeError(
            f"mixed_ba shape mismatch: got {mixed_ba.shape[1]}, expected {ba_dim}."
        )
    return mixed_qkvz.shape[0], qkv_dim, ba_dim


def _fused_qkvzba_split_reshape_cat_contiguous_impl(
    mixed_qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
) -> None:
    if mixed_qkv.shape[0] >= 1024 and head_qk % 128 == 0 and head_v % 128 == 0:
        if mixed_qkv.shape[0] >= 8192:
            kernel = _fused_qkvzba_split_reshape_cat_contiguous_vec_kernel(
                num_heads_qk,
                num_heads_v,
                head_qk,
                head_v,
                tilelang_dtype(mixed_qkvz.dtype),
                tilelang_dtype(mixed_ba.dtype),
                32,
                32,
            )
        else:
            kernel = _fused_qkvzba_split_reshape_cat_contiguous_row_kernel(
                num_heads_qk,
                num_heads_v,
                head_qk,
                head_v,
                tilelang_dtype(mixed_qkvz.dtype),
                tilelang_dtype(mixed_ba.dtype),
                1024,
            )
        kernel(mixed_qkv, z, b, a, mixed_qkvz, mixed_ba)
        return

    kernel = _fused_qkvzba_split_reshape_cat_contiguous_kernel(
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        tilelang_dtype(mixed_qkvz.dtype),
        tilelang_dtype(mixed_ba.dtype),
    )
    kernel(mixed_qkv, z, b, a, mixed_qkvz, mixed_ba)


@register_custom_op(
    op_name="musa_fused_qkvzba_split_reshape_cat_contiguous",
    mutates_args=["mixed_qkv", "z", "b", "a"],
)
def _fused_qkvzba_split_reshape_cat_contiguous_custom(
    mixed_qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
) -> None:
    # Keep the TileLang executable launch opaque to Dynamo.
    _fused_qkvzba_split_reshape_cat_contiguous_impl(
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
    )


@debug_kernel_api
def fused_qkvzba_split_reshape_cat_contiguous(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
):
    batch, qkv_dim, _ = _check_fused_qkvzba_inputs(
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
    )
    mixed_qkv = torch.empty(
        (batch, qkv_dim), dtype=mixed_qkvz.dtype, device=mixed_qkvz.device
    )
    z = torch.empty(
        (batch, num_heads_v, head_v), dtype=mixed_qkvz.dtype, device=mixed_qkvz.device
    )
    b = torch.empty((batch, num_heads_v), dtype=mixed_ba.dtype, device=mixed_ba.device)
    a = torch.empty_like(b)
    _fused_qkvzba_split_reshape_cat_contiguous_custom(
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
    )
    return mixed_qkv, z, b, a

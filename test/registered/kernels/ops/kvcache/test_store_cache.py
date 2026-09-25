import itertools
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.kvcache.cache_move import store_k_slots
from sglang.kernels.ops.kvcache.cache_ops import launch_reshape_and_cache_flash
from sglang.kernels.ops.kvcache.kvcache import can_use_store_cache, store_cache
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseMHAMainPool
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKOnlyPool,
    MHATokenToKVPool,
    _as_token_head_dim,
    _has_dense_kv_rows,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=34, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Nightly is not redundant here: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=46, stage="nightly", runner_config="1-gpu-large")
register_amd_ci(est_time=62, stage="jit-kernel-unit", runner_config="amd")

BS_LIST = [2**n for n in range(0, 15)]
BS_LIST += [x + 1 + i for i, x in enumerate(BS_LIST)]
BS_LIST = get_ci_test_range(BS_LIST, [1, 9, 256, 16399])
HIDDEN_DIMS = get_ci_test_range(
    [64, 128, 256, 512, 1024, 96, 97, 100], [64, 512, 1024, 97]
)
CACHE_SIZE = 1024 * 1024
DTYPE = torch.bfloat16
DEVICE = "cuda"


@pytest.mark.parametrize(
    "batch_size,element_dim",
    list(itertools.product(BS_LIST, HIDDEN_DIMS)),
)
def test_store_cache(batch_size: int, element_dim: int) -> None:
    k = torch.randn((batch_size, element_dim), dtype=DTYPE, device=DEVICE)
    v = torch.randn((batch_size, element_dim), dtype=DTYPE, device=DEVICE)
    k_cache = torch.randn((CACHE_SIZE, element_dim), dtype=DTYPE, device=DEVICE)
    v_cache = torch.randn((CACHE_SIZE, element_dim), dtype=DTYPE, device=DEVICE)
    indices = torch.randperm(CACHE_SIZE - 1, device=DEVICE)[:batch_size] + 1

    store_cache(k, v, k_cache, v_cache, indices)

    assert torch.all(k_cache[indices] == k)
    assert torch.all(v_cache[indices] == v)


# Smaller subset for targeted tests below
REPR_BS = get_ci_test_range([1, 7, 128], [1, 128])
REPR_DIMS = get_ci_test_range([64, 128, 512, 1024, 96], [64, 1024, 96])
SMALL_CACHE = 4096


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "batch_size,element_dim",
    list(itertools.product(REPR_BS, REPR_DIMS)),
)
def test_store_cache_dtypes(
    batch_size: int, element_dim: int, dtype: torch.dtype
) -> None:
    k = torch.randn((batch_size, element_dim), dtype=dtype, device=DEVICE)
    v = torch.randn((batch_size, element_dim), dtype=dtype, device=DEVICE)
    k_cache = torch.randn((SMALL_CACHE, element_dim), dtype=dtype, device=DEVICE)
    v_cache = torch.randn((SMALL_CACHE, element_dim), dtype=dtype, device=DEVICE)
    indices = torch.randperm(SMALL_CACHE - 1, device=DEVICE)[:batch_size] + 1

    store_cache(k, v, k_cache, v_cache, indices)

    assert torch.all(k_cache[indices] == k)
    assert torch.all(v_cache[indices] == v)


@pytest.mark.parametrize(
    "batch_size,element_dim",
    list(itertools.product(REPR_BS, REPR_DIMS)),
)
def test_store_cache_int32_indices(batch_size: int, element_dim: int) -> None:
    k = torch.randn((batch_size, element_dim), dtype=DTYPE, device=DEVICE)
    v = torch.randn((batch_size, element_dim), dtype=DTYPE, device=DEVICE)
    k_cache = torch.randn((SMALL_CACHE, element_dim), dtype=DTYPE, device=DEVICE)
    v_cache = torch.randn((SMALL_CACHE, element_dim), dtype=DTYPE, device=DEVICE)
    # int32 indices exercise a different CUDA template instantiation than default int64
    indices = (torch.randperm(SMALL_CACHE - 1, device=DEVICE)[:batch_size] + 1).to(
        torch.int32
    )

    store_cache(k, v, k_cache, v_cache, indices)

    assert torch.all(k_cache[indices.long()] == k)
    assert torch.all(v_cache[indices.long()] == v)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_store_cache_reserved_skip_index(index_dtype: torch.dtype) -> None:
    element_dim = 1024
    k = torch.randn((4, element_dim), dtype=DTYPE, device=DEVICE)
    v = torch.randn((4, element_dim), dtype=DTYPE, device=DEVICE)
    # Model kernels may leave CUDA-graph padding rows undefined. Reproduce the
    # dangerous case directly instead of requiring a full model checkpoint.
    k[[0, 2]] = torch.nan
    v[[0, 2]] = torch.nan
    k_cache = torch.randn((SMALL_CACHE, element_dim), dtype=DTYPE, device=DEVICE)
    v_cache = torch.randn((SMALL_CACHE, element_dim), dtype=DTYPE, device=DEVICE)
    reserved_k_before = k_cache[0].clone()
    reserved_v_before = v_cache[0].clone()
    indices = torch.tensor([0, 7, 0, 9], dtype=index_dtype, device=DEVICE)

    store_cache(
        k,
        v,
        k_cache,
        v_cache,
        indices,
    )

    torch.testing.assert_close(k_cache[0], reserved_k_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(v_cache[0], reserved_v_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(k_cache[indices[1].long()], k[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(v_cache[indices[1].long()], v[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(k_cache[indices[3].long()], k[3], rtol=0.0, atol=0.0)
    torch.testing.assert_close(v_cache[indices[3].long()], v[3], rtol=0.0, atol=0.0)


def test_store_cache_zero_index_can_be_written_when_skip_disabled() -> None:
    element_dim = 64
    k = torch.randn((1, element_dim), dtype=DTYPE, device=DEVICE)
    v = torch.randn((1, element_dim), dtype=DTYPE, device=DEVICE)
    k_cache = torch.randn((SMALL_CACHE, element_dim), dtype=DTYPE, device=DEVICE)
    v_cache = torch.randn((SMALL_CACHE, element_dim), dtype=DTYPE, device=DEVICE)
    indices = torch.zeros(1, dtype=torch.int64, device=DEVICE)

    store_cache(k, v, k_cache, v_cache, indices, reserved_skip_index=-1)

    torch.testing.assert_close(k_cache[0], k[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(v_cache[0], v[0], rtol=0.0, atol=0.0)


# Asymmetric K/V (head_dim != v_head_dim): different row widths AND cache strides.
# MiMoV2 is 192/128. Both orderings, since nothing may assume K is the wider one.
ASYM_DIM_PAIRS = get_ci_test_range(
    [(192, 128), (128, 192), (1024, 512), (512, 1024), (96, 64), (2048, 1024)],
    [(192, 128), (512, 1024)],
)


# The kernel is a byte copier specialized on (k_row_bytes, v_row_bytes) -- no dtype
# in its template args -- so equal-itemsize dtypes share one instantiation. bf16 and
# fp32 are the two distinct itemsizes; fp16 would just re-run the bf16 one.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("k_dim,v_dim", ASYM_DIM_PAIRS)
def test_store_cache_asymmetric(k_dim: int, v_dim: int, dtype: torch.dtype) -> None:
    batch_size = 128
    k = torch.randn((batch_size, k_dim), dtype=dtype, device=DEVICE)
    v = torch.randn((batch_size, v_dim), dtype=dtype, device=DEVICE)
    k_cache = torch.randn((SMALL_CACHE, k_dim), dtype=dtype, device=DEVICE)
    v_cache = torch.randn((SMALL_CACHE, v_dim), dtype=dtype, device=DEVICE)
    k_before, v_before = k_cache.clone(), v_cache.clone()
    indices = torch.randperm(SMALL_CACHE - 1, device=DEVICE)[:batch_size] + 1

    store_cache(k, v, k_cache, v_cache, indices)

    assert torch.all(k_cache[indices] == k)
    assert torch.all(v_cache[indices] == v)
    # Applying K's stride to V (or vice versa) would corrupt neighbouring slots,
    # which the target-slot assertions above cannot see.
    untouched = torch.ones(SMALL_CACHE, dtype=torch.bool, device=DEVICE)
    untouched[indices] = False
    assert torch.all(k_cache[untouched] == k_before[untouched])
    assert torch.all(v_cache[untouched] == v_before[untouched])


def test_can_use_store_cache() -> None:
    assert can_use_store_cache(128)
    assert can_use_store_cache(256)
    assert can_use_store_cache(1024)
    assert can_use_store_cache(2048)
    # asymmetric widths, and the documented default (v falls back to k)
    assert can_use_store_cache(384, 256)
    assert can_use_store_cache(256, 384)
    assert can_use_store_cache(1024, 0) == can_use_store_cache(1024)


# ---------------------------------------------------------------------------
# store_k_slots -- Triton row scatter replacing `k_buffer[loc] = src`
# (ATen advanced indexing / index_put) in the MiniMax-M3 sparse index-K store.
# ---------------------------------------------------------------------------

# (head_num, head_dim) pairs chosen so ROW_DIM = head_num * head_dim lands below,
# on, and above the kernel's block width, including a non-multiple. This pins the
# row-block grid behaviour without pinning BLOCK itself.
K_SLOT_SHAPES = get_ci_test_range(
    [(1, 64), (2, 64), (1, 128), (4, 128), (8, 128), (1, 96), (3, 100), (16, 128)],
    [(1, 64), (4, 128), (3, 100)],
)
K_SLOT_BS = get_ci_test_range([1, 7, 128, 1000], [1, 128])


def _ref_scatter(
    k_buffer: torch.Tensor, src: torch.Tensor, loc: torch.Tensor
) -> torch.Tensor:
    """Reference: what `k_buffer[loc] = src` produces, for non-negative loc."""
    out = k_buffer.clone()
    out[loc.long()] = src
    return out


@pytest.mark.parametrize("head_num,head_dim", K_SLOT_SHAPES)
@pytest.mark.parametrize("batch_size", K_SLOT_BS)
def test_store_k_slots_matches_advanced_indexing(
    head_num: int, head_dim: int, batch_size: int
) -> None:
    src = torch.randn((batch_size, head_num, head_dim), dtype=DTYPE, device=DEVICE)
    k_buffer = torch.randn(
        (SMALL_CACHE, head_num, head_dim), dtype=DTYPE, device=DEVICE
    )
    loc = torch.randperm(SMALL_CACHE, device=DEVICE)[:batch_size]
    expected = _ref_scatter(k_buffer, src, loc)

    store_k_slots(k_buffer, src, loc)

    # Bit-equality: this is a byte move, not arithmetic.
    torch.testing.assert_close(k_buffer, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_store_k_slots_skips_negative_slots(index_dtype: torch.dtype) -> None:
    head_num, head_dim = 4, 128
    src = torch.randn((4, head_num, head_dim), dtype=DTYPE, device=DEVICE)
    # A padded row may hold anything; NaN makes an erroneous write visible.
    src[[0, 2]] = torch.nan
    k_buffer = torch.randn(
        (SMALL_CACHE, head_num, head_dim), dtype=DTYPE, device=DEVICE
    )
    before = k_buffer.clone()
    loc = torch.tensor([-1, 7, -1, 9], dtype=index_dtype, device=DEVICE)

    store_k_slots(k_buffer, src, loc)

    torch.testing.assert_close(k_buffer[7], src[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(k_buffer[9], src[3], rtol=0.0, atol=0.0)
    # Everything else, including the rows the NaN sources would have hit, is intact.
    untouched = torch.ones(SMALL_CACHE, dtype=torch.bool, device=DEVICE)
    untouched[[7, 9]] = False
    torch.testing.assert_close(
        k_buffer[untouched], before[untouched], rtol=0.0, atol=0.0
    )


def _random_like_cache(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=dtype, device=DEVICE)
    return torch.randn(shape, dtype=dtype, device=DEVICE)


# (index dtype, store dtype): the bf16 default and the fp8 index cache, whose
# pool stores raw bytes.
@pytest.mark.parametrize(
    "index_dtype,store_dtype,layout",
    [
        (torch.bfloat16, torch.bfloat16, "contiguous"),
        (torch.bfloat16, torch.bfloat16, "token_strided"),
        (torch.bfloat16, torch.bfloat16, "head_dim_sliced"),
        (torch.float8_e4m3fn, torch.uint8, "contiguous"),
    ],
)
def test_k_only_pool_set_k_buffer_matches_advanced_indexing(
    index_dtype: torch.dtype, store_dtype: torch.dtype, layout: str
) -> None:
    head_num, head_dim, batch_size = 1, 128, 32
    pool = SimpleNamespace(
        dtype=index_dtype,
        store_dtype=store_dtype,
        head_num=head_num,
        head_dim=head_dim,
        size=SMALL_CACHE - 1,
        page_size=1,
        k_buffer=[_random_like_cache((SMALL_CACHE, head_num, head_dim), store_dtype)],
    )
    rows = 2 * batch_size if layout == "token_strided" else batch_size
    width = 2 * head_dim if layout == "head_dim_sliced" else head_dim
    cache_k = torch.randn((rows, head_num, width), dtype=DTYPE, device=DEVICE)
    if layout == "token_strided":
        cache_k = cache_k[::2]
    elif layout == "head_dim_sliced":
        cache_k = cache_k[..., :head_dim]
    # Only the sliced source misses store_k_slots and takes the indexing fallback.
    assert _has_dense_kv_rows(cache_k, head_num, head_dim, batch_size) == (
        layout != "head_dim_sliced"
    )
    loc = torch.randperm(SMALL_CACHE, device=DEVICE)[:batch_size]
    expected = pool.k_buffer[0].clone()
    expected[loc] = cache_k.to(index_dtype).view(store_dtype)

    MHATokenToKOnlyPool.set_k_buffer(pool, 0, loc, cache_k)

    torch.testing.assert_close(pool.k_buffer[0], expected, rtol=0.0, atol=0.0)


def test_store_k_slots_empty_loc_is_noop() -> None:
    head_num, head_dim = 4, 128
    src = torch.randn((0, head_num, head_dim), dtype=DTYPE, device=DEVICE)
    k_buffer = torch.randn(
        (SMALL_CACHE, head_num, head_dim), dtype=DTYPE, device=DEVICE
    )
    before = k_buffer.clone()
    loc = torch.empty(0, dtype=torch.int64, device=DEVICE)

    store_k_slots(k_buffer, src, loc)

    torch.testing.assert_close(k_buffer, before, rtol=0.0, atol=0.0)


def test_store_k_slots_accepts_token_strided_source() -> None:
    head_num, head_dim = 4, 128
    k_buffer = torch.randn(
        (SMALL_CACHE, head_num, head_dim), dtype=DTYPE, device=DEVICE
    )
    # Strided along tokens but dense within each row: the kernel is passed
    # src.stride(0), so this is in contract even though is_contiguous() is False.
    src = torch.randn((256, head_num, head_dim), dtype=DTYPE, device=DEVICE)[::2]
    assert not src.is_contiguous()
    loc = torch.randperm(SMALL_CACHE, device=DEVICE)[: src.shape[0]]
    expected = _ref_scatter(k_buffer, src, loc)

    store_k_slots(k_buffer, src, loc)

    torch.testing.assert_close(k_buffer, expected, rtol=0.0, atol=0.0)


def test_store_k_slots_rejects_non_dense_rows() -> None:
    head_num, head_dim = 4, 128
    k_buffer = torch.randn(
        (SMALL_CACHE, head_num, head_dim), dtype=DTYPE, device=DEVICE
    )
    # Sliced along head_dim: the trailing block is no longer a flat run, which the
    # kernel cannot express. The caller must fall back to advanced indexing.
    src = torch.randn((8, head_num, head_dim * 2), dtype=DTYPE, device=DEVICE)[
        :, :, :head_dim
    ]
    loc = torch.arange(8, device=DEVICE)

    with pytest.raises(AssertionError):
        store_k_slots(k_buffer, src, loc)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.uint8]
)
def test_store_k_slots_dtypes(dtype: torch.dtype) -> None:
    head_num, head_dim = 4, 128
    batch_size = 32
    src = _random_like_cache((batch_size, head_num, head_dim), dtype)
    k_buffer = _random_like_cache((SMALL_CACHE, head_num, head_dim), dtype)
    loc = torch.randperm(SMALL_CACHE, device=DEVICE)[:batch_size]
    expected = _ref_scatter(k_buffer, src, loc)

    store_k_slots(k_buffer, src, loc)

    torch.testing.assert_close(k_buffer, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("layout", ["two_dimensional", "noncontiguous", "float"])
def test_store_k_slots_rejects_unsupported_loc(layout: str) -> None:
    if layout == "two_dimensional":
        loc = torch.arange(8, device=DEVICE).view(2, 4)
    elif layout == "noncontiguous":
        loc = torch.arange(16, device=DEVICE)[::2]
    else:
        loc = torch.arange(8, device=DEVICE, dtype=torch.float32)
    src = torch.randn((8, 1, 64), dtype=DTYPE, device=DEVICE)
    k_buffer = torch.randn((SMALL_CACHE, 1, 64), dtype=DTYPE, device=DEVICE)

    with pytest.raises(AssertionError):
        store_k_slots(k_buffer, src, loc)


# ---------------------------------------------------------------------------
# The guards that route a store to the kernels above: `_has_dense_kv_rows` for
# both, plus the pool-side conditions in `can_store_kv_fused_cast`. Shape and
# stride logic only, so these need no launch.
# ---------------------------------------------------------------------------

GUARD_HEAD_NUM, GUARD_HEAD_DIM = 2, 64
GUARD_ROW = GUARD_HEAD_NUM * GUARD_HEAD_DIM
QKV_WIDTH = 2304  # q + k + v at TP4, the width a K column slice strides by


def _qkv_k_slice(num_tokens: int) -> torch.Tensor:
    """A K slice as production produces it: `qkv.split(..., dim=-1)`."""
    qkv = torch.zeros((num_tokens, QKV_WIDTH), dtype=DTYPE, device=DEVICE)
    return qkv[:, GUARD_ROW : 2 * GUARD_ROW]


def _dense(t: torch.Tensor, num_tokens: int) -> bool:
    return _has_dense_kv_rows(t, GUARD_HEAD_NUM, GUARD_HEAD_DIM, num_tokens)


def test_dense_kv_rows_accepts_qkv_column_slice() -> None:
    k = _qkv_k_slice(8)
    assert not k.is_contiguous()
    assert k.stride() == (QKV_WIDTH, 1)
    assert _dense(k, 8)


def test_dense_kv_rows_accepts_contiguous_and_token_strided() -> None:
    flat = torch.zeros((8, GUARD_ROW), dtype=DTYPE, device=DEVICE)
    split = torch.zeros((8, GUARD_HEAD_NUM, GUARD_HEAD_DIM), dtype=DTYPE, device=DEVICE)
    # Dense (head, dim) block with gaps between tokens: in contract.
    strided = torch.zeros(
        (16, GUARD_HEAD_NUM, GUARD_HEAD_DIM), dtype=DTYPE, device=DEVICE
    )[::2]
    assert not strided.is_contiguous()
    assert _dense(flat, 8) and _dense(split, 8) and _dense(strided, 8)


@pytest.mark.parametrize("layout", ["qkv_slice", "flat", "split"])
def test_dense_kv_rows_rejects_wrong_token_count(layout: str) -> None:
    # Right row layout, more tokens than loc. reshape_and_cache_flash sizes its
    # grid from shape[0] and reads the slot index unmasked, so this overruns loc.
    if layout == "qkv_slice":
        t = _qkv_k_slice(8)
    elif layout == "flat":
        t = torch.zeros((8, GUARD_ROW), dtype=DTYPE, device=DEVICE)
    else:
        t = torch.zeros((8, GUARD_HEAD_NUM, GUARD_HEAD_DIM), dtype=DTYPE, device=DEVICE)
    assert _dense(t, 8)
    assert not _dense(t, 4)


def test_dense_kv_rows_rejects_non_dense_trailing_block() -> None:
    # Sliced along head_dim: the trailing block is no longer a flat run.
    sliced = torch.zeros(
        (8, GUARD_HEAD_NUM, 2 * GUARD_HEAD_DIM), dtype=DTYPE, device=DEVICE
    )[:, :, :GUARD_HEAD_DIM]
    transposed = torch.zeros(
        (8, GUARD_HEAD_DIM, GUARD_HEAD_NUM), dtype=DTYPE, device=DEVICE
    ).transpose(1, 2)
    assert tuple(transposed.shape) == (8, GUARD_HEAD_NUM, GUARD_HEAD_DIM)
    assert not _dense(sliced, 8)
    assert not _dense(transposed, 8)


def test_dense_kv_rows_rejects_unsupported_rank() -> None:
    t4 = torch.zeros((2, 4, GUARD_HEAD_NUM, GUARD_HEAD_DIM), dtype=DTYPE, device=DEVICE)
    assert not _dense(t4[:, ::2], 4)


def test_as_token_head_dim_does_not_copy() -> None:
    k = _qkv_k_slice(8)
    view = _as_token_head_dim(k, GUARD_HEAD_NUM, GUARD_HEAD_DIM)
    assert tuple(view.shape) == (8, GUARD_HEAD_NUM, GUARD_HEAD_DIM)
    assert view.data_ptr() == k.data_ptr()
    assert view.stride() == (QKV_WIDTH, GUARD_HEAD_DIM, 1)


class _PoolStub:
    """The pool state can_store_kv_fused_cast and store_kv_fused_cast read."""

    def __init__(self, buffer_dims: int = 3):
        self.kv_cache_layout = "nhd"
        self.use_hnd = False
        self.is_quantized_kv_cache = False
        self.dtype = torch.float8_e4m3fn
        self.store_dtype = torch.uint8
        self.head_num = GUARD_HEAD_NUM
        self.head_dim = GUARD_HEAD_DIM
        self.v_head_dim = GUARD_HEAD_DIM
        self.start_layer = 0
        shape = (
            (SMALL_CACHE, GUARD_HEAD_NUM, GUARD_HEAD_DIM)
            if buffer_dims == 3
            # NPUMHATokenToKVPool's paged per-layer buffer.
            else (SMALL_CACHE, 1, GUARD_HEAD_NUM, GUARD_HEAD_DIM)
        )
        self.k_buffer = [torch.zeros(shape, dtype=torch.uint8, device=DEVICE)]
        self.v_buffer = [torch.zeros(shape, dtype=torch.uint8, device=DEVICE)]
        self.size = SMALL_CACHE - 1
        self.page_size = 1

    def _get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.k_buffer[layer_id].view(self.dtype)

    def _get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffer[layer_id].view(self.dtype)


def _can_store(pool: _PoolStub, num_tokens: int = 8, **kwargs) -> bool:
    loc = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    return MHATokenToKVPool.can_store_kv_fused_cast(
        pool, 0, loc, _qkv_k_slice(num_tokens), _qkv_k_slice(num_tokens), **kwargs
    )


def test_can_store_kv_fused_cast_accepts_flat_slot_indexed_buffer() -> None:
    assert _can_store(_PoolStub())
    assert _can_store(_PoolStub(), k_scale=1.0, v_scale=1.0)


def test_can_store_kv_fused_cast_rejects_paged_per_layer_buffer() -> None:
    # store_kv_fused_cast presents the buffer as page-size 1 via unsqueeze(1),
    # which mis-indexes a buffer that is already paged.
    assert not _can_store(_PoolStub(buffer_dims=4))


def test_can_store_kv_fused_cast_rejects_scales_that_divide() -> None:
    assert not _can_store(_PoolStub(), k_scale=0.5)
    assert not _can_store(_PoolStub(), v_scale=0.5)
    # A tensor scale cannot be inspected without a sync, so it must divide.
    assert not _can_store(_PoolStub(), k_scale=torch.ones(1, device=DEVICE))


@pytest.mark.parametrize(
    "attr,value",
    [
        ("dtype", DTYPE),  # source already in cache dtype
        ("v_head_dim", GUARD_HEAD_DIM // 2),
        ("kv_cache_layout", "hnd"),
        ("use_hnd", True),
        ("is_quantized_kv_cache", True),
        ("store_dtype", DTYPE),
    ],
)
def test_can_store_kv_fused_cast_rejects_unsupported_pool(attr: str, value) -> None:
    pool = _PoolStub()
    setattr(pool, attr, value)
    assert not _can_store(pool)


def test_can_store_kv_fused_cast_rejects_hisparse_main_pool() -> None:
    # The fused store would skip HiSparse's logical-to-device slot translation.
    loc = torch.arange(8, dtype=torch.int64, device=DEVICE)
    k, v = _qkv_k_slice(8), _qkv_k_slice(8)
    assert MHATokenToKVPool.can_store_kv_fused_cast(_PoolStub(), 0, loc, k, v)
    assert not HiSparseMHAMainPool.can_store_kv_fused_cast(_PoolStub(), 0, loc, k, v)


@pytest.mark.parametrize("layout", ["contiguous", "qkv_slice"])
def test_store_kv_fused_cast_every_bf16_value(layout: str) -> None:
    # Byte-identical to `.to(fp8)` wherever that is finite. Where ATen overflows to
    # NaN (|x| > 464, inf), Triton's cast saturates to +-448 on every backend.
    values = torch.arange(-(2**15), 2**15, dtype=torch.int32, device=DEVICE)
    values = values.to(torch.int16).view(DTYPE)
    num_tokens = values.numel() // GUARD_ROW
    k_rows = values.view(num_tokens, GUARD_ROW)
    v_rows = values.flip(0).view(num_tokens, GUARD_ROW)
    if layout == "qkv_slice":
        cache_k, cache_v = _qkv_k_slice(num_tokens), _qkv_k_slice(num_tokens)
        cache_k.copy_(k_rows)
        cache_v.copy_(v_rows)
    else:
        cache_k = k_rows.view(num_tokens, GUARD_HEAD_NUM, GUARD_HEAD_DIM)
        cache_v = v_rows.view(num_tokens, GUARD_HEAD_NUM, GUARD_HEAD_DIM)
    pool = _PoolStub()
    # Slot 0 is the reserved padding slot, which the store skips.
    loc = torch.randperm(SMALL_CACHE - 1, device=DEVICE)[:num_tokens] + 1
    before_k, before_v = pool.k_buffer[0].clone(), pool.v_buffer[0].clone()

    MHATokenToKVPool.store_kv_fused_cast(pool, 0, loc, cache_k, cache_v)

    fp8_max = torch.finfo(pool.dtype).max
    saturated = torch.tensor([fp8_max, -fp8_max], device=DEVICE).to(pool.dtype)
    pos_max, neg_max = saturated.view(torch.uint8).tolist()
    untouched = torch.ones(SMALL_CACHE, dtype=torch.bool, device=DEVICE)
    untouched[loc] = False
    for src, buffer, before in (
        (k_rows, pool.k_buffer[0], before_k),
        (v_rows, pool.v_buffer[0], before_v),
    ):
        got = buffer[loc].view(num_tokens, GUARD_ROW)
        aten = src.to(pool.dtype).view(torch.uint8)
        src_nan = src.isnan()
        aten_nan = (aten & 0x7F) == 0x7F
        torch.testing.assert_close(got[~aten_nan], aten[~aten_nan], rtol=0, atol=0)
        overflow = aten_nan & ~src_nan
        want = torch.where(src < 0, neg_max, pos_max).to(torch.uint8)
        torch.testing.assert_close(got[overflow], want[overflow], rtol=0, atol=0)
        assert torch.all((got[src_nan] & 0x7F) == 0x7F), "NaN must stay NaN"
        torch.testing.assert_close(
            buffer[untouched], before[untouched], rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_store_kv_fused_cast_skips_reserved_slot(index_dtype: torch.dtype) -> None:
    pool = _PoolStub()
    shape = (4, GUARD_HEAD_NUM, GUARD_HEAD_DIM)
    cache_k = torch.randn(shape, dtype=DTYPE, device=DEVICE)
    cache_v = torch.randn(shape, dtype=DTYPE, device=DEVICE)
    # Padded rows may hold anything; NaN makes a write to slot 0 visible.
    cache_k[[0, 2]] = torch.nan
    cache_v[[0, 2]] = torch.nan
    loc = torch.tensor([0, 7, 0, 9], dtype=index_dtype, device=DEVICE)
    expected_k, expected_v = pool.k_buffer[0].clone(), pool.v_buffer[0].clone()
    expected_k[[7, 9]] = cache_k[[1, 3]].to(pool.dtype).view(torch.uint8)
    expected_v[[7, 9]] = cache_v[[1, 3]].to(pool.dtype).view(torch.uint8)

    MHATokenToKVPool.store_kv_fused_cast(pool, 0, loc, cache_k, cache_v)

    torch.testing.assert_close(pool.k_buffer[0], expected_k, rtol=0.0, atol=0.0)
    torch.testing.assert_close(pool.v_buffer[0], expected_v, rtol=0.0, atol=0.0)


def test_reshape_and_cache_flash_writes_slot_zero_by_default() -> None:
    # The AITER backend's fused KV write relies on the default skipping no slot.
    key = torch.randn((2, GUARD_HEAD_NUM, GUARD_HEAD_DIM), dtype=DTYPE, device=DEVICE)
    key_cache = torch.zeros(
        (SMALL_CACHE, 1, GUARD_HEAD_NUM, GUARD_HEAD_DIM), dtype=DTYPE, device=DEVICE
    )
    value_cache = torch.zeros_like(key_cache)
    loc = torch.tensor([0, 5], dtype=torch.int64, device=DEVICE)

    launch_reshape_and_cache_flash(key, key, key_cache, value_cache, loc)

    torch.testing.assert_close(key_cache[[0, 5], 0], key, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

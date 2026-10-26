import itertools
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.kvcache.cache_move import store_k_slots
from sglang.kernels.ops.kvcache.cache_ops import (
    launch_reshape_and_cache_flash,
    reshape_and_cache_flash,
)
from sglang.kernels.ops.kvcache.kvcache import can_use_store_cache, store_cache
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseMHAMainPool
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKOnlyPool,
    MHATokenToKVPool,
    _as_token_head_dim,
    _has_dense_kv_rows,
)
from sglang.srt.utils import is_hip
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


@pytest.fixture
def async_assert_off(monkeypatch):
    # Pool stores probe `loc` only when this is set (CI sets it); the in-kernel
    # bounds must hold without it.
    monkeypatch.setenv("SGLANG_ENABLE_ASYNC_ASSERT", "false")


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


def _random_like_cache(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=dtype, device=DEVICE)
    return torch.randn(shape, dtype=dtype, device=DEVICE)


GUARD = 4
# Slots outside [0, SMALL_CACHE) that stay inside _guarded's sentinel rows.
OUT_OF_RANGE_SLOTS = [-1, -GUARD, SMALL_CACHE, SMALL_CACHE + GUARD - 1]


def _guarded(shape: tuple, dtype: torch.dtype) -> tuple:
    """``(backing, view)``: ``view`` has ``shape``, with GUARD sentinel rows around it.

    An unchecked slot in ``[-GUARD, shape[0] + GUARD)`` lands in a sentinel row, so a
    missing bound shows up as corruption rather than an illegal memory access.
    """
    backing = _random_like_cache((shape[0] + 2 * GUARD, *shape[1:]), dtype)
    return backing, backing[GUARD : GUARD + shape[0]]


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_store_k_slots_skips_out_of_range_slots(index_dtype: torch.dtype) -> None:
    head_num, head_dim = 4, 128
    backing, k_buffer = _guarded((SMALL_CACHE, head_num, head_dim), DTYPE)
    # The first and last slots are in range and must still be written.
    in_range = [0, SMALL_CACHE - 1]
    loc = torch.tensor(in_range + OUT_OF_RANGE_SLOTS, dtype=index_dtype, device=DEVICE)
    src = torch.randn((loc.numel(), head_num, head_dim), dtype=DTYPE, device=DEVICE)
    # A padded or stale row may hold anything; NaN makes an erroneous write visible.
    src[len(in_range) :] = torch.nan
    expected = backing.clone()
    expected[[GUARD + slot for slot in in_range]] = src[: len(in_range)]

    store_k_slots(k_buffer, src, loc)

    torch.testing.assert_close(backing, expected, rtol=0.0, atol=0.0)


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
    # Only the sliced source is not dense, so set_k_buffer copies it first.
    assert _has_dense_kv_rows(cache_k, head_num, head_dim, batch_size) == (
        layout != "head_dim_sliced"
    )
    loc = torch.randperm(SMALL_CACHE, device=DEVICE)[:batch_size]
    expected = pool.k_buffer[0].clone()
    expected[loc] = cache_k.to(index_dtype).view(store_dtype)

    MHATokenToKOnlyPool.set_k_buffer(pool, 0, loc, cache_k)

    torch.testing.assert_close(pool.k_buffer[0], expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("layout", ["contiguous", "head_dim_sliced"])
def test_k_only_pool_set_k_buffer_skips_out_of_range_slots(
    layout: str, async_assert_off
) -> None:
    head_num, head_dim = 1, 128
    backing, k_buffer = _guarded((SMALL_CACHE, head_num, head_dim), DTYPE)
    pool = SimpleNamespace(
        dtype=DTYPE,
        store_dtype=DTYPE,
        head_num=head_num,
        head_dim=head_dim,
        size=SMALL_CACHE - 1,
        page_size=1,
        k_buffer=[k_buffer],
    )
    in_range = [0, SMALL_CACHE - 1]
    loc = torch.tensor(in_range + OUT_OF_RANGE_SLOTS, device=DEVICE)
    width = 2 * head_dim if layout == "head_dim_sliced" else head_dim
    cache_k = torch.randn((loc.numel(), head_num, width), dtype=DTYPE, device=DEVICE)
    cache_k[len(in_range) :] = torch.nan
    cache_k = cache_k[..., :head_dim]
    expected = backing.clone()
    expected[[GUARD + slot for slot in in_range]] = cache_k[: len(in_range)]

    MHATokenToKOnlyPool.set_k_buffer(pool, 0, loc, cache_k)

    torch.testing.assert_close(backing, expected, rtol=0.0, atol=0.0)


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


def test_store_k_slots_accepts_strided_loc() -> None:
    head_num, head_dim, batch_size = 4, 128, 64
    k_buffer = torch.randn(
        (SMALL_CACHE, head_num, head_dim), dtype=DTYPE, device=DEVICE
    )
    src = torch.randn((batch_size, head_num, head_dim), dtype=DTYPE, device=DEVICE)
    loc = torch.randperm(SMALL_CACHE, device=DEVICE)[: 2 * batch_size : 2]
    assert not loc.is_contiguous()
    expected = _ref_scatter(k_buffer, src, loc)

    store_k_slots(k_buffer, src, loc)

    torch.testing.assert_close(k_buffer, expected, rtol=0.0, atol=0.0)


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


@pytest.mark.parametrize("case", ["non_dense_rows", "two_dimensional_loc", "float_loc"])
def test_store_k_slots_rejects_unsupported_inputs(case: str) -> None:
    head_num, head_dim = 4, 128
    k_buffer = torch.randn(
        (SMALL_CACHE, head_num, head_dim), dtype=DTYPE, device=DEVICE
    )
    src = torch.randn((8, head_num, head_dim), dtype=DTYPE, device=DEVICE)
    loc = torch.arange(8, device=DEVICE)
    if case == "non_dense_rows":
        # Sliced along head_dim: the trailing block is no longer a flat run, which
        # the kernel cannot express. set_k_buffer makes such a source contiguous first.
        src = torch.randn((8, head_num, 2 * head_dim), dtype=DTYPE, device=DEVICE)[
            ..., :head_dim
        ]
    elif case == "two_dimensional_loc":
        loc = loc.view(2, 4)
    else:
        loc = loc.float()

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


def _kv_rows(layout: str) -> torch.Tensor:
    """Eight tokens of K in a layout a store can be handed."""
    n, h, d = 8, GUARD_HEAD_NUM, GUARD_HEAD_DIM
    if layout == "qkv_slice":
        return _qkv_k_slice(n)
    if layout == "flat":
        return torch.zeros((n, h * d), dtype=DTYPE, device=DEVICE)
    if layout == "split":
        return torch.zeros((n, h, d), dtype=DTYPE, device=DEVICE)
    if layout == "token_strided":
        return torch.zeros((2 * n, h, d), dtype=DTYPE, device=DEVICE)[::2]
    if layout == "head_dim_sliced":
        return torch.zeros((n, h, 2 * d), dtype=DTYPE, device=DEVICE)[..., :d]
    if layout == "transposed":
        return torch.zeros((n, d, h), dtype=DTYPE, device=DEVICE).transpose(1, 2)
    return torch.zeros((2, n, h, d), dtype=DTYPE, device=DEVICE)[:, ::2]


# (layout, tokens checked, contiguous, dense). Holding more rows than loc is rejected:
# reshape_and_cache_flash sizes its grid from shape[0] and would read loc past its end.
@pytest.mark.parametrize(
    "layout,num_tokens,contiguous,dense",
    [
        ("qkv_slice", 8, False, True),
        ("flat", 8, True, True),
        ("split", 8, True, True),
        ("token_strided", 8, False, True),
        ("qkv_slice", 4, False, False),
        ("flat", 4, True, False),
        ("split", 4, True, False),
        ("head_dim_sliced", 8, False, False),
        ("transposed", 8, False, False),
        ("rank_4", 8, False, False),
    ],
)
def test_dense_kv_rows(
    layout: str, num_tokens: int, contiguous: bool, dense: bool
) -> None:
    t = _kv_rows(layout)
    assert t.numel() == 8 * GUARD_ROW and t.is_contiguous() == contiguous
    assert _dense(t, num_tokens) == dense


def test_as_token_head_dim_does_not_copy() -> None:
    k = _qkv_k_slice(8)
    view = _as_token_head_dim(k, GUARD_HEAD_NUM, GUARD_HEAD_DIM)
    assert tuple(view.shape) == (8, GUARD_HEAD_NUM, GUARD_HEAD_DIM)
    assert view.data_ptr() == k.data_ptr()
    assert view.stride() == (QKV_WIDTH, GUARD_HEAD_DIM, 1)


class _PoolStub:
    """The pool state can_store_kv_fused_cast and store_kv_fused_cast read."""

    requires_physical_write_loc = False
    _check_physical_write_loc = MHATokenToKVPool._check_physical_write_loc

    def __init__(self, buffer_dims: int = 3, dtype: torch.dtype = torch.float8_e4m3fn):
        self.kv_cache_layout = "nhd"
        self.use_hnd = False
        self.is_quantized_kv_cache = False
        self.dtype = dtype
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


def test_can_store_kv_fused_cast_rejects_scales_that_divide() -> None:
    assert not _can_store(_PoolStub(), k_scale=0.5)
    assert not _can_store(_PoolStub(), v_scale=0.5)
    # A tensor scale cannot be inspected without a sync, so it must divide.
    assert not _can_store(_PoolStub(), k_scale=torch.ones(1, device=DEVICE))


class _SetKVPoolStub(_PoolStub):
    """Adds the state set_kv_buffer's unfused store reads."""

    _store_kv_layer = MHATokenToKVPool._store_kv_layer

    def __init__(self) -> None:
        super().__init__()
        self.row_dim = self.v_row_dim = GUARD_ROW
        self.device_module = torch.cuda
        self.alt_stream = None


@pytest.mark.parametrize("scale_kind", ["host_one", "host", "tensor"])
def test_set_kv_buffer_divides_unless_scale_is_host_one(scale_kind: str) -> None:
    # A host 1.0 is never divided, so the caller's K/V stay untouched; host and tensor
    # scales divide. Dividing by 1.0 can clear a NaN's sign, so NaN need only stay NaN.
    pool = _SetKVPoolStub()
    values = torch.arange(-(2**15), 2**15, dtype=torch.int32, device=DEVICE)
    num_tokens = values.numel() // GUARD_ROW
    cache_k = (
        values.to(torch.int16)
        .view(DTYPE)
        .view(num_tokens, GUARD_HEAD_NUM, GUARD_HEAD_DIM)
    )
    cache_v = cache_k.flip(0)
    scale = {
        "host_one": 1.0,
        "host": 2.0,
        "tensor": torch.tensor(2.0, device=DEVICE),
    }[scale_kind]
    divisor = 1.0 if scale_kind == "host_one" else 2.0
    sources = (cache_k, cache_v)
    expected = [(src / divisor).to(pool.dtype).view(torch.uint8) for src in sources]
    src_nan = [src.isnan() for src in sources]
    versions = (cache_k._version, cache_v._version)
    # store_cache skips the reserved padding slot 0.
    loc = torch.arange(1, num_tokens + 1, dtype=torch.int64, device=DEVICE)

    MHATokenToKVPool.set_kv_buffer(
        pool,
        SimpleNamespace(layer_id=0),
        loc,
        cache_k,
        cache_v,
        k_scale=scale,
        v_scale=scale,
    )

    for buffer, want, nan in zip(
        (pool.k_buffer[0], pool.v_buffer[0]), expected, src_nan
    ):
        got = buffer[loc]
        torch.testing.assert_close(got[~nan], want[~nan], rtol=0.0, atol=0.0)
        assert torch.all(_fp8_isnan(got[nan], pool.dtype)), "NaN must stay NaN"
    if scale_kind == "host_one":
        assert (cache_k._version, cache_v._version) == versions


@pytest.mark.parametrize(
    "buffer_dims,attr,value",
    [
        # store_kv_fused_cast presents the buffer as page-size 1 via unsqueeze(1),
        # which mis-indexes a buffer that is already paged.
        (4, None, None),
        (3, "dtype", torch.float8_e5m2),
        (3, "dtype", DTYPE),  # source already in cache dtype
        (3, "v_head_dim", GUARD_HEAD_DIM // 2),
        (3, "kv_cache_layout", "hnd"),
        (3, "use_hnd", True),
        (3, "is_quantized_kv_cache", True),
        (3, "store_dtype", DTYPE),
    ],
)
def test_can_store_kv_fused_cast_rejects_unsupported_pool(
    buffer_dims: int, attr, value
) -> None:
    pool = _PoolStub(buffer_dims=buffer_dims)
    if attr is not None:
        setattr(pool, attr, value)
    assert not _can_store(pool)


def test_can_store_kv_fused_cast_rejects_hisparse_main_pool() -> None:
    # The fused store would skip HiSparse's logical-to-device slot translation.
    loc = torch.arange(8, dtype=torch.int64, device=DEVICE)
    k, v = _qkv_k_slice(8), _qkv_k_slice(8)
    assert MHATokenToKVPool.can_store_kv_fused_cast(_PoolStub(), 0, loc, k, v)
    assert not HiSparseMHAMainPool.can_store_kv_fused_cast(_PoolStub(), 0, loc, k, v)
    with pytest.raises(NotImplementedError):
        HiSparseMHAMainPool.store_kv_fused_cast(_PoolStub(), 0, loc, k, v)


def test_store_kv_fused_cast_rejects_unmarked_loc_on_physical_id_pool() -> None:
    # As in set_kv_buffer: a pool that takes physical ids rejects an unmarked loc.
    pool = _PoolStub()
    pool.requires_physical_write_loc = True
    loc = torch.arange(1, 9, dtype=torch.int64, device=DEVICE)
    k, v = _qkv_k_slice(8), _qkv_k_slice(8)
    before_k, before_v = pool.k_buffer[0].clone(), pool.v_buffer[0].clone()

    with pytest.raises(ValueError):
        MHATokenToKVPool.store_kv_fused_cast(pool, 0, loc, k, v)

    torch.testing.assert_close(pool.k_buffer[0], before_k, rtol=0.0, atol=0.0)
    torch.testing.assert_close(pool.v_buffer[0], before_v, rtol=0.0, atol=0.0)


FP8_KV_DTYPES = [
    torch.float8_e4m3fn,
    pytest.param(
        torch.float8_e4m3fnuz,
        marks=pytest.mark.skipif(
            not is_hip(), reason="float8_e4m3fnuz is the ROCm gfx942 KV-cache dtype"
        ),
    ),
]


def _fp8_isnan(raw: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return raw.view(dtype).float().isnan()


@pytest.mark.parametrize("dtype", FP8_KV_DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "qkv_slice"])
def test_store_kv_fused_cast_every_bf16_value(layout: str, dtype: torch.dtype) -> None:
    # Byte-identical to `.to(fp8)` wherever that is finite. Where ATen overflows to
    # NaN, the cast saturates to the fp8 max; fnuz may keep NaN (gfx950 does for inf).
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
    pool = _PoolStub(dtype=dtype)
    # Slot 0 is the reserved padding slot, which the store skips.
    loc = torch.randperm(SMALL_CACHE - 1, device=DEVICE)[:num_tokens] + 1
    before_k, before_v = pool.k_buffer[0].clone(), pool.v_buffer[0].clone()

    MHATokenToKVPool.store_kv_fused_cast(pool, 0, loc, cache_k, cache_v)

    fp8_max = torch.finfo(dtype).max
    saturated = torch.tensor([fp8_max, -fp8_max], device=DEVICE).to(dtype)
    pos_max, neg_max = saturated.view(torch.uint8).tolist()
    untouched = torch.ones(SMALL_CACHE, dtype=torch.bool, device=DEVICE)
    untouched[loc] = False
    for src, buffer, before in (
        (k_rows, pool.k_buffer[0], before_k),
        (v_rows, pool.v_buffer[0], before_v),
    ):
        got = buffer[loc].view(num_tokens, GUARD_ROW)
        aten = src.to(dtype).view(torch.uint8)
        src_nan = src.isnan()
        aten_nan = _fp8_isnan(aten, dtype)
        got_nan = _fp8_isnan(got, dtype)
        torch.testing.assert_close(got[~aten_nan], aten[~aten_nan], rtol=0, atol=0)
        overflow = aten_nan & ~src_nan
        want = torch.where(src < 0, neg_max, pos_max).to(torch.uint8)
        overflow_ok = got == want
        if dtype == torch.float8_e4m3fnuz:
            overflow_ok |= got_nan
        assert torch.all(overflow_ok[overflow]), "unexpected byte where .to() overflows"
        assert torch.all(got_nan[src_nan]), "NaN must stay NaN"
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


def test_store_kv_fused_cast_skips_out_of_range_slots(async_assert_off) -> None:
    pool = _PoolStub()
    shape = tuple(pool.k_buffer[0].shape)
    k_backing, pool.k_buffer[0] = _guarded(shape, torch.uint8)
    v_backing, pool.v_buffer[0] = _guarded(shape, torch.uint8)
    assert pool.size + pool.page_size == shape[0]
    # Slot 0 is reserved, so slot 1 is the first one written.
    in_range = [1, SMALL_CACHE - 1]
    loc = torch.tensor(in_range + OUT_OF_RANGE_SLOTS, device=DEVICE)
    rows = (loc.numel(), GUARD_HEAD_NUM, GUARD_HEAD_DIM)
    cache_k = torch.randn(rows, dtype=DTYPE, device=DEVICE)
    cache_v = torch.randn(rows, dtype=DTYPE, device=DEVICE)
    expected_k, expected_v = k_backing.clone(), v_backing.clone()
    for src, expected in ((cache_k, expected_k), (cache_v, expected_v)):
        src[len(in_range) :] = torch.nan
        written = src[: len(in_range)].to(pool.dtype).view(torch.uint8)
        expected[[GUARD + slot for slot in in_range]] = written

    MHATokenToKVPool.store_kv_fused_cast(pool, 0, loc, cache_k, cache_v)

    torch.testing.assert_close(k_backing, expected_k, rtol=0.0, atol=0.0)
    torch.testing.assert_close(v_backing, expected_v, rtol=0.0, atol=0.0)


def test_reshape_and_cache_flash_kernel_launches_with_original_arguments() -> None:
    # The raw kernel is public: the slot-skip and bound parameters must stay
    # optional and off by default.
    key = torch.randn((2, GUARD_HEAD_NUM, GUARD_HEAD_DIM), dtype=DTYPE, device=DEVICE)
    key_cache = torch.zeros(
        (SMALL_CACHE, 1, GUARD_HEAD_NUM, GUARD_HEAD_DIM), dtype=DTYPE, device=DEVICE
    )
    value_cache = torch.zeros_like(key_cache)
    loc = torch.tensor([0, 5], dtype=torch.int64, device=DEVICE)

    reshape_and_cache_flash[(loc.numel(), 1)](
        key,
        key,
        key_cache,
        value_cache,
        loc,
        None,
        key,
        key,
        key_cache.stride(0),
        value_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(1),
        value_cache.stride(2),
        key.stride(0),
        key.stride(0),
        GUARD_HEAD_NUM,
        GUARD_HEAD_DIM,
        key_cache.shape[1],
        HEAD_BLOCK=4,
        BLOCK_D=GUARD_HEAD_DIM,
        HAS_SWA=False,
        USE_SCALE=False,
    )

    torch.testing.assert_close(key_cache[loc, 0], key, rtol=0.0, atol=0.0)
    torch.testing.assert_close(value_cache[loc, 0], key, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="default"),
        pytest.param({"size_limit": SMALL_CACHE}, id="size_limit"),
        pytest.param({"skip_slot_zero": True}, id="skip_slot_zero"),
        pytest.param(
            {"skip_slot_zero": True, "size_limit": SMALL_CACHE},
            id="skip_slot_zero_and_size_limit",
        ),
    ],
)
def test_reshape_and_cache_flash_writes_only_allowed_slots(kwargs: dict) -> None:
    # The default writes slot 0, which the AITER backend's KV write relies on; a
    # bound at the slot count still writes the last slot and skips any past it.
    shape = (SMALL_CACHE, 1, GUARD_HEAD_NUM, GUARD_HEAD_DIM)
    k_backing, key_cache = _guarded(shape, DTYPE)
    v_backing, value_cache = _guarded(shape, DTYPE)
    in_range = [0, 5, SMALL_CACHE - 1]
    out_of_range = OUT_OF_RANGE_SLOTS if "size_limit" in kwargs else []
    loc = torch.tensor(in_range + out_of_range, dtype=torch.int64, device=DEVICE)
    key = torch.randn((loc.numel(), *shape[2:]), dtype=DTYPE, device=DEVICE)
    key[len(in_range) :] = torch.nan
    skip_zero = kwargs.get("skip_slot_zero", False)
    written = [i for i, slot in enumerate(in_range) if slot != 0 or not skip_zero]
    expected_k, expected_v = k_backing.clone(), v_backing.clone()
    for expected in (expected_k, expected_v):
        expected[[GUARD + in_range[i] for i in written], 0] = key[written]

    launch_reshape_and_cache_flash(key, key, key_cache, value_cache, loc, **kwargs)

    torch.testing.assert_close(k_backing, expected_k, rtol=0.0, atol=0.0)
    torch.testing.assert_close(v_backing, expected_v, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "case", ["size_limit_past_cache", "size_limit_with_swa", "strided_slot_mapping"]
)
def test_reshape_and_cache_flash_rejects_unsupported_arguments(case: str) -> None:
    key = torch.randn((2, GUARD_HEAD_NUM, GUARD_HEAD_DIM), dtype=DTYPE, device=DEVICE)
    key_cache = torch.zeros(
        (SMALL_CACHE, 1, GUARD_HEAD_NUM, GUARD_HEAD_DIM), dtype=DTYPE, device=DEVICE
    )
    loc = torch.zeros(2, dtype=torch.int64, device=DEVICE)
    kwargs = {}
    if case == "size_limit_past_cache":
        kwargs["size_limit"] = SMALL_CACHE + 1
    elif case == "size_limit_with_swa":
        kwargs["swa_slot_mapping"] = torch.zeros(
            SMALL_CACHE, dtype=torch.int64, device=DEVICE
        )
        kwargs["size_limit"] = SMALL_CACHE
    else:
        loc = torch.zeros(4, dtype=torch.int64, device=DEVICE)[::2]

    with pytest.raises(AssertionError):
        launch_reshape_and_cache_flash(
            key, key, key_cache, key_cache.clone(), loc, **kwargs
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

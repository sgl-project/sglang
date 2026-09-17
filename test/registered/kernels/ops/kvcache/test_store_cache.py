import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.kvcache.cache_move import store_k_slots
from sglang.kernels.ops.kvcache.kvcache import can_use_store_cache, store_cache
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=34, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Nightly is not redundant here: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=46, stage="nightly", runner_config="1-gpu-large")
register_amd_ci(est_time=62, stage="jit-kernel-unit", runner_config="amd")

BS_LIST = [2**n for n in range(0, 15)]
BS_LIST += [x + 1 + i for i, x in enumerate(BS_LIST)]
BS_LIST = get_ci_test_range(BS_LIST, [1, 9, 256, 16399])
HIDDEN_DIMS = get_ci_test_range(
    [64, 128, 256, 512, 1024, 96, 98, 100], [64, 512, 1024, 98]
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

    # AOT store cache
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
@pytest.mark.parametrize("num_split", [1, 2, 4])
def test_store_cache_reserved_skip_index(
    index_dtype: torch.dtype, num_split: int
) -> None:
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
        num_split=num_split,
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


def _valid_num_splits(element_dim: int, dtype: torch.dtype) -> list:
    """Return the list of valid num_split values for a given element_dim/dtype."""
    row_bytes = element_dim * dtype.itemsize
    splits = [1]
    if row_bytes % (2 * 128) == 0:
        splits.append(2)
    if row_bytes % (4 * 128) == 0:
        splits.append(4)
    return splits


_NUM_SPLIT_CASES = [
    (_dim, _ns, _dtype)
    for _dtype in [torch.float16, torch.bfloat16, torch.float32]
    for _dim in REPR_DIMS
    for _ns in _valid_num_splits(_dim, _dtype)
]


@pytest.mark.parametrize("element_dim,num_split,dtype", _NUM_SPLIT_CASES)
def test_store_cache_num_split(
    element_dim: int, num_split: int, dtype: torch.dtype
) -> None:
    batch_size = 128
    k = torch.randn((batch_size, element_dim), dtype=dtype, device=DEVICE)
    v = torch.randn((batch_size, element_dim), dtype=dtype, device=DEVICE)
    k_cache = torch.randn((SMALL_CACHE, element_dim), dtype=dtype, device=DEVICE)
    v_cache = torch.randn((SMALL_CACHE, element_dim), dtype=dtype, device=DEVICE)
    indices = torch.randperm(SMALL_CACHE - 1, device=DEVICE)[:batch_size] + 1

    # Verify each num_split kernel path (1, 2, 4) produces correct results
    store_cache(k, v, k_cache, v_cache, indices, num_split=num_split)

    assert torch.all(k_cache[indices] == k)
    assert torch.all(v_cache[indices] == v)


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


def _valid_asym_num_splits(k_dim: int, v_dim: int, dtype: torch.dtype) -> list:
    """num_split values valid for BOTH rows; a split must divide each of them."""
    k_bytes, v_bytes = k_dim * dtype.itemsize, v_dim * dtype.itemsize
    splits = [1]
    if k_bytes % (2 * 128) == 0 and v_bytes % (2 * 128) == 0:
        splits.append(2)
    if k_bytes % (4 * 128) == 0 and v_bytes % (4 * 128) == 0:
        splits.append(4)
    return splits


def _default_num_split(k_dim: int, v_dim: int, dtype: torch.dtype) -> int:
    """Mirrors the heuristic in store_cache(); the default is already exercised
    by test_store_cache_asymmetric, which does not pass num_split."""
    k_bytes, v_bytes = k_dim * dtype.itemsize, v_dim * dtype.itemsize
    if k_bytes % 2048 == 0 and v_bytes % 2048 == 0:
        return 4
    if k_bytes % 1024 == 0 and v_bytes % 1024 == 0:
        return 2
    return 1


# Only splits the default heuristic would NOT pick: the split gate is two-sided
# (K and V must both align), so the off-default branches are what needs pinning.
_ASYM_NUM_SPLIT_CASES = [
    (_k, _v, _ns)
    for _k, _v in ASYM_DIM_PAIRS
    for _ns in _valid_asym_num_splits(_k, _v, DTYPE)
    if _ns != _default_num_split(_k, _v, DTYPE)
]


@pytest.mark.parametrize("k_dim,v_dim,num_split", _ASYM_NUM_SPLIT_CASES)
def test_store_cache_asymmetric_num_split(
    k_dim: int, v_dim: int, num_split: int
) -> None:
    batch_size = 128
    k = torch.randn((batch_size, k_dim), dtype=DTYPE, device=DEVICE)
    v = torch.randn((batch_size, v_dim), dtype=DTYPE, device=DEVICE)
    k_cache = torch.randn((SMALL_CACHE, k_dim), dtype=DTYPE, device=DEVICE)
    v_cache = torch.randn((SMALL_CACHE, v_dim), dtype=DTYPE, device=DEVICE)
    indices = torch.randperm(SMALL_CACHE - 1, device=DEVICE)[:batch_size] + 1

    store_cache(k, v, k_cache, v_cache, indices, num_split=num_split)

    assert torch.all(k_cache[indices] == k)
    assert torch.all(v_cache[indices] == v)


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


def test_store_k_slots_leaves_other_slots_untouched() -> None:
    head_num, head_dim = 8, 128
    batch_size = 64
    src = torch.randn((batch_size, head_num, head_dim), dtype=DTYPE, device=DEVICE)
    k_buffer = torch.randn(
        (SMALL_CACHE, head_num, head_dim), dtype=DTYPE, device=DEVICE
    )
    before = k_buffer.clone()
    loc = torch.randperm(SMALL_CACHE, device=DEVICE)[:batch_size]

    store_k_slots(k_buffer, src, loc)

    # A wrong slot stride would corrupt neighbours, which target-slot checks miss.
    untouched = torch.ones(SMALL_CACHE, dtype=torch.bool, device=DEVICE)
    untouched[loc] = False
    torch.testing.assert_close(
        k_buffer[untouched], before[untouched], rtol=0.0, atol=0.0
    )


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_store_k_slots_dtypes(dtype: torch.dtype) -> None:
    head_num, head_dim = 4, 128
    batch_size = 32
    src = torch.randn((batch_size, head_num, head_dim), dtype=dtype, device=DEVICE)
    k_buffer = torch.randn((SMALL_CACHE, head_num, head_dim), dtype=dtype, device=DEVICE)
    loc = torch.randperm(SMALL_CACHE, device=DEVICE)[:batch_size]
    expected = _ref_scatter(k_buffer, src, loc)

    store_k_slots(k_buffer, src, loc)

    torch.testing.assert_close(k_buffer, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

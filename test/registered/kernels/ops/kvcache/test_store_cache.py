import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.kvcache.kvcache import (
    can_use_store_cache,
    can_use_store_cache_quant,
    is_store_cache_quant_aligned,
    store_cache,
    store_cache_quant,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=28, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Nightly is not redundant here: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=40, stage="nightly", runner_config="1-gpu-large")
register_amd_ci(est_time=55, stage="jit-kernel-unit", runner_config="amd")

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
SMALL_CACHE = 4097  # 4096 usable slots plus reserved CUDA-graph padding slot 0


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


# --- store_cache_quant (fused FP8 quantize + store) ---

QUANT_DST = torch.float8_e4m3fn
QUANT_SRC_DTYPES_TESTED = [torch.bfloat16, torch.float16, torch.float32]
requires_cuda_quant = pytest.mark.skipif(
    torch.version.hip is not None, reason="store_cache_quant is CUDA-only"
)
# 104 is not a multiple of vec_width * warp_size -> exercises the epilogue tail
QUANT_DIMS = get_ci_test_range([64, 128, 512, 1024, 96, 104], [64, 1024, 104])
QUANT_BS = get_ci_test_range([1, 7, 128, 4096], [1, 128])


@requires_cuda_quant
def test_store_cache_quant_empty_batch() -> None:
    k = torch.empty((0, 64), dtype=DTYPE, device=DEVICE)
    v = torch.empty_like(k)
    k_cache = torch.zeros((1, 64), dtype=QUANT_DST, device=DEVICE)
    v_cache = torch.zeros_like(k_cache)

    store_cache_quant(
        k,
        v,
        k_cache,
        v_cache,
        torch.empty(0, dtype=torch.int64, device=DEVICE),
    )

    assert k_cache.view(torch.uint8).sum().item() == 0
    assert v_cache.view(torch.uint8).sum().item() == 0


def _quant_ref(x: torch.Tensor, inv_scale: torch.Tensor) -> torch.Tensor:
    """fp32 multiply by the reciprocal, clip to the finite range, RNE convert —
    the kernel's documented conversion order."""
    fp8_max = torch.finfo(QUANT_DST).max
    xf = x.float() * inv_scale
    return torch.clamp(xf, -fp8_max, fp8_max).to(QUANT_DST)


@pytest.mark.parametrize("src_dtype", QUANT_SRC_DTYPES_TESTED)
@pytest.mark.parametrize(
    "batch_size,element_dim",
    list(itertools.product(QUANT_BS, QUANT_DIMS)),
)
@requires_cuda_quant
def test_store_cache_quant(
    batch_size: int, element_dim: int, src_dtype: torch.dtype
) -> None:
    assert can_use_store_cache_quant(element_dim, src_dtype, QUANT_DST)
    k = torch.randn((batch_size, element_dim), dtype=src_dtype, device=DEVICE) * 3
    v = torch.randn((batch_size, element_dim), dtype=src_dtype, device=DEVICE) * 3
    k_cache = torch.zeros((SMALL_CACHE, element_dim), dtype=QUANT_DST, device=DEVICE)
    v_cache = torch.zeros((SMALL_CACHE, element_dim), dtype=QUANT_DST, device=DEVICE)
    indices = torch.randperm(SMALL_CACHE - 1, device=DEVICE)[:batch_size] + 1

    store_cache_quant(k, v, k_cache, v_cache, indices)

    one = torch.ones((), dtype=torch.float32, device=DEVICE)
    k_ref = _quant_ref(k, one)
    v_ref = _quant_ref(v, one)
    assert torch.equal(k_cache[indices].view(torch.uint8), k_ref.view(torch.uint8))
    assert torch.equal(v_cache[indices].view(torch.uint8), v_ref.view(torch.uint8))
    # untouched rows stay zero (scatter must not smear across rows)
    mask = torch.ones(SMALL_CACHE, dtype=torch.bool, device=DEVICE)
    mask[indices] = False
    assert k_cache.view(torch.uint8)[mask].sum().item() == 0


@pytest.mark.parametrize("scale_form", ["tensor", "host_reciprocal"])
@requires_cuda_quant
def test_store_cache_quant_scale_forms(scale_form: str) -> None:
    """The kernel accepts scales as a device scalar (read on GPU, no host sync)
    or a host-precomputed reciprocal; both must scale before the FP8 convert."""
    batch_size, element_dim = 128, 1024
    k_scale, v_scale = 1.7, 0.9
    k = torch.randn((batch_size, element_dim), dtype=DTYPE, device=DEVICE) * 3
    v = torch.randn((batch_size, element_dim), dtype=DTYPE, device=DEVICE) * 3
    k_cache = torch.zeros((SMALL_CACHE, element_dim), dtype=QUANT_DST, device=DEVICE)
    v_cache = torch.zeros((SMALL_CACHE, element_dim), dtype=QUANT_DST, device=DEVICE)
    indices = torch.randperm(SMALL_CACHE - 1, device=DEVICE)[:batch_size] + 1

    if scale_form == "tensor":
        k_scale_t = torch.tensor([k_scale], dtype=torch.float32, device=DEVICE)
        v_scale_t = torch.tensor([v_scale], dtype=torch.float32, device=DEVICE)
        store_cache_quant(k, v, k_cache, v_cache, indices, k_scale_t, v_scale_t)
        k_inv = 1.0 / k_scale_t
        v_inv = 1.0 / v_scale_t
    else:
        store_cache_quant(
            k,
            v,
            k_cache,
            v_cache,
            indices,
            k_inv_scale=1.0 / k_scale,
            v_inv_scale=1.0 / v_scale,
        )
        k_inv = torch.tensor(1.0 / k_scale, dtype=torch.float32, device=DEVICE)
        v_inv = torch.tensor(1.0 / v_scale, dtype=torch.float32, device=DEVICE)

    k_ref = _quant_ref(k, k_inv)
    v_ref = _quant_ref(v, v_inv)
    assert torch.equal(k_cache[indices].view(torch.uint8), k_ref.view(torch.uint8))
    assert torch.equal(v_cache[indices].view(torch.uint8), v_ref.view(torch.uint8))


@requires_cuda_quant
def test_store_cache_quant_does_not_mutate_inputs() -> None:
    """Unlike the eager quantize path (in-place div_), the fused kernel must
    leave k/v untouched — callers reuse them for the attention compute."""
    k = torch.randn((16, 1024), dtype=DTYPE, device=DEVICE)
    v = torch.randn((16, 1024), dtype=DTYPE, device=DEVICE)
    k_orig, v_orig = k.clone(), v.clone()
    k_cache = torch.zeros((SMALL_CACHE, 1024), dtype=QUANT_DST, device=DEVICE)
    v_cache = torch.zeros((SMALL_CACHE, 1024), dtype=QUANT_DST, device=DEVICE)
    indices = torch.randperm(SMALL_CACHE - 1, device=DEVICE)[:16] + 1
    scale = torch.tensor([1.7], dtype=torch.float32, device=DEVICE)

    store_cache_quant(k, v, k_cache, v_cache, indices, scale, scale)

    assert torch.equal(k, k_orig)
    assert torch.equal(v, v_orig)


@requires_cuda_quant
def test_store_cache_quant_rejects_misaligned_source_rows() -> None:
    storage_k = torch.randn(259, dtype=torch.float16, device=DEVICE)
    storage_v = torch.randn(259, dtype=torch.float16, device=DEVICE)
    k = torch.as_strided(storage_k, (4, 1, 64), (65, 64, 1)).view(-1, 64)
    v = torch.as_strided(storage_v, (4, 1, 64), (65, 64, 1)).view(-1, 64)
    k_cache = torch.zeros((SMALL_CACHE, 64), dtype=QUANT_DST, device=DEVICE)
    v_cache = torch.zeros_like(k_cache)
    indices = torch.arange(1, 5, dtype=torch.int64, device=DEVICE)

    assert not is_store_cache_quant_aligned(k, v)
    with pytest.raises(ValueError, match="16-byte-aligned"):
        store_cache_quant(k, v, k_cache, v_cache, indices)


@requires_cuda_quant
def test_store_cache_quant_clips_out_of_range() -> None:
    """Values beyond the finite FP8 range must saturate to +-448, not overflow
    to NaN (the eager .to(fp8) path NaNs; the kernel clips first)."""
    fp8_max = torch.finfo(QUANT_DST).max
    k = torch.full((1, 64), 30000.0, dtype=DTYPE, device=DEVICE)
    v = torch.full((1, 64), -30000.0, dtype=DTYPE, device=DEVICE)
    k_cache = torch.zeros((SMALL_CACHE, 64), dtype=QUANT_DST, device=DEVICE)
    v_cache = torch.zeros((SMALL_CACHE, 64), dtype=QUANT_DST, device=DEVICE)
    indices = torch.tensor([3], device=DEVICE)

    store_cache_quant(k, v, k_cache, v_cache, indices)

    assert torch.all(k_cache[indices].float() == fp8_max)
    assert torch.all(v_cache[indices].float() == -fp8_max)


@requires_cuda_quant
def test_store_cache_quant_int32_indices() -> None:
    k = torch.randn((64, 512), dtype=DTYPE, device=DEVICE)
    v = torch.randn((64, 512), dtype=DTYPE, device=DEVICE)
    k_cache = torch.zeros((SMALL_CACHE, 512), dtype=QUANT_DST, device=DEVICE)
    v_cache = torch.zeros((SMALL_CACHE, 512), dtype=QUANT_DST, device=DEVICE)
    # int32 indices exercise a different CUDA template instantiation than default int64
    indices = (torch.randperm(SMALL_CACHE - 1, device=DEVICE)[:64] + 1).to(torch.int32)

    store_cache_quant(k, v, k_cache, v_cache, indices)

    one = torch.ones((), dtype=torch.float32, device=DEVICE)
    assert torch.equal(
        k_cache[indices.long()].view(torch.uint8), _quant_ref(k, one).view(torch.uint8)
    )
    assert torch.equal(
        v_cache[indices.long()].view(torch.uint8), _quant_ref(v, one).view(torch.uint8)
    )


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("reserved_skip_index", [None, -1])
@requires_cuda_quant
def test_store_cache_quant_reserved_skip_index(
    index_dtype: torch.dtype, reserved_skip_index: int | None
) -> None:
    element_dim = 64
    k = torch.randn((2, element_dim), dtype=DTYPE, device=DEVICE)
    v = torch.randn((2, element_dim), dtype=DTYPE, device=DEVICE)
    k_cache = torch.zeros((SMALL_CACHE, element_dim), dtype=QUANT_DST, device=DEVICE)
    v_cache = torch.zeros_like(k_cache)
    indices = torch.tensor([0, 3], dtype=index_dtype, device=DEVICE)
    one = torch.ones((), dtype=torch.float32, device=DEVICE)

    if reserved_skip_index is None:
        store_cache_quant(k, v, k_cache, v_cache, indices)
    else:
        store_cache_quant(
            k,
            v,
            k_cache,
            v_cache,
            indices,
            reserved_skip_index=reserved_skip_index,
        )

    if reserved_skip_index is None:
        assert torch.equal(
            k_cache[0].view(torch.uint8),
            torch.zeros(element_dim, dtype=torch.uint8, device=DEVICE),
        )
        assert torch.equal(
            v_cache[0].view(torch.uint8),
            torch.zeros(element_dim, dtype=torch.uint8, device=DEVICE),
        )
    else:
        assert torch.equal(
            k_cache[0].view(torch.uint8), _quant_ref(k[:1], one).view(torch.uint8)[0]
        )
        assert torch.equal(
            v_cache[0].view(torch.uint8), _quant_ref(v[:1], one).view(torch.uint8)[0]
        )
    assert torch.equal(
        k_cache[3].view(torch.uint8), _quant_ref(k[1:], one).view(torch.uint8)[0]
    )
    assert torch.equal(
        v_cache[3].view(torch.uint8), _quant_ref(v[1:], one).view(torch.uint8)[0]
    )


@requires_cuda_quant
def test_can_use_store_cache_quant() -> None:
    assert can_use_store_cache_quant(1024, torch.bfloat16, QUANT_DST)
    assert can_use_store_cache_quant(104, torch.bfloat16, QUANT_DST)
    # row not a multiple of the vector width (16B / itemsize)
    assert not can_use_store_cache_quant(100, torch.bfloat16, QUANT_DST)
    # unsupported source/destination dtypes fall back to the unfused path
    assert not can_use_store_cache_quant(1024, torch.bfloat16, torch.float8_e5m2)
    assert not can_use_store_cache_quant(1024, torch.int8, QUANT_DST)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

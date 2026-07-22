import itertools
import sys

import pytest
import torch

from sglang.jit_kernel.kvcache import (
    can_use_store_cache,
    can_use_store_cache_quant,
    store_cache,
    store_cache_quant,
)
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=28, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Nightly is not redundant here: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)
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
    indices = torch.randperm(CACHE_SIZE, device=DEVICE)[:batch_size]

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
    indices = torch.randperm(SMALL_CACHE, device=DEVICE)[:batch_size]

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
    indices = torch.randperm(SMALL_CACHE, device=DEVICE)[:batch_size].to(torch.int32)

    store_cache(k, v, k_cache, v_cache, indices)

    assert torch.all(k_cache[indices.long()] == k)
    assert torch.all(v_cache[indices.long()] == v)


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
    indices = torch.randperm(SMALL_CACHE, device=DEVICE)[:batch_size]

    # Verify each num_split kernel path (1, 2, 4) produces correct results
    store_cache(k, v, k_cache, v_cache, indices, num_split=num_split)

    assert torch.all(k_cache[indices] == k)
    assert torch.all(v_cache[indices] == v)


def test_can_use_store_cache() -> None:
    assert can_use_store_cache(128)
    assert can_use_store_cache(256)
    assert can_use_store_cache(1024)
    assert can_use_store_cache(2048)


# --- store_cache_quant (fused FP8 quantize + store) ---

QUANT_DST = torch.float8_e4m3fn
QUANT_SRC_DTYPES_TESTED = [torch.bfloat16, torch.float16, torch.float32]
# 104 is not a multiple of vec_width * warp_size -> exercises the epilogue tail
QUANT_DIMS = get_ci_test_range([64, 128, 512, 1024, 96, 104], [64, 1024, 104])
QUANT_BS = get_ci_test_range([1, 7, 128, 4096], [1, 128])


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
def test_store_cache_quant(
    batch_size: int, element_dim: int, src_dtype: torch.dtype
) -> None:
    assert can_use_store_cache_quant(element_dim, src_dtype, QUANT_DST)
    k = torch.randn((batch_size, element_dim), dtype=src_dtype, device=DEVICE) * 3
    v = torch.randn((batch_size, element_dim), dtype=src_dtype, device=DEVICE) * 3
    k_cache = torch.zeros((SMALL_CACHE, element_dim), dtype=QUANT_DST, device=DEVICE)
    v_cache = torch.zeros((SMALL_CACHE, element_dim), dtype=QUANT_DST, device=DEVICE)
    indices = torch.randperm(SMALL_CACHE, device=DEVICE)[:batch_size]

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
def test_store_cache_quant_scale_forms(scale_form: str) -> None:
    """The kernel accepts scales as a device scalar (read on GPU, no host sync)
    or a host-precomputed reciprocal; both must scale before the FP8 convert."""
    batch_size, element_dim = 128, 1024
    k_scale, v_scale = 1.7, 0.9
    k = torch.randn((batch_size, element_dim), dtype=DTYPE, device=DEVICE) * 3
    v = torch.randn((batch_size, element_dim), dtype=DTYPE, device=DEVICE) * 3
    k_cache = torch.zeros((SMALL_CACHE, element_dim), dtype=QUANT_DST, device=DEVICE)
    v_cache = torch.zeros((SMALL_CACHE, element_dim), dtype=QUANT_DST, device=DEVICE)
    indices = torch.randperm(SMALL_CACHE, device=DEVICE)[:batch_size]

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


def test_store_cache_quant_does_not_mutate_inputs() -> None:
    """Unlike the eager quantize path (in-place div_), the fused kernel must
    leave k/v untouched — callers reuse them for the attention compute."""
    k = torch.randn((16, 1024), dtype=DTYPE, device=DEVICE)
    v = torch.randn((16, 1024), dtype=DTYPE, device=DEVICE)
    k_orig, v_orig = k.clone(), v.clone()
    k_cache = torch.zeros((SMALL_CACHE, 1024), dtype=QUANT_DST, device=DEVICE)
    v_cache = torch.zeros((SMALL_CACHE, 1024), dtype=QUANT_DST, device=DEVICE)
    indices = torch.randperm(SMALL_CACHE, device=DEVICE)[:16]
    scale = torch.tensor([1.7], dtype=torch.float32, device=DEVICE)

    store_cache_quant(k, v, k_cache, v_cache, indices, scale, scale)

    assert torch.equal(k, k_orig)
    assert torch.equal(v, v_orig)


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


def test_store_cache_quant_int32_indices() -> None:
    k = torch.randn((64, 512), dtype=DTYPE, device=DEVICE)
    v = torch.randn((64, 512), dtype=DTYPE, device=DEVICE)
    k_cache = torch.zeros((SMALL_CACHE, 512), dtype=QUANT_DST, device=DEVICE)
    v_cache = torch.zeros((SMALL_CACHE, 512), dtype=QUANT_DST, device=DEVICE)
    # int32 indices exercise a different CUDA template instantiation than default int64
    indices = torch.randperm(SMALL_CACHE, device=DEVICE)[:64].to(torch.int32)

    store_cache_quant(k, v, k_cache, v_cache, indices)

    one = torch.ones((), dtype=torch.float32, device=DEVICE)
    assert torch.equal(
        k_cache[indices.long()].view(torch.uint8), _quant_ref(k, one).view(torch.uint8)
    )
    assert torch.equal(
        v_cache[indices.long()].view(torch.uint8), _quant_ref(v, one).view(torch.uint8)
    )


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

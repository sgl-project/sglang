"""
Tests for the DeepSeek-V4 fused KV store kernels
(``python/sglang/kernels/jit/csrc/deepseek_v4/store.cuh``).

Focus: padding-slot / sentinel-index guarding. Both
``fused_store_flashmla_cache`` and ``fused_store_indexer_cache`` say
"always load the value from input (don't store if invalid)" but must also
skip the store itself for invalid rows:

  * CUDA-graph padding rows carry ``out_cache_loc == 0`` — the pool's
    reserved sink slot (real allocations start at 1, see
    ``PagedTokenToKVPoolAllocator.clear()``). Storing an undefined padding
    row into slot 0 poisons the page that page-table padding also points
    at, and partially-valid tiles read the garbage back before masking.
  * The SWA page LUT deliberately ends with a live ``-1`` ("no page")
    entry. Unguarded, ``page = index >> kPageBits`` goes negative and the
    store becomes an out-of-bounds write *before* the cache base.

The ``-1`` case is made deterministic without compute-sanitizer by placing
a sentinel-filled guard page immediately before the cache allocation: the
unguarded kernel's negative-page store lands in the guard page.

Same invariant as ``test_store_cache_reserved_skip_index`` in
``test/registered/kernels/ops/kvcache/test_store_cache.py``, applied to the
DeepSeek-V4 fused store kernels.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

try:
    from sglang.kernels.ops.attention.dsv4 import fused_store_cache

    HAS_FUSED_STORE = True
except ImportError:
    HAS_FUSED_STORE = False

try:
    from sglang.srt.utils import is_hip

    _is_hip = is_hip()
except ImportError:
    _is_hip = False

register_cuda_ci(est_time=16, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE_SIZE = 64
DEVICE = "cuda"
SENTINEL = 0xAB

# Per-type layout constants, mirroring store.cuh:
#   flashmla: input (N, 512) bf16; page bytes = ceil(584 * 64 / 576) * 576;
#             token slot = 576 value bytes at offset*576 (448 fp8 + 128 bf16)
#             + 8 scale bytes at 576*page_size + offset*8.
#   indexer:  input (N, 128) bf16; page bytes = 132 * 64;
#             token slot = 128 value bytes at offset*128
#             + 4 scale bytes at 128*page_size + offset*4.
LAYOUT = {
    "flashmla": {
        "input_dim": 512,
        "page_bytes": -(-584 * PAGE_SIZE // 576) * 576,
        "value_stride": 576,
        "scale_base": 576 * PAGE_SIZE,
        "scale_stride": 8,
    },
    "indexer": {
        "input_dim": 128,
        "page_bytes": 132 * PAGE_SIZE,
        "value_stride": 128,
        "scale_base": 128 * PAGE_SIZE,
        "scale_stride": 4,
    },
}


def _skip_if_unavailable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if _is_hip:
        pytest.skip("Fused store JIT kernel is CUDA-specific (HIP uses triton path)")
    if not HAS_FUSED_STORE:
        pytest.skip("fused_store_cache not importable")


def _make_input(num_tokens: int, store_type: str) -> torch.Tensor:
    dim = LAYOUT[store_type]["input_dim"]
    return torch.randn((num_tokens, dim), dtype=torch.bfloat16, device=DEVICE)


def _make_cache(num_pages: int, store_type: str) -> torch.Tensor:
    return torch.full(
        (num_pages, LAYOUT[store_type]["page_bytes"]),
        SENTINEL,
        dtype=torch.uint8,
        device=DEVICE,
    )


def _slot_bytes(cache: torch.Tensor, index: int, store_type: str) -> torch.Tensor:
    """All cache bytes the kernel would touch for a token slot (values + scales)."""
    layout = LAYOUT[store_type]
    page, offset = index // PAGE_SIZE, index % PAGE_SIZE
    values = cache[page][
        offset * layout["value_stride"] : (offset + 1) * layout["value_stride"]
    ]
    scale_start = layout["scale_base"] + offset * layout["scale_stride"]
    scales = cache[page][scale_start : scale_start + layout["scale_stride"]]
    return torch.cat([values, scales])


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("store_type", ["flashmla", "indexer"])
def test_fused_store_padding_index_zero_not_written(
    store_type: str, index_dtype: torch.dtype
) -> None:
    """Padding rows (out_cache_loc == 0) must not store into the reserved slot."""
    _skip_if_unavailable()
    inp = _make_input(4, store_type)
    # CUDA-graph padding rows carry undefined values. Reproduce the dangerous
    # case directly instead of requiring a full model checkpoint.
    inp[[0, 2]] = torch.nan
    cache = _make_cache(3, store_type)
    indices = torch.tensor([0, 7, 0, PAGE_SIZE + 9], dtype=index_dtype, device=DEVICE)

    fused_store_cache(inp, cache, indices, page_size=PAGE_SIZE, type=store_type)
    torch.cuda.synchronize()

    assert torch.all(
        _slot_bytes(cache, 0, store_type) == SENTINEL
    ), "padding row (index 0) was stored into the reserved sink slot"
    for valid_index in (7, PAGE_SIZE + 9):
        assert not torch.all(
            _slot_bytes(cache, valid_index, store_type) == SENTINEL
        ), f"valid row (index {valid_index}) was not stored"


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("store_type", ["flashmla", "indexer"])
def test_fused_store_negative_index_no_oob_write(
    store_type: str, index_dtype: torch.dtype
) -> None:
    """The SWA LUT's ``-1`` sentinel must not become a negative-page OOB write.

    The cache view starts one page into a larger allocation; an unguarded
    ``page = -1 >> kPageBits`` store lands in the sentinel-filled guard page
    immediately before the cache base.
    """
    _skip_if_unavailable()
    inp = _make_input(4, store_type)
    inp[[0, 2]] = torch.nan
    full = _make_cache(4, store_type)
    guard_page, cache = full[0], full[1:]
    indices = torch.tensor([-1, 7, -1, PAGE_SIZE + 9], dtype=index_dtype, device=DEVICE)

    fused_store_cache(inp, cache, indices, page_size=PAGE_SIZE, type=store_type)
    torch.cuda.synchronize()

    assert torch.all(
        guard_page == SENTINEL
    ), "index -1 produced an out-of-bounds write before the cache base"
    for valid_index in (7, PAGE_SIZE + 9):
        assert not torch.all(
            _slot_bytes(cache, valid_index, store_type) == SENTINEL
        ), f"valid row (index {valid_index}) was not stored"


@pytest.mark.parametrize("store_type", ["flashmla", "indexer"])
def test_fused_store_padding_rows_are_pure_noops(store_type: str) -> None:
    """Interleaving padding rows must leave the cache byte-identical to a
    run with only the valid rows."""
    _skip_if_unavailable()
    valid = _make_input(2, store_type)
    valid_indices = [7, PAGE_SIZE + 9]

    cache_a = _make_cache(3, store_type)
    indices_a = torch.tensor(valid_indices, dtype=torch.int64, device=DEVICE)
    fused_store_cache(valid, cache_a, indices_a, page_size=PAGE_SIZE, type=store_type)

    padded = _make_input(5, store_type)
    padded[[0, 2, 4]] = torch.nan
    padded[1], padded[3] = valid[0], valid[1]
    cache_b = _make_cache(3, store_type)
    indices_b = torch.tensor(
        [0, valid_indices[0], -1, valid_indices[1], 0],
        dtype=torch.int64,
        device=DEVICE,
    )
    fused_store_cache(padded, cache_b, indices_b, page_size=PAGE_SIZE, type=store_type)
    torch.cuda.synchronize()

    assert torch.equal(cache_a, cache_b)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

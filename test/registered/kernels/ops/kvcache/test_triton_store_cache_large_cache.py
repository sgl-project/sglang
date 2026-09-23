import pytest
import torch

from sglang.kernels.ops.kvcache.triton_store_cache import (
    triton_fused_store_flashmla,
    triton_fused_store_indexer,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=90, stage="jit-kernel-unit", runner_config="amd")

PAGE_SIZE = 256
MLA_BYTES_PER_PAGE = 149760  # DeepSeek-V4.1 geometry: 256 slots x 585 B
MLA_HEAD_DIM = 512
INDEXER_HEAD_DIM = 128
DEVICE = "cuda"
INT32_MAX = 2**31 - 1


def _pages_for(stride: int) -> int:
    """Pages needed for the highest page index to overflow int32 at `stride`."""
    return INT32_MAX // stride + 2


def _skip_unless_free(num_bytes: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    free, _ = torch.cuda.mem_get_info()
    if free < num_bytes + (1 << 30):
        pytest.skip(f"needs {num_bytes / 2**30:.1f} GiB free, has {free / 2**30:.1f}")


def _store_at(store, cache, head_dim, slot, token):
    idx = torch.tensor([slot], dtype=torch.int32, device=DEVICE)
    store(token, cache, idx, PAGE_SIZE)


@pytest.mark.parametrize(
    "store,head_dim,stride_div,bytes_per_page",
    [
        # the rope tile addresses the cache through its bf16 view, so its page
        # stride is bytes_per_page // 2 and it overflows on a cache past 4 GiB
        (triton_fused_store_flashmla, MLA_HEAD_DIM, 2, MLA_BYTES_PER_PAGE),
        # the indexer scales address it through an f32 view: bytes_per_page // 4
        (
            triton_fused_store_indexer,
            INDEXER_HEAD_DIM,
            4,
            PAGE_SIZE * (INDEXER_HEAD_DIM + 4),
        ),
    ],
)
def test_store_past_int32_page_offset(store, head_dim, stride_div, bytes_per_page):
    """A slot whose page offset exceeds int32 must still land in its own slot.

    `loc` used to be narrowed to int32, so `page * BYTES_PER_PAGE` wrapped and
    the store scattered outside the cache. Fixed upstream in #41159; this
    pins the behaviour so it cannot regress.
    """
    num_pages = _pages_for(bytes_per_page // stride_div)
    total = num_pages * bytes_per_page
    _skip_unless_free(total)

    cache = torch.zeros((num_pages, bytes_per_page), dtype=torch.uint8, device=DEVICE)
    torch.manual_seed(0)
    token = torch.randn((1, head_dim), dtype=torch.bfloat16, device=DEVICE)

    low_slot = PAGE_SIZE + 7
    high_slot = (num_pages - 1) * PAGE_SIZE + 7
    assert (high_slot // PAGE_SIZE) * (bytes_per_page // stride_div) > INT32_MAX

    _store_at(store, cache, head_dim, low_slot, token)
    _store_at(store, cache, head_dim, high_slot, token)
    torch.cuda.synchronize()

    low_page = cache[low_slot // PAGE_SIZE]
    high_page = cache[high_slot // PAGE_SIZE]
    assert low_page.any(), "low slot wrote nothing; the fixture itself is wrong"
    torch.testing.assert_close(high_page, low_page)


def test_flashmla_high_slot_does_not_clobber_other_pages():
    """The wrapped store landed in an unrelated page. Nothing else may change."""
    stride = MLA_BYTES_PER_PAGE // 2
    num_pages = _pages_for(stride)
    total = num_pages * MLA_BYTES_PER_PAGE
    _skip_unless_free(total)

    cache = torch.zeros(
        (num_pages, MLA_BYTES_PER_PAGE), dtype=torch.uint8, device=DEVICE
    )
    torch.manual_seed(0)
    token = torch.randn((1, MLA_HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)

    high_page = num_pages - 1
    _store_at(
        triton_fused_store_flashmla, cache, MLA_HEAD_DIM, high_page * PAGE_SIZE, token
    )
    torch.cuda.synchronize()

    assert cache[high_page].any(), "the store did not reach its own page"
    assert not cache[:high_page].any(), "the store wrote outside its page"

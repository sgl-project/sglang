from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.kvcache.dsv4_fp4_index_k_cache import store_fp4_index_k_cache
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.common import (
    quantize_fp4_indexer_ref,
    store_fp4_index_cache_ref,
)

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD_DIM = 128
FP4_DIM = HEAD_DIM // 2
SCALE_BYTES = 4
PAGE_SIZE = 64


@pytest.mark.parametrize("num_tokens", [1, 16, 96])
def test_fp4_index_cache_store_layout(num_tokens: int) -> None:
    torch.manual_seed(num_tokens)
    num_pages = max(1, (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE)
    x = torch.randn(num_tokens, HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    loc = torch.randperm(num_pages * PAGE_SIZE, device=get_device())[:num_tokens].to(
        torch.int64
    )
    cache = torch.zeros(
        num_pages,
        PAGE_SIZE * (FP4_DIM + SCALE_BYTES),
        device=get_device(),
        dtype=torch.uint8,
    )

    store_fp4_index_k_cache(x, cache, loc, page_size=PAGE_SIZE)

    ref_fp4, ref_sf = quantize_fp4_indexer_ref(x)
    expected = store_fp4_index_cache_ref(ref_fp4, ref_sf, loc, num_pages)
    torch.testing.assert_close(cache, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

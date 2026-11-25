import sys
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.dsa import dequant_k_cache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_sparse_kv_gather_preserves_selection_and_padding():
    # A strided table includes a shared token, unsorted positions and an idle row.
    indices = (
        torch.tensor([[5, 2, -1], [2, 0, 1], [-1, -1, -1]], dtype=torch.int32)
        .T.contiguous()
        .T
    )
    cache = torch.randn(8, 1, 16, generator=torch.Generator().manual_seed(42))

    def gather(cache, slots):
        assert slots.is_contiguous() and slots.ndim == 1
        assert slots.numel() == indices.numel()
        assert torch.all(slots >= 0)
        return cache.index_select(0, slots)

    with patch.object(dequant_k_cache, "dequantize_k_cache_paged_dsa", gather):
        selected, remapped = dequant_k_cache.gather_sparse_decode_kv(cache, indices)

    valid = indices >= 0
    assert remapped.shape == indices.shape and remapped.dtype == torch.int32
    assert torch.equal(remapped[~valid], indices[~valid])
    torch.testing.assert_close(selected[remapped[valid]], cache[indices[valid]])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

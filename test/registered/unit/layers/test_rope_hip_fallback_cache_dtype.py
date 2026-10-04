import sys
from unittest import mock

import pytest
import torch

from sglang.srt.layers.rotary_embedding import base
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_hip_fallback_gets_query_dtype_cache(monkeypatch, dtype):
    monkeypatch.setattr(base, "_is_cpu", True)
    monkeypatch.setattr(base, "_is_hip", True)
    monkeypatch.setattr(base, "publish_role", lambda: None)
    rope = base.RotaryEmbedding(
        head_size=64,
        rotary_dim=64,
        max_position_embeddings=256,
        base=10000,
        is_neox_style=True,
        dtype=dtype,
    )
    # The HIP fallback is sgl_kernel.rotary_embedding, which needs the cache in
    # the query dtype.
    rope.use_fallback_kernel = True
    rope.fallback_rotary_embedding = mock.Mock()
    fp32_cache = rope.cos_sin_cache
    # HIP keeps the buffer fp32 for the fused QSA indexer.
    assert fp32_cache.dtype == torch.float32

    positions = torch.arange(4)
    query = torch.zeros(4, 2 * 64, dtype=dtype)
    key = torch.zeros(4, 64, dtype=dtype)
    for _ in range(2):
        rope.forward_cuda(positions, query, key)

    first, second = (
        call.args[4] for call in rope.fallback_rotary_embedding.call_args_list
    )
    assert first.dtype == dtype
    assert torch.equal(first, fp32_cache.to(dtype))
    assert second is first
    assert rope.cos_sin_cache is fp32_cache

    # Extending the buffer invalidates the cast copy.
    rope._ensure_cos_sin_cache_length(len(fp32_cache) + 16)
    rope.forward_cuda(positions, query, key)
    third = rope.fallback_rotary_embedding.call_args.args[4]
    assert rope.cos_sin_cache.dtype == torch.float32
    assert third.dtype == dtype
    assert torch.equal(third, rope.cos_sin_cache.to(dtype))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

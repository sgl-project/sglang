import sys

import pytest
import torch
from torch.testing import assert_close

from sglang.srt.layers.rotary_embedding import base, factory, rope_variant
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("kind", ("yarn", "deepseek_yarn", "mrope"))
@pytest.mark.parametrize("factor", (1.0, 2.0))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_cache_extension(monkeypatch, kind, factor, dtype):
    monkeypatch.setattr(base, "_is_cpu", True)
    monkeypatch.setattr(base, "publish_role", lambda: None)
    monkeypatch.setattr(rope_variant, "get_device", lambda: "cpu")
    monkeypatch.setattr(rope_variant, "_is_npu", True)
    monkeypatch.setattr(factory, "_ROPE_DICT", {})
    scaling = {"rope_type": kind, "factor": factor, "attn_factor": 1.3}
    if kind == "mrope":
        scaling.update(rope_type="yarn", mrope_section=[16, 8, 8])
    rope = factory.get_rope(
        head_size=64,
        rotary_dim=64,
        max_position=4096,
        base=10000,
        dtype=torch.float32,
        rope_scaling=scaling,
    )
    rope.cos_sin_cache = rope.cos_sin_cache.to(dtype)
    prefix = rope.cos_sin_cache.clone()
    for needed in (len(prefix) - 1, len(prefix), len(prefix) + 256):
        rope._ensure_cos_sin_cache_length(needed)
        assert len(rope.cos_sin_cache) > needed
        assert rope.max_position_embeddings == 4096
        positions = torch.arange(len(rope.cos_sin_cache), dtype=torch.float32)
        phase = torch.outer(positions, rope._compute_inv_freq(factor))
        expected = torch.cat((phase.cos(), phase.sin()), dim=-1) * rope.mscale
        assert_close(rope.cos_sin_cache, expected.to(dtype), rtol=0, atol=0)
        assert torch.equal(rope.cos_sin_cache[: len(prefix)], prefix)
        if kind == "deepseek_yarn":
            assert_close(rope.cos_cached_total, expected[:, :32].repeat(1, 2))
            assert_close(rope.sin_cached_total, expected[:, 32:].repeat(1, 2))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

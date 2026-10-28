# SPDX-License-Identifier: Apache-2.0
"""AITer attention impl construction guards (ROCm-only; skipped elsewhere)."""

import unittest
from unittest.mock import patch

import pytest
import torch

HEAD_SIZE = 128


def _impl_cls():
    # `aiter` ships with ROCm only, and importing the backend module needs it.
    pytest.importorskip("aiter", reason="AITer is a ROCm-only dependency")
    from sglang.multimodal_gen.runtime.layers.attention.backends.aiter import AITerImpl

    return AITerImpl


def _build(num_heads: int, num_kv_heads: int | None):
    return _impl_cls()(
        num_heads=num_heads,
        head_size=HEAD_SIZE,
        softmax_scale=HEAD_SIZE**-0.5,
        num_kv_heads=num_kv_heads,
    )


@pytest.mark.parametrize("num_kv_heads", [32, 8, 1, None])
def test_accepts_grouped_and_multi_query_kv_heads(num_kv_heads):
    # aiter's mha entry points broadcast each KV head across its group of
    # query heads, so Cosmos3-style GQA cross-attention is supported.
    assert _build(32, num_kv_heads).softmax_scale == pytest.approx(HEAD_SIZE**-0.5)


def test_rejects_kv_heads_that_do_not_divide_the_query_heads():
    with pytest.raises(ValueError, match="multiple of num_kv_heads"):
        _build(32, 5)


class TestGfx1250BF16AttentionUnderCompile(unittest.TestCase):
    """On gfx1250, flash_attn_func runs a FlyDSL kernel that JIT-compiles in Python and
    fails when Dynamo traces it; a compiled DiT block must call it eagerly."""

    def test_flash_attn_func_runs_outside_dynamo_on_gfx1250(self):
        pytest.importorskip("aiter", reason="AITer is a ROCm-only dependency")
        from sglang.multimodal_gen.runtime.layers.attention.backends import (
            aiter as aiter_backend,
        )

        traced_calls = []

        def fake_flash_attn_func(q, k, v, **kwargs):
            traced_calls.append(torch.compiler.is_compiling())
            return q.clone()

        impl = _build(4, 4)
        q = torch.randn(1, 8, 4, HEAD_SIZE)
        compiled_forward = torch.compile(impl.forward, backend="eager")
        with (
            patch.object(aiter_backend, "USE_AITER_GFX1250", True),
            patch.object(aiter_backend.aiter, "flash_attn_func", fake_flash_attn_func),
        ):
            out = compiled_forward(q, q, q)
        torch.testing.assert_close(out, q)
        self.assertEqual(traced_calls, [False])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

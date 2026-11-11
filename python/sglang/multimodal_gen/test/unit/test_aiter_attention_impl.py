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


class TestBF16AttentionUnderCompile(unittest.TestCase):
    """BF16 flash_attn_func must run outside Dynamo when tracing it would skip the DiT.

    gfx1250 dispatches to a FlyDSL kernel that JIT-compiles in Python. ENABLE_CK=0
    dispatches to aiter's Triton kernel, which opens a config JSON. Either graph
    break sits inside the block loop and drops the compiled forward.
    """

    def _compiled_call(self, *, gfx1250: bool, enable_ck: bool) -> list[bool]:
        pytest.importorskip("aiter", reason="AITer is a ROCm-only dependency")
        from sglang.multimodal_gen.runtime.layers.attention.backends import (
            aiter as aiter_backend,
        )

        torch._dynamo.reset()
        traced_calls = []

        def fake_flash_attn_func(q, k, v, **kwargs):
            traced_calls.append(torch.compiler.is_compiling())
            return q.clone()

        impl = _build(4, 4)
        q = torch.randn(1, 8, 4, HEAD_SIZE)
        compiled_forward = torch.compile(impl.forward, backend="eager")
        with (
            patch.object(aiter_backend, "USE_AITER_GFX1250", gfx1250),
            patch.object(aiter_backend, "_AITER_ENABLE_CK", enable_ck),
            patch.object(aiter_backend.aiter, "flash_attn_func", fake_flash_attn_func),
        ):
            out = compiled_forward(q, q, q)
        torch.testing.assert_close(out, q)
        return traced_calls

    def test_flash_attn_func_runs_outside_dynamo_on_gfx1250(self):
        self.assertEqual(self._compiled_call(gfx1250=True, enable_ck=True), [False])

    def test_flash_attn_func_runs_outside_dynamo_when_ck_is_disabled(self):
        self.assertEqual(self._compiled_call(gfx1250=False, enable_ck=False), [False])

    def test_ck_flash_attn_stays_inside_the_compiled_graph(self):
        self.assertEqual(self._compiled_call(gfx1250=False, enable_ck=True), [True])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a ROCm GPU")
@pytest.mark.parametrize("pad", [0, 24])
def test_one_live_sequence_matches_the_varlen_kernel(pad):
    """A packed single sequence (plus trailing padding) runs the batch kernel:
    its live rows get the varlen kernel's bytes (MHA v4's own bytes where that
    kernel serves the call) and the padding rows are zero."""
    impl = _impl_cls()(
        num_heads=4,
        head_size=HEAD_SIZE,
        softmax_scale=HEAD_SIZE**-0.5,
        packed_trailing_padding=True,
    )
    torch.manual_seed(0)
    total = 4096
    used = total - pad
    packed = torch.randn(total, 4, 3 * HEAD_SIZE, device="cuda", dtype=torch.bfloat16)
    # strided views, as MiniMax-H3's Ulysses exchange hands them over
    q, k, v = packed.split(HEAD_SIZE, dim=-1)
    bounds = (0, used, total) if pad else (0, total)
    cu_seqlens = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    out = impl.forward_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=used, cu_seqlens_host=bounds
    )
    # without host bounds the call keeps the varlen kernel
    ref = impl.forward_varlen(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=used)
    from sglang.multimodal_gen.runtime.layers.attention.backends import aiter as backend

    if backend._can_use_mha_v4_bf16(q, k, v, False, 0.0):
        live = (t[None, :used].contiguous() for t in (q, k, v))
        expected = backend._mha_v4_bf16(*live, HEAD_SIZE**-0.5)
        # strided views and contiguous copies give the same bytes
        assert torch.equal(out[:used], expected[0])
        torch.testing.assert_close(out[:used], ref[:used], atol=2e-2, rtol=2e-2)
    else:
        assert torch.equal(out[:used], ref[:used])
    assert not out[used:].any()

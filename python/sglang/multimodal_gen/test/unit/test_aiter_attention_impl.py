# SPDX-License-Identifier: Apache-2.0
"""AITer attention impl construction guards (ROCm-only; skipped elsewhere)."""

import sys
import types
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


@pytest.mark.parametrize("racy", [True, False])
def test_mha_v4_needs_the_aiter_race_fix(racy):
    """ROCm/aiter#6841 fixed the BF16 kernel's races and dropped ``sparse`` here."""
    _impl_cls()
    from sglang.multimodal_gen.runtime.layers.attention.backends import aiter as backend

    fake = types.ModuleType("aiter.ops.mha_v4")
    if racy:
        fake._check_lse_capable = lambda q_format, v_format, sparse: None
    else:
        fake._check_lse_capable = lambda q_format, v_format: None
    with (
        patch.object(backend, "USE_AITER_GFX95", True),
        patch.dict(sys.modules, {"aiter.ops.mha_v4": fake}),
        patch.object(sys.modules["aiter.ops"], "mha_v4", fake, create=True),
    ):
        assert (backend._load_mha_v4() is None) == racy


def test_one_live_sequence_keeps_the_varlen_kernel_off_gfx950():
    """The batch kernel and the key split were only measured on gfx950."""
    impl = _build(4, 4)
    from sglang.multimodal_gen.runtime.layers.attention.backends import aiter as backend

    rows = 8
    q = torch.randn(rows, 4, HEAD_SIZE)
    bounds = (0, rows)
    with (
        patch.object(backend, "USE_AITER_GFX95", False),
        patch.object(
            backend.aiter, "flash_attn_varlen_func", lambda **kwargs: kwargs["q"] + 1
        ),
    ):
        out = impl.forward_varlen(
            q,
            q,
            q,
            cu_seqlens=torch.tensor(bounds, dtype=torch.int32),
            max_seqlen=rows,
            cu_seqlens_host=bounds,
        )
    torch.testing.assert_close(out, q + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a ROCm GPU")
@pytest.mark.parametrize("pad", [0, 24])
def test_one_live_sequence_matches_the_varlen_kernel(pad):
    """A packed single sequence (plus trailing padding) runs the batch kernel
    on strided q/k/v: its live rows get the bytes of the same kernel on
    contiguous inputs, match the varlen kernel (its bytes under fmha v3), and
    the padding rows are zero."""
    impl = _impl_cls()(
        num_heads=4,
        head_size=HEAD_SIZE,
        softmax_scale=HEAD_SIZE**-0.5,
        packed_trailing_padding=True,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends import aiter as backend

    if not backend.USE_AITER_GFX95:
        pytest.skip("the batch kernel path is gfx950-only")
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
    live = (t[None, :used].contiguous() for t in (q, k, v))
    assert torch.equal(out[:used], backend._flash(*live, HEAD_SIZE**-0.5)[0])
    # without host bounds the call keeps the varlen kernel
    ref = impl.forward_varlen(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=used)
    if backend._MHA_V4 is None:
        assert torch.equal(out[:used], ref[:used])
    torch.testing.assert_close(out[:used], ref[:used], atol=2e-2, rtol=2e-2)
    assert not out[used:].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a ROCm GPU")
def test_tail_rows_with_split_keys_match_one_launch():
    """Rows past the last full wave run with their keys split and an LSE
    merge; they match a single launch within BF16 rounding, and repeat
    bitwise."""
    impl = _impl_cls()(
        num_heads=7,
        head_size=HEAD_SIZE,
        softmax_scale=HEAD_SIZE**-0.5,
        packed_trailing_padding=True,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends import aiter as backend

    if not backend.USE_AITER_GFX95:
        pytest.skip("the key split is gfx950-only")
    heads, device = 7, torch.device("cuda", torch.cuda.current_device())
    rows = 4 * backend._num_cus(device.index) // heads * 256 + 300
    total = rows + 64
    torch.manual_seed(0)
    packed = torch.randn(
        total, heads, 3 * HEAD_SIZE, device=device, dtype=torch.bfloat16
    )
    q, k, v = packed.split(HEAD_SIZE, dim=-1)
    assert backend._tail_split_rows(rows, heads, device) is not None
    bounds = (0, rows, total)
    cu_seqlens = torch.tensor(bounds, dtype=torch.int32, device=device)
    out = impl.forward_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=rows, cu_seqlens_host=bounds
    )
    live = (t[None, :rows] for t in (q, k, v))
    single = backend._flash(*live, HEAD_SIZE**-0.5)[0]
    torch.testing.assert_close(out[:rows], single, atol=2e-2, rtol=2e-2)
    assert not out[rows:].any()
    again = impl.forward_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=rows, cu_seqlens_host=bounds
    )
    assert torch.equal(out, again)


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
    sys.exit(pytest.main([__file__, "-v"]))

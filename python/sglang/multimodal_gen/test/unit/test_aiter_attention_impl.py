# SPDX-License-Identifier: Apache-2.0
"""AITer attention impl construction guards (ROCm-only; skipped elsewhere)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.test_utils import CustomTestCase

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


NUM_HEADS = 4
TOTAL_ROWS = 16
LIVE_ROWS = 10


class TestPackedTrailingPaddingVarlen(CustomTestCase):
    """A packed caller's padding tail must not reach the grouped varlen kernel.

    MiniMax H3 pads its single packed sequence up to a 64-row multiple, so
    cu_seqlens carries a trailing segment of nothing but padding that the
    kernel still schedules work over. Attending the live prefix alone is
    invisible in the live rows, so only a dispatch-level assertion catches a
    regression back to the two-segment call.
    """

    def _varlen_call(
        self,
        *,
        packed_trailing_padding: bool = True,
        cu_seqlens_host: tuple[int, ...] | None = (0, LIVE_ROWS, TOTAL_ROWS),
        max_seqlen: int = LIVE_ROWS,
        return_tuple: bool = False,
    ) -> SimpleNamespace:
        pytest.importorskip("aiter", reason="AITer is a ROCm-only dependency")
        from sglang.multimodal_gen.runtime.layers.attention.backends import (
            aiter as aiter_backend,
        )

        calls = []
        kernel_outputs = []

        def fake_flash_attn_varlen_func(**kwargs):
            calls.append(kwargs)
            # Derived from q so the assertions can tell which rows were fed in
            # and where the kernel's output landed.
            out = kwargs["q"] * 2 + 1
            kernel_outputs.append(out)
            return (out, None) if return_tuple else out

        impl = _impl_cls()(
            num_heads=NUM_HEADS,
            head_size=HEAD_SIZE,
            softmax_scale=HEAD_SIZE**-0.5,
            num_kv_heads=NUM_HEADS,
            packed_trailing_padding=packed_trailing_padding,
        )
        qkv = torch.randn(TOTAL_ROWS, NUM_HEADS, HEAD_SIZE)
        cu_seqlens = torch.tensor(
            (
                cu_seqlens_host
                if cu_seqlens_host is not None
                else (0, LIVE_ROWS, TOTAL_ROWS)
            ),
            dtype=torch.int32,
        )
        with patch.object(
            aiter_backend.aiter,
            "flash_attn_varlen_func",
            fake_flash_attn_varlen_func,
        ):
            out = impl.forward_varlen(
                qkv,
                qkv,
                qkv,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )
        return SimpleNamespace(
            calls=calls, kernel_outputs=kernel_outputs, qkv=qkv, out=out
        )

    def test_live_prefix_is_attended_as_a_single_segment(self):
        run = self._varlen_call()

        (call,) = run.calls
        for operand in ("q", "k", "v"):
            self.assertEqual(call[operand].shape[0], LIVE_ROWS)
            torch.testing.assert_close(call[operand], run.qkv[:LIVE_ROWS])
        for bounds in ("cu_seqlens_q", "cu_seqlens_k"):
            self.assertEqual(call[bounds].tolist(), [0, LIVE_ROWS])
        self.assertEqual(call["max_seqlen_q"], LIVE_ROWS)
        self.assertEqual(call["max_seqlen_k"], LIVE_ROWS)

    def test_padding_rows_are_zeroed_and_live_rows_keep_their_offsets(self):
        run = self._varlen_call()

        self.assertEqual(run.out.shape, (TOTAL_ROWS, NUM_HEADS, HEAD_SIZE))
        torch.testing.assert_close(run.out[:LIVE_ROWS], run.qkv[:LIVE_ROWS] * 2 + 1)
        torch.testing.assert_close(
            run.out[LIVE_ROWS:], torch.zeros_like(run.out[LIVE_ROWS:])
        )

    def test_tuple_return_is_unwrapped_on_the_live_prefix_path(self):
        # aiter returns (out, lse) for some configurations; the padded path
        # has its own unwrap, and a tuple reaching new_zeros() raises.
        run = self._varlen_call(return_tuple=True)

        torch.testing.assert_close(run.out[:LIVE_ROWS], run.qkv[:LIVE_ROWS] * 2 + 1)
        torch.testing.assert_close(
            run.out[LIVE_ROWS:], torch.zeros_like(run.out[LIVE_ROWS:])
        )

    def test_empty_trailing_segment_is_collapsed_to_one(self):
        # H3's token refiner always packs [0, N, N]: the rows are trimmed, so
        # the trailing segment holds nothing and only costs the kernel a
        # partition. There is no tail to drop, so the result comes back as is.
        run = self._varlen_call(
            cu_seqlens_host=(0, TOTAL_ROWS, TOTAL_ROWS), max_seqlen=TOTAL_ROWS
        )

        (call,) = run.calls
        self.assertEqual(call["q"].shape[0], TOTAL_ROWS)
        self.assertEqual(call["cu_seqlens_q"].tolist(), [0, TOTAL_ROWS])
        self.assertEqual(call["max_seqlen_q"], TOTAL_ROWS)
        # Returned straight through: allocating and copying here would cost
        # more than the empty segment it saves.
        self.assertIs(run.out, run.kernel_outputs[0])

    def test_full_sequence_is_attended_when_the_caller_does_not_opt_in(self):
        run = self._varlen_call(packed_trailing_padding=False, max_seqlen=TOTAL_ROWS)

        (call,) = run.calls
        self.assertEqual(call["q"].shape[0], TOTAL_ROWS)
        self.assertEqual(call["cu_seqlens_q"].tolist(), [0, LIVE_ROWS, TOTAL_ROWS])
        torch.testing.assert_close(run.out, run.qkv * 2 + 1)

    def test_full_sequence_is_attended_without_host_bounds(self):
        # The live length is a host value; without it there is nothing to slice
        # at, and reading it off the device tensor would sync every call.
        run = self._varlen_call(cu_seqlens_host=None)

        (call,) = run.calls
        self.assertEqual(call["q"].shape[0], TOTAL_ROWS)
        self.assertEqual(call["cu_seqlens_q"].tolist(), [0, LIVE_ROWS, TOTAL_ROWS])
        torch.testing.assert_close(run.out, run.qkv * 2 + 1)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

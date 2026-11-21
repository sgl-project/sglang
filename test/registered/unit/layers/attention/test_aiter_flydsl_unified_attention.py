"""FlyDSL unified attention in front of the AITER backend's Triton unified attention, on CPU."""

import sys
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import aiter_flydsl_unified_attention as fly_ua
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeFlyDSL:
    """Stands in for aiter.ops.flydsl.unified_attention_bf16."""

    def __init__(self, accept=True):
        self.accept = accept
        self.calls = []

    def unified_attention_flydsl(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["out"] if self.accept else None

    def flydsl_decline_reason(self, **kwargs):
        return None if self.accept else "requires page size in (64, 128), got 1"


def _kwargs(seqused_k=None, **extra):
    return dict(
        q=torch.zeros(16, 16, 192),
        k=torch.zeros(4, 128, 1, 192),
        v=torch.zeros(4, 128, 1, 128),
        out=torch.zeros(16, 16, 128),
        cu_seqlens_q=torch.tensor([0, 8, 16], dtype=torch.int32),
        seqused_k=(
            torch.tensor([300, 400], dtype=torch.int64)
            if seqused_k is None
            else seqused_k
        ),
        max_seqlen_q=8,
        max_seqlen_k=512,
        softmax_scale=0.1,
        causal=True,
        window_size=(-1, -1),
        block_table=torch.zeros(2, 4, dtype=torch.int32),
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        sinks=None,
        **extra,
    )


class TestFlyDSLUnifiedAttention(CustomTestCase):
    def setUp(self):
        fly_ua.reset_flydsl_unified_attention_cache()
        self.triton_calls = []
        self.capturing = False
        capture = patch.object(
            torch.cuda, "is_current_stream_capturing", lambda: self.capturing
        )
        capture.start()
        self.addCleanup(capture.stop)

        def triton(**kwargs):
            self.triton_calls.append(kwargs)
            return kwargs["out"]

        self._triton = triton

    def _wrap(self, flydsl):
        module = types.ModuleType("aiter.ops.flydsl.unified_attention_bf16")
        module.unified_attention_flydsl = flydsl.unified_attention_flydsl
        module.flydsl_decline_reason = flydsl.flydsl_decline_reason
        parent = types.ModuleType("aiter.ops.flydsl")
        parent.unified_attention_bf16 = module
        with (
            envs.SGLANG_USE_AITER_FLYDSL_UNIFIED_ATTN.override(True),
            patch.dict(
                sys.modules,
                {
                    "aiter.ops.flydsl": parent,
                    "aiter.ops.flydsl.unified_attention_bf16": module,
                },
            ),
        ):
            return fly_ua.with_flydsl_unified_attention(self._triton)

    def test_off_by_default_returns_triton_unchanged(self):
        self.assertIs(fly_ua.with_flydsl_unified_attention(self._triton), self._triton)

    def test_missing_flydsl_module_falls_back_to_triton(self):
        with (
            envs.SGLANG_USE_AITER_FLYDSL_UNIFIED_ATTN.override(True),
            patch.dict(sys.modules, {"aiter.ops.flydsl": None}),
        ):
            wrapped = fly_ua.with_flydsl_unified_attention(self._triton)
        self.assertIs(wrapped, self._triton)

    def test_accepted_call_skips_triton(self):
        flydsl = _FakeFlyDSL(accept=True)
        kwargs = _kwargs(equal_seqlen_q=True)
        out = self._wrap(flydsl)(**kwargs)
        self.assertIs(out, kwargs["out"])
        self.assertEqual(self.triton_calls, [])
        (call,) = flydsl.calls
        self.assertNotIn("equal_seqlen_q", call)
        self.assertEqual(call["seqused_k"].dtype, torch.int32)
        self.assertEqual(call["seqused_k"].tolist(), [300, 400])

    def test_declined_call_runs_triton_with_callers_arguments(self):
        flydsl = _FakeFlyDSL(accept=False)
        kwargs = _kwargs(equal_seqlen_q=True)
        self._wrap(flydsl)(**kwargs)
        (call,) = self.triton_calls
        self.assertIs(call["seqused_k"], kwargs["seqused_k"])
        self.assertEqual(call["seqused_k"].dtype, torch.int64)
        self.assertTrue(call["equal_seqlen_q"])

    def test_int32_copy_is_shared_by_layers_and_dropped_per_forward(self):
        flydsl = _FakeFlyDSL(accept=True)
        wrapped = self._wrap(flydsl)
        seqused_k = torch.tensor([300, 400], dtype=torch.int64)
        wrapped(**_kwargs(seqused_k))
        wrapped(**_kwargs(seqused_k))
        self.assertIs(flydsl.calls[0]["seqused_k"], flydsl.calls[1]["seqused_k"])

        seqused_k += 1
        fly_ua.reset_flydsl_unified_attention_cache()
        wrapped(**_kwargs(seqused_k))
        self.assertEqual(flydsl.calls[2]["seqused_k"].tolist(), [301, 401])

    def test_graph_capture_does_not_reuse_an_eager_copy(self):
        flydsl = _FakeFlyDSL(accept=True)
        wrapped = self._wrap(flydsl)
        seqused_k = torch.tensor([300, 400], dtype=torch.int64)
        wrapped(**_kwargs(seqused_k))
        self.capturing = True
        wrapped(**_kwargs(seqused_k))
        wrapped(**_kwargs(seqused_k))
        self.assertIsNot(flydsl.calls[0]["seqused_k"], flydsl.calls[1]["seqused_k"])
        self.assertIs(flydsl.calls[1]["seqused_k"], flydsl.calls[2]["seqused_k"])

    def test_int32_seqused_k_is_passed_through(self):
        flydsl = _FakeFlyDSL(accept=True)
        seqused_k = torch.tensor([300, 400], dtype=torch.int32)
        self._wrap(flydsl)(**_kwargs(seqused_k))
        self.assertIs(flydsl.calls[0]["seqused_k"], seqused_k)


if __name__ == "__main__":
    unittest.main()

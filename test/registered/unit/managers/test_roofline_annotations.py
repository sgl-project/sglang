"""Unit tests for roofline profiling annotations (#24911) — no server, no model loading.

Covers ``SchedulerProfilerManager._build_profile_annotation`` (the per-iteration
sq/sk/sqsq/sqsk aggregation + ``torch.profiler.record_function`` marker) and the
``roofline_annotations`` plumbing through ``ProfileReq``.
"""

import json
import unittest
from contextlib import nullcontext
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import ProfileReq
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _make_manager(*, roofline_annotations: bool, profiler_active: bool):
    """Build a manager without running __post_init__ (which needs envs/GPU)."""
    mgr = SchedulerProfilerManager.__new__(SchedulerProfilerManager)
    mgr.roofline_annotations = roofline_annotations
    mgr.torch_profiler = object() if profiler_active else None
    mgr._profile_manager = None
    return mgr


def _decode_req(rid: str, seqlen: int):
    return SimpleNamespace(rid=rid, seqlen=seqlen)


def _prefill_req(rid: str, extend_input_len: int, prefix_len: int):
    return SimpleNamespace(
        rid=rid,
        extend_input_len=extend_input_len,
        prefix_indices=[0] * prefix_len,
    )


def _batch(forward_mode, reqs, decoding_reqs=None):
    return SimpleNamespace(
        forward_mode=forward_mode, reqs=reqs, decoding_reqs=decoding_reqs
    )


def _expected(bs, ctx, gen):
    """ctx/gen are (R, sq, sk, sqsq, sqsk) tuples."""
    return (
        f"execute_{bs}"
        f"_context_{ctx[0]}(sq{ctx[1]}sk{ctx[2]}sqsq{ctx[3]}sqsk{ctx[4]})"
        f"_generation_{gen[0]}(sq{gen[1]}sk{gen[2]}sqsq{gen[3]}sqsk{gen[4]})"
    )


class TestRooflineAnnotationString(CustomTestCase):
    def _annotate(self, batch):
        mgr = _make_manager(roofline_annotations=True, profiler_active=True)
        return mgr._build_profile_annotation(batch).name

    def test_pure_decode_batch(self):
        # Two decode reqs: each contributes nq=1, nkv=seqlen.
        batch = _batch(
            ForwardMode.DECODE,
            [_decode_req("a", 10), _decode_req("b", 20)],
        )
        # g: R=2, sq=2, sk=30, sqsq=1+1=2, sqsk=1*10+1*20=30; bs=2
        expected = _expected(2, (0, 0, 0, 0, 0), (2, 2, 30, 2, 30))
        self.assertEqual(self._annotate(batch), expected)

    def test_pure_prefill_batch(self):
        # nq=extend_input_len, nkv=prefix_len+extend_input_len.
        batch = _batch(
            ForwardMode.EXTEND,
            [_prefill_req("a", 8, 2), _prefill_req("b", 4, 6)],
        )
        # req a: nq=8, nkv=10, sqsq=64, sqsk=80
        # req b: nq=4, nkv=10, sqsq=16, sqsk=40
        # p: R=2, sq=12, sk=20, sqsq=80, sqsk=120; bs=12
        expected = _expected(12, (2, 12, 20, 80, 120), (0, 0, 0, 0, 0))
        self.assertEqual(self._annotate(batch), expected)

    def test_mixed_batch_uses_decoding_reqs(self):
        # MIXED: reqs listed in decoding_reqs are generation, the rest context.
        ctx = _prefill_req("ctx", 5, 3)  # nq=5, nkv=8, sqsq=25, sqsk=40
        gen = _decode_req("gen", 12)  # nq=1, nkv=12, sqsq=1, sqsk=12
        batch = _batch(
            ForwardMode.MIXED,
            [ctx, gen],
            decoding_reqs=[SimpleNamespace(rid="gen")],
        )
        # bs = 5 (ctx) + 1 (gen) = 6
        expected = _expected(6, (1, 5, 8, 25, 40), (1, 1, 12, 1, 12))
        self.assertEqual(self._annotate(batch), expected)

    def test_mixed_batch_with_no_decoding_reqs_is_all_context(self):
        batch = _batch(
            ForwardMode.MIXED,
            [_prefill_req("a", 3, 0)],
            decoding_reqs=None,
        )
        # nq=3, nkv=3, sqsq=9, sqsk=9; bs=3
        expected = _expected(3, (1, 3, 3, 9, 9), (0, 0, 0, 0, 0))
        self.assertEqual(self._annotate(batch), expected)

    def test_empty_batch(self):
        batch = _batch(ForwardMode.DECODE, [])
        expected = _expected(0, (0, 0, 0, 0, 0), (0, 0, 0, 0, 0))
        self.assertEqual(self._annotate(batch), expected)


class TestRooflineAnnotationGating(CustomTestCase):
    def test_disabled_flag_returns_nullcontext(self):
        mgr = _make_manager(roofline_annotations=False, profiler_active=True)
        batch = _batch(ForwardMode.DECODE, [_decode_req("a", 10)])
        self.assertIsInstance(mgr._build_profile_annotation(batch), nullcontext)

    def test_no_active_profiler_returns_nullcontext(self):
        mgr = _make_manager(roofline_annotations=True, profiler_active=False)
        batch = _batch(ForwardMode.DECODE, [_decode_req("a", 10)])
        self.assertIsInstance(mgr._build_profile_annotation(batch), nullcontext)

    def test_enabled_and_active_returns_record_function(self):
        mgr = _make_manager(roofline_annotations=True, profiler_active=True)
        batch = _batch(ForwardMode.DECODE, [_decode_req("a", 10)])
        result = mgr._build_profile_annotation(batch)
        self.assertNotIsInstance(result, nullcontext)
        # Usable as a context manager (the scheduler wraps run_batch with it).
        with result:
            pass


class TestRooflineAnnotationPlumbing(CustomTestCase):
    def test_default_is_false(self):
        self.assertFalse(ProfileReq().roofline_annotations)

    def test_json_round_trip(self):
        req = ProfileReq(output_dir="/tmp/x", roofline_annotations=True)
        payload = {"roofline_annotations": req.roofline_annotations}
        parsed = json.loads(json.dumps(payload))
        self.assertTrue(parsed["roofline_annotations"])
        self.assertTrue(ProfileReq(**parsed).roofline_annotations)


if __name__ == "__main__":
    unittest.main()

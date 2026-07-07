"""Unit tests for roofline profiling annotations (#24911) — no server, no model loading.

The roofline aggregates are folded into SGLang's existing per-forward ``step[...]``
span (see ``sglang.srt.model_executor.step_span_utils.build_step_span_name``): only
the genuinely-new terms (``sqsq``/``sqsk``/``sk``, plus the context/generation split
for MIXED) are appended, and ``bs``/``toks`` are left to the base label so nothing is
duplicated. This also covers the ``roofline_annotations`` plumbing on ``ProfileReq``.
"""

import json
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import ProfileReq
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.step_span_utils import build_step_span_name

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _CpuMirror:
    """Minimal stand-in for the ``seq_lens_cpu`` tensor (only ``tolist`` used)."""

    def __init__(self, data):
        self._data = list(data)

    def tolist(self):
        return list(self._data)


def _fb(
    forward_mode,
    *,
    batch_size,
    extend_num_tokens=None,
    seq_lens_cpu=None,
    extend_seq_lens_cpu=None,
    extend_prefix_lens_cpu=None,
):
    return SimpleNamespace(
        forward_mode=forward_mode,
        batch_size=batch_size,
        extend_num_tokens=extend_num_tokens,
        seq_lens_cpu=None if seq_lens_cpu is None else _CpuMirror(seq_lens_cpu),
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        extend_prefix_lens_cpu=extend_prefix_lens_cpu,
    )


class TestStepSpanRoofline(CustomTestCase):
    def _name(self, fb):
        return build_step_span_name(fb, roofline_annotations=True)

    def test_pure_decode_batch(self):
        # Two decode reqs: each nq=1, nkv=seqlen.
        # sk=30, sqsq=1+1=2, sqsk=1*10+1*20=30.
        fb = _fb(ForwardMode.DECODE, batch_size=2, seq_lens_cpu=[10, 20])
        self.assertEqual(
            self._name(fb), "step[DECODE bs=2 g_sqsq=2 g_sqsk=30 g_sk=30]"
        )

    def test_pure_prefill_batch(self):
        # req a: nq=8, nkv=10 -> sqsq=64, sqsk=80
        # req b: nq=4, nkv=10 -> sqsq=16, sqsk=40
        # sk=20, sqsq=80, sqsk=120; toks left to the base label.
        fb = _fb(
            ForwardMode.EXTEND,
            batch_size=2,
            extend_num_tokens=12,
            extend_seq_lens_cpu=[8, 4],
            extend_prefix_lens_cpu=[2, 6],
        )
        self.assertEqual(
            self._name(fb), "step[EXTEND bs=2 toks=12 c_sqsq=80 c_sqsk=120 c_sk=20]"
        )

    def test_mixed_batch_splits_context_and_generation(self):
        # ctx: nq=5, nkv=8 -> sqsq=25, sqsk=40; gen (len-1 extend): nq=1, nkv=12.
        fb = _fb(
            ForwardMode.MIXED,
            batch_size=2,
            extend_seq_lens_cpu=[5, 1],
            extend_prefix_lens_cpu=[3, 11],
        )
        self.assertEqual(
            self._name(fb),
            "step[MIXED bs=2 c=1 g=1 "
            "c_sq=5 c_sk=8 c_sqsq=25 c_sqsk=40 "
            "g_sq=1 g_sk=12 g_sqsq=1 g_sqsk=12]",
        )

    def test_mixed_batch_all_context(self):
        fb = _fb(
            ForwardMode.MIXED,
            batch_size=1,
            extend_seq_lens_cpu=[3],
            extend_prefix_lens_cpu=[0],
        )
        self.assertEqual(
            self._name(fb),
            "step[MIXED bs=1 c=1 g=0 "
            "c_sq=3 c_sk=3 c_sqsq=9 c_sqsk=9 "
            "g_sq=0 g_sk=0 g_sqsq=0 g_sqsk=0]",
        )

    def test_missing_cpu_mirror_falls_back_to_base(self):
        # No seq_lens_cpu (some overlap paths) -> emit the base label unchanged.
        fb = _fb(ForwardMode.DECODE, batch_size=2, seq_lens_cpu=None)
        self.assertEqual(self._name(fb), "step[DECODE bs=2]")


class TestStepSpanGating(CustomTestCase):
    def test_disabled_flag_emits_base_label(self):
        fb = _fb(ForwardMode.DECODE, batch_size=2, seq_lens_cpu=[10, 20])
        self.assertEqual(
            build_step_span_name(fb, roofline_annotations=False), "step[DECODE bs=2]"
        )

    def test_disabled_flag_is_default(self):
        fb = _fb(
            ForwardMode.EXTEND,
            batch_size=1,
            extend_num_tokens=4,
            extend_seq_lens_cpu=[4],
            extend_prefix_lens_cpu=[0],
        )
        self.assertEqual(build_step_span_name(fb), "step[EXTEND bs=1 toks=4]")


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

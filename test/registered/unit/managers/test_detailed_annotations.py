"""Unit tests for detailed profiling annotations (#24911) — no server, no model loading.

The detailed-annotation aggregates are folded into SGLang's existing per-forward ``step[...]``
span (see ``sglang.srt.utils.profile_utils.build_step_span_name``): the
per-phase ``sq``/``sqsq``/``sqsk``/``sk`` terms (with the context/generation split
for MIXED) are appended and are self-contained, so ``sq`` is emitted even where it
duplicates the base label's ``bs``/``toks``. This also covers the
``detailed_annotations`` plumbing on ``ProfileReq``.
"""

import json
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import ProfileReq
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.step_span_utils import (
    detailed_annotations_enabled,
    set_detailed_annotations_enabled,
)
from sglang.srt.utils.profile_utils import build_step_span_name

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
    num_tokens_per_req=None,
):
    # A spec input (EAGLE/MTP) only needs to expose ``num_tokens_per_req`` for
    # the detailed-annotation suffix; None -> no spec_info (vanilla decode, N_Q == 1).
    spec_info = (
        None
        if num_tokens_per_req is None
        else SimpleNamespace(num_tokens_per_req=num_tokens_per_req)
    )
    return SimpleNamespace(
        forward_mode=forward_mode,
        batch_size=batch_size,
        extend_num_tokens=extend_num_tokens,
        seq_lens_cpu=None if seq_lens_cpu is None else _CpuMirror(seq_lens_cpu),
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        extend_prefix_lens_cpu=extend_prefix_lens_cpu,
        spec_info=spec_info,
    )


class TestStepSpanDetailedAnnotations(CustomTestCase):
    def _name(self, fb):
        return build_step_span_name(fb, detailed_annotations=True)

    def test_pure_decode_batch(self):
        # Two decode reqs: each nq=1, nkv=seqlen.
        # sk=30, sqsq=1+1=2, sqsk=1*10+1*20=30.
        fb = _fb(ForwardMode.DECODE, batch_size=2, seq_lens_cpu=[10, 20])
        self.assertEqual(
            self._name(fb), "step[DECODE bs=2 g_sq=2 g_sqsq=2 g_sqsk=30 g_sk=30]"
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
            self._name(fb),
            "step[EXTEND bs=2 toks=12 c_sq=12 c_sqsq=80 c_sqsk=120 c_sk=20]",
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

    def test_spec_draft_decode_uses_num_tokens_per_req(self):
        # EAGLE draft-decode: N_Q per req = topk (num_tokens_per_req), not 1.
        # topk=4, seqs=[10,20]: sq=4*2=8, sk=30, sqsq=16+16=32,
        # sqsk=4*10+4*20=120. g_sq is emitted because it != bs.
        fb = _fb(
            ForwardMode.DECODE,
            batch_size=2,
            seq_lens_cpu=[10, 20],
            num_tokens_per_req=4,
        )
        self.assertEqual(
            self._name(fb),
            "step[DECODE bs=2 g_sq=8 g_sqsq=32 g_sqsk=120 g_sk=30]",
        )

    def test_target_verify_uses_draft_token_width(self):
        # MTP/EAGLE target-verify: N_Q per req = num_draft_tokens. It is
        # classified as generation (``g_``) by request phase;
        # its quadratic self-attention is still captured in g_sqsq.
        # ndt=3, seqs=[10,20]: sq=3*2=6, sk=30, sqsq=9+9=18,
        # sqsk=3*10+3*20=90.
        fb = _fb(
            ForwardMode.TARGET_VERIFY,
            batch_size=2,
            seq_lens_cpu=[10, 20],
            num_tokens_per_req=3,
        )
        self.assertEqual(
            self._name(fb),
            "step[TARGET_VERIFY bs=2 g_sq=6 g_sqsq=18 g_sqsk=90 g_sk=30]",
        )

    def test_target_verify_without_cpu_mirror_falls_back_to_base(self):
        fb = _fb(
            ForwardMode.TARGET_VERIFY,
            batch_size=2,
            seq_lens_cpu=None,
            num_tokens_per_req=3,
        )
        self.assertEqual(self._name(fb), "step[TARGET_VERIFY bs=2]")

    def test_draft_extend_v2_uses_extend_mirrors_with_context_prefix(self):
        # EAGLE/MTP draft-extend is extend-shaped
        # req a: nq=2, nkv=12 -> sqsq=4, sqsk=24
        # req b: nq=3, nkv=23 -> sqsq=9, sqsk=69
        # sq=5, sk=35, sqsq=13, sqsk=93.
        fb = _fb(
            ForwardMode.DRAFT_EXTEND_V2,
            batch_size=2,
            extend_seq_lens_cpu=[2, 3],
            extend_prefix_lens_cpu=[10, 20],
        )
        self.assertEqual(
            self._name(fb),
            "step[DRAFT_EXTEND_V2 bs=2 c_sq=5 c_sqsq=13 c_sqsk=93 c_sk=35]",
        )

    def test_draft_extend_v2_without_extend_mirrors_falls_back_to_base(self):
        fb = _fb(ForwardMode.DRAFT_EXTEND_V2, batch_size=2)
        self.assertEqual(self._name(fb), "step[DRAFT_EXTEND_V2 bs=2]")

    def test_missing_cpu_mirror_falls_back_to_base(self):
        # No seq_lens_cpu (some overlap paths) -> emit the base label unchanged.
        fb = _fb(ForwardMode.DECODE, batch_size=2, seq_lens_cpu=None)
        self.assertEqual(self._name(fb), "step[DECODE bs=2]")


class TestStepSpanGating(CustomTestCase):
    def test_disabled_flag_emits_base_label(self):
        fb = _fb(ForwardMode.DECODE, batch_size=2, seq_lens_cpu=[10, 20])
        self.assertEqual(
            build_step_span_name(fb, detailed_annotations=False), "step[DECODE bs=2]"
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


class TestDetailedAnnotationPlumbing(CustomTestCase):
    def test_default_is_false(self):
        self.assertFalse(ProfileReq().detailed_annotations)

    def test_json_round_trip(self):
        req = ProfileReq(output_dir="/tmp/x", detailed_annotations=True)
        payload = {"detailed_annotations": req.detailed_annotations}
        parsed = json.loads(json.dumps(payload))
        self.assertTrue(parsed["detailed_annotations"])
        self.assertTrue(ProfileReq(**parsed).detailed_annotations)


class TestDetailedAnnotationsToggle(CustomTestCase):
    """The process-wide toggle (set by the profiler manager) is the default source
    for build_step_span_name when no explicit flag is passed."""

    def tearDown(self):
        set_detailed_annotations_enabled(False)

    def test_default_off_no_suffix(self):
        set_detailed_annotations_enabled(False)
        self.assertFalse(detailed_annotations_enabled())
        fb = _fb(ForwardMode.DECODE, batch_size=2, seq_lens_cpu=[10, 20])
        self.assertEqual(build_step_span_name(fb), "step[DECODE bs=2]")

    def test_toggle_on_folds_suffix(self):
        set_detailed_annotations_enabled(True)
        self.assertTrue(detailed_annotations_enabled())
        fb = _fb(ForwardMode.DECODE, batch_size=2, seq_lens_cpu=[10, 20])
        # sq=2, sk=30, sqsq=2, sqsk=30
        self.assertEqual(
            build_step_span_name(fb),
            "step[DECODE bs=2 g_sq=2 g_sqsq=2 g_sqsk=30 g_sk=30]",
        )

    def test_explicit_arg_overrides_toggle(self):
        set_detailed_annotations_enabled(True)
        fb = _fb(ForwardMode.DECODE, batch_size=2, seq_lens_cpu=[10, 20])
        # explicit False wins over the enabled toggle
        self.assertEqual(
            build_step_span_name(fb, detailed_annotations=False), "step[DECODE bs=2]"
        )


if __name__ == "__main__":
    unittest.main()

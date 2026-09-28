"""Unit tests for the ForwardBatch attention plan marker / plan record.

Covers the contract behind ``skip_attn_backend_init`` deprecation:
  * fresh batches need planning; marked batches don't
  * the plan record (planned bs / num tokens) snapshots mark-time shapes
  * reshape after marking triggers a re-plan only for sites that opted
    into ``replan_equivalent``; re-marking re-records the new shapes
  * the deprecated kwarg shim maps explicit values onto the marker
    (mapped, not ignored) and warns once per process

Pure dataclass logic — CPU only.
"""

import unittest
import warnings

import torch

import sglang.srt.model_executor.forward_batch_info as fbi
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


def _make_batch(bs: int = 2, num_tokens: int = 2) -> ForwardBatch:
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=bs,
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        req_pool_indices=torch.zeros(bs, dtype=torch.long),
        seq_lens=torch.ones(bs, dtype=torch.long),
        out_cache_loc=torch.zeros(bs, dtype=torch.long),
        seq_lens_sum=bs,
    )


class TestForwardMetadataPlanRecord(CustomTestCase):
    def test_fresh_batch_needs_planning(self):
        fb = _make_batch()
        self.assertTrue(fb.needs_forward_metadata_init())
        self.assertFalse(fb.forward_metadata_ready)

    def test_mark_records_shapes_and_skips_planning(self):
        fb = _make_batch(bs=3, num_tokens=7)
        fb.mark_forward_metadata_ready()
        self.assertFalse(fb.needs_forward_metadata_init())
        self.assertEqual(fb.forward_metadata_planned_bs, 3)
        self.assertEqual(fb.forward_metadata_planned_num_tokens, 7)

    def test_reshape_without_opt_in_keeps_skipping(self):
        # Wrapper regimes must never auto-re-plan (would clobber per-step metadata).
        fb = _make_batch(bs=2)
        fb.mark_forward_metadata_ready()
        fb.batch_size = 4  # DP padding reshapes the batch
        self.assertFalse(fb.needs_forward_metadata_init())

    def test_reshape_with_opt_in_replans(self):
        fb = _make_batch(bs=2, num_tokens=2)
        fb.mark_forward_metadata_ready(replan_equivalent=True)
        self.assertFalse(fb.needs_forward_metadata_init())

        fb.batch_size = 4  # bs drift (prepare_mlp_sync_batch decode pad)
        self.assertTrue(fb.needs_forward_metadata_init())

        fb.batch_size = 2
        fb.input_ids = torch.zeros(6, dtype=torch.long)  # token drift
        self.assertTrue(fb.needs_forward_metadata_init())

    def test_remark_re_records_padded_shapes(self):
        # Per-step loops re-mark each plan; the re-mark must snapshot padded shapes.
        fb = _make_batch(bs=2)
        fb.mark_forward_metadata_ready(replan_equivalent=True)
        fb.batch_size = 4
        self.assertTrue(fb.needs_forward_metadata_init())
        fb.mark_forward_metadata_ready(replan_equivalent=True)
        self.assertFalse(fb.needs_forward_metadata_init())
        self.assertEqual(fb.forward_metadata_planned_bs, 4)


class TestDeprecatedSkipKwargShim(CustomTestCase):
    def setUp(self):
        self._saved_warned = fbi._skip_attn_backend_init_warned
        fbi._skip_attn_backend_init_warned = False

    def tearDown(self):
        fbi._skip_attn_backend_init_warned = self._saved_warned

    def test_none_is_a_silent_no_op(self):
        fb = _make_batch()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fb.apply_deprecated_skip_attn_backend_init(None)
        self.assertTrue(fb.needs_forward_metadata_init())

    def test_true_maps_onto_marker_and_warns(self):
        # Mapped, not ignored: a no-op would silently re-plan multi-step metadata.
        fb = _make_batch()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fb.apply_deprecated_skip_attn_backend_init(True)
        self.assertFalse(fb.needs_forward_metadata_init())
        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, DeprecationWarning))

    def test_false_warns_but_does_not_mark(self):
        fb = _make_batch()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fb.apply_deprecated_skip_attn_backend_init(False)
        self.assertTrue(fb.needs_forward_metadata_init())
        self.assertEqual(len(caught), 1)

    def test_warns_once_per_process(self):
        # Hot-loop guard: per-forward callers must not pay warnings.warn repeatedly.
        fb = _make_batch()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fb.apply_deprecated_skip_attn_backend_init(True)
            fb.apply_deprecated_skip_attn_backend_init(True)
            _make_batch().apply_deprecated_skip_attn_backend_init(False)
        self.assertEqual(len(caught), 1)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for RepetitionTruncationLogitProcessor."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import FINISH_REPEAT_TRUNCATION
from sglang.srt.sampling.custom_logit_processor import (
    RepetitionTruncationLogitProcessor,
)
from sglang.test.test_utils import CustomTestCase

EOS = 99


class DummyReq:
    def __init__(self, output_ids, rid="req", eos_token_id=EOS):
        self.output_ids = list(output_ids)
        self.repetition_detected = False
        self.to_finish = None
        self.rid = rid
        self.eos_token_ids = {eos_token_id} if eos_token_id is not None else set()


class DummySamplingBatchInfo:
    def __init__(
        self,
        custom_params,
        custom_logit_processor,
        custom_logit_processor_params=None,
        custom_logit_processor_indices=None,
        custom_logit_processor_indices_device=None,
    ):
        self.custom_params = custom_params
        self.custom_logit_processor = custom_logit_processor
        self.custom_logit_processor_params = custom_logit_processor_params
        self.custom_logit_processor_indices = custom_logit_processor_indices
        self.custom_logit_processor_indices_device = (
            custom_logit_processor_indices_device
        )

    def __len__(self):
        return len(self.custom_params)


class TestRepetitionTruncationLogitProcessor(CustomTestCase):
    vocab_size = 128

    def _base_params(self, req, **overrides):
        params = {
            RepetitionTruncationLogitProcessor.REQ_PARAM_KEY: req,
            "ngram_size": 3,
            "window_size": 8,
            "min_content_length": 3,
            "min_repeat": 2,
        }
        params.update(overrides)
        return params

    def _run(self, req, logits, params, req_count=1):
        processor = RepetitionTruncationLogitProcessor()
        return processor(logits, [params] * req_count)

    def _assert_unchanged(self, original, result):
        self.assertTrue(torch.equal(original, result))

    def _assert_skip(self, output_ids, expected_detected=False, **param_overrides):
        logits = torch.randn((1, self.vocab_size))
        req = DummyReq(output_ids)
        params = self._base_params(req, **param_overrides)
        out = self._run(req, logits.clone(), params)
        self._assert_unchanged(logits, out)
        self.assertEqual(req.repetition_detected, expected_detected)
        self.assertIsNone(req.to_finish)

    def _assert_force_stop(self, output_ids, **param_overrides):
        logits = torch.zeros((1, self.vocab_size))
        req = DummyReq(output_ids)
        params = self._base_params(req, **param_overrides)
        out = self._run(req, logits, params)

        self.assertTrue(req.repetition_detected)
        self.assertIsInstance(req.to_finish, FINISH_REPEAT_TRUNCATION)
        self.assertEqual(out[0, EOS].item(), 0.0)
        mask = torch.ones(self.vocab_size, dtype=torch.bool)
        mask[EOS] = False
        self.assertTrue(torch.isinf(out[0][mask]).all())
        self.assertTrue((out[0][mask] < 0).all())

    # --- defaults / config ---

    def test_default_min_repeat_clamped_to_two(self):
        """A min_repeat below 2 is clamped, so a single occurrence never triggers."""
        # [1,2,3] forms one (1,2,3) trigram → one occurrence, must not truncate.
        self._assert_skip([1, 2, 3, 4, 5, 6], min_repeat=1)

    def test_module_defaults_used_when_params_absent(self):
        """When ngram/window params are omitted, the module-level defaults apply."""
        from sglang.srt.sampling import custom_logit_processor as clp

        # A short output is well below the default ngram_size / min_content_length,
        # so with only the req handle provided the processor skips it.
        req = DummyReq([1, 2, 3])
        logits = torch.randn((1, self.vocab_size))
        params = {RepetitionTruncationLogitProcessor.REQ_PARAM_KEY: req}
        out = self._run(req, logits.clone(), params)
        self._assert_unchanged(logits, out)
        self.assertGreaterEqual(clp.DEFAULT_REPETITION_TRUNCATION_MIN_REPEAT, 2)

    def test_invalid_config_skips(self):
        self._assert_skip([1, 2, 3, 1, 2, 3], ngram_size=1, window_size=0)

    def test_window_smaller_than_ngram_skips(self):
        self._assert_skip([1, 2, 3, 4, 1, 2, 3, 4], ngram_size=4, window_size=3)

    def test_min_content_length_skip(self):
        self._assert_skip([1, 2, 3, 4, 5, 6], min_content_length=10)

    def test_seq_len_shorter_than_ngram(self):
        self._assert_skip([1, 2, 3], min_content_length=1, ngram_size=4)

    def test_min_repeat_threshold(self):
        self._assert_skip([1, 2, 3, 1, 2, 3, 4, 5], min_repeat=3)

    def test_repeat_outside_window_skips(self):
        self._assert_skip([1, 2, 3, 7, 8, 9, 1, 2, 3], window_size=3, min_repeat=2)

    # --- detection / truncation ---

    def test_force_stop_on_ngram_repeat(self):
        self._assert_force_stop([1, 2, 3, 1, 2, 3, 4, 5])

    def test_eos_token_id_param_overrides_req(self):
        """custom_params['eos_token_id'] takes precedence over req.eos_token_ids."""
        logits = torch.zeros((1, self.vocab_size))
        req = DummyReq([1, 2, 3, 1, 2, 3, 4, 5], eos_token_id=5)
        params = self._base_params(req, eos_token_id=7)
        out = self._run(req, logits, params)
        self.assertTrue(req.repetition_detected)
        self.assertEqual(out[0, 7].item(), 0.0)
        self.assertTrue(torch.isinf(out[0, 5]) and out[0, 5] < 0)

    def test_debug_detect_only_marks_without_truncating(self):
        with envs.SGLANG_DEBUG_REPETITION_TRUNCATION_DETECT_ONLY.override(True):
            logits = torch.randn((1, self.vocab_size))
            req = DummyReq([1, 2, 3, 1, 2, 3, 4, 5])
            params = self._base_params(req)
            out = self._run(req, logits.clone(), params)
            self._assert_unchanged(logits, out)
            self.assertTrue(req.repetition_detected)
            self.assertIsNone(req.to_finish)

    def test_cache_reset_on_seq_shrink(self):
        with envs.SGLANG_DEBUG_REPETITION_TRUNCATION_DETECT_ONLY.override(True):
            req = DummyReq([1, 2, 3, 4, 5, 6, 7])
            logits = torch.randn((1, self.vocab_size))
            params = self._base_params(req, min_repeat=3)
            out = self._run(req, logits.clone(), params)
            self._assert_unchanged(logits, out)
            self.assertFalse(req.repetition_detected)

            # Shrink the sequence to trigger a cache reset; still no repeat.
            req.output_ids = [1, 2, 3]
            out = self._run(req, logits.clone(), params)
            self._assert_unchanged(logits, out)
            self.assertFalse(req.repetition_detected)

    def test_cached_action_reused_across_rows(self):
        """The same request appearing in multiple rows reuses one decision."""
        req_count = 2
        logits = torch.zeros((req_count, self.vocab_size))
        req = DummyReq([1, 2, 3, 1, 2, 3, 4, 5])
        params = self._base_params(req)
        out = self._run(req, logits, params, req_count=req_count)

        self.assertTrue(req.repetition_detected)
        self.assertIsInstance(req.to_finish, FINISH_REPEAT_TRUNCATION)
        for row in range(req_count):
            self.assertEqual(out[row, EOS].item(), 0.0)
            mask = torch.ones(self.vocab_size, dtype=torch.bool)
            mask[EOS] = False
            self.assertTrue(torch.isinf(out[row][mask]).all())

    # --- integration with the sampler fan-out (spec decoding) ---

    def test_sampler_masks_only_target_rows_with_spec_tokens(self):
        draft_token_num = 2
        batch_size = 3
        logits = torch.arange(
            batch_size * draft_token_num * self.vocab_size, dtype=torch.float
        ).reshape(batch_size * draft_token_num, self.vocab_size)
        original = logits.clone()

        req = DummyReq([1, 2, 3, 1, 2, 3, 4, 5], rid="masked-req")
        params = self._base_params(req)
        custom_params = [None, params, None]
        batch_mask = torch.tensor([False, True, False])

        processor = RepetitionTruncationLogitProcessor()
        sampling_info = DummySamplingBatchInfo(
            custom_params=custom_params,
            custom_logit_processor={1: (processor, batch_mask)},
        )
        apply_custom_logit_processor(
            logits, sampling_info, num_tokens_in_batch=draft_token_num
        )

        changed_rows = [2, 3]
        unchanged_rows = [0, 1, 4, 5]
        self.assertTrue(torch.equal(logits[unchanged_rows], original[unchanged_rows]))
        for row in changed_rows:
            self.assertEqual(logits[row, EOS].item(), 0.0)
            mask = torch.ones(self.vocab_size, dtype=torch.bool)
            mask[EOS] = False
            self.assertTrue(torch.isinf(logits[row][mask]).all())
            self.assertTrue((logits[row][mask] < 0).all())


if __name__ == "__main__":
    unittest.main()

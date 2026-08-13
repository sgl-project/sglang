"""Regression for stop/EOS vs max_new_tokens ordering under speculative decoding
(multi-token commits): a stop token (or stop string) committed mid-run must
finish the request and trim at the stop even when the same run crosses
max_new_tokens. The old length-first ordering finished such steps as
FINISH_LENGTH with finished_len == max_new_tokens, so tokens over-accepted
after the EOS (e.g. the target's degenerate post-EOS prediction — speculative
decoding always appends a bonus token after an accepted EOS draft) leaked into
the emitted output as ``[..., <eos>, <junk>]``. Conversely, a stop matched
*beyond* the cap must not extend the output past the cap: it is demoted to a
length finish. Drives the real `Req.update_finish_state`; pure CPU."""

import unittest
from array import array

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    FINISH_MATCHED_STR,
    FINISH_MATCHED_TOKEN,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

EOS_ID = 2
STOP_ID = 1
ID_TO_TEXT = {
    STOP_ID: "STOP",
    EOS_ID: "",
    **{i: chr(ord("a") + i % 26) for i in range(10, 40)},
}


class _FakeTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None

    def decode(self, ids):
        return "".join(ID_TO_TEXT[int(i)] for i in ids)


class _MockTokenizerForNormalize:
    """Mock tokenizer for normalize() - returns char-count as token list."""

    def encode(self, s, add_special_tokens=False):
        return list(range(len(s)))  # One "token" per character


def _make_req(
    output_ids,
    *,
    max_new_tokens,
    stop=None,
    eos_token_ids=frozenset({EOS_ID}),
    vocab_size=10_000,
):
    sp = SamplingParams(max_new_tokens=max_new_tokens, stop=stop)
    sp.normalize(tokenizer=_MockTokenizerForNormalize())
    req = Req(
        rid="t",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sp,
        eos_token_ids=set(eos_token_ids),
        vocab_size=vocab_size,
    )
    req.tokenizer = _FakeTokenizer()
    req.output_ids = array("q", output_ids)
    return req


class TestFinishLengthSpeculative(CustomTestCase):
    def test_eos_mid_run_beats_length_cap(self):
        # One spec step commits [12, EOS, 20], crossing max_new_tokens=5 in the
        # same step the EOS lands. Length-first ordering finished this as
        # FINISH_LENGTH(finished_len=5) and emitted the over-accepted token
        # after the EOS; the EOS match must win and trim it.
        req = _make_req([10, 11, 12, EOS_ID, 20], max_new_tokens=5)
        req.update_finish_state(new_accepted_len=3)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_reason.matched, EOS_ID)
        self.assertEqual(req.finished_len, 4)
        self.assertEqual(list(req.output_ids_through_stop), [10, 11, 12, EOS_ID])

    def test_stop_str_mid_run_beats_length_cap(self):
        # Same ordering bug for the stop-string branch: "STOP" (token index 2)
        # is inside the run that crosses the cap; the str match must win.
        req = _make_req([10, 11, STOP_ID, 20, 21], max_new_tokens=5, stop=["STOP"])
        req.update_finish_state(new_accepted_len=3)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_STR)
        self.assertEqual(req.finished_reason.matched, "STOP")
        self.assertEqual(req.finished_len, 3)
        self.assertEqual(list(req.output_ids_through_stop), [10, 11, STOP_ID])

    def test_eos_beyond_cap_demoted_to_length(self):
        # The EOS lands past max_new_tokens=4 (position 4, finished_len would be
        # 5): the stop is not emittable within the budget, so the finish must be
        # demoted to FINISH_LENGTH at the cap instead of exceeding it.
        req = _make_req([10, 11, 12, 20, EOS_ID], max_new_tokens=4)
        req.update_finish_state(new_accepted_len=3)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.finished_len, 4)
        self.assertEqual(list(req.output_ids_through_stop), [10, 11, 12, 20])

    def test_no_stop_still_finishes_by_length(self):
        # Negative branch: a run crossing the cap with no stop anywhere must
        # still finish as FINISH_LENGTH at the cap.
        req = _make_req([10, 11, 12, 20, 21], max_new_tokens=4)
        req.update_finish_state(new_accepted_len=3)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.finished_len, 4)
        self.assertEqual(list(req.output_ids_through_stop), [10, 11, 12, 20])

    def test_missing_vocab_size_still_finishes_by_length(self):
        # Prefill-only embedding and scoring requests do not set vocab_size.
        # They must reach the length check without attempting an upper-bound
        # comparison against None.
        req = _make_req([10], max_new_tokens=0, vocab_size=None)
        req.update_finish_state()
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.finished_len, 0)
        self.assertEqual(list(req.output_ids_through_stop), [])

    def test_eos_at_cap_boundary_reports_stop(self):
        # Tie case (also non-spec: new_accepted_len=1): the EOS is exactly the
        # max_new_tokens-th token. The emitted tokens are identical either way;
        # the finish must report the stop match, not the length cap.
        req = _make_req([10, EOS_ID], max_new_tokens=2)
        req.update_finish_state(new_accepted_len=1)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_len, 2)
        self.assertEqual(list(req.output_ids_through_stop), [10, EOS_ID])


# Incident replay with production token ids. Spec verification has no EOS
# awareness: when the draft proposes <|eot|> (200008) and the target accepts it
# mid-run, a bonus token is still sampled one position after the EOS. The
# target's post-EOS argmax is degenerate — deterministically the raw byte
# token 2 — so accept runs end [..., 200008, 2]. The finished_len trim must
# hide that junk: every emitted output must end at [..., 200008].
EOT_ID = 200008  # <|eot|>
POST_EOS_JUNK_ID = 2  # raw byte token: target's degenerate post-EOS prediction
VOCAB_SIZE = 202_048


class TestPostEosBonusTokenIncident(CustomTestCase):
    def _eot_req(self, output_ids, *, max_new_tokens):
        return _make_req(
            output_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=frozenset({EOT_ID}),
            vocab_size=VOCAB_SIZE,
        )

    def test_eot_then_bonus_junk_within_cap_is_trimmed(self):
        # Run [12, 200008, 2] with budget left. This case was correct even
        # without the ordering fix: the EOS match trims the bonus junk.
        req = self._eot_req([10, 11, 12, EOT_ID, POST_EOS_JUNK_ID], max_new_tokens=100)
        req.update_finish_state(new_accepted_len=3)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_reason.matched, EOT_ID)
        self.assertEqual(list(req.output_ids_through_stop), [10, 11, 12, EOT_ID])

    def test_eot_then_bonus_junk_crossing_cap_is_trimmed(self):
        # The incident: the same run crosses max_new_tokens in the step the EOS
        # lands. Without the fix the length check ran first and emitted
        # [..., 200008, 2] (FINISH_LENGTH, finished_len == max_new_tokens);
        # the EOS match must win and the output must end at 200008.
        req = self._eot_req([10, 11, 12, EOT_ID, POST_EOS_JUNK_ID], max_new_tokens=5)
        req.update_finish_state(new_accepted_len=3)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_reason.matched, EOT_ID)
        self.assertEqual(list(req.output_ids_through_stop), [10, 11, 12, EOT_ID])
        self.assertNotIn(POST_EOS_JUNK_ID, req.output_ids_through_stop)

    def test_run_ending_exactly_at_eot_keeps_eot(self):
        # Run [12, 200008] (the EOS itself is the last committed token, e.g. it
        # was the bonus). Same result with and without the fix; pins that the
        # trim keeps the stop token itself and drops nothing else.
        req = self._eot_req([10, 11, 12, EOT_ID], max_new_tokens=100)
        req.update_finish_state(new_accepted_len=2)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_len, 4)
        self.assertEqual(list(req.output_ids_through_stop), [10, 11, 12, EOT_ID])


if __name__ == "__main__":
    unittest.main()

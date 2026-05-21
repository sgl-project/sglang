"""Unit tests for srt/managers/schedule_batch.py — no server, no model loading."""

import unittest
from unittest.mock import MagicMock

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import (  # noqa: E402
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_STR,
    FINISH_MATCHED_TOKEN,
    FINISHED_MATCHED_REGEX,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


_VOCAB_SIZE = 65536


def make_req(
    input_ids=None,
    max_new_tokens=10,
    stop=None,
    stop_token_ids=None,
    ignore_eos=False,
    eos_token_ids=None,
) -> Req:
    sp = SamplingParams(
        max_new_tokens=max_new_tokens,
        stop=stop,
        stop_token_ids=stop_token_ids,
        ignore_eos=ignore_eos,
    )
    sp.normalize(None)
    return Req(
        rid="test-rid",
        origin_input_text="hello",
        origin_input_ids=input_ids if input_ids is not None else [1, 2, 3],
        sampling_params=sp,
        eos_token_ids=eos_token_ids,
        vocab_size=_VOCAB_SIZE,
    )


class TestFinishReasonToJson(CustomTestCase):
    def test_finish_matched_token(self):
        reason = FINISH_MATCHED_TOKEN(matched=42)
        j = reason.to_json()
        self.assertEqual(j["type"], "stop")
        self.assertEqual(j["matched"], 42)
        self.assertFalse(reason.is_error)

    def test_finish_matched_str(self):
        reason = FINISH_MATCHED_STR(matched="</s>")
        j = reason.to_json()
        self.assertEqual(j["type"], "stop")
        self.assertEqual(j["matched"], "</s>")
        self.assertFalse(reason.is_error)

    def test_finished_matched_regex(self):
        reason = FINISHED_MATCHED_REGEX(matched=r"\n\n")
        j = reason.to_json()
        self.assertEqual(j["type"], "stop")
        self.assertEqual(j["matched"], r"\n\n")
        self.assertFalse(reason.is_error)

    def test_finish_length(self):
        reason = FINISH_LENGTH(length=128)
        j = reason.to_json()
        self.assertEqual(j["type"], "length")
        self.assertEqual(j["length"], 128)
        self.assertFalse(reason.is_error)

    def test_finish_abort_defaults(self):
        reason = FINISH_ABORT()
        j = reason.to_json()
        self.assertEqual(j["type"], "abort")
        self.assertTrue(reason.is_error)
        self.assertIsNone(j["status_code"])

    def test_finish_abort_with_fields(self):
        reason = FINISH_ABORT(message="OOM", status_code=500, err_type="oom")
        j = reason.to_json()
        self.assertEqual(j["message"], "OOM")
        self.assertEqual(j["status_code"], 500)
        self.assertEqual(j["err_type"], "oom")


class TestReqProperties(CustomTestCase):
    def test_seqlen_no_output(self):
        req = make_req(input_ids=[10, 20, 30])
        self.assertEqual(req.seqlen, 3)

    def test_seqlen_with_output(self):
        req = make_req(input_ids=[10, 20, 30])
        req.output_ids = [100, 200]
        self.assertEqual(req.seqlen, 5)

    def test_finished_false_initially(self):
        req = make_req()
        self.assertFalse(req.finished())

    def test_finished_true_after_reason_set(self):
        req = make_req()
        req.finished_reason = FINISH_LENGTH(length=10)
        self.assertTrue(req.finished())

    def test_output_ids_through_stop_no_finish_len(self):
        req = make_req()
        req.output_ids = [1, 2, 3, 4, 5]
        req.finished_len = None
        self.assertEqual(req.output_ids_through_stop, [1, 2, 3, 4, 5])

    def test_output_ids_through_stop_with_finish_len(self):
        req = make_req()
        req.output_ids = [1, 2, 3, 4, 5]
        req.finished_len = 3
        self.assertEqual(req.output_ids_through_stop, [1, 2, 3])


class TestCheckFinished(CustomTestCase):
    def test_already_finished_is_noop(self):
        req = make_req()
        req.finished_reason = FINISH_LENGTH(length=5)
        original = req.finished_reason
        req.check_finished()
        self.assertIs(req.finished_reason, original)

    def test_to_finish_transfers_reason(self):
        req = make_req()
        abort = FINISH_ABORT(message="abort")
        req.to_finish = abort
        req.check_finished()
        self.assertIs(req.finished_reason, abort)
        self.assertIsNone(req.to_finish)

    def test_max_new_tokens_triggers_finish_length(self):
        req = make_req(max_new_tokens=3)
        req.output_ids = [1, 2, 3]
        req.check_finished()
        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.finished_reason.length, 3)

    def test_not_finished_when_under_max_new_tokens(self):
        req = make_req(max_new_tokens=5)
        req.output_ids = [1, 2]
        req.check_finished()
        self.assertIsNone(req.finished_reason)

    def test_stop_token_id_triggers_finish(self):
        req = make_req(max_new_tokens=10, stop_token_ids=[99])
        req.output_ids = [10, 20, 99]
        req.check_finished()
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_reason.matched, 99)

    def test_eos_token_ids_triggers_finish(self):
        req = make_req(max_new_tokens=10, eos_token_ids={2})
        req.output_ids = [10, 2]
        req.check_finished()
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_reason.matched, 2)

    def test_ignore_eos_bypasses_token_check(self):
        req = make_req(max_new_tokens=10, stop_token_ids=[99], ignore_eos=True)
        req.output_ids = [99]
        req.check_finished()
        self.assertIsNone(req.finished_reason)

    def test_stop_string_in_decoded_text_triggers_finish(self):
        req = make_req(max_new_tokens=10, stop=["END"])
        req.output_ids = [5, 6, 7]
        req.decoded_text = "some generated text END here"
        mock_tok = MagicMock()
        mock_tok.decode.return_value = ""
        req.tokenizer = mock_tok
        req.check_finished()
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_STR)
        self.assertEqual(req.finished_reason.matched, "END")

    def test_no_finish_condition(self):
        req = make_req(max_new_tokens=10)
        req.output_ids = [1]
        req.check_finished()
        self.assertIsNone(req.finished_reason)

    def test_finished_len_set_on_length_finish(self):
        req = make_req(max_new_tokens=2)
        req.output_ids = [10, 20]
        req.check_finished()
        self.assertEqual(req.finished_len, 2)

    def test_finished_len_set_on_token_finish(self):
        req = make_req(max_new_tokens=5, stop_token_ids=[7])
        req.output_ids = [1, 2, 7]
        req.check_finished()
        self.assertEqual(req.finished_len, 3)


class TestResetForRetract(CustomTestCase):
    def test_retraction_count_increments(self):
        req = make_req()
        self.assertEqual(req.retraction_count, 0)
        req.reset_for_retract()
        self.assertEqual(req.retraction_count, 1)
        req.reset_for_retract()
        self.assertEqual(req.retraction_count, 2)

    def test_prefix_indices_cleared(self):
        import torch

        req = make_req()
        req.prefix_indices = torch.tensor([10, 20, 30], dtype=torch.int64)
        req.reset_for_retract()
        self.assertEqual(req.prefix_indices.numel(), 0)

    def test_extend_input_len_zeroed(self):
        req = make_req()
        req.extend_input_len = 42
        req.reset_for_retract()
        self.assertEqual(req.extend_input_len, 0)

    def test_cache_protected_len_zeroed(self):
        req = make_req()
        req.cache_protected_len = 99
        req.reset_for_retract()
        self.assertEqual(req.cache_protected_len, 0)

    def test_is_retracted_flag_set(self):
        req = make_req()
        self.assertFalse(req.is_retracted)
        req.reset_for_retract()
        self.assertTrue(req.is_retracted)

    def test_retracted_stain_set(self):
        req = make_req()
        self.assertFalse(req.retracted_stain)
        req.reset_for_retract()
        self.assertTrue(req.retracted_stain)

    def test_kv_lengths_zeroed(self):
        req = make_req()
        req.kv_committed_len = 50
        req.kv_allocated_len = 100
        req.reset_for_retract()
        self.assertEqual(req.kv_committed_len, 0)
        self.assertEqual(req.kv_allocated_len, 0)

    def test_kv_freed_flags_reset(self):
        req = make_req()
        req.kv_committed_freed = True
        req.kv_overallocated_freed = True
        req.reset_for_retract()
        self.assertFalse(req.kv_committed_freed)
        self.assertFalse(req.kv_overallocated_freed)

    def test_output_ids_preserved_without_input_embeds(self):
        req = make_req()
        req.output_ids = [10, 20, 30]
        req.input_embeds = None
        req.reset_for_retract()
        self.assertEqual(req.output_ids, [10, 20, 30])

    def test_output_ids_cleared_with_input_embeds(self):
        req = make_req()
        req.output_ids = [10, 20, 30]
        req.input_embeds = [[0.1, 0.2]]
        req.reset_for_retract()
        self.assertEqual(req.output_ids, [])


class TestSetExtendInputLen(CustomTestCase):
    def test_basic_set(self):
        req = make_req()
        req.fill_ids = [1, 2, 3, 4, 5]
        req.set_extend_input_len(5)
        self.assertEqual(req.extend_input_len, 5)

    def test_logprob_start_len_zero_no_prefix(self):
        # logprob_start_len=0, prefix_indices empty → extend_logprob_start_len=0
        req = make_req()
        req.fill_ids = [1, 2, 3, 4]
        req.logprob_start_len = 0
        req.set_extend_input_len(4)
        self.assertEqual(req.extend_logprob_start_len, 0)

    def test_logprob_start_len_positive_with_no_prefix(self):
        # logprob_start_len=2, prefix empty → extend_logprob_start_len = min(2, extend_input_len)
        req = make_req()
        req.fill_ids = [1, 2, 3, 4]
        req.logprob_start_len = 2
        req.set_extend_input_len(4)
        self.assertEqual(req.extend_logprob_start_len, 2)

    def test_logprob_start_len_minus_one_uses_fill_ids(self):
        # logprob_start_len=-1 means start from end of fill_ids (only new tokens get logprobs)
        req = make_req()
        req.fill_ids = [1, 2, 3, 4, 5]
        req.logprob_start_len = -1
        req.set_extend_input_len(3)
        # logprob_start_len = len(fill_ids) = 5, prefix_indices = empty (len=0)
        # extend_logprob_start_len = min(5 - 0, 3) = 3
        self.assertEqual(req.extend_logprob_start_len, 3)

    def test_extend_logprob_start_len_clamped_at_extend_input_len(self):
        # logprob_start_len > extend_input_len → clamped
        req = make_req()
        req.fill_ids = [1, 2, 3, 4, 5]
        req.logprob_start_len = 100
        req.set_extend_input_len(3)
        self.assertLessEqual(req.extend_logprob_start_len, req.extend_input_len)


class TestUpdateReasoningTokens(CustomTestCase):
    def test_no_think_end_accumulates_all(self):
        req = make_req()
        req.update_reasoning_tokens([10, 20, 30], think_end_id=99)
        self.assertEqual(req.reasoning_tokens, 3)
        self.assertFalse(req._is_reasoning_over)

    def test_think_end_in_middle_stops_counting(self):
        # think_end_id at index 1 → count 2 tokens (0 and 1), mark over
        req = make_req()
        req.update_reasoning_tokens([10, 99, 30, 40], think_end_id=99)
        self.assertEqual(req.reasoning_tokens, 2)
        self.assertTrue(req._is_reasoning_over)

    def test_think_end_is_first_token(self):
        req = make_req()
        req.update_reasoning_tokens([99, 10, 20], think_end_id=99)
        self.assertEqual(req.reasoning_tokens, 1)
        self.assertTrue(req._is_reasoning_over)

    def test_think_end_is_last_token(self):
        req = make_req()
        req.update_reasoning_tokens([10, 20, 99], think_end_id=99)
        self.assertEqual(req.reasoning_tokens, 3)
        self.assertTrue(req._is_reasoning_over)

    def test_noop_after_reasoning_over(self):
        req = make_req()
        req.update_reasoning_tokens([99], think_end_id=99)
        self.assertEqual(req.reasoning_tokens, 1)
        # second call must be a no-op
        req.update_reasoning_tokens([10, 20, 30], think_end_id=99)
        self.assertEqual(req.reasoning_tokens, 1)

    def test_single_int_input_treated_as_list(self):
        req = make_req()
        req.update_reasoning_tokens(42, think_end_id=99)
        self.assertEqual(req.reasoning_tokens, 1)
        self.assertFalse(req._is_reasoning_over)

    def test_single_int_is_think_end(self):
        req = make_req()
        req.update_reasoning_tokens(99, think_end_id=99)
        self.assertEqual(req.reasoning_tokens, 1)
        self.assertTrue(req._is_reasoning_over)

    def test_accumulates_across_multiple_calls(self):
        req = make_req()
        req.update_reasoning_tokens([1, 2], think_end_id=99)
        req.update_reasoning_tokens([3, 4], think_end_id=99)
        self.assertEqual(req.reasoning_tokens, 4)


class TestVocabBoundaryFinish(CustomTestCase):
    def test_all_valid_tokens_returns_false(self):
        req = make_req()
        req.output_ids = [1, 2, 3]
        result = req._check_vocab_boundary_finish([1, 2, 3])
        self.assertFalse(result)
        self.assertIsNone(req.finished_reason)

    def test_token_above_vocab_size_triggers_finish(self):
        req = make_req()
        req.output_ids = [100000]
        result = req._check_vocab_boundary_finish([100000])
        self.assertTrue(result)
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_STR)
        self.assertEqual(req.finished_reason.matched, "NaN happened")
        self.assertEqual(req.finished_len, 1)

    def test_negative_token_triggers_finish(self):
        req = make_req()
        req.output_ids = [-1]
        result = req._check_vocab_boundary_finish([-1])
        self.assertTrue(result)
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_STR)

    def test_bad_token_replaced_by_stop_token_id(self):
        req = make_req(stop_token_ids=[7])
        req.output_ids = [100000]
        req._check_vocab_boundary_finish([100000])
        self.assertEqual(req.output_ids[0], 7)

    def test_bad_token_replaced_by_eos_token_id(self):
        req = make_req(eos_token_ids={2})
        req.output_ids = [100000]
        req._check_vocab_boundary_finish([100000])
        self.assertEqual(req.output_ids[0], 2)

    def test_eos_wins_over_stop_token_when_both_set(self):
        # eos_token_ids assignment comes last in the source, so it wins
        req = make_req(stop_token_ids=[7], eos_token_ids={2})
        req.output_ids = [100000]
        req._check_vocab_boundary_finish([100000])
        self.assertEqual(req.output_ids[0], 2)

    def test_bad_token_mid_sequence_sets_correct_finished_len(self):
        req = make_req()
        req.output_ids = [1, 2, 100000, 4]
        req._check_vocab_boundary_finish([1, 2, 100000, 4])
        # bad token is at index 2 → finished_len = 3
        self.assertEqual(req.finished_len, 3)

    def test_stops_at_first_bad_token(self):
        req = make_req()
        req.output_ids = [100000, 200000]
        req._check_vocab_boundary_finish([100000, 200000])
        # only the first bad token triggers finish
        self.assertEqual(req.finished_len, 1)


class TestIncrementalDetokenize(CustomTestCase):
    def test_first_call_sets_read_and_surr_offset(self):
        req = make_req(input_ids=list(range(10)))
        req.output_ids = [100, 101]
        ids, rel_offset = req.init_incremental_detokenize()
        # read_offset = len(origin_input_ids_unpadded) = 10
        self.assertEqual(req.read_offset, 10)
        # surr_offset = max(10 - 5, 0) = 5
        self.assertEqual(req.surr_offset, 5)
        # rel_offset = read_offset - surr_offset = 5
        self.assertEqual(rel_offset, 5)
        # ids = input_ids[5:] + output_ids = [5,6,7,8,9,100,101]
        self.assertEqual(ids, [5, 6, 7, 8, 9, 100, 101])

    def test_first_call_short_input_surr_offset_is_zero(self):
        req = make_req(input_ids=[1, 2])  # len=2 < OFFSET(5)
        req.output_ids = []
        _, rel_offset = req.init_incremental_detokenize()
        self.assertEqual(req.surr_offset, 0)
        self.assertEqual(req.read_offset, 2)
        self.assertEqual(rel_offset, 2)

    def test_first_call_empty_output(self):
        req = make_req(input_ids=[10, 20, 30])
        req.output_ids = []
        ids, _ = req.init_incremental_detokenize()
        # surr_and_decode_ids = input_ids[surr_offset:] + [] = [10,20,30] (surr_offset=0)
        self.assertEqual(ids, [10, 20, 30])

    def test_second_call_extends_with_new_tokens(self):
        req = make_req(input_ids=list(range(10)))
        req.output_ids = [100]
        req.init_incremental_detokenize()  # first call

        req.output_ids = [100, 101, 102]
        ids, rel_offset = req.init_incremental_detokenize()  # second call
        # surr_and_decode_ids should have the 2 new tokens appended
        self.assertIn(101, ids)
        self.assertIn(102, ids)
        # rel_offset stays the same (read_offset and surr_offset don't change)
        self.assertEqual(rel_offset, req.read_offset - req.surr_offset)

    def test_second_call_no_new_tokens_is_stable(self):
        req = make_req(input_ids=[1, 2, 3])
        req.output_ids = [10, 11]
        ids_first, offset_first = req.init_incremental_detokenize()

        ids_second, offset_second = req.init_incremental_detokenize()
        self.assertEqual(ids_first, ids_second)
        self.assertEqual(offset_first, offset_second)


class TestCheckMatchStopStrPrefix(CustomTestCase):
    def _req_with_mock_tokenizer(self, stop, decoded_output):
        req = make_req(stop=stop)
        mock_tok = MagicMock()
        mock_tok.decode.return_value = decoded_output
        req.tokenizer = mock_tok
        req.output_ids = [1]  # non-empty so tail_str doesn't short-circuit on len
        return req

    def test_no_stop_strs_returns_false(self):
        req = make_req()  # stop=None → stop_strs=[]
        self.assertFalse(req.check_match_stop_str_prefix())

    def test_empty_tail_str_returns_false(self):
        req = self._req_with_mock_tokenizer(stop=["END"], decoded_output="")
        self.assertFalse(req.check_match_stop_str_prefix())

    def test_full_stop_str_contained_returns_true(self):
        req = self._req_with_mock_tokenizer(
            stop=["END"], decoded_output="some END here"
        )
        self.assertTrue(req.check_match_stop_str_prefix())

    def test_partial_suffix_matches_stop_prefix_returns_true(self):
        # tail ends with "EN", stop is "END" → "EN" == "END"[:2]
        req = self._req_with_mock_tokenizer(stop=["END"], decoded_output="EN")
        self.assertTrue(req.check_match_stop_str_prefix())

    def test_single_char_overlap_returns_true(self):
        # tail ends with "E", stop is "END" → "E" == "END"[:1]
        req = self._req_with_mock_tokenizer(stop=["END"], decoded_output="E")
        self.assertTrue(req.check_match_stop_str_prefix())

    def test_no_overlap_returns_false(self):
        req = self._req_with_mock_tokenizer(stop=["END"], decoded_output="xyz")
        self.assertFalse(req.check_match_stop_str_prefix())

    def test_multiple_stop_strs_any_match_returns_true(self):
        req = self._req_with_mock_tokenizer(stop=["FOO", "END"], decoded_output="EN")
        self.assertTrue(req.check_match_stop_str_prefix())

    def test_multiple_stop_strs_no_match_returns_false(self):
        req = self._req_with_mock_tokenizer(stop=["FOO", "BAR"], decoded_output="xyz")
        self.assertFalse(req.check_match_stop_str_prefix())


if __name__ == "__main__":
    unittest.main(verbosity=2)

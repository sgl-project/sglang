"""Unit tests for the streaming-session in-place token-array share protocol
(`Session.create_req` / `finish_req` / `abort_req`):

- token arrays are extended in place and shared across turns (no per-turn copy);
- committed_* lengths recorded at finish_req trim away tokens appended by a
  turn that aborted before finishing (mid-turn and first-turn aborts);
- max_new_tokens overshoot falls back to a fill_ids rebuild instead of
  carrying an inconsistent array.

And for non-streaming sessions, that a close (explicit or by the timeout
reaper) does not free media an unfinished request reads.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.io_struct import CloseSessionReqInput
from sglang.srt.managers.req_types import (
    FINISH_LENGTH,
    FINISH_MATCHED_STR,
    FINISH_MATCHED_TOKEN,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session, SessionController
from sglang.test.test_utils import CustomTestCase

VOCAB = 1 << 20
EOS_ID = 2

_STOP_STR_ID_TO_TEXT = {
    10: "p",
    41: "a",
    42: "E",
    43: "N",
    44: "D",
    45: "x",
    46: "y",
    90: "q",
}


class _FakeStopStrTokenizer:
    bos_token_id = None
    eos_token_id = -1
    additional_stop_token_ids = None

    def decode(self, ids):
        return "".join(_STOP_STR_ID_TO_TEXT[int(i)] for i in ids)

    def encode(self, s, add_special_tokens=False):
        return list(range(len(s)))


def _recv(
    rid,
    input_ids,
    max_new_tokens=8,
    parent=None,
    *,
    offset=None,
    drop_previous_output=False,
    sampling_params=None,
):
    if sampling_params is None:
        sampling_params = SamplingParams(max_new_tokens=max_new_tokens)
        sampling_params.normalize(tokenizer=None)
    return SimpleNamespace(
        rid=rid,
        input_ids=array("q", input_ids),
        mm_inputs=None,
        session_params=SimpleNamespace(
            id="s",
            rid=parent,
            offset=offset,
            replace=False,
            drop_previous_output=drop_previous_output,
        ),
        sampling_params=sampling_params,
        lora_id=None,
        custom_logit_processor=None,
        stream=False,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        return_sampling_mask=False,
        sampling_logprobs_mode="selected",
        require_reasoning=False,
        return_hidden_states=False,
        return_routed_experts=False,
        routed_experts_start_len=0,
        bootstrap_host=None,
        bootstrap_port=None,
        bootstrap_room=None,
        routed_dp_rank=None,
        disagg_prefill_dp_rank=None,
        priority=None,
        routing_key=None,
        extra_key=None,
        cache_salt=None,
        http_worker_ipc=None,
        time_stats=None,
        filler_seed=None,
    )


class TestSessionTokenShare(CustomTestCase):
    def setUp(self):
        self.session = Session(capacity_of_str_len=0, session_id="s", streaming=True)

    def _create(self, rid, input_ids, max_new_tokens=8, *, eos_token_ids=None):
        return self.session.create_req(
            _recv(rid, input_ids, max_new_tokens=max_new_tokens),
            tokenizer=None,
            vocab_size=VOCAB,
            eos_token_ids=eos_token_ids,
        )

    def _decode_and_finish(self, req, output, baked=None):
        """Simulate decode then a successful finish.

        `baked` output tokens are folded into the fill array before the rest
        arrive (mix_with_running refreshes mid-decode, so the bake is often
        partial).
        """
        if baked is None:
            baked = len(output)
        req.output_ids.extend(output[:baked])
        req.refresh_fill_ids()
        req.output_ids.extend(output[baked:])
        self.session.finish_req(req)

    def test_full_media_namespace_survives_text_only_turns(self):
        first = self._create("first", [1, 2, 3])
        first.extend_image_inputs(
            MultimodalInputs(
                mm_items=[
                    MultimodalDataItem(modality=Modality.IMAGE, identity=b"A" * 32)
                ]
            )
        )
        first_salt = first.cache_salt
        self._decode_and_finish(first, [4])
        second = self._create("second", [5])
        self.assertEqual(second.cache_salt, first_salt)
        second.extend_image_inputs(
            MultimodalInputs(
                mm_items=[
                    MultimodalDataItem(modality=Modality.IMAGE, identity=b"B" * 32)
                ]
            )
        )
        self.assertNotEqual(second.cache_salt, first_salt)
        self.assertEqual(first.cache_salt, first_salt)
        self._decode_and_finish(second, [6])
        third = self._create("third", [7])
        self.assertEqual(third.cache_salt, second.cache_salt)

    def test_normal_multi_turn_share_and_carry(self):
        in1, out1 = list(range(100, 110)), [1, 2, 3]
        r1 = self._create("r1", in1)
        self.assertEqual(list(r1.origin_input_ids), in1)
        self._decode_and_finish(r1, out1, baked=2)  # partial bake
        self.assertEqual(self.session.committed_origin_len, len(in1))
        self.assertEqual(self.session.committed_fill_len, len(in1) + 2)

        in2, out2 = [7, 8], [4, 5]
        r2 = self._create("r2", in2)
        # In-place share: same objects, extended to the new prompt.
        self.assertIs(r2.origin_input_ids, r1.origin_input_ids)
        self.assertEqual(list(r2.origin_input_ids), in1 + out1 + in2)
        # Carry: the fill array handed over and equal to the new origin.
        self.assertIs(r2.full_untruncated_fill_ids, r1.full_untruncated_fill_ids)
        self.assertEqual(list(r2.full_untruncated_fill_ids), list(r2.origin_input_ids))
        self._decode_and_finish(r2, out2)

        r3 = self._create("r3", [9])
        self.assertEqual(list(r3.origin_input_ids), in1 + out1 + in2 + out2 + [9])
        self.assertEqual(list(r3.full_untruncated_fill_ids), list(r3.origin_input_ids))

    def test_multimodal_counts_follow_committed_history(self):
        """An aborted turn's count edits must not leak into the next turn."""
        r1 = self._create("r1", [1])
        r1.mm_image_tokens, r1.mm_audio_tokens, r1.mm_video_tokens = (2, 3, 4)
        r1.mm_token_counts_ready = True
        self._decode_and_finish(r1, [])

        r2 = self._create("r2", [2])
        self.assertEqual(r2.get_mm_token_counts(), (2, 3, 4))
        r2.mm_image_tokens = 5
        self.session.abort_req()

        r3 = self._create("r3", [3])
        self.assertEqual(r3.get_mm_token_counts(), (2, 3, 4))

    def test_mid_turn_abort_then_continue(self):
        in1, out1 = list(range(200, 210)), [1, 2, 3]
        r1 = self._create("r1", in1)
        self._decode_and_finish(r1, out1)

        # Turn 2 extends the shared arrays, decodes a bit, then aborts:
        # finish_req never runs, turns still points at r1.
        r2 = self._create("r2", [50, 51])
        self.assertEqual(list(r2.origin_input_ids), in1 + out1 + [50, 51])
        r2.output_ids.extend([6, 7])
        r2.refresh_fill_ids()
        self.session.abort_req()
        self.assertEqual(self.session.committed_origin_len, len(in1))

        # Turn 3 must see exactly r1's history — no [50, 51], no doubled out1.
        r3 = self._create("r3", [60])
        self.assertEqual(list(r3.origin_input_ids), in1 + out1 + [60])
        self.assertEqual(list(r3.full_untruncated_fill_ids), list(r3.origin_input_ids))

        # Two aborted attempts in a row heal idempotently.
        self.session.abort_req()
        r4 = self._create("r4", [70])
        self.assertEqual(list(r4.origin_input_ids), in1 + out1 + [70])
        self.assertEqual(list(r4.full_untruncated_fill_ids), list(r4.origin_input_ids))

    def test_first_turn_abort(self):
        self._create("r1", [1, 2, 3])
        self.assertTrue(self.session._inflight)
        self.session.abort_req()
        self.assertFalse(self.session._inflight)
        # No finish_req ran: nothing committed, next turn starts from scratch.
        self.assertIsNone(self.session.committed_origin_len)
        r2 = self._create("r2", [4, 5])
        self.assertEqual(list(r2.origin_input_ids), [4, 5])
        self._decode_and_finish(r2, [9])
        r3 = self._create("r3", [6])
        self.assertEqual(list(r3.origin_input_ids), [4, 5, 9, 6])

    def test_max_new_tokens_overshoot_falls_back(self):
        in1 = list(range(300, 310))
        r1 = self._create("r1", in1, max_new_tokens=4)
        # Spec-decode overshoot: 6 tokens decoded and baked into the fill
        # array, then output trimmed to finished_len (like _trim_overshoot)
        # before finish.
        r1.output_ids.extend([1, 2, 3, 4, 5, 6])
        r1.refresh_fill_ids()
        del r1.output_ids[4:]
        self.session.finish_req(r1)
        self.assertEqual(
            self.session.committed_fill_len, len(in1) + 6
        )  # fill kept the overshoot

        # Next turn: out_tail is output[:max_new]; the carried fill has more
        # baked than out_tail, so the carry is dropped and the fill rebuilds.
        r2 = self._create("r2", [50])
        self.assertEqual(list(r2.origin_input_ids), in1 + [1, 2, 3, 4] + [50])
        self.assertEqual(len(r2.full_untruncated_fill_ids), 0)  # carry skipped
        r2.refresh_fill_ids()
        self.assertEqual(list(r2.full_untruncated_fill_ids), list(r2.origin_input_ids))

    def test_finished_len_without_inplace_trim_and_carry_fallback(self):
        """_share_token_arrays respects finished_len even without prior in-place
        output_ids trimming, dropping carry_fill when baked exceeds out_tail."""
        # Case 1: baked fill includes post-stop overshoot -> carry_fill falls back.
        in1 = list(range(400, 405))
        r1 = self._create("r1", in1, max_new_tokens=8, eos_token_ids={EOS_ID})
        r1.output_ids.extend([41, EOS_ID, 77, 78])
        r1.refresh_fill_ids()
        r1.update_finish_state(new_accepted_len=4)
        self.assertEqual(r1.finished_len, 2)
        self.assertEqual(list(r1.output_ids), [41, EOS_ID, 77, 78])
        self.session.finish_req(r1)

        r2 = self._create("r2", [50])
        self.assertIs(r2.origin_input_ids, r1.origin_input_ids)
        self.assertEqual(list(r2.origin_input_ids), in1 + [41, EOS_ID, 50])
        self.assertEqual(list(r2.origin_input_ids_unpadded), in1 + [41, EOS_ID, 50])
        self.assertEqual(len(r2.full_untruncated_fill_ids), 0)
        r2.refresh_fill_ids()
        self.assertEqual(list(r2.full_untruncated_fill_ids), list(r2.origin_input_ids))

        # Case 2: baked fill within trimmed out_tail -> carry_fill succeeds.
        self.setUp()
        r1 = self._create("r1", in1, max_new_tokens=8, eos_token_ids={EOS_ID})
        r1.output_ids.extend([41])
        r1.refresh_fill_ids()
        r1.output_ids.extend([EOS_ID, 77, 78])
        r1.update_finish_state(new_accepted_len=3)
        self.assertEqual(r1.finished_len, 2)
        self.session.finish_req(r1)

        r2 = self._create("r2", [50])
        self.assertEqual(list(r2.origin_input_ids), in1 + [41, EOS_ID, 50])
        self.assertIs(r2.full_untruncated_fill_ids, r1.full_untruncated_fill_ids)
        self.assertEqual(list(r2.full_untruncated_fill_ids), in1 + [41, EOS_ID, 50])

        # Case 3: finished_len == 0 with non-empty output_ids and max_new_tokens > 0.
        self.setUp()
        r1 = self._create("r1", in1, max_new_tokens=8)
        r1.output_ids.extend([41, 42])
        r1.refresh_fill_ids()
        r1.finished_reason = FINISH_LENGTH(length=0)
        r1.finished_len = 0
        self.session.finish_req(r1)

        r2 = self._create("r2", [50])
        self.assertEqual(list(r2.origin_input_ids), in1 + [50])
        self.assertEqual(list(r2.origin_input_ids_unpadded), in1 + [50])
        self.assertEqual(len(r2.full_untruncated_fill_ids), 0)

        # Case 4: finished_len is None with len(output_ids) > max_new_tokens.
        self.setUp()
        r1 = self._create("r1", in1, max_new_tokens=2)
        r1.output_ids.extend([41, 42, 43, 44])
        r1.refresh_fill_ids()
        r1.finish_outside_step(FINISH_LENGTH(length=2))
        self.assertIsNone(r1.finished_len)
        self.session.finish_req(r1)

        r2 = self._create("r2", [50])
        self.assertEqual(list(r2.origin_input_ids), in1 + [41, 42, 50])
        self.assertEqual(len(r2.full_untruncated_fill_ids), 0)


class TestSessionFinishedLenConcat(CustomTestCase):
    """Non-streaming sessions splice prior outputs via _concat_token_arrays and
    must stop at Req.output_ids_through_stop (capped at max_new_tokens)."""

    def _new_session(self):
        return Session(capacity_of_str_len=0, session_id="s", streaming=False)

    def _create(
        self,
        session,
        rid,
        input_ids,
        *,
        parent=None,
        max_new_tokens=8,
        offset=None,
        drop_previous_output=False,
        sampling_params=None,
        tokenizer=None,
        vocab_size=VOCAB,
        eos_token_ids=None,
    ):
        return session.create_req(
            _recv(
                rid,
                input_ids,
                max_new_tokens=max_new_tokens,
                parent=parent,
                offset=offset,
                drop_previous_output=drop_previous_output,
                sampling_params=sampling_params,
            ),
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            eos_token_ids=eos_token_ids,
        )

    def test_stop_token_overshoot_matches_step_by_step(self):
        """Multi-token commit past EOS matches step-by-step decoding on Turn 2."""
        spec_session = self._new_session()
        t1_spec = self._create(
            spec_session, "t1", [10], max_new_tokens=8, eos_token_ids={EOS_ID}
        )
        t1_spec.output_ids.extend([41, EOS_ID, 77, 78])
        t1_spec.update_finish_state(new_accepted_len=4)
        self.assertIsInstance(t1_spec.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(t1_spec.finished_len, 2)
        self.assertEqual(list(t1_spec.output_ids), [41, EOS_ID, 77, 78])
        t2_spec = self._create(spec_session, "t2", [90], parent="t1")

        step_session = self._new_session()
        t1_step = self._create(
            step_session, "t1", [10], max_new_tokens=8, eos_token_ids={EOS_ID}
        )
        for tok in (41, EOS_ID):
            t1_step.output_ids.append(tok)
            t1_step.update_finish_state(new_accepted_len=1)
        self.assertEqual(t1_step.finished_len, 2)
        t2_step = self._create(step_session, "t2", [90], parent="t1")

        expected = [10, 41, EOS_ID, 90]
        self.assertEqual(list(t2_spec.origin_input_ids), expected)
        self.assertEqual(list(t2_spec.origin_input_ids_unpadded), expected)
        self.assertEqual(list(t2_spec.origin_input_ids), list(t2_step.origin_input_ids))
        self.assertEqual(
            list(t2_spec.origin_input_ids_unpadded),
            list(t2_step.origin_input_ids_unpadded),
        )

    def test_stop_string_overshoot_excludes_post_stop_tokens(self):
        """Stop string matched mid-step carries only tokens through the stop."""
        tok = _FakeStopStrTokenizer()
        sp = SamplingParams(max_new_tokens=8, stop=["END"])
        sp.normalize(tokenizer=tok)

        spec_session = self._new_session()
        t1_spec = self._create(
            spec_session, "t1", [10], sampling_params=sp, tokenizer=tok
        )
        # [41, 42, 43, 44, 45, 46] decodes to "aENDxy".
        t1_spec.output_ids.extend([41, 42, 43, 44, 45, 46])
        t1_spec.update_finish_state(new_accepted_len=6)
        self.assertIsInstance(t1_spec.finished_reason, FINISH_MATCHED_STR)
        self.assertEqual(t1_spec.finished_reason.matched, "END")
        self.assertEqual(t1_spec.finished_len, 4)
        self.assertEqual(list(t1_spec.output_ids), [41, 42, 43, 44, 45, 46])
        t2_spec = self._create(spec_session, "t2", [90], parent="t1", tokenizer=tok)

        step_session = self._new_session()
        t1_step = self._create(
            step_session, "t1", [10], sampling_params=sp, tokenizer=tok
        )
        for token_id in (41, 42, 43, 44):
            t1_step.output_ids.append(token_id)
            t1_step.update_finish_state(new_accepted_len=1)
        t2_step = self._create(step_session, "t2", [90], parent="t1", tokenizer=tok)

        expected = [10, 41, 42, 43, 44, 90]
        self.assertEqual(list(t2_spec.origin_input_ids), expected)
        self.assertEqual(list(t2_spec.origin_input_ids_unpadded), expected)
        self.assertEqual(list(t2_spec.origin_input_ids), list(t2_step.origin_input_ids))
        self.assertEqual(
            list(t2_spec.origin_input_ids_unpadded),
            list(t2_step.origin_input_ids_unpadded),
        )

    def test_vocab_boundary_overshoot_does_not_leak_unsanitized_tokens(self):
        """Vocab-boundary/NaN stop mid-step must not leak unsanitized trailing
        negative or out-of-vocab tokens into Turn 2."""
        vocab_size = 128
        for name, committed in (
            ("oov_then_negative", [41, vocab_size, -5]),
            ("negative_then_oov", [41, -1, vocab_size + 7]),
        ):
            with self.subTest(name):
                session = self._new_session()
                t1 = self._create(
                    session,
                    "t1",
                    [10],
                    max_new_tokens=8,
                    vocab_size=vocab_size,
                    eos_token_ids={EOS_ID},
                )
                t1.output_ids.extend(committed)
                t1.update_finish_state(new_accepted_len=len(committed))
                self.assertIsInstance(t1.finished_reason, FINISH_MATCHED_STR)
                self.assertEqual(t1.finished_reason.matched, "NaN happened")
                self.assertEqual(t1.finished_len, 2)
                # The trailing token past finished_len stays unsanitized in output_ids.
                self.assertEqual(t1.output_ids[2], committed[2])

                t2 = self._create(
                    session, "t2", [90], parent="t1", vocab_size=vocab_size
                )
                self.assertEqual(list(t2.origin_input_ids), [10, 41, EOS_ID, 90])
                self.assertEqual(
                    list(t2.origin_input_ids_unpadded), [10, 41, EOS_ID, 90]
                )
                self.assertTrue(
                    all(0 <= tid < vocab_size for tid in t2.origin_input_ids)
                )

    def test_offset_drop_previous_output_and_distinct_unpadded_arrays(self):
        """Negative/positive offset, drop_previous_output, and distinct padded vs
        unpadded origins all operate on output_ids_through_stop."""
        session = self._new_session()
        t1 = self._create(session, "t1", [10], max_new_tokens=8, eos_token_ids={EOS_ID})
        t1.output_ids.extend([41, EOS_ID, 77, 78])
        t1.update_finish_state(new_accepted_len=4)
        self.assertEqual(t1.finished_len, 2)

        # Negative offset=-1 drops the stop token (EOS_ID), not overshoot token 78.
        t2_neg = self._create(session, "t2_neg", [90], parent="t1", offset=-1)
        self.assertEqual(list(t2_neg.origin_input_ids), [10, 41, 90])
        self.assertEqual(list(t2_neg.origin_input_ids_unpadded), [10, 41, 90])

        # Positive offset within trimmed history slices normally.
        t2_pos_in = self._create(session, "t2_pos_in", [90], parent="t1", offset=2)
        self.assertEqual(list(t2_pos_in.origin_input_ids), [10, 41, 90])
        self.assertEqual(list(t2_pos_in.origin_input_ids_unpadded), [10, 41, 90])

        # Positive offset past len(origin) + finished_len must not resurrect overshoot.
        t2_pos_past = self._create(session, "t2_pos_past", [90], parent="t1", offset=4)
        self.assertEqual(list(t2_pos_past.origin_input_ids), [10, 41, EOS_ID, 90])
        self.assertEqual(
            list(t2_pos_past.origin_input_ids_unpadded), [10, 41, EOS_ID, 90]
        )

        # drop_previous_output=True omits Turn 1 output entirely.
        t2_drop = self._create(
            session, "t2_drop", [90], parent="t1", drop_previous_output=True
        )
        self.assertEqual(list(t2_drop.origin_input_ids), [10, 90])
        self.assertEqual(list(t2_drop.origin_input_ids_unpadded), [10, 90])

        # Distinct origin_input_ids vs origin_input_ids_unpadded both use out_tail.
        t1_mm = self._create(
            session, "t1_mm", [10, 11, 12], max_new_tokens=8, eos_token_ids={EOS_ID}
        )
        t1_mm.origin_input_ids_unpadded = array("q", [10, 12])
        t1_mm.output_ids.extend([41, EOS_ID, 77, 78])
        t1_mm.update_finish_state(new_accepted_len=4)

        t2_mm = self._create(session, "t2_mm", [90], parent="t1_mm")
        self.assertEqual(list(t2_mm.origin_input_ids), [10, 11, 12, 41, EOS_ID, 90])
        self.assertEqual(
            list(t2_mm.origin_input_ids_unpadded), [10, 12, 41, EOS_ID, 90]
        )

        t2_mm_neg = self._create(session, "t2_mm_neg", [90], parent="t1_mm", offset=-1)
        self.assertEqual(list(t2_mm_neg.origin_input_ids), [10, 11, 12, 41, 90])
        self.assertEqual(list(t2_mm_neg.origin_input_ids_unpadded), [10, 12, 41, 90])

        t2_mm_drop = self._create(
            session, "t2_mm_drop", [90], parent="t1_mm", drop_previous_output=True
        )
        self.assertEqual(list(t2_mm_drop.origin_input_ids), [10, 11, 12, 90])
        self.assertEqual(list(t2_mm_drop.origin_input_ids_unpadded), [10, 12, 90])

    def test_finished_len_zero_and_none_boundaries(self):
        """finished_len == 0 must not be treated as falsy; finished_len is None
        must still obey the max_new_tokens cap."""
        session = self._new_session()

        # finished_len == 0 with non-empty output_ids and max_new_tokens > 0.
        t1_zero = self._create(session, "t1_zero", [10], max_new_tokens=8)
        t1_zero.output_ids.extend([41, 42])
        t1_zero.finished_reason = FINISH_LENGTH(length=0)
        t1_zero.finished_len = 0
        t2_zero = self._create(session, "t2_zero", [90], parent="t1_zero")
        self.assertEqual(list(t2_zero.origin_input_ids), [10, 90])
        self.assertEqual(list(t2_zero.origin_input_ids_unpadded), [10, 90])

        # finished_len is None with len(output_ids) > max_new_tokens.
        t1_none = self._create(session, "t1_none", [10], max_new_tokens=2)
        t1_none.output_ids.extend([41, 42, 43, 44])
        t1_none.finish_outside_step(FINISH_LENGTH(length=2))
        self.assertIsNone(t1_none.finished_len)
        t2_none = self._create(session, "t2_none", [90], parent="t1_none")
        self.assertEqual(list(t2_none.origin_input_ids), [10, 41, 42, 90])
        self.assertEqual(list(t2_none.origin_input_ids_unpadded), [10, 41, 42, 90])


class TestSessionTeardown(CustomTestCase):
    """A non-streaming session's turns share their parents' media items; the
    session frees them all at close."""

    def setUp(self):
        self.controller = SessionController(MagicMock())
        self.controller.open(
            SimpleNamespace(
                session_id="s", capacity_of_str_len=0, streaming=False, timeout=None
            )
        )
        self.session = self.controller.get("s")
        self.now = 0.0

    def _create(self, rid, parent=None, image=None):
        req = self.session.create_req(
            _recv(rid, [1, 2], parent=parent),
            tokenizer=None,
            vocab_size=VOCAB,
        )
        if image is not None:
            req.extend_image_inputs(
                MultimodalInputs(
                    mm_items=[
                        MultimodalDataItem(
                            modality=Modality.IMAGE,
                            identity=image * 32,
                            feature=torch.ones(1),
                        )
                    ]
                )
            )
        return req

    @staticmethod
    def _finish(req):
        req.finish_outside_step(FINISH_LENGTH(length=1))

    def _reap(self):
        self.now += 2.0
        self.controller.maybe_reap(self.now)

    def _close(self):
        self.controller.close(CloseSessionReqInput(session_id="s"))

    def test_close_waits_for_unfinished_reader(self):
        """A close, explicit or by timeout, waits for an unfinished child that
        reads its parent's image, then frees the image."""
        for by_timeout in (False, True):
            with self.subTest(by_timeout=by_timeout):
                self.setUp()
                parent = self._create("parent", image=b"A")
                self._finish(parent)
                child = self._create("child", parent="parent")
                [item] = child.multimodal_inputs.mm_items
                if by_timeout:
                    self.session.timeout = 1.0
                    self.session.last_active_time -= 2.0
                    self._reap()
                else:
                    self._close()
                self.assertIn("s", self.controller)
                self.assertIsNotNone(item.feature)
                self._finish(child)
                self._reap()
                self.assertNotIn("s", self.controller)
                self.assertIsNone(item.feature)


if __name__ == "__main__":
    unittest.main()

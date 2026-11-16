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
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session, SessionController
from sglang.test.test_utils import CustomTestCase

VOCAB = 1 << 20


def _recv(rid, input_ids, max_new_tokens=8, parent=None):
    return SimpleNamespace(
        rid=rid,
        input_ids=array("q", input_ids),
        mm_inputs=None,
        session_params=SimpleNamespace(
            id="s",
            rid=parent,
            offset=None,
            replace=False,
            drop_previous_output=False,
        ),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
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
    )


class TestSessionTokenShare(CustomTestCase):
    def setUp(self):
        self.session = Session(capacity_of_str_len=0, session_id="s", streaming=True)

    def _create(self, rid, input_ids, max_new_tokens=8):
        return self.session.create_req(
            _recv(rid, input_ids, max_new_tokens=max_new_tokens),
            tokenizer=None,
            vocab_size=VOCAB,
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

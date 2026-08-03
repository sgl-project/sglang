"""Unit tests for the streaming-session in-place token-array share protocol
(`Session.create_req` / `finish_req` / `abort_req`):

- token arrays are extended in place and shared across turns (no per-turn copy);
- committed_* lengths recorded at finish_req trim away tokens appended by a
  turn that aborted before finishing (mid-turn and first-turn aborts);
- max_new_tokens overshoot falls back to a fill_ids rebuild instead of
  carrying an inconsistent array.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.schedule_batch import (
    MultimodalInputs,
    StreamingSessionAbortPolicy,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
from sglang.srt.session.streaming_session import SessionSlot, StreamingSession
from sglang.test.test_utils import CustomTestCase

VOCAB = 1 << 20


def _recv(rid, input_ids, max_new_tokens=8, mm_inputs=None):
    return SimpleNamespace(
        rid=rid,
        input_ids=array("q", input_ids),
        mm_inputs=mm_inputs,
        session_params=SimpleNamespace(
            id="s",
            rid=None,
            offset=None,
            replace=False,
            drop_previous_output=False,
            drop_trailing_stop_token=False,
        ),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        lora_id=None,
        custom_logit_processor=None,
        stream=False,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        return_sampling_mask=False,
        require_reasoning=False,
        return_hidden_states=False,
        return_routed_experts=False,
        routed_experts_start_len=0,
        priority=None,
        routing_key=None,
        extra_key=None,
        http_worker_ipc=None,
        time_stats=None,
    )


class TestSessionTokenShare(CustomTestCase):

    def setUp(self):
        self.session = Session(capacity_of_str_len=0, session_id="s", streaming=True)

    def _create(self, rid, input_ids, max_new_tokens=8, mm_inputs=None):
        return self.session.create_req(
            _recv(
                rid,
                input_ids,
                max_new_tokens=max_new_tokens,
                mm_inputs=mm_inputs,
            ),
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
        req._refresh_fill_ids()
        req.output_ids.extend(output[baked:])
        self.session.finish_req(req)

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

    def test_bos_only_turn_forwards_retained_tail_before_boundary(self):
        bos, empty_audio = 200000, 201472
        first = self._create("first", [10], max_new_tokens=1)
        self._decode_and_finish(first, [99])

        direct_append = (
            list(first.origin_input_ids) + list(first.output_ids) + [empty_audio]
        )
        self.assertEqual(direct_append[1:], [99, empty_audio])

        drain = self.session.create_req(
            _recv("drain", [bos], max_new_tokens=0),
            tokenizer=SimpleNamespace(bos_token_id=bos),
            vocab_size=VOCAB,
        )
        self.assertEqual(list(drain.origin_input_ids), [10, 99])
        self.session.finish_req(drain)

        boundary = self._create("boundary", [empty_audio])
        self.assertEqual(list(boundary.origin_input_ids), [10, 99, empty_audio])
        self.assertEqual(
            list(boundary.origin_input_ids)[self.session.committed_origin_len :],
            [empty_audio],
        )

    def test_streaming_boundary_replacement_rejects_none(self):
        with self.assertRaisesRegex(ValueError, "cannot be None"):
            self.session.replace_streaming_boundary(None)

    def test_mid_turn_abort_then_continue(self):
        in1, out1 = list(range(200, 210)), [1, 2, 3]
        r1 = self._create("r1", in1)
        self._decode_and_finish(r1, out1)

        # Turn 2 extends the shared arrays, decodes a bit, then aborts:
        # finish_req never runs, req_nodes still points at r1.
        r2 = self._create("r2", [50, 51])
        self.assertEqual(list(r2.origin_input_ids), in1 + out1 + [50, 51])
        r2.output_ids.extend([6, 7])
        r2._refresh_fill_ids()
        self.session.abort_req(r2)
        self.assertEqual(self.session.committed_origin_len, len(in1))

        # Turn 3 must see exactly r1's history — no [50, 51], no doubled out1.
        r3 = self._create("r3", [60])
        self.assertEqual(list(r3.origin_input_ids), in1 + out1 + [60])
        self.assertEqual(list(r3.full_untruncated_fill_ids), list(r3.origin_input_ids))

        # Two aborted attempts in a row heal idempotently.
        self.session.abort_req(r3)
        r4 = self._create("r4", [70])
        self.assertEqual(list(r4.origin_input_ids), in1 + out1 + [70])
        self.assertEqual(list(r4.full_untruncated_fill_ids), list(r4.origin_input_ids))

    def test_retraction_checkpoint_keeps_turn_inflight(self):
        r1 = self._create("r1", [1, 2, 3])
        r1.output_ids.extend([4, 5])
        r1._refresh_fill_ids()

        self.session.checkpoint_retracted_req(r1)

        self.assertTrue(self.session._inflight)
        [node] = self.session.req_nodes.values()
        self.assertIs(node.req, r1)
        with get_parallel().override(tp_rank=0):
            overlapping = self._create("overlap", [6])
        self.assertIsNotNone(overlapping.to_finish)
        self.assertIn("already has an active request", overlapping.to_finish.message)

        self.session.abort_req(r1)
        resumed = self._create("resumed", [7])
        self.assertEqual(list(resumed.origin_input_ids), [1, 2, 3, 4, 5, 7])

    def test_detaching_rejected_concurrent_request_preserves_owner(self):
        owner = self._create("owner", [1, 2, 3])
        with get_parallel().override(tp_rank=0):
            rejected = self._create("rejected", [4])
        self.assertIsNotNone(rejected.to_finish)

        cache = StreamingSession(SimpleNamespace())
        control = Mock()
        cache.attach_session_lifecycle(control)

        self.assertTrue(cache.detach_queued_request(rejected))
        self.assertIsNone(rejected.session)
        self.assertTrue(self.session._inflight)
        self.assertIs(self.session._inflight_req, owner)
        control.on_session_released.assert_not_called()

        self.assertTrue(cache.detach_queued_request(owner))
        self.assertFalse(self.session._inflight)
        self.assertIsNone(self.session._inflight_req)
        control.on_session_released.assert_called_once_with("s")

    def test_preaborted_concurrent_request_preserves_owner(self):
        owner = self._create("owner", [1, 2, 3])
        with get_parallel().override(tp_rank=0):
            rejected = self._create("rejected", [4])
        cache = StreamingSession(SimpleNamespace())
        slot = SessionSlot(
            req_pool_idx=1,
            kv=SimpleNamespace(kv_allocated_len=0, swa_evicted_seqlen=0),
        )
        cache.slots["s"] = slot

        self.assertIsNone(cache.find_active_slot(rejected))

        self.assertIsNone(rejected.session)
        self.assertTrue(self.session._inflight)
        self.assertIs(self.session._inflight_req, owner)
        self.assertIs(cache.slots["s"], slot)

    def test_finishing_rejected_concurrent_request_preserves_owner(self):
        owner = self._create("owner", [1, 2, 3])
        with get_parallel().override(tp_rank=0):
            rejected = self._create("rejected", [4])
        rejected.finished_reason = rejected.to_finish

        cache = StreamingSession(SimpleNamespace())
        control = Mock()
        cache.attach_session_lifecycle(control)
        release_session = Mock(wraps=cache.release_session)
        cache.release_session = release_session

        self.assertTrue(cache.try_cache_finished_req(rejected))

        self.assertIsNone(rejected.session)
        self.assertIsNone(rejected.req_pool_idx)
        self.assertIsNone(rejected.kv)
        self.assertTrue(self.session._inflight)
        self.assertIs(self.session._inflight_req, owner)
        release_session.assert_not_called()
        control.on_session_released.assert_not_called()

    def test_canceling_rejected_concurrent_request_preserves_owner_slot(self):
        retained_item = SimpleNamespace(feature=object())
        boundary = self._create("boundary", [1, 2])
        boundary.multimodal_inputs = MultimodalInputs(mm_items=[retained_item])
        self._decode_and_finish(boundary, [])

        owner = self._create("owner", [3])
        with get_parallel().override(tp_rank=0):
            rejected = self._create("rejected", [4])
        rejected.finished_reason = rejected.to_finish
        rejected.streaming_abort_policy = StreamingSessionAbortPolicy.COMMIT_FORWARDED
        rejected_item = SimpleNamespace(feature=object())
        rejected.multimodal_inputs = MultimodalInputs(
            mm_items=[retained_item, rejected_item]
        )

        cache = StreamingSession(SimpleNamespace())
        slot = SessionSlot(
            req_pool_idx=1,
            kv_committed_len=3,
            kv=SimpleNamespace(kv_allocated_len=3, swa_evicted_seqlen=0),
        )
        cache.slots["s"] = slot
        control = Mock()
        cache.attach_session_lifecycle(control)
        release_session = Mock(wraps=cache.release_session)
        cache.release_session = release_session

        self.assertTrue(cache.try_cache_finished_req(rejected))

        self.assertIsNone(rejected.session)
        self.assertTrue(self.session._inflight)
        self.assertIs(self.session._inflight_req, owner)
        self.assertIs(cache.slots["s"], slot)
        self.assertIsNotNone(retained_item.feature)
        self.assertIsNone(rejected_item.feature)
        self.assertIsNone(rejected.multimodal_inputs)
        release_session.assert_not_called()
        control.on_request_committed.assert_not_called()
        control.on_session_released.assert_not_called()

    def test_resumed_retracted_request_finishes_without_self_detach(self):
        r1 = self._create("r1", [1, 2, 3])
        r1.output_ids.extend([4])
        r1._refresh_fill_ids()
        self.session.checkpoint_retracted_req(r1)

        self.session.finish_req(r1)

        self.assertFalse(self.session._inflight)
        self.assertIs(r1.session, self.session)
        [node] = self.session.req_nodes.values()
        self.assertIs(node.req, r1)

    def test_retracted_queue_rejection_retains_partial_append_boundary(self):
        r1 = self._create("r1", [1, 2, 3])
        r1.output_ids.extend([4, 5])
        r1._refresh_fill_ids()
        retained_mm = MultimodalInputs(mm_items=["retained-audio"])
        r1.multimodal_inputs = retained_mm
        r1.req_pool_idx = 0
        r1.kv_committed_len = 5
        r1.kv = SimpleNamespace(kv_allocated_len=5, swa_evicted_seqlen=0)
        cache = StreamingSession(SimpleNamespace())

        cache.cache_finished_req(r1, is_insert=False)
        r1.reset_for_retract()

        self.assertTrue(self.session._inflight)
        self.assertTrue(cache.detach_queued_request(r1))
        self.assertFalse(self.session._inflight)
        self.assertIsNone(r1.session)
        self.assertIs(r1.multimodal_inputs, retained_mm)
        self.assertEqual(cache.slots["s"].kv_committed_len, 5)

        r2 = self._create("r2", [6])
        self.assertEqual(list(r2.origin_input_ids), [1, 2, 3, 4, 5, 6])
        self.assertIs(r2.multimodal_inputs, retained_mm)

    def test_first_turn_abort(self):
        r1 = self._create("r1", [1, 2, 3])
        self.assertTrue(self.session._inflight)
        self.session.abort_req(r1)
        self.assertFalse(self.session._inflight)
        # No finish_req ran: nothing committed, next turn starts from scratch.
        self.assertIsNone(self.session.committed_origin_len)
        r2 = self._create("r2", [4, 5])
        self.assertEqual(list(r2.origin_input_ids), [4, 5])
        self._decode_and_finish(r2, [9])
        r3 = self._create("r3", [6])
        self.assertEqual(list(r3.origin_input_ids), [4, 5, 9, 6])

    def test_aborted_turn_does_not_mutate_committed_multimodal_history(self):
        committed_mm = MultimodalInputs(mm_items=["committed"])
        r1 = self._create("r1", [1])
        r1.multimodal_inputs = committed_mm
        self._decode_and_finish(r1, [2])

        incoming_mm = MultimodalInputs(mm_items=["aborted"])
        r2 = self._create("r2", [3], mm_inputs=incoming_mm)
        self.assertIsNot(r2.multimodal_inputs, r1.multimodal_inputs)
        r2.multimodal_inputs.merge(incoming_mm)
        self.assertEqual(r2.multimodal_inputs.mm_items, ["committed", "aborted"])
        self.session.abort_req(r2)

        self.assertEqual(r1.multimodal_inputs.mm_items, ["committed"])
        r3 = self._create("r3", [4], mm_inputs=MultimodalInputs(mm_items=["next"]))
        self.assertEqual(r3.multimodal_inputs.mm_items, ["committed"])

    def test_max_new_tokens_overshoot_falls_back(self):
        in1 = list(range(300, 310))
        r1 = self._create("r1", in1, max_new_tokens=4)
        # Spec-decode overshoot: 6 tokens decoded and baked into the fill
        # array, then output trimmed to finished_len (like _trim_overshoot)
        # before finish.
        r1.output_ids.extend([1, 2, 3, 4, 5, 6])
        r1._refresh_fill_ids()
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
        r2._refresh_fill_ids()
        self.assertEqual(list(r2.full_untruncated_fill_ids), list(r2.origin_input_ids))


if __name__ == "__main__":
    unittest.main()

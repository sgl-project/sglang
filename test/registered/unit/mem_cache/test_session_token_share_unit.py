"""Unit tests for the streaming-session in-place token-array share protocol
(`Session.create_req` / `finish_req` / `abort_req`):

- token arrays are extended in place and shared across turns (no per-turn copy);
- committed_* lengths recorded at finish_req trim away tokens appended by a
  turn that aborted before finishing (mid-turn and first-turn aborts);
- max_new_tokens overshoot falls back to a fill_ids rebuild instead of
  carrying an inconsistent array.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.managers.io_struct import SessionParams
from sglang.srt.managers.schedule_batch import FINISH_LENGTH
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
from sglang.test.test_utils import CustomTestCase

VOCAB = 1 << 20


def _recv(rid, input_ids, max_new_tokens=8):
    return SimpleNamespace(
        rid=rid,
        lifecycle_id=None,
        input_ids=array("q", input_ids),
        mm_inputs=None,
        session_params=SimpleNamespace(
            id="s", rid=None, offset=None, replace=False, drop_previous_output=False
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

    def test_routing_snapshot_restores_committed_history_without_mutation(self):
        first = self._create("r1", [1, 2, 3])
        self._decode_and_finish(first, [4, 5])
        self._create("aborted", [99])
        self.session.abort_req()
        before = list(first.origin_input_ids)
        params = SessionParams(id="s")
        snapshot, multimodal = self.session.routing_input_ids(params, [], None)
        self.assertEqual(snapshot, [1, 2, 3, 4, 5])
        self.assertFalse(multimodal)
        self.assertEqual(list(first.origin_input_ids), before)
        self.assertFalse(self.session._inflight)
        resumed = self._create("r3", [])
        self.assertEqual(list(resumed.origin_input_ids), snapshot)
        with self.assertRaisesRegex(ValueError, "active request"):
            self.session.routing_input_ids(params, [], None)

    def test_tree_routing_matches_execution_for_native_history_options(self):
        for options, expected in [
            ({}, [10, 11, 20, 21, 30]),
            ({"offset": 1}, [10, 30]),
            ({"offset": -1}, [10, 11, 20, 30]),
            ({"drop_previous_output": True}, [10, 11, 30]),
            ({"replace": True}, [10, 11, 20, 21, 30]),
            ({"replace": True, "rid": None}, [1, 30]),
        ]:
            with self.subTest(options=options):
                session = Session(0, "s")
                first = session.create_req(_recv("r1", [10, 11], 2), None, VOCAB)
                first.output_ids.extend([20, 21, 22])
                first.finished_reason = FINISH_LENGTH(2)
                child = _recv("r2", [1, 30])
                child.session_params = SessionParams(
                    id="s", **({"rid": "r1"} | options)
                )
                tokenizer = SimpleNamespace(bos_token_id=1)
                snapshot, _ = session.routing_input_ids(
                    child.session_params, child.input_ids, tokenizer
                )
                self.assertEqual(snapshot, expected)
                self.assertEqual(list(first.origin_input_ids), [10, 11])
                self.assertEqual(list(session.req_nodes), ["r1"])
                actual = session.create_req(child, tokenizer, VOCAB)
                self.assertEqual(list(actual.origin_input_ids), snapshot)

    def test_empty_session_and_missing_parent_do_not_produce_routing_tokens(self):
        with self.assertRaisesRegex(ValueError, "restoring history"):
            self.session.routing_input_ids(SessionParams(id="s"), [], None)
        tree = Session(0, "s")
        with self.assertRaisesRegex(ValueError, "Invalid request session id"):
            tree.routing_input_ids(SessionParams(id="s", rid="missing"), [1], None)


if __name__ == "__main__":
    unittest.main()

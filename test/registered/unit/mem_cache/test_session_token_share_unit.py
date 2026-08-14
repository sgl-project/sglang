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

from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
from sglang.test.test_utils import CustomTestCase

VOCAB = 1 << 20


def _recv(rid, input_ids, max_new_tokens=8, custom_inputs=None):
    return SimpleNamespace(
        rid=rid,
        input_ids=array("q", input_ids),
        mm_inputs=None,
        session_params=SimpleNamespace(
            id="s", rid=None, offset=None, replace=False, drop_previous_output=False
        ),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        lora_id=None,
        custom_logit_processor=None,
        custom_inputs=custom_inputs,
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
        cache_salt=None,
        http_worker_ipc=None,
        time_stats=None,
    )


class TestSessionTokenShare(CustomTestCase):

    def setUp(self):
        self.session = Session(capacity_of_str_len=0, session_id="s", streaming=True)

    def _create(self, rid, input_ids, max_new_tokens=8, custom_inputs=None):
        return self.session.create_req(
            _recv(
                rid,
                input_ids,
                max_new_tokens=max_new_tokens,
                custom_inputs=custom_inputs,
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

    def test_empty_continuation_custom_inputs_survive_abort(self):
        first_custom_inputs = {"acoustic_embedding": [[1.0]], "frame": 1}
        r1 = self._create("r1", [10], custom_inputs=first_custom_inputs)
        self.assertIs(r1.custom_inputs, first_custom_inputs)
        self._decode_and_finish(r1, [20])

        second_custom_inputs = {"acoustic_embedding": [[2.0]], "frame": 2}
        r2 = self._create("r2", [], custom_inputs=second_custom_inputs)
        self.assertEqual(list(r2.origin_input_ids), [10, 20])
        self.assertIs(r2.custom_inputs, second_custom_inputs)

        # An aborted frame must not become part of session history. The next
        # empty continuation resumes from the last successfully committed turn
        # and receives its own per-frame inputs.
        r2.output_ids.append(30)
        r2._refresh_fill_ids()
        self.session.abort_req()
        third_custom_inputs = {"acoustic_embedding": [[3.0]], "frame": 3}
        r3 = self._create("r3", [], custom_inputs=third_custom_inputs)
        self.assertEqual(list(r3.origin_input_ids), [10, 20])
        self.assertIs(r3.custom_inputs, third_custom_inputs)

    def test_empty_first_streaming_request_is_aborted(self):
        with get_parallel().override(tp_rank=0):
            req = self._create("r1", [], custom_inputs={"frame": 1})

        self.assertIsNotNone(req.to_finish)
        self.assertFalse(self.session._inflight)
        self.assertIn("empty session continuation", req.to_finish.message.lower())

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


if __name__ == "__main__":
    unittest.main()

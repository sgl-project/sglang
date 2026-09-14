from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

from array import array
from types import SimpleNamespace

from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
from sglang.test.test_utils import CustomTestCase

VOCAB = 1 << 20


def _recv(rid, input_ids, max_new_tokens=8):
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
        stream=False,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
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

    def _create(self, rid, input_ids, max_new_tokens=8):
        return self.session.create_req(
            _recv(rid, input_ids, max_new_tokens=max_new_tokens),
            tokenizer=None,
            vocab_size=VOCAB,
        )

    def _decode_and_finish(self, req, output):
        req.extend_output_ids(output)
        self.session.finish_req(req)

    def test_normal_multi_turn_reuses_buffer_zero_copy(self):
        """A clean continuation hands the buffer over without copying it."""
        in1, out1 = list(range(100, 110)), [1, 2, 3]
        r1 = self._create("r1", in1)
        self.assertEqual(list(r1.origin_input_ids), in1)
        self._decode_and_finish(r1, out1)

        in2 = [7, 8]
        r2 = self._create("r2", in2)
        self.assertEqual(list(r2.origin_input_ids), in1 + out1 + in2)
        self.assertIs(r2.token_buf._data, r1.token_buf._data)
        self.assertIs(
            r2.origin_input_ids_unpadded._data, r1.origin_input_ids_unpadded._data
        )

    def test_mid_turn_abort_then_continue(self):
        """An aborted turn must not pollute the next turn's reconstructed history."""
        in1, out1 = list(range(200, 210)), [1, 2, 3]
        r1 = self._create("r1", in1)
        self._decode_and_finish(r1, out1)

        r2 = self._create("r2", [50, 51])
        self.assertEqual(list(r2.origin_input_ids), in1 + out1 + [50, 51])
        r2.extend_output_ids([6, 7])
        self.session.abort_req()
        self.assertEqual(list(r1.origin_input_ids), in1)
        self.assertEqual(list(r1.output_ids), out1)

        r3 = self._create("r3", [60])
        self.assertEqual(list(r3.origin_input_ids), in1 + out1 + [60])

        self.session.abort_req()
        r4 = self._create("r4", [70])
        self.assertEqual(list(r4.origin_input_ids), in1 + out1 + [70])

    def test_first_turn_abort_starts_fresh(self):
        """Aborting the very first turn leaves the next turn starting from scratch."""
        self._create("r1", [1, 2, 3])
        self.assertTrue(self.session._inflight)
        self.session.abort_req()
        self.assertFalse(self.session._inflight)

        r2 = self._create("r2", [4, 5])
        self.assertEqual(list(r2.origin_input_ids), [4, 5])
        self._decode_and_finish(r2, [9])
        r3 = self._create("r3", [6])
        self.assertEqual(list(r3.origin_input_ids), [4, 5, 9, 6])


if __name__ == "__main__":
    import unittest

    unittest.main()

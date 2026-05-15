import unittest

from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestReqFinish(CustomTestCase):
    def _req(self, rid: str):
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=[1],
            sampling_params=SamplingParams(max_new_tokens=2, stop_token_ids=[7]),
        )
        req.output_ids = [5, 7]
        return req

    def test_update_finish_state_keeps_length_priority_by_default(self):
        req = self._req("length-first")

        req.update_finish_state(new_accepted_len=2)

        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.finished_len, 2)

    def test_check_finished_can_check_stop_token_before_length(self):
        req = self._req("stop-token-first")

        req.check_finished_stop_before_length(new_accepted_len=2)

        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_len, 2)


if __name__ == "__main__":
    unittest.main()

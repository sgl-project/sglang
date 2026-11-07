import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.dllm.algorithm.low_confidence import LowConfidence
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers.schedule_batch import Req as ScheduleReq
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Req:
    def __init__(
        self, ids, *, prefill=True, prefix=0, incomplete=False, state=None, end=None
    ):
        self.ids = array("q", ids)
        self.prefix_len = prefix
        self.origin_input_ids = array("q", ids)
        self.extend_end = len(ids) if end is None else end
        self.dllm_incomplete_ids = array("q", [1] if incomplete else [])
        self.dllm_algo_state = state
        self.prefill = prefill

    def is_dllm_prefill(self):
        return self.prefill

    def get_fill_ids(self):
        return self.ids


class TestPrefillTokens(unittest.TestCase):
    def setUp(self):
        self.alg = LowConfidence(
            SimpleNamespace(
                block_size=4,
                mask_id=99,
                first_done_first_out_mode=True,
                algorithm_config={},
            )
        )

    def check(self, *reqs):
        return self.alg.prefill_tokens(SimpleNamespace(reqs=reqs))

    def test_full_blocks(self):
        self.assertEqual(
            self.check(Req([4, 5, 6, 7]), Req([8, 9, 10, 11])),
            [[4, 5, 6, 7], [8, 9, 10, 11]],
        )
        self.assertEqual(self.check(Req([0, 4, 5, 6, 7], prefix=1)), [[4, 5, 6, 7]])

    def test_real_request_fields(self):
        req = ScheduleReq.__new__(ScheduleReq)
        req.origin_input_ids = array("q", [0, 4, 5, 6, 7])
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.prefix_len = 1
        req.extend_end = 5
        req.dllm_phase = DllmReqPhase.STAGING_PREFILL
        req.dllm_incomplete_ids = array("q")
        req.dllm_algo_state = None
        self.assertEqual(self.check(req), [[4, 5, 6, 7]])

    def test_falls_back_for_unsafe_batches(self):
        for r in (
            Req([1, 2, 3, 99]),
            Req([1, 2, 3, 4], prefill=False),
            Req([1, 2, 3]),
            Req([1, 2, 3, 4], incomplete=True),
            Req([1, 2, 3, 4], state={}),
            Req([1, 2, 3, 4], end=5),
        ):
            self.assertIsNone(self.check(Req([4, 5, 6, 7]), r))
        self.assertIsNone(self.alg.prefill_tokens(None))
        self.assertIsNone(self.check())


if __name__ == "__main__":
    unittest.main()

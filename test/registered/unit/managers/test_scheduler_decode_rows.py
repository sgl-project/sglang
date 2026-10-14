"""Scheduler.get_decode_step reuses the previous decode step's rows only while
its requests keep their slots and penalty rows."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.managers.scheduler_components.pending_steps import PendingSteps
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.penalties import PenaltyStore
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _new_scheduler() -> Scheduler:
    s = Scheduler.__new__(Scheduler)
    s.device = "cpu"
    s.decoding = []
    s.decode_rows = None
    s.pending = PendingSteps()
    s.flat_slot = ScheduleSlot()
    s.enable_skip_finishing_decode = False
    s.spec_algorithm = SpeculativeAlgorithm.NONE
    s.forward_ct = 1
    s.tree_cache = MagicMock()
    s.token_to_kv_pool_allocator = MagicMock(page_size=1)
    s.token_to_kv_pool_allocator.check_decode_capacity.return_value = True
    s.new_token_ratio_tracker = MagicMock()
    s.penalty_store = PenaltyStore(8, "cpu")
    s._build_sampling_info = MagicMock(side_effect=lambda reqs: object())
    s._build_decode_step = MagicMock(
        side_effect=lambda reqs, indices: SimpleNamespace(
            slots=indices[1].tolist(), sampling_info=None
        )
    )
    return s


def _make_req(rid: str, slot: int, **sampling_kwargs) -> Req:
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(**sampling_kwargs),
    )
    req.kv.req_pool_idx = slot
    return req


class TestDecodeRowsReuse(CustomTestCase):
    def setUp(self):
        super().setUp()
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)

    def test_reuse_while_requests_keep_their_slots(self):
        s = _new_scheduler()
        s.decoding = [_make_req("a", 1), _make_req("b", 2)]

        first = s.get_decode_step(s.flat_slot)
        second = s.get_decode_step(s.flat_slot)

        self.assertEqual(second.slots, [1, 2])
        self.assertIs(second.sampling_info, first.sampling_info)

    def test_readmitted_request_decodes_with_its_new_slot(self):
        s = _new_scheduler()
        req = _make_req("a", 1)
        s.decoding = [req]
        first = s.get_decode_step(s.flat_slot)

        # Everything leaves decoding without a decode step (a retracting
        # pause, say), then the same request object is re-admitted elsewhere.
        s.decoding = []
        self.assertIsNone(s.get_decode_step(s.flat_slot))
        req.kv.req_pool_idx = 4
        s.decoding = [req]
        step = s.get_decode_step(s.flat_slot)

        self.assertEqual(step.slots, [4])
        self.assertIsNot(step.sampling_info, first.sampling_info)

    def test_readmitted_request_in_the_same_slot_with_a_new_penalty_row(self):
        s = _new_scheduler()
        store = s.penalty_store
        req_a = _make_req("a", 1, frequency_penalty=0.5)
        req_b = _make_req("b", 2, frequency_penalty=0.5)
        store.alloc(req_a)
        store.alloc(req_b)
        s.decoding = [req_a]
        first = s.get_decode_step(s.flat_slot)

        # Both are retracted, and req_a comes back to slot 1 but takes the row
        # req_b freed last.
        s.decoding = []
        store.free(1)
        store.free(2)
        store.alloc(req_a)
        self.assertNotEqual(store.row_of_slot[1], 0)
        s.decoding = [req_a]
        step = s.get_decode_step(s.flat_slot)

        self.assertEqual(step.slots, [1])
        self.assertIsNot(step.sampling_info, first.sampling_info)


if __name__ == "__main__":
    unittest.main()

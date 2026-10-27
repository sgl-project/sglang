"""StepBuilder.build_decode_step reuses the previous decode step's rows only
while its requests keep their slots."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.engine_state import LaunchCounters
from sglang.srt.managers.scheduler_components.step_builder import StepBuilder
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.penalties import PenaltyStore
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _new_step_builder() -> StepBuilder:
    builder = StepBuilder(
        kv=SimpleNamespace(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            tree_cache=None,
            penalty_store=PenaltyStore(8, 8, "cpu"),
        ),
        future_map=None,
        counters=LaunchCounters(),
        workers=SimpleNamespace(tp_worker=None, model_worker=None),
        hisparse_coordinator=None,
        model_config=None,
        device="cpu",
    )
    builder.build_sampling_info = MagicMock(
        side_effect=lambda reqs, req_pool_indices: object()
    )
    builder.new_decode_step = MagicMock(
        side_effect=lambda reqs, indices: SimpleNamespace(
            slots=indices[1].tolist(), sampling_info=None
        )
    )
    return builder


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
        builder = _new_step_builder()
        decoding = [_make_req("a", 1), _make_req("b", 2)]

        first = builder.build_decode_step(list(decoding))
        second = builder.build_decode_step(list(decoding))

        self.assertEqual(second.slots, [1, 2])
        self.assertIs(second.sampling_info, first.sampling_info)

    def test_readmitted_request_decodes_with_its_new_slot(self):
        builder = _new_step_builder()
        req = _make_req("a", 1)
        first = builder.build_decode_step([req])

        # The same request object is re-admitted elsewhere.
        req.kv.req_pool_idx = 4
        step = builder.build_decode_step([req])

        self.assertEqual(step.slots, [4])
        self.assertIsNot(step.sampling_info, first.sampling_info)


if __name__ == "__main__":
    unittest.main()

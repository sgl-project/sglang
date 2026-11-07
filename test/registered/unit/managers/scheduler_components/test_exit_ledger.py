import unittest
from unittest.mock import MagicMock

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.output_streamer import (
    OutputSink,
    _GenerationStreamAccumulator,
)
from sglang.srt.managers.scheduler_components.request_list import ListKind
from sglang.srt.managers.scheduler_components.state_check import CheckPhase
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import FakeReq
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _accumulator() -> _GenerationStreamAccumulator:
    return _GenerationStreamAccumulator(
        return_logprob=False,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        disaggregation_mode=DisaggregationMode.NULL,
        default_stream_interval=1,
        default_force_stream_interval=1,
        build_request_stats=lambda req: None,
        output_sink=OutputSink.DETOKENIZER,
        is_request_level_customized_info_key=lambda key: False,
        current_weight_version=None,
    )


def _checker(lists) -> SchedulerInvariantChecker:
    return SchedulerInvariantChecker(
        disaggregation_mode=DisaggregationMode.NULL,
        page_size=1,
        tree_cache=None,
        token_to_kv_pool_allocator=None,
        req_to_token_pool=None,
        pool_stats_observer=None,
        get_request_lists=lambda: lists,
        get_skip_idle_pool_checks=lambda: True,
        scheduler_stage_metrics=None,
    )


class TestRetireOnce(CustomTestCase):
    def test_a_finished_request_streamed_after_its_abort_output_raises(self):
        req = FakeReq(
            rid="r0",
            beam_group=None,
            finished_output=False,
            finished=lambda: True,
        )
        scheduler = MagicMock(rust_server=None)
        with envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.override(True):
            Scheduler.send_abort_output(scheduler, AbortReq(rid="r0"), req)
            with self.assertRaisesRegex(RuntimeError, "r0 sent its final output"):
                _accumulator().accept(req=req)

    def test_a_request_lost_from_every_list_is_reported_unless_a_step_holds_it(self):
        lists = {ListKind.DECODING: []}
        checker = _checker(lists)
        stepping, lost = FakeReq(rid="stepping"), FakeReq(rid="lost")
        lists[ListKind.DECODING] = [stepping, lost]
        with (
            envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.override(True),
            envs.SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK.override(False),
        ):
            checker.run(CheckPhase.BUSY, batch=MagicMock())

            stepping.inflight_steps = 1
            lists[ListKind.DECODING] = []
            with self.assertRaisesRegex(RuntimeError, r"\['lost'\]"):
                checker.run(CheckPhase.BUSY, batch=MagicMock())

            # The step is processed and the request exits.
            stepping.inflight_steps = 0
            stepping.exit_via = "release"
            checker.run(CheckPhase.IDLE)


if __name__ == "__main__":
    unittest.main()

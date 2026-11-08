import unittest
from unittest.mock import MagicMock

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.exit_ledger import record_send
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.output_streamer import (
    OutputSink,
    SchedulerOutputStreamer,
    _GenerationStreamAccumulator,
)
from sglang.srt.managers.scheduler_components.request_list import ListKind
from sglang.srt.managers.scheduler_components.state_check import (
    CheckPhase,
    StateCheckError,
)
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

    def test_a_finished_embedding_streamed_twice_raises(self):
        req = FakeReq(
            rid="e0",
            finished=lambda: True,
            finished_reason=MagicMock(),
            http_worker_ipc=None,
            embedding=[0.0],
            pooled_hidden_state=None,
        )
        streamer = MagicMock()
        with envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.override(True):
            SchedulerOutputStreamer._stream_output_embedding(streamer, [req])
            with self.assertRaisesRegex(RuntimeError, "e0 sent its final output"):
                SchedulerOutputStreamer._stream_output_embedding(streamer, [req])


class TestSendAtIdle(CustomTestCase):
    def test_a_taken_request_in_no_list_that_never_sent_is_reported(self):
        sent, held, lost = (FakeReq(rid=rid) for rid in ("sent", "held", "lost"))
        checker = SchedulerInvariantChecker(
            disaggregation_mode=DisaggregationMode.NULL,
            page_size=1,
            tree_cache=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(),
            req_to_token_pool=MagicMock(),
            pool_stats_observer=MagicMock(),
            scheduler_stage_metrics=None,
            # Idle leaves held requests listed.
            get_request_lists=lambda: {ListKind.PD_HELD: [held]},
            get_skip_idle_pool_checks=lambda: True,
        )
        with envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.override(True):
            for req in (sent, held, lost):
                checker.unsent_outputs.record_take(req)
            record_send(sent, "stream")
            with self.assertRaisesRegex(StateCheckError, r"output: \['lost'\]$"):
                checker.run(CheckPhase.IDLE)


if __name__ == "__main__":
    unittest.main()

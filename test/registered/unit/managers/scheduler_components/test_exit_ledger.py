import unittest
from unittest.mock import MagicMock

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.output_streamer import (
    OutputSink,
    _GenerationStreamAccumulator,
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


if __name__ == "__main__":
    unittest.main()

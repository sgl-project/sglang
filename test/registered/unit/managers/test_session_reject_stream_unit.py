import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSessionRejectStream(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="scheduler")

    def test_streams_preaborted_session_request_without_queueing(self):
        finish_reason = FINISH_ABORT("Streaming session already has an active request.")
        trace_ctx = SimpleNamespace(abort=Mock())
        time_stats = SimpleNamespace(trace_ctx=trace_ctx, set_quick_finish_time=Mock())
        req = SimpleNamespace(
            finished_reason=None,
            return_logprob=False,
            to_finish=finish_reason,
            time_stats=time_stats,
        )
        session = SimpleNamespace(
            close_on_finish=False,
            create_req=Mock(return_value=req),
        )
        output_streamer = SimpleNamespace(stream_output=Mock())
        self_obj = SimpleNamespace(
            enable_session_radix_cache=False,
            session_controller={
                "session-a": session,
            },
            tokenizer=None,
            model_config=SimpleNamespace(
                vocab_size=16,
                hf_eos_token_id=[],
            ),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            output_streamer=output_streamer,
            disaggregation_mode=DisaggregationMode.NULL,
            _add_request_to_queue=Mock(side_effect=AssertionError),
        )
        recv_req = SimpleNamespace(
            session_params=SimpleNamespace(id="session-a"),
            session_id=None,
            bootstrap_port=None,
            mm_inputs=None,
            return_logprob=False,
        )

        Scheduler.handle_generate_request(self_obj, recv_req)

        output_streamer.stream_output.assert_called_once_with([req], False)
        self.assertIs(req.finished_reason, finish_reason)
        self.assertIsNone(req.to_finish)
        self_obj._add_request_to_queue.assert_not_called()
        trace_ctx.abort.assert_called_once_with(
            abort_info={"reason": finish_reason.message}
        )
        time_stats.set_quick_finish_time.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

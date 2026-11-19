"""The Rust frontend waits for one reply per control request; the control
handlers Python's TokenizerManager fire-and-forgets must not leave it waiting."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import scheduler as scheduler_module  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    CloseSessionReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    ControlAckOutput,
    FlushCacheReqInput,
    FreezeGCReq,
    PauseGenerationReqInput,
)
from sglang.srt.runtime_context import publish, reset_context  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

NO_REPLY_REQUESTS = [
    PauseGenerationReqInput(mode="in_place"),
    ContinueGenerationReqInput(),
    FreezeGCReq(),
    ConfigureLoggingReq(),
    CloseSessionReqInput(session_id="s"),
]


def _scheduler(rust_server):
    """A scheduler whose request dispatcher runs no handler (returns None), so
    only the dispatch loop's own reply logic is under test."""
    scheduler = object.__new__(scheduler_module.Scheduler)
    scheduler.scheduler_stage_metrics = None
    scheduler.session_controller = SimpleNamespace(maybe_reap=MagicMock())
    scheduler._request_dispatcher = MagicMock(return_value=None)
    scheduler.flush_wrapper = SimpleNamespace(check_pending=MagicMock())
    scheduler.external_corpus_manager = None
    scheduler.rust_server = rust_server
    scheduler.ipc_channels = MagicMock()
    scheduler._event_loop_should_restart = False
    return scheduler


class TestRustControlAck(CustomTestCase):
    def setUp(self):
        super().setUp()
        # The dispatch loop's tail reads the published config.
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)

    def test_no_reply_handlers_answer_a_success_output_under_the_rust_frontend(self):
        rust_server = MagicMock()
        scheduler = _scheduler(rust_server)
        with (
            patch.object(
                scheduler_module, "is_health_check_generate_req", return_value=False
            ),
            patch.object(
                scheduler_module,
                "get_mm",
                return_value=SimpleNamespace(mm_feature_transport=""),
            ),
        ):
            scheduler.process_input_requests(NO_REPLY_REQUESTS)

        replies = [call.args for call in rust_server.push_control_output.call_args_list]
        self.assertEqual([req for req, _ in replies], NO_REPLY_REQUESTS)
        for req, output in replies:
            self.assertEqual(output, ControlAckOutput(), type(req).__name__)

    def test_python_frontend_and_replying_handlers_are_unchanged(self):
        # Python frontend: these requests still produce no output at all.
        scheduler = _scheduler(rust_server=None)
        with (
            patch.object(
                scheduler_module, "is_health_check_generate_req", return_value=False
            ),
            patch.object(
                scheduler_module,
                "get_mm",
                return_value=SimpleNamespace(mm_feature_transport=""),
            ),
        ):
            scheduler.process_input_requests(NO_REPLY_REQUESTS)
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

        # A request outside the table gets no synthesized reply, under either frontend.
        rust_server = MagicMock()
        scheduler = _scheduler(rust_server)
        with (
            patch.object(
                scheduler_module, "is_health_check_generate_req", return_value=False
            ),
            patch.object(
                scheduler_module,
                "get_mm",
                return_value=SimpleNamespace(mm_feature_transport=""),
            ),
        ):
            scheduler.process_input_requests([FlushCacheReqInput()])
        rust_server.push_control_output.assert_not_called()


if __name__ == "__main__":
    unittest.main()

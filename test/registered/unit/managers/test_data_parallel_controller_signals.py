"""Signal propagation tests for the data-parallel controller process."""

import signal
import unittest
from unittest.mock import ANY, MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDataParallelControllerSignalForwarding(CustomTestCase):
    def test_forwards_sigquit_to_parent_once(self):
        scheduler_proc = MagicMock(pid=10, exitcode=0)
        controller = MagicMock(
            scheduler_procs=[scheduler_proc],
            max_total_num_tokens=1024,
            max_req_input_len=512,
            startup_time={},
        )
        parent_process = MagicMock()
        setup_order = []

        with (
            get_context().override_server_args() as server_args,
            patch(
                "sglang.srt.managers.data_parallel_controller.DataParallelController"
            ) as controller_type,
            patch(
                "sglang.srt.managers.data_parallel_controller.psutil.Process"
            ) as process_type,
            patch(
                "sglang.srt.managers.data_parallel_controller.signal.signal"
            ) as register_signal,
            patch(
                "sglang.srt.managers.data_parallel_controller."
                "kill_itself_when_parent_died"
            ),
            patch("sglang.srt.managers.data_parallel_controller.publish"),
            patch("sglang.srt.managers.data_parallel_controller.configure_logger"),
            patch("sglang.srt.managers.data_parallel_controller.faulthandler.enable"),
            patch(
                "sglang.srt.managers.data_parallel_controller.setproctitle.setproctitle"
            ),
        ):
            process_type.return_value.parent.return_value = parent_process
            register_signal.side_effect = lambda *_: setup_order.append("signal")
            controller_type.side_effect = lambda *_: (
                setup_order.append("controller") or controller
            )
            run_data_parallel_controller_process(
                server_args,
                MagicMock(),
                MagicMock(),
            )

        self.assertEqual(setup_order, ["signal", "controller"])
        register_signal.assert_called_once_with(signal.SIGQUIT, ANY)
        sigquit_handler = register_signal.call_args.args[1]
        sigquit_handler(signal.SIGQUIT, None)
        sigquit_handler(signal.SIGQUIT, None)

        parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)
        controller.event_loop.assert_called_once_with()
        scheduler_proc.join.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

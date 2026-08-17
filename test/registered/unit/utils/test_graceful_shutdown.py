import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from unittest.mock import Mock, patch

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.tokenizer_manager import (
    shutdown_scheduler_and_child_processes,
)
from sglang.srt.utils.common import install_graceful_sigterm_handler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

logger = logging.getLogger(__name__)


def _wait_for_sigterm(ready: mp.Event) -> None:
    install_graceful_sigterm_handler(logger, "test worker")
    ready.set()
    while True:
        time.sleep(1)


def _finish_cleanup_after_sigterm(
    cleanup_started: mp.Event,
    cleanup_finished: mp.Event,
) -> None:
    install_graceful_sigterm_handler(
        logger,
        "test worker",
        is_shutting_down=lambda: True,
    )
    try:
        return
    finally:
        cleanup_started.set()
        time.sleep(0.5)
        cleanup_finished.set()


class TestGracefulShutdown(CustomTestCase):
    def test_unexpected_sigterm_remains_nonzero(self):
        ready = mp.Event()
        process = mp.Process(target=_wait_for_sigterm, args=(ready,))
        process.start()
        try:
            self.assertTrue(ready.wait(timeout=3))
            process.terminate()
            process.join(timeout=3)
            self.assertEqual(process.exitcode, 128 + signal.SIGTERM)
        finally:
            if process.is_alive():
                process.kill()
                process.join()

    def test_sigterm_does_not_interrupt_cleanup_in_progress(self):
        cleanup_started = mp.Event()
        cleanup_finished = mp.Event()
        process = mp.Process(
            target=_finish_cleanup_after_sigterm,
            args=(cleanup_started, cleanup_finished),
        )
        process.start()
        try:
            self.assertTrue(cleanup_started.wait(timeout=3))
            process.terminate()
            self.assertTrue(cleanup_finished.wait(timeout=3))
            process.join(timeout=3)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.kill()
                process.join()

    @patch("sglang.srt.managers.tokenizer_manager.graceful_kill_process_tree")
    @patch(
        "sglang.srt.managers.tokenizer_manager.collect_scheduler_processes",
        return_value=[],
    )
    def test_shutdown_dispatches_before_terminating_children(
        self,
        collect_scheduler_processes,
        graceful_kill_process_tree,
    ):
        dispatch_shutdown = Mock()

        with envs.SGLANG_CHILD_PROCESS_SHUTDOWN_TIMEOUT.override(2.0):
            shutdown_scheduler_and_child_processes(dispatch_shutdown)

        dispatch_shutdown.assert_called_once_with()
        collect_scheduler_processes.assert_called_once_with()
        graceful_kill_process_tree.assert_called_once_with(
            os.getpid(),
            include_parent=False,
            timeout=2.0,
        )

    @patch("sglang.srt.managers.tokenizer_manager.graceful_kill_process_tree")
    @patch("sglang.srt.managers.tokenizer_manager.collect_scheduler_processes")
    def test_dispatch_failure_uses_process_signal_fallback(
        self,
        collect_scheduler_processes,
        graceful_kill_process_tree,
    ):
        dispatch_shutdown = Mock(side_effect=RuntimeError("send failed"))

        with envs.SGLANG_CHILD_PROCESS_SHUTDOWN_TIMEOUT.override(2.0):
            shutdown_scheduler_and_child_processes(dispatch_shutdown)

        collect_scheduler_processes.assert_not_called()
        graceful_kill_process_tree.assert_called_once_with(
            os.getpid(),
            include_parent=False,
            timeout=2.0,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

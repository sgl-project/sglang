"""Graceful-drain signal topology.

Server shutdown is coordinated by the tokenizer manager: on a stop signal it
drains in-flight requests, then explicitly stops the worker subprocesses.
These tests pin the two properties that make the drain reachable:

1. worker subprocesses ignore group-delivered SIGINT/SIGTERM instead of dying
   mid-forward at signal time, and
2. the main-process handler maps the first stop signal to the drain flag and a
   repeated stop signal to force exit.
"""

import signal
import unittest
from types import SimpleNamespace

from sglang.srt.utils.common import ignore_external_stop_signals
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestGracefulDrainSignals(CustomTestCase):
    def test_ignore_external_stop_signals_sets_sig_ign(self):
        old_int = signal.getsignal(signal.SIGINT)
        old_term = signal.getsignal(signal.SIGTERM)
        try:
            ignore_external_stop_signals()
            self.assertIs(signal.getsignal(signal.SIGINT), signal.SIG_IGN)
            self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_IGN)
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)

    def test_stop_signal_sets_drain_flag_and_escalates(self):
        from sglang.srt.managers.tokenizer_manager import SignalHandler

        tokenizer_manager = SimpleNamespace(
            gracefully_exit=False, drain_force_exit=False
        )
        handler = SignalHandler(tokenizer_manager)

        handler.sigterm_handler(signal.SIGTERM, None)
        self.assertTrue(tokenizer_manager.gracefully_exit)
        self.assertFalse(tokenizer_manager.drain_force_exit)

        # Orchestrators deliver one stop event as a bundle of DISTINCT signals
        # (e.g. Modal sends SIGTERM and SIGINT together); a signal not yet
        # seen joins the active drain instead of escalating.
        handler.sigterm_handler(signal.SIGINT, None)
        self.assertFalse(tokenizer_manager.drain_force_exit)

        # The SAME signal arriving again is the operator insisting: force exit.
        handler.sigterm_handler(signal.SIGINT, None)
        self.assertTrue(tokenizer_manager.drain_force_exit)


if __name__ == "__main__":
    unittest.main()

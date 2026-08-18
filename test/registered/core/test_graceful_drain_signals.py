"""Graceful-drain signal topology.

Server shutdown is coordinated by the tokenizer manager: on a stop signal it
drains in-flight requests, then explicitly stops the worker subprocesses.
These tests pin the properties that make the drain reachable:

1. worker subprocesses survive group-delivered SIGINT/SIGTERM (verified
   against a real process group, not just the handler table),
2. the main-process handler maps the first stop signal to the drain flag,
   folds the rest of the same stop event's signal bundle into the active
   drain, and treats a repeated signal as operator escalation, and
3. the drain-timeout knob tolerates unparsable values instead of killing the
   shutdown watchdog.
"""

import os
import signal
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace

from sglang.srt.utils.common import ignore_external_stop_signals
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


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

    @unittest.skipUnless(sys.platform != "win32", "process groups are POSIX-only")
    def test_child_survives_group_delivered_stop_signals(self):
        """Signal a real process group; the protected child must outlive it.

        This is the deployment scenario: the orchestrator signals the whole
        group, and workers must keep running so the coordinator can drain
        in-flight requests before stopping them explicitly.
        """
        child_src = textwrap.dedent(
            """
            import sys, time
            from sglang.srt.utils.common import ignore_external_stop_signals

            ignore_external_stop_signals()
            print("armed", flush=True)
            time.sleep(60)
            """
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", child_src],
            stdout=subprocess.PIPE,
            start_new_session=True,  # own process group, so killpg can't hit us
        )
        try:
            # Wait until the handlers are installed before signaling.
            self.assertEqual(proc.stdout.readline().strip(), b"armed")
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            os.killpg(pgid, signal.SIGINT)
            try:
                proc.wait(timeout=2)
                self.fail(
                    f"child exited with {proc.returncode} after group stop signals"
                )
            except subprocess.TimeoutExpired:
                pass  # still alive: signals were ignored
            self.assertIsNone(proc.poll())
        finally:
            proc.kill()
            proc.wait(timeout=10)

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

    def test_drain_timeout_tolerates_garbage(self):
        """The knob is registered in environ.py, whose reader warns and falls
        back to the default on unparsable values — a bad setting must not be
        able to kill the shutdown watchdog mid-drain."""
        from sglang.srt.environ import envs

        with envs.SGLANG_GRACEFUL_SHUTDOWN_TIMEOUT.override("120"):
            self.assertEqual(envs.SGLANG_GRACEFUL_SHUTDOWN_TIMEOUT.get(), 120.0)
        with envs.SGLANG_GRACEFUL_SHUTDOWN_TIMEOUT.override("not-a-float"):
            self.assertEqual(envs.SGLANG_GRACEFUL_SHUTDOWN_TIMEOUT.get(), 0.0)
        # Default (unset): no timeout.
        os.environ.pop("SGLANG_GRACEFUL_SHUTDOWN_TIMEOUT", None)
        self.assertEqual(envs.SGLANG_GRACEFUL_SHUTDOWN_TIMEOUT.get(), 0.0)


if __name__ == "__main__":
    unittest.main()

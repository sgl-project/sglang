"""Test --custom-sigquit-handler parameter: registers a user-provided
SIGQUIT handler for additional cleanup (e.g., crash dumps) when the
server receives SIGQUIT from a failing child process.

This parameter is ONLY available via the Engine API (programmatic), not
via CLI.
"""

import multiprocessing
import signal
import unittest

from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)


def _custom_sigquit_handler(signum, frame):
    """Module-level SIGQUIT handler for picklability in multiprocessing.

    Simulates a production handler that logs the signal and writes a
    crash-dump marker before re-raising.  If the environment variable
    SGLANG_SIGQUIT_MARKER is set, the handler writes a marker file at
    that path as a persistent side-effect for test verification.
    """
    import os
    import sys

    sys.stderr.write(f"[custom_sigquit_handler] PID={os.getpid()} received SIGQUIT\n")
    marker_path = os.environ.get("SGLANG_SIGQUIT_MARKER_FILE")
    if marker_path:
        try:
            with open(marker_path, "w") as f:
                f.write(f"pid={os.getpid()}\nsignum={signum}\n")
        except OSError:
            pass
    # Re-raise so the process still exits with the expected signal behaviour
    signal.signal(signal.SIGQUIT, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGQUIT)


def _engine_sigquit_worker(marker_path, model_path, port):
    """Worker process: starts Engine with custom handler, then self-triggers SIGQUIT.

    Runs in a separate process so that SIGQUIT does not kill the test runner.
    The handler writes a marker file which the parent process verifies.
    """
    import os
    import signal
    import time

    os.environ["SGLANG_SIGQUIT_MARKER_FILE"] = marker_path
    try:
        from sglang.srt.entrypoints.engine import Engine

        Engine(
            model_path=model_path,
            attention_backend="ascend",
            disable_cuda_graph=True,
            port=port,
            custom_sigquit_handler=_custom_sigquit_handler,
        )
        time.sleep(2)
        os.kill(os.getpid(), signal.SIGQUIT)
    except Exception:
        import sys
        import traceback

        traceback.print_exc()
        sys.exit(1)


class TestNpuCustomSigquitHandler(CustomTestCase):
    """Test --custom-sigquit-handler parameter via Engine API.

    Business scenario: Production deployments that need custom cleanup
    (crash dumps, metric flushing, alert triggering) when sglang
    receives SIGQUIT from a dying child process.  The default handler
    kills the process tree; a custom handler can add logging, save
    state, or notify operators before exiting.

    [Test Category] Parameter
    [Test Target] --custom-sigquit-handler
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    # Distinct ports per test to avoid TIME_WAIT conflicts between
    # sequential Engine startups.
    _next_port = 31400

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _alloc_port(cls):
        port = cls._next_port
        cls._next_port += 1
        return port

    def setUp(self):
        self._saved_sigquit = signal.getsignal(signal.SIGQUIT)

    def tearDown(self):
        signal.signal(signal.SIGQUIT, self._saved_sigquit)

    def _create_engine(self, **extra_kwargs):
        """Create an Engine instance with NPU-appropriate defaults.

        Returns the Engine object.  Caller MUST call ``engine.shutdown()``
        in a finally block to clean up subprocesses.
        """
        from sglang.srt.entrypoints.engine import Engine

        return Engine(
            model_path=self.model,
            attention_backend="ascend",
            disable_cuda_graph=True,
            port=self._alloc_port(),
            **extra_kwargs,
        )

    # ------------------------------------------------------------------
    # TC-SH-01 (P1): Engine API with custom callable handler
    #   Branch: engine.py:1298  custom_sigquit_handler is not None
    #           engine.py:1316  signal.signal(SIGQUIT, handler)
    #   Priority: P1 — requires model loading via Engine()
    # ------------------------------------------------------------------

    def test_custom_handler_registered(self):
        """Engine(custom_sigquit_handler=my_func) → my_func is registered
        as the SIGQUIT handler.

        Verifies the custom handler REPLACES the default
        launch_phase_sigquit_handler.
        """
        self.assertIsNot(
            signal.getsignal(signal.SIGQUIT),
            _custom_sigquit_handler,
            "SIGQUIT handler should NOT already be _custom_sigquit_handler "
            "before Engine starts",
        )
        engine = None
        try:
            engine = self._create_engine(
                custom_sigquit_handler=_custom_sigquit_handler,
            )
            self.assertIs(
                signal.getsignal(signal.SIGQUIT),
                _custom_sigquit_handler,
                "SIGQUIT handler should be the custom callable",
            )
        finally:
            if engine is not None:
                engine.shutdown()

    # ------------------------------------------------------------------
    # TC-SH-02 (P1): Custom handler actually executes on SIGQUIT
    #   Starts Engine in a subprocess, sends SIGQUIT, verifies the
    #   handler's marker file is written.
    # ------------------------------------------------------------------

    def test_custom_handler_invoked_on_sigquit(self):
        """Start Engine with custom handler → SIGQUIT → handler writes marker file.

        Spawns Engine in a child process so that SIGQUIT delivery does
        not kill the test runner.
        """
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".marker") as f:
            marker_path = f.name

        proc = multiprocessing.Process(
            target=_engine_sigquit_worker,
            args=(marker_path, self.model, self._alloc_port()),
        )
        proc.start()
        proc.join(timeout=120)

        # Process must have been killed by SIGQUIT (Unix: negated signal number)
        self.assertIsNotNone(proc.exitcode, "Worker process should have terminated")
        self.assertLess(
            proc.exitcode,
            0,
            f"Worker should die by signal, not exit cleanly (got exitcode={proc.exitcode})",
        )

        # Marker file is the persistent side-effect from the custom handler
        self.assertTrue(
            os.path.exists(marker_path),
            f"Marker file {marker_path} should exist after SIGQUIT, "
            f"but was not written.",
        )
        if os.path.exists(marker_path):
            with open(marker_path) as f:
                content = f.read()
            self.assertIn("pid=", content, "Marker should record PID")
            self.assertIn(
                f"signum={signal.SIGQUIT}",
                content,
                f"Marker should record signum={signal.SIGQUIT} (SIGQUIT)",
            )
        try:
            os.unlink(marker_path)
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()

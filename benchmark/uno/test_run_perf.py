"""CPU-only process lifecycle regressions: python -m unittest benchmark.uno.test_run_perf."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.uno.run_perf import (
    _live_process_group_members,
    _stop_process_group,
    run_job,
)


class TestEngineProcessCleanup(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def job(self, code):
        return {
            "benchmark": "gsm8k",
            "concurrency": 1,
            "graph": "graph",
            "repeat": 1,
            "mode": "ar",
            "width": None,
            "directory": str(self.root / "run"),
            "command": [sys.executable, "-c", code],
        }

    def test_success_reclaims_orphan_even_when_it_ignores_sigterm(self):
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        self.addCleanup(_stop_process_group, unrelated)
        identity = self.root / "identity"
        ready = self.root / "ready"
        child = (
            "import signal,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"Path({str(ready)!r}).touch(); time.sleep(60)"
        )
        summary = {
            "num_tokens": 100,
            "num_forwards": 100,
            "num_completions": 1,
            "generation_seconds": 1,
            "accuracy": 0,
            "num_prompts": 1,
            "dataset_selection": {},
            "max_tokens": 100,
            "tokens_per_second": 100,
            "tokens_per_second_per_request": 100,
            "tokens_per_forward": 1,
            "num_length_capped": 1,
        }
        code = f"""
import json, os, subprocess, sys, time
from pathlib import Path
Path({str(identity)!r}).write_text(str(os.getpid()))
subprocess.Popen([sys.executable, '-c', {child!r}])
deadline = time.monotonic() + 10
while not Path({str(ready)!r}).exists():
    if time.monotonic() > deadline:
        raise RuntimeError('child failed to start')
    time.sleep(0.01)
Path({str(self.root / "run" / "summary.json")!r}).write_text({json.dumps(summary)!r})
"""
        try:
            result = run_job(self.job(code), timeout=15)
            self.assertEqual(result["output_tokens"], 100)
            self.assertEqual(_live_process_group_members(int(identity.read_text())), [])
            self.assertIsNone(unrelated.poll())
        finally:
            if identity.exists():
                try:
                    os.killpg(int(identity.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_nonzero_exit_includes_log_and_cleans_up(self):
        with self.assertRaisesRegex(
            RuntimeError, "(?s)code 7.*UNO initialization failed"
        ):
            run_job(
                self.job("import sys; print('UNO initialization failed'); sys.exit(7)"),
                timeout=10,
            )

    def test_timeout_reaps_launcher(self):
        identity = self.root / "identity"
        code = (
            "import os,time; from pathlib import Path; "
            f"Path({str(identity)!r}).write_text(str(os.getpid())); time.sleep(60)"
        )
        with self.assertRaisesRegex(RuntimeError, "timed out.*engine.log"):
            run_job(self.job(code), timeout=1)
        self.assertEqual(_live_process_group_members(int(identity.read_text())), [])

    def test_uncleanable_group_blocks_next_run(self):
        process = unittest.mock.Mock(pid=123456)
        with (
            patch(
                "benchmark.uno.run_perf._live_process_group_members", return_value=[42]
            ),
            patch("benchmark.uno.run_perf._wait_process_group", return_value=[42]),
            patch("benchmark.uno.run_perf.os.killpg") as kill,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "refusing to start the next engine"
            ):
                _stop_process_group(process)
        self.assertEqual(kill.call_args_list[-1].args, (123456, signal.SIGKILL))


if __name__ == "__main__":
    unittest.main()

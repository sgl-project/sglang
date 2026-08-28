import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from sglang.test.ci import warm_test_worker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestWarmTestWorker(CustomTestCase):
    def test_reuses_interpreter_and_restores_process_state(self):
        result_read_fd, result_write_fd = os.pipe()
        process = subprocess.Popen(
            [
                sys.executable,
                warm_test_worker.__file__,
                "--result-fd",
                str(result_write_fd),
            ],
            stdin=subprocess.PIPE,
            text=True,
            pass_fds=(result_write_fd,),
        )
        os.close(result_write_fd)

        try:
            with (
                tempfile.TemporaryDirectory() as tmpdir,
                os.fdopen(result_read_fd) as result_stream,
            ):
                first = Path(tmpdir) / "first.py"
                first.write_text(textwrap.dedent("""
                        import builtins
                        import os

                        builtins._sglang_warm_worker_marker = 41
                        os.environ["SGLANG_WARM_WORKER_TEST"] = "leaked"
                        raise SystemExit(0)
                        """))
                second = Path(tmpdir) / "second.py"
                second.write_text(textwrap.dedent("""
                        import builtins
                        import os

                        assert builtins._sglang_warm_worker_marker == 41
                        assert "SGLANG_WARM_WORKER_TEST" not in os.environ
                        del builtins._sglang_warm_worker_marker
                        raise SystemExit(3)
                        """))

                results = []
                for filename in (first, second):
                    process.stdin.write(json.dumps({"filename": str(filename)}) + "\n")
                    process.stdin.flush()
                    results.append(json.loads(result_stream.readline()))

                self.assertEqual([result["returncode"] for result in results], [0, 3])
                self.assertTrue(all(result["elapsed"] >= 0 for result in results))

                process.stdin.write(json.dumps({"command": "stop"}) + "\n")
                process.stdin.flush()
                self.assertEqual(process.wait(timeout=10), 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == "__main__":
    unittest.main()

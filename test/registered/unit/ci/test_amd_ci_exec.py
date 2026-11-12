import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SCRIPT = Path(__file__).resolve().parents[4] / "scripts/ci/amd/amd_ci_exec.sh"


class TestAmdCiExec(CustomTestCase):
    def _run_wrapper(self, *, first_exit, retry_exit=0, args=("python3", "test.py")):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls_path = root / "calls.jsonl"
            docker = root / "docker"
            docker.write_text(
                f"#!{sys.executable}\n"
                + textwrap.dedent(
                    """\
                    import json
                    import os
                    import sys
                    from pathlib import Path

                    calls = Path(os.environ["MOCK_DOCKER_CALLS"])
                    is_retry = calls.exists()
                    with calls.open("a") as stream:
                        stream.write(json.dumps(sys.argv[1:]) + "\\n")
                    key = "MOCK_RETRY_EXIT" if is_retry else "MOCK_FIRST_EXIT"
                    sys.exit(int(os.environ[key]))
                    """
                )
            )
            docker.chmod(0o755)
            result = subprocess.run(
                ["bash", str(SCRIPT), *args],
                env={
                    **os.environ,
                    "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                    "AMD_CI_GPU_ARCH": "mi35x",
                    "CI_CONTAINER_NAME": "test-container",
                    "MOCK_DOCKER_CALLS": str(calls_path),
                    "MOCK_FIRST_EXIT": str(first_exit),
                    "MOCK_RETRY_EXIT": str(retry_exit),
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
        return result, calls

    def test_failure_cannot_be_hidden_by_a_successful_retry(self):
        """Test-runner errors and crashes must not disappear after an offline retry."""
        for exit_code in (1, 2, 3, 4, 5, 6, 125, 126, 127, 134, 137, 139, 143, 255):
            with self.subTest(exit_code=exit_code):
                result, calls = self._run_wrapper(first_exit=exit_code)
                self.assertEqual(result.returncode, exit_code, result.stdout)
                self.assertEqual(len(calls), 1)

    def test_offline_retry_preserves_arguments_and_final_status(self):
        command = ["python3", "test.py", "--filter", "value with spaces"]
        args = ["-w", "/work dir", "-e", "TEST_VALUE=a b=c", "--", *command]
        for retry_exit in (0, 1):
            with self.subTest(retry_exit=retry_exit):
                result, calls = self._run_wrapper(
                    first_exit=28, retry_exit=retry_exit, args=args
                )
                self.assertEqual(result.returncode, retry_exit, result.stdout)
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0][:3], ["exec", "-w", "/work dir"])
                self.assertIn("TEST_VALUE=a b=c", calls[0])
                tail = ["test-container", *command]
                self.assertEqual(calls[0][-len(tail) :], tail)
                self.assertEqual(
                    calls[1],
                    calls[0][: -len(tail)] + ["-e", "HF_HUB_OFFLINE=1"] + tail,
                )


if __name__ == "__main__":
    unittest.main()

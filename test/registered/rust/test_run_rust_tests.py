"""Run the repository's native Rust unit tests from the CPU CI suite."""

import shutil
import subprocess
import unittest
from pathlib import Path

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

BUILD_AND_RUN_TIMEOUT_S = 900
RUST_WORKSPACE = Path(__file__).resolve().parents[3] / "rust"
register_cpu_ci(est_time=900, suite="base-a-test-cpu")


# Exported by _pr-test-stage-cpu.yml as the negation of the check-changes
# rust_workspace paths filter; it defaults to false, so only a CI run that
# positively detected no rust/ changes skips the cargo build.
@unittest.skipIf(
    envs.SGLANG_SKIP_RUST_TESTS.get(),
    "SGLANG_SKIP_RUST_TESTS is set (no rust/ workspace changes per CI check-changes)",
)
class TestCargoWorkspace(CustomTestCase):
    def _run_cargo(self, args: list[str], *, cwd: Path, env: dict | None = None):
        proc = subprocess.run(
            ["cargo", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=BUILD_AND_RUN_TIMEOUT_S,
        )
        # Print unconditionally so a green run still shows which tests ran.
        print(proc.stdout)
        self.assertEqual(
            proc.returncode,
            0,
            f"`cargo {' '.join(args)}` failed in {cwd}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
        )

    def test_cargo_test_workspace(self):
        # Not skipUnless: cargo is a hard dependency of the editable install
        # (setuptools-rust builds sglang-grpc), so a missing toolchain is a
        # broken environment, and a silently-skipped CI test is worthless.
        self.assertIsNotNone(
            shutil.which("cargo"),
            "cargo not found on PATH; install a Rust toolchain "
            "(scripts/ci/utils/install_rust_protoc.sh)",
        )
        self.assertTrue(
            (RUST_WORKSPACE / "Cargo.toml").is_file(),
            f"rust workspace manifest not found at {RUST_WORKSPACE}",
        )

        self._run_cargo(["test", "--workspace"], cwd=RUST_WORKSPACE)


if __name__ == "__main__":
    unittest.main()

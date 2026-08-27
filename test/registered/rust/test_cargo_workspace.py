"""Run the `rust/` Cargo workspace's unit tests from the CPU CI suite.

The `rust/` workspace (sglang-grpc, sglang-mm, sglang-server) is compiled into
the wheel by setuptools-rust, but until now nothing ran `cargo test` in CI --
`.github/workflows/pr-test-rust.yml` and `pr-benchmark-rust.yml` are both
path-scoped to `sgl-model-gateway/**`, a different workspace. `lint.yml` covers
rustfmt/clippy via the pre-commit hooks, so this file only adds the test run.

The debug profile is deliberate: these are pure-logic tests (no timing or
codegen assertions), and the release profile costs a full LTO build for the
same coverage.
"""

import shutil
import subprocess
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# base-c-test-cpu is where this was asked for, and it matches the repo's
# base-a + base-c dual-registration convention -- but base-c-test-cpu currently
# has no runner job in any workflow (it was carved out of base-b in #28623 to
# *reduce* CPU CI scope), so base-a-test-cpu is what actually executes.
register_cpu_ci(est_time=300, suite="base-a-test-cpu")

# repo root: test/registered/rust/<this file>
RUST_WORKSPACE = Path(__file__).resolve().parents[3] / "rust"

# Not `est_time`: that is a scheduling hint for partition balancing (a rough
# average), this is a hard ceiling for the worst case. The 136 tests run in ~1s;
# what varies is the build. Cache-warm the workspace crates recompile in ~15s,
# but a Swatinem/rust-cache miss rebuilds all ~370 dependencies -- measured at
# 48s on 4 fast cores, so several minutes on a hosted runner.
#
# Capped below the 600s `timeout-minutes` on the suite's "Run test" step so a
# hang fails here, with output, instead of being killed as an opaque job
# timeout. The harness `--timeout-per-file` (1200s) is looser still.
BUILD_AND_RUN_TIMEOUT_S = 300


class TestCargoWorkspace(CustomTestCase):
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

        proc = subprocess.run(
            ["cargo", "test", "--workspace"],
            cwd=RUST_WORKSPACE,
            capture_output=True,
            text=True,
            timeout=BUILD_AND_RUN_TIMEOUT_S,
        )
        # Print unconditionally so a green run still shows which tests ran.
        print(proc.stdout)
        self.assertEqual(
            proc.returncode,
            0,
            f"`cargo test --workspace` failed in {RUST_WORKSPACE}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
        )


if __name__ == "__main__":
    unittest.main()

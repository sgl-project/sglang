"""Regression tests for scripts/ci/amd/amd_ci_install_dependency.sh.

The MORI reinstall runs its apt steps inside `docker exec ... bash -c` under
`set -euo pipefail`, where a bare `apt-get` is fatal. `apt-get update` exits
100 when any single index is unreachable, even though it keeps every index it
did fetch -- so one dead source in the ROCm base image fails the whole
"Install dependencies" step on every AMD runner at once. That is what took
out ~25 of 27 jobs in pr-test-amd run 32399046576 when AMD's internal
rocm-osdb artifactory started 404ing.

The packages those steps install are optional (rocm.Dockerfile builds MORI
without them), so none of them may abort the run.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "ci" / "amd" / "amd_ci_install_dependency.sh"

BUILD_MARKER = "MORI_BUILD_REACHED"
INSTALL_MARKER = "APT_INSTALL_RAN"

# Replays the observed failure: the rocm-osdb index 404s, every other index is
# fetched fine, and apt-get update still exits 100. APT_UPDATE_OK /
# APT_INSTALL_OK select which half of the world is healthy.
APT_STUB = """#!/bin/bash
case "$1" in
  update)
    echo "Get:1 https://archive.ubuntu.com/ubuntu jammy InRelease [270 kB]"
    echo "Fetched 48.0 MB in 3s (18.2 MB/s)"
    [ "${APT_UPDATE_OK}" = 1 ] && exit 0
    echo "Err:13 http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-22.04-deb compute-rocm-rel-7.0/38 amd64 Packages" >&2
    echo "  404  Not Found [IP: 10.216.51.87 80]" >&2
    echo "E: Some index files failed to download. They have been ignored, or old ones used instead." >&2
    exit 100
    ;;
  install)
    [ "${APT_INSTALL_OK}" = 1 ] || { echo "E: Unable to locate package" >&2; exit 100; }
    echo "APT_INSTALL_RAN"
    ;;
esac
"""


def _mori_shell_body(script: str) -> str:
    """Body of the `docker exec ... bash -c "..."` that reinstalls MORI."""
    marker = 'docker exec ci_sglang bash -c "'
    start = script.index(marker, script.index("[MORI] Reinstalling MORI")) + len(marker)
    body = script[start:]
    return body[: body.index('\n  "')]


class TestAmdCiInstallDependencyApt(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        body = _mori_shell_body(INSTALL_SCRIPT.read_text())
        cls.apt_lines = [
            line for line in body.splitlines() if line.strip().startswith("apt-get")
        ]
        # Without this the tests below would pass vacuously if the block were
        # ever restructured out from under them.
        assert cls.apt_lines, "found no apt-get lines in the MORI reinstall block"
        # The block is a double-quoted host string, so a `$` in these lines
        # would be expanded before the container ever sees them; replaying them
        # verbatim would then not reflect what really runs.
        assert not any("$" in line for line in cls.apt_lines), cls.apt_lines

    def run_apt_lines(self, *, update_ok: str, install_ok: str):
        """Run the real apt lines in the context the container gives them."""
        with tempfile.TemporaryDirectory() as stub_dir:
            stub = Path(stub_dir) / "apt-get"
            stub.write_text(APT_STUB)
            stub.chmod(0o755)
            return subprocess.run(
                [
                    "bash",
                    "-c",
                    "set -euo pipefail\n"
                    + "\n".join(self.apt_lines)
                    + f"\necho {BUILD_MARKER}\n",
                ],
                env={
                    **os.environ,
                    "PATH": stub_dir + os.pathsep + os.environ["PATH"],
                    "APT_UPDATE_OK": update_ok,
                    "APT_INSTALL_OK": install_ok,
                },
                capture_output=True,
                text=True,
            )

    def test_dead_apt_index_does_not_abort_the_dependency_install(self):
        result = self.run_apt_lines(update_ok="0", install_ok="1")
        self.assertIn(
            BUILD_MARKER,
            result.stdout,
            f"a 404 on one apt index aborted the MORI build:\n{result.stderr}",
        )
        self.assertEqual(result.returncode, 0)

    def test_unavailable_optional_package_does_not_abort_the_dependency_install(self):
        result = self.run_apt_lines(update_ok="0", install_ok="0")
        self.assertIn(
            BUILD_MARKER,
            result.stdout,
            f"an unavailable optional package aborted the MORI build:\n{result.stderr}",
        )
        self.assertEqual(result.returncode, 0)

    def test_healthy_apt_still_installs_the_optional_packages(self):
        result = self.run_apt_lines(update_ok="1", install_ok="1")
        self.assertIn(INSTALL_MARKER, result.stdout)
        self.assertIn(BUILD_MARKER, result.stdout)
        self.assertEqual(result.returncode, 0)

    def test_every_mori_apt_line_stays_guarded(self):
        for line in self.apt_lines:
            self.assertIn(
                "||",
                line,
                "unguarded apt step under `set -e` will fail every AMD job "
                f"whenever a single apt source is unreachable: {line.strip()}",
            )


if __name__ == "__main__":
    unittest.main()

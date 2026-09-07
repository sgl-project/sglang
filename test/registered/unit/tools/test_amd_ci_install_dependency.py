"""Guard on the apt calls in scripts/ci/amd/amd_ci_install_dependency.sh.

Those calls run under `set -euo pipefail`, and `apt-get update` exits 100 when
any single index is unreachable -- even though it keeps every index it did
fetch. An unguarded call therefore fails the whole "Install dependencies" step
on every AMD runner at once, which is what took out ~25 of 27 jobs in
pr-test-amd run 32399046576 when AMD's internal rocm-osdb artifactory started
404ing on an index this repo never installs from.

The packages involved are optional -- rocm.Dockerfile builds MORI without them
-- so no apt call here may be able to abort the run.
"""

import re
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

INSTALL_SCRIPT = (
    Path(__file__).resolve().parents[4] / "scripts/ci/amd/amd_ci_install_dependency.sh"
)


class TestAmdCiInstallDependencyApt(CustomTestCase):
    def test_apt_calls_cannot_abort_the_dependency_install(self):
        unguarded = [
            line.strip()
            for line in INSTALL_SCRIPT.read_text().splitlines()
            if re.match(r"\s*(sudo\s+)?apt-get\b", line) and "||" not in line
        ]
        self.assertEqual(
            unguarded,
            [],
            "an unguarded apt-get under `set -e` fails the dependency install on "
            "every AMD runner whenever one apt source is unreachable; give it an "
            "`|| echo ...` fallback",
        )


if __name__ == "__main__":
    unittest.main()

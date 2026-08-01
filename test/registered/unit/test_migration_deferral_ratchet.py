"""Ratchet guard: config-namespace-migration test deferrals may only decrease.

A set of unit tests was module-skipped ("Temporarily skipped during the
ServerArgs config-namespace migration") because their fixtures inject config
in ways the namespace accessors cannot see; they are recovered together with
the reader migration. Nothing else enforces that list: without this pin a new
skip can ride in unnoticed, and the already-skipped files keep growing test
code nobody has ever run. The count is exact: un-skipping a file must lower
the baseline to lock in the recovery, and no new deferral may appear.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from pathlib import Path

from sglang.test.test_utils import CustomTestCase

# test/registered/unit/<this file> -> test/
_TEST_ROOT = Path(__file__).resolve().parents[2]

_MARKER = "config-namespace migration"

_BASELINE = 15


class TestMigrationDeferralRatchet(CustomTestCase):
    def test_deferred_test_files_match_the_baseline(self):
        deferred = sorted(
            p.relative_to(_TEST_ROOT).as_posix()
            for p in _TEST_ROOT.rglob("*.py")
            if p.name != Path(__file__).name and _MARKER in p.read_text(errors="ignore")
        )
        count = len(deferred)
        if count > _BASELINE:
            self.fail(
                f"deferred (module-skipped) migration tests grew: {count} > "
                f"baseline {_BASELINE}: {deferred}. Do not add new deferrals — "
                "seed the context (override_server_args / publish a real "
                "ServerArgs) instead of skipping the module."
            )
        if count < _BASELINE:
            self.fail(
                f"deferred migration tests shrank: {count} < baseline "
                f"{_BASELINE}. Lower the baseline in this file to lock in the "
                "recovery."
            )


if __name__ == "__main__":
    unittest.main()

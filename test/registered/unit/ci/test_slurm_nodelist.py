import subprocess
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

EXPAND_NODELIST = (
    Path(__file__).resolve().parents[4] / "scripts/ci/slurm/expand_slurm_nodelist.py"
)


class TestSlurmNodeList(CustomTestCase):
    def test_cleanup_receives_complete_hostnames(self):
        """A compressed NodeList must not become invalid SSH targets."""
        for nodelist, expected in (
            ("pit2-p03-g16", ["pit2-p03-g16"]),
            ("pit2-p03-g16,pit2-p03-g26", ["pit2-p03-g16", "pit2-p03-g26"]),
            ("pit2-p03-g[16,26]", ["pit2-p03-g16", "pit2-p03-g26"]),
            (
                "node[007-009,012],other1",
                ["node007", "node008", "node009", "node012", "other1"],
            ),
            ("rack[1-2]n[01-02]", ["rack1n01", "rack1n02", "rack2n01", "rack2n02"]),
            ("(null)", []),
            ("", []),
        ):
            with self.subTest(nodelist=nodelist):
                completed = subprocess.run(
                    [sys.executable, str(EXPAND_NODELIST)],
                    input=nodelist,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                self.assertEqual(completed.stdout.splitlines(), expected)

    def test_invalid_hostlist_emits_no_partial_cleanup_targets(self):
        for nodelist in (
            "good,node[1-",
            "node[3-1]",
            "node[1,,2]",
            "-option",
            "node;cmd",
        ):
            with self.subTest(nodelist=nodelist):
                completed = subprocess.run(
                    [sys.executable, str(EXPAND_NODELIST)],
                    input=nodelist,
                    text=True,
                    capture_output=True,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(completed.stdout, "")
                self.assertTrue(completed.stderr)


if __name__ == "__main__":
    unittest.main()

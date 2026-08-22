"""Unit tests for scripts/ci/amd/amd_ci_latest_release_tag.py"""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[4]
CI_REGISTER_PATH = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"
HELPER_PATH = REPO_ROOT / "scripts" / "ci" / "amd" / "amd_ci_latest_release_tag.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Loaded directly, like the sibling get_version_tag test: this covers CI tooling
# and should not need the sglang package importable to run.
register_cpu_ci = _load_module("ci_register", CI_REGISTER_PATH).register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAmdCiLatestReleaseTag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = _load_module("amd_ci_latest_release_tag", HELPER_PATH)

    def test_remote_tags_are_listed_with_refs_only(self):
        ls_remote_output = (
            "aaa\trefs/tags/v0.5.9\n"
            "bbb\trefs/tags/v0.5.10\n"
            "ccc\trefs/heads/main\n"
        )

        with patch.object(
            self.helper, "run_git", return_value=ls_remote_output
        ) as run_git:
            tags = self.helper.list_remote_tags()

        # ls-remote is what keeps this from transferring objects into the shallow
        # CI checkout, and --refs is what keeps annotated tags from also yielding
        # a peeled `<tag>^{}` entry.
        run_git.assert_called_once_with(
            "ls-remote", "--tags", "--refs", "origin", "v*.*.*"
        )
        self.assertEqual(tags, ["v0.5.9", "v0.5.10"])

    def test_tag_ordering_matches_the_shared_release_helper(self):
        # The image tag only resolves if this picks the same tag the nightly
        # release workflow published under, so stable/post must sort above rc.
        with patch.object(
            self.helper,
            "list_remote_tags",
            return_value=["v0.5.10rc0", "v0.5.9", "v0.5.10.post1", "v0.5.10"],
        ):
            self.assertEqual(self.helper.get_latest_release_tag(), "v0.5.10.post1")

    def test_unreachable_remote_falls_back_to_local_tags(self):
        with (
            patch.object(self.helper, "list_remote_tags", return_value=[]),
            patch.object(
                self.helper, "list_local_tags", return_value=["v0.5.8", "v0.5.9"]
            ),
        ):
            self.assertEqual(self.helper.get_latest_release_tag(), "v0.5.9")

    def test_no_tags_anywhere_yields_empty_so_callers_keep_their_default(self):
        with (
            patch.object(self.helper, "list_remote_tags", return_value=[]),
            patch.object(self.helper, "list_local_tags", return_value=[]),
        ):
            self.assertEqual(self.helper.get_latest_release_tag(), "")

    def test_failed_git_invocation_is_reported_but_not_fatal(self):
        failed = Mock(returncode=128, stdout="", stderr="no such remote")

        with patch.object(self.helper.subprocess, "run", return_value=failed):
            self.assertEqual(self.helper.run_git("ls-remote"), "")


if __name__ == "__main__":
    unittest.main()

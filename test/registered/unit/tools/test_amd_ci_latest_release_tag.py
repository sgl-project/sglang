"""Unit tests for scripts/ci/amd/amd_ci_latest_release_tag.py

The AMD container scripts resolve the nightly image tag by reading tag *names*
off the remote with `git ls-remote`, never by fetching tag objects into the
depth-1 CI checkout. These tests pin both halves of that: the helper's own
behavior, and the fact that the two container scripts still call it.
"""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]
HELPER_PATH = REPO_ROOT / "scripts/ci/amd/amd_ci_latest_release_tag.py"
CONTAINER_SCRIPTS = [
    REPO_ROOT / "scripts/ci/amd/amd_ci_start_container.sh",
    REPO_ROOT / "scripts/ci/amd/amd_ci_start_container_disagg.sh",
]


class TestAmdCiLatestReleaseTag(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "amd_ci_latest_release_tag", HELPER_PATH
        )
        cls.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.helper)

    def test_remote_tags_are_listed_without_transferring_objects(self):
        ls_remote_output = (
            "aaa\trefs/tags/v0.5.9\n"
            "bbb\trefs/tags/v0.5.10\n"
            "ccc\trefs/heads/main\n"
        )

        with patch.object(
            self.helper, "run_git", return_value=ls_remote_output
        ) as run_git:
            tags = self.helper.list_remote_tags()

        # ls-remote is what keeps this from pulling tag objects into the depth-1
        # CI checkout, and --refs is what keeps an annotated tag from also
        # yielding a peeled `<tag>^{}` entry.
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

    def test_missing_release_helper_yields_empty_instead_of_a_traceback(self):
        # Branches from before #35196 keep the ordering helper at
        # python/tools/get_version_tag.py, so a cherry-pick of this script onto
        # one of them finds nothing at VERSION_HELPER_PATH. Guessing an order
        # there could name an image the nightly never published.
        with patch.object(
            self.helper, "VERSION_HELPER_PATH", Path("/nonexistent/get_version_tag.py")
        ):
            self.assertIsNone(self.helper.load_parse_version_tuple())
            self.assertEqual(self.helper.get_latest_release_tag(), "")

    def test_failed_git_invocation_is_reported_but_not_fatal(self):
        failed = Mock(returncode=128, stdout="", stderr="no such remote")

        with patch.object(self.helper.subprocess, "run", return_value=failed):
            self.assertEqual(self.helper.run_git("ls-remote"), "")

    def test_container_scripts_resolve_the_tag_without_fetching_tags(self):
        for path in CONTAINER_SCRIPTS:
            with self.subTest(script=path.name):
                lines = path.read_text().splitlines()
                self.assertTrue(
                    any("amd_ci_latest_release_tag.py" in line for line in lines),
                    f"{path.name} no longer resolves the image tag through the "
                    "helper, so it is back to whatever the tag lookup was before",
                )
                self.assertEqual(
                    [line.strip() for line in lines if "fetch --tags" in line],
                    [],
                    f"{path.name} fetches tags into the depth-1 CI checkout: that "
                    "transfers ~170MB of objects no later step reads, and still "
                    "cannot make `git describe` work from a single-commit HEAD; "
                    "read the tag name off the remote instead",
                )


if __name__ == "__main__":
    unittest.main()

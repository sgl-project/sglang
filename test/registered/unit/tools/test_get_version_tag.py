import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[4]
CI_REGISTER_PATH = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"
VERSION_HELPER_PATH = REPO_ROOT / "python" / "tools" / "get_version_tag.py"
PYPROJECT_PATHS = [
    REPO_ROOT / "python" / "pyproject.toml",
    REPO_ROOT / "python" / "pyproject_cpu.toml",
    REPO_ROOT / "python" / "pyproject_npu.toml",
    REPO_ROOT / "python" / "pyproject_other.toml",
    REPO_ROOT / "python" / "pyproject_xpu.toml",
    REPO_ROOT / "3rdparty" / "amd" / "wheel" / "sglang" / "pyproject.toml",
]
DESCRIBE_COMMAND = (
    'git_describe_command = ["python3", "python/tools/get_version_tag.py"]'
)
TAG_ONLY_DESCRIBE_COMMAND = (
    'git_describe_command = ["python3", "python/tools/get_version_tag.py", '
    '"--tag-only"]'
)
FALLBACK_VERSION = 'fallback_version = "0.0.0.dev0"'


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module("ci_register", CI_REGISTER_PATH).register_cpu_ci
register_cpu_ci(est_time=0, suite="base-a-test-cpu")


class TestGetVersionTag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.version_helper = _load_module("get_version_tag", VERSION_HELPER_PATH)

    def test_parse_version_tuple_sorts_stable_above_rc_and_post_above_stable(self):
        tags = ["v0.5.10rc0", "v0.5.9", "v0.5.10.post1", "v0.5.10"]

        self.assertEqual(
            sorted(tags, key=self.version_helper.parse_version_tuple, reverse=True),
            ["v0.5.10.post1", "v0.5.10", "v0.5.10rc0", "v0.5.9"],
        )

    def test_exact_version_tag_takes_precedence_over_latest_tag(self):
        with (
            patch.object(
                self.version_helper, "get_exact_version_tag", return_value="v0.5.9"
            ),
            patch.object(
                self.version_helper, "get_latest_version_tag_describe"
            ) as latest_describe,
        ):
            self.assertEqual(self.version_helper.get_version_describe(), "v0.5.9")

        latest_describe.assert_not_called()

    def test_pyprojects_use_describe_mode_for_setuptools_scm(self):
        for path in PYPROJECT_PATHS:
            with self.subTest(path=path):
                content = path.read_text()
                self.assertIn(DESCRIBE_COMMAND, content)
                self.assertNotIn(TAG_ONLY_DESCRIBE_COMMAND, content)
                self.assertIn(FALLBACK_VERSION, content)

    def test_strip_post_suffix(self):
        """strip_post_suffix removes .postN so setuptools-scm bumps micro."""
        strip = self.version_helper.strip_post_suffix
        self.assertEqual(strip("v0.5.10.post1"), "v0.5.10")
        self.assertEqual(strip("v0.5.10.post99"), "v0.5.10")
        self.assertEqual(strip("v0.5.10"), "v0.5.10")
        self.assertEqual(strip("v0.5.10rc1"), "v0.5.10rc1")

    def test_ancestor_tag_excludes_non_ancestor_tags(self):
        """Highest global tag (v0.5.11) should be ignored if not an ancestor."""

        def fake_run_git(*args, **kwargs):
            # git tag --merged HEAD --list v*.*.*  →  only ancestor tags
            if "--merged" in args:
                return "v0.5.9\nv0.5.10\nv0.5.6"
            return ""

        with patch.object(
            self.version_helper,
            "run_git",
            side_effect=fake_run_git,
        ):
            result = self.version_helper.get_latest_ancestor_version_tag()
            self.assertEqual(result, "v0.5.10")

    def test_describe_uses_ancestor_tag_when_it_is_latest(self):
        """Ancestor tag used when it matches global latest, with .postN stripped."""
        with patch.object(
            self.version_helper,
            "get_latest_version_tag",
            return_value="v0.5.10",
        ), patch.object(
            self.version_helper,
            "get_latest_ancestor_version_tag",
            return_value="v0.5.10",
        ), patch.object(
            self.version_helper,
            "run_git",
            return_value="v0.5.10-5-gabcdef0",
        ):
            result = self.version_helper.get_latest_version_tag_describe()
            self.assertEqual(result, "v0.5.10-5-gabcdef0")

    def test_describe_strips_post_suffix_from_ancestor_tag(self):
        """v0.5.10.post1 ancestor → describe relative to v0.5.10.

        setuptools-scm will then produce 0.5.11.devN (correct),
        not 0.5.10.post2.devN (wrong).
        """
        with patch.object(
            self.version_helper,
            "get_latest_version_tag",
            return_value="v0.5.10.post1",
        ), patch.object(
            self.version_helper,
            "get_latest_ancestor_version_tag",
            return_value="v0.5.10.post1",
        ), patch.object(
            self.version_helper,
            "run_git",
            return_value="v0.5.10-8-gabcdef0",
        ):
            result = self.version_helper.get_latest_version_tag_describe()
            # Must be v0.5.10-based, NOT v0.5.10.post1-based
            self.assertEqual(result, "v0.5.10-8-gabcdef0")

    def test_describe_uses_global_tag_on_main_with_release_branches(self):
        """On main where release tags live on release branches, use global tag.

        main's highest ancestor tag is v0.5.6.post2 but the global latest
        is v0.5.10.post1 (on release/v0.5.10).  The global tag wins because
        it's newer, and .postN is stripped so setuptools-scm produces
        0.5.11.devN.
        """

        def fake_run_git(*args, **kwargs):
            # _describe_from_tag for global v0.5.10.post1:
            #   git describe --match v0.5.10 HEAD → fails (not ancestor)
            if "describe" in args:
                return ""
            #   merge-base v0.5.10.post1 HEAD → aaa111
            if args == ("merge-base", "v0.5.10.post1", "HEAD"):
                return "aaa111"
            if args == ("rev-list", "--count", "aaa111..HEAD"):
                return "3924"
            if args == ("rev-parse", "--short", "HEAD"):
                return "222eda1"
            return ""

        with patch.object(
            self.version_helper,
            "get_latest_version_tag",
            return_value="v0.5.10.post1",
        ), patch.object(
            self.version_helper,
            "get_latest_ancestor_version_tag",
            return_value="v0.5.6.post2",
        ), patch.object(
            self.version_helper,
            "run_git",
            side_effect=fake_run_git,
        ):
            result = self.version_helper.get_latest_version_tag_describe()
            # Global v0.5.10.post1 used, .post1 stripped → v0.5.10
            self.assertEqual(result, "v0.5.10-3924-g222eda1")

    def test_describe_falls_back_to_global_tag_when_no_ancestor_tags(self):
        """When no ancestor tags exist, fall back to globally highest tag."""

        def fake_run_git(*args, **kwargs):
            if args == ("merge-base", "v0.5.11", "HEAD"):
                return "aaa111"
            if args == ("rev-list", "--count", "aaa111..HEAD"):
                return "42"
            if args == ("rev-parse", "--short", "HEAD"):
                return "bbb222"
            return ""

        with patch.object(
            self.version_helper,
            "get_latest_version_tag",
            return_value="v0.5.11",
        ), patch.object(
            self.version_helper,
            "get_latest_ancestor_version_tag",
            return_value="",
        ), patch.object(
            self.version_helper,
            "run_git",
            side_effect=fake_run_git,
        ):
            result = self.version_helper.get_latest_version_tag_describe()
            self.assertEqual(result, "v0.5.11-42-gbbb222")

    def test_tag_only_cli_mode_remains_available_for_callers_that_need_latest_tag(self):
        with (
            patch.object(sys, "argv", ["get_version_tag.py", "--tag-only"]),
            patch.object(
                self.version_helper, "get_latest_version_tag", return_value="v0.5.10"
            ),
            patch.object(
                self.version_helper, "get_version_describe"
            ) as version_describe,
            patch("builtins.print") as print_mock,
        ):
            self.version_helper.main()

        version_describe.assert_not_called()
        print_mock.assert_called_once_with("v0.5.10")


if __name__ == "__main__":
    unittest.main()

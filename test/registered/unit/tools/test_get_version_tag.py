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
register_cpu_ci(est_time=0, suite="stage-a-test-cpu")


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

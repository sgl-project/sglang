import importlib.util
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CI_REGISTER_PATH = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"

# pyproject variants where easydict used to be a non-optional core dependency and
# has been moved to the opt-in `remote-models` extra. easydict is LGPL-3.0 and is
# not imported by sglang itself; keeping it out of the default install keeps a
# plain `pip install sglang` Apache-2.0 clean (issue #29177).
CORE_DEP_PYPROJECTS = [
    REPO_ROOT / "python" / "pyproject.toml",
    REPO_ROOT / "python" / "pyproject_cpu.toml",
    REPO_ROOT / "python" / "pyproject_npu.toml",
    REPO_ROOT / "python" / "pyproject_xpu.toml",
]
# pyproject_other.toml keeps easydict inside its optional `runtime_common` extra
# (its core `dependencies` list is already minimal), so it is intentionally exempt.
OTHER_PYPROJECT = REPO_ROOT / "python" / "pyproject_other.toml"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module("ci_register", CI_REGISTER_PATH).register_cpu_ci
register_cpu_ci(est_time=0, suite="base-a-test-cpu")


def _extract_array(text, key):
    """Return the raw text between a top-level ``key = [`` and its matching ``]``.

    Uses a bracket-depth counter so both single-line (``x = ["a"]``) and
    multi-line arrays are captured. The assignment must start at the beginning
    of a line so a key nested inside another table is not matched. Parsed as
    text (no tomllib) to stay compatible with the ``requires-python >= 3.10``
    floor, matching the other pyproject tests in this directory.
    """
    m = re.search(r"(?m)^" + re.escape(key) + r"\s*=\s*\[", text)
    if not m:
        return None
    depth = 1
    i = m.end()
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
        i += 1
    return text[m.end() : i - 1]


class TestPyprojectLicenseExtras(unittest.TestCase):
    """Regression guard for issue #29177 (easydict LGPL-3.0 compliance).

    easydict must never be pulled into a default install; it lives only behind
    an opt-in extra. These tests fail if a future edit re-adds it as a core
    dependency.
    """

    def test_easydict_not_in_core_dependencies(self):
        for path in CORE_DEP_PYPROJECTS:
            with self.subTest(pyproject=path.name):
                core = _extract_array(path.read_text(), "dependencies")
                self.assertIsNotNone(
                    core, f"no core `dependencies` array found in {path.name}"
                )
                self.assertNotIn(
                    "easydict",
                    core,
                    f"easydict (LGPL-3.0) must not be a core dependency in "
                    f"{path.name}; use the opt-in `remote-models` extra (issue #29177)",
                )

    def test_remote_models_extra_provides_easydict(self):
        for path in CORE_DEP_PYPROJECTS:
            with self.subTest(pyproject=path.name):
                extra = _extract_array(path.read_text(), "remote-models")
                self.assertIsNotNone(
                    extra, f"`remote-models` extra missing in {path.name}"
                )
                self.assertIn(
                    "easydict",
                    extra,
                    f"`remote-models` extra must provide easydict in {path.name}",
                )

    def test_other_pyproject_keeps_easydict_in_runtime_common(self):
        runtime_common = _extract_array(
            OTHER_PYPROJECT.read_text(), "runtime_common"
        )
        self.assertIsNotNone(
            runtime_common, "`runtime_common` extra missing in pyproject_other.toml"
        )
        self.assertIn(
            "easydict",
            runtime_common,
            "pyproject_other.toml is expected to keep easydict in its optional "
            "`runtime_common` extra",
        )


if __name__ == "__main__":
    unittest.main()

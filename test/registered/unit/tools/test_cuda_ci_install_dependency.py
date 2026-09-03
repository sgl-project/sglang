import importlib.util
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CI_REGISTER_PATH = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "ci" / "cuda" / "ci_install_dependency.sh"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module("ci_register", CI_REGISTER_PATH).register_cpu_ci
register_cpu_ci(est_time=0, suite="base-a-test-cpu")


class TestCudaCiInstallDependencyTorchCache(unittest.TestCase):
    def test_rebuilds_torch_extensions_before_editable_install(self):
        script = INSTALL_SCRIPT.read_text()

        self.assertRegex(
            script,
            re.compile(
                r"setup_cargo_cache\s*\n"
                r"\s*invalidate_torch_rust_cache\s*\n"
                r"\s*install_sglang"
            ),
        )
        self.assertIn('"${SGLANG_BUILD_RUST_EXTS:-}" = "none"', script)
        self.assertRegex(
            script,
            re.compile(
                r"cargo clean --release --manifest-path "
                r"\"\$\{REPO_ROOT\}/rust/sglang-radix-tree/Cargo\.toml\"\s*\\\n"
                r"\s*-p torch-sys -p sglang-radix-tree"
            ),
        )


if __name__ == "__main__":
    unittest.main()

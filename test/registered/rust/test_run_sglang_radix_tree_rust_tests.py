"""Run the standalone sglang-radix-tree crate's native Rust unit tests."""

import shutil
import subprocess
import unittest
from pathlib import Path

from sglang.srt.environ import envs
from sglang.srt.rust_extensions.torch_build import torch_build_configuration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

BUILD_AND_RUN_TIMEOUT_S = 900
RUST_WORKSPACE = Path(__file__).resolve().parents[3] / "rust"
SGLANG_RADIX_TREE_MANIFEST = RUST_WORKSPACE / "sglang-radix-tree" / "Cargo.toml"

register_cpu_ci(est_time=900, suite="base-a-test-cpu")


@unittest.skipIf(
    envs.SGLANG_SKIP_RUST_TESTS.get(),
    "SGLANG_SKIP_RUST_TESTS is set (no rust/ workspace changes per CI check-changes)",
)
class TestSGLangRadixTreeCargo(CustomTestCase):
    def test_sglang_radix_tree_native_tests(self):
        self.assertIsNotNone(
            shutil.which("cargo"),
            "cargo not found on PATH; install a Rust toolchain "
            "(scripts/ci/utils/install_rust_protoc.sh)",
        )
        self.assertTrue(
            SGLANG_RADIX_TREE_MANIFEST.is_file(),
            f"sglang-radix-tree manifest not found at {SGLANG_RADIX_TREE_MANIFEST}",
        )
        build = torch_build_configuration(
            compat_header=SGLANG_RADIX_TREE_MANIFEST.parent / "torch_2_13_compat.h",
            python_module="sglang.srt.mem_cache.rust_tree_core.mem_cache",
        )
        proc = subprocess.run(
            [
                "cargo",
                "test",
                "--manifest-path",
                str(SGLANG_RADIX_TREE_MANIFEST),
                "--locked",
                "--no-default-features",
            ],
            cwd=RUST_WORKSPACE,
            env=build.environment,
            capture_output=True,
            text=True,
            timeout=BUILD_AND_RUN_TIMEOUT_S,
        )
        print(proc.stdout)
        self.assertEqual(
            proc.returncode,
            0,
            f"sglang-radix-tree native tests failed\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
        )


if __name__ == "__main__":
    unittest.main()

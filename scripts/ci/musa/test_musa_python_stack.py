from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).with_name("musa_python_stack.py")
SPEC = importlib.util.spec_from_file_location("musa_python_stack", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
musa_python_stack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(musa_python_stack)


class MusaPythonStackTest(unittest.TestCase):
    def test_installer_upgrades_tvm_ffi_without_replacing_vendor_torch(self) -> None:
        script = MODULE_PATH.with_name("musa_install_dependency.sh").read_text()
        start = script.index("VENDOR_WHEELS=(")
        end = script.index('if [ -z "$MUSA_CI_ISOLATED_USERBASE" ]; then', start)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            bundle.mkdir()
            old_ffi = bundle / "apache_tvm_ffi-0.1.9.post2+musa.1.whl"
            torch_wheel = bundle / "torch-2.9.0.whl"
            musa_wheel = bundle / "torch_musa-2.9.0.whl"
            for wheel in (old_ffi, torch_wheel, musa_wheel):
                wheel.touch()
            ffi = root / "apache_tvm_ffi-0.1.11.post1+musa.1.whl"
            torchada = root / "torchada-0.1.90.whl"
            result = subprocess.run(
                ["bash"],
                input=(
                    "set -euo pipefail\n"
                    + script[start:end]
                    + '\nprintf "%s\\n" "${VENDOR_WHEELS[@]}"\n'
                ),
                env={
                    **os.environ,
                    "WHL_DIR": str(bundle),
                    "MUSA_TORCHADA_WHEEL": str(torchada),
                    "MUSA_TRITON_WHEEL": "",
                    "MUSA_TVM_FFI_WHEEL": str(ffi),
                },
                text=True,
                capture_output=True,
                check=True,
            )
        self.assertCountEqual(
            result.stdout.splitlines(),
            [str(torchada), str(ffi), str(torch_wheel), str(musa_wheel)],
        )

    def test_legacy_runner_installs_runtime_and_tests_without_new_vendor_pins(
        self,
    ) -> None:
        self.assertEqual(
            musa_python_stack.ci_install_extras("2.9.0"),
            "runtime_common,diffusion_musa,test",
        )

    def test_current_runner_uses_public_musa_dependency_set(self) -> None:
        self.assertEqual(
            musa_python_stack.ci_install_extras("2.11.0.post1+musa5.2.0"),
            "dev_musa",
        )

    def test_install_extras_reject_unknown_torch_line(self) -> None:
        with self.assertRaisesRegex(
            musa_python_stack.StackError, "unsupported MUSA Torch line 2.10"
        ):
            musa_python_stack.ci_install_extras("2.10.0")

    def test_compressed_tensors_for_torch_29(self) -> None:
        self.assertEqual(
            musa_python_stack.compressed_tensors_version("2.9.0+musa"),
            "0.15.0",
        )

    def test_compressed_tensors_for_torch_211(self) -> None:
        self.assertEqual(
            musa_python_stack.compressed_tensors_version("2.11.0.post1+musa5.2.0"),
            "0.17.0",
        )

    def test_unknown_torch_line_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            musa_python_stack.StackError, "unsupported MUSA Torch line 2.10"
        ):
            musa_python_stack.compressed_tensors_version("2.10.0")

    def test_mismatched_torch_and_torch_musa_lines_fail(self) -> None:
        with self.assertRaisesRegex(
            musa_python_stack.StackError,
            "Torch and Torch-MUSA lines do not match",
        ):
            musa_python_stack.validate_core_versions(
                {
                    "torch": "2.11.0.post1+musa5.2.0",
                    "torch-musa": "2.9.0+musa4.3.0",
                }
            )

    def test_constraints_pin_core_and_optional_vendor_packages(self) -> None:
        versions = {
            "torch": "2.9.0",
            "torch-musa": "2.9.0",
            "torchada": "0.1.82",
            "triton": "3.2.0",
            "apache-tvm-ffi": "0.1.9.post3+musa.1",
            "deep-gemm": "0.2.4+musa",
            "mate": "0.2.0+musa",
        }

        def version(name: str) -> str:
            try:
                return versions[name]
            except KeyError as exc:
                raise musa_python_stack.importlib.metadata.PackageNotFoundError(
                    name
                ) from exc

        with mock.patch.object(
            musa_python_stack.importlib.metadata, "version", side_effect=version
        ):
            pins = musa_python_stack.build_constraints()

        self.assertIn("torch==2.9.0", pins)
        self.assertIn("torch-musa==2.9.0", pins)
        self.assertIn("torchada==0.1.82", pins)
        self.assertIn("triton==3.2.0", pins)
        self.assertIn("apache-tvm-ffi==0.1.9.post3+musa.1", pins)
        self.assertIn("deep-gemm==0.2.4+musa", pins)
        self.assertIn("mate==0.2.0+musa", pins)
        self.assertIn("compressed-tensors==0.15.0", pins)


if __name__ == "__main__":
    unittest.main()

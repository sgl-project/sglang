import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


class TestDiffusionRustGate(unittest.TestCase):
    def test_build_gate(self):
        jobs = yaml.safe_load((ROOT / ".github/workflows/pr-test.yml").read_text())[
            "jobs"
        ]
        for name in ("rust-ext-build", "rust-ext-build-aarch64"):
            for event in ("pull_request", "schedule", "workflow_dispatch"):
                for main, jit, kernel, diffusion in (
                    (False, False, False, True),
                    (True, False, False, True),
                    (False, True, False, True),
                    (False, False, True, True),
                    (False, False, False, False),
                ):
                    with self.subTest(
                        job=name, event=event, flags=(main, jit, kernel, diffusion)
                    ):
                        expression = jobs[name]["if"]
                        values = {
                            "github.event_name": event,
                            "needs.check-changes.result": "success",
                            "needs.call-gate.result": "success",
                            "needs.check-changes.outputs.main_package": str(
                                main
                            ).lower(),
                            "needs.check-changes.outputs.jit_kernel": str(jit).lower(),
                            "needs.check-changes.outputs.sgl_kernel": str(
                                kernel
                            ).lower(),
                            "needs.check-changes.outputs.multimodal_gen": str(
                                diffusion
                            ).lower(),
                        }
                        for key, value in values.items():
                            expression = expression.replace(key, repr(value))
                        expression = expression.replace("!cancelled()", "True")
                        expression = expression.replace("&&", " and ").replace(
                            "||", " or "
                        )
                        actual = eval(
                            " ".join(expression.split()), {"__builtins__": {}}
                        )
                        expected = (
                            event != "pull_request"
                            or not diffusion
                            or main
                            or jit
                            or kernel
                        )
                        self.assertEqual(actual, expected)

        caller = jobs["call-multimodal-gen-tests"]
        self.assertNotIn("rust-ext-build", caller["needs"])
        self.assertNotIn("rust_ext_artifact", caller["with"])
        callee = (ROOT / ".github/workflows/pr-test-multimodal-gen.yml").read_text()
        self.assertNotIn("download-rust-ext", callee)
        self.assertNotIn("rust_ext_artifact", callee)

    def test_installer_preserves_srt_fallback(self):
        source = (ROOT / "scripts/ci/cuda/ci_install_dependency.sh").read_text()
        # exercise the actual installer functions, replacing only package-manager I/O
        functions = "\n".join(
            re.search(rf"^{name}\(\) \{{\n.*?^\}}", source, re.M | re.S)[0]
            for name in (
                "configure_environment",
                "require_prebuilt_rust_exts",
                "setup_cargo_cache",
            )
        )
        for extra, expected in (("diffusion", "none:never"), ("", ":auto")):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                script = (
                    functions
                    + """
mark_step_done() { :; }
python3() { if [ "$1" = "-c" ]; then echo .test.so; fi; }
uv() { :; }
pip() { :; }
configure_environment "$1"
require_prebuilt_rust_exts
if [ "$1" = diffusion ]; then setup_cargo_cache; fi
printf 'RESULT=%s:%s\n' "$SGLANG_BUILD_RUST_EXTS" "$SGLANG_RUST_BUILD_MODE"
"""
                )
                env = dict(
                    os.environ,
                    GITHUB_ENV=f"{tmp}/env",
                    USE_VENV="0",
                    SGLANG_BUILD_RUST_EXTS="none",
                    SGLANG_RUST_BUILD_MODE="never",
                )
                result = subprocess.run(
                    ["bash", "-eu", "-c", script, "test", extra],
                    cwd=tmp,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn(f"RESULT={expected}", result.stdout)
                if extra == "diffusion":
                    self.assertIn(
                        "SGLANG_BUILD_RUST_EXTS=none",
                        Path(env["GITHUB_ENV"]).read_text(),
                    )


if __name__ == "__main__":
    unittest.main()

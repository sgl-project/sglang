import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.test_utils import CustomTestCase

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/ci/ppu/check_dependencies.py"
ADDITIONS = {
    "sailify": "1.0.0",
    "sgl-eval": "0.1.2",
    "math-verify": "0.9.0",
    "latex2sympy2-extended": "1.11.0",
    "antlr4-python3-runtime": "4.9.3",
    "editdistance": "0.8.1",
}
WHEELS = {
    "sailify": "sailify-1.0.0-py3-none-any.whl",
    "sgl-eval": "sgl_eval-0.1.2-py3-none-any.whl",
    "math-verify": "Math.Verify-0.9.0-py3-none-any.whl",
    "latex2sympy2-extended": "latex2sympy2_extended-1.11.0-py3-none-any.whl",
    "antlr4-python3-runtime": "antlr4_python3_runtime-4.9.3-1-py3-none-any.whl",
    "editdistance": "editdistance-0.8.1-cp312-cp312-linux_x86_64.whl",
}


class TestPPUCIDependencies(CustomTestCase):
    def check_snapshots(self, before, after):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("before.json", "after.json")]
            for path, snapshot in zip(paths, (before, after)):
                path.write_text(json.dumps(snapshot))
            return subprocess.run(
                [sys.executable, str(SCRIPT), "check", *map(str, paths)],
                capture_output=True,
                text=True,
            )

    def test_built_replacements_and_pinned_runtime_additions_are_allowed(self):
        before = {
            "torch": {"version": "2.13.0", "record": "compatible"},
            "triton": {"version": "3.7.0", "record": "vendor"},
            "sglang": {"version": "0.5.17", "record": "preinstalled"},
            "sglang-kernel": {"version": "0.4.8", "record": "preinstalled"},
        }
        after = {
            **before,
            "sglang": {"version": "0.5.18", "record": "built"},
            "sglang-kernel": {"version": "0.4.9", "record": "built"},
            **{
                name: {"version": version, "record": "added"}
                for name, version in ADDITIONS.items()
            },
        }
        result = self.check_snapshots(before, after)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unpinned_runtime_additions_are_rejected(self):
        for name, version in ADDITIONS.items():
            with self.subTest(name=name):
                result = self.check_snapshots(
                    {}, {name: {"version": version + ".1", "record": "added"}}
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name, result.stderr)

    def test_existing_dependencies_are_preserved(self):
        before = {
            "torch": {"version": "2.13.0", "record": "compatible"},
            "triton": {"version": "3.7.0", "record": "vendor"},
            "sailify": {"version": "vendor-version", "record": "vendor"},
            "sgl-eval": {"version": ADDITIONS["sgl-eval"], "record": "vendor"},
        }
        result = self.check_snapshots(before, before)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name, replacement in (
            ("torch", {"version": "2.13.0", "record": "cuda-free"}),
            ("triton", {"version": "3.8.0", "record": "vendor"}),
            ("triton", {"version": "3.7.0", "record": "upstream"}),
            ("triton", None),
            ("sailify", {"version": ADDITIONS["sailify"], "record": "vendor"}),
            ("sailify", {"version": "vendor-version", "record": "different"}),
            ("sailify", None),
            ("sgl-eval", {"version": "0.1.3", "record": "vendor"}),
            ("sgl-eval", {"version": ADDITIONS["sgl-eval"], "record": "different"}),
            ("sgl-eval", None),
            ("nvidia-cublas-cu13", {"version": "13", "record": "new"}),
        ):
            with self.subTest(name=name, replacement=replacement):
                after = before.copy()
                if replacement is None:
                    del after[name]
                else:
                    after[name] = replacement
                result = self.check_snapshots(before, after)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Unexpected dependency changes", result.stderr)
                self.assertIn(name, result.stderr)

    def run_isolated_cli(self, command, installed=None, wheel_names=()):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            site = root / "site"
            site.mkdir()
            for name, version in (installed or {}).items():
                info = site / f"{name.replace('-', '_')}-{version}.dist-info"
                info.mkdir()
                (info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
                (info / "RECORD").write_text("preinstalled")
            wheel_dir = root / "wheel house"
            wheel_dir.mkdir()
            for name in wheel_names:
                (wheel_dir / name).touch()
            args = [str(wheel_dir)] if command == "missing-wheels" else []
            result = subprocess.run(
                [sys.executable, "-S", str(SCRIPT), command, *args],
                env={**os.environ, "PYTHONPATH": str(site)},
                capture_output=True,
                text=True,
            )
            return result, wheel_dir

    def test_requirements_cli_emits_only_pinned_evaluators(self):
        result, _ = self.run_isolated_cli(
            "requirements", {"sailify": "vendor-version", "sgl-eval": "99.0.0"}
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "sgl-eval==0.1.2\n"
            "math-verify==0.9.0\n"
            "latex2sympy2-extended==1.11.0\n"
            "antlr4-python3-runtime==4.9.3\n"
            "editdistance==0.8.1\n",
        )
        self.assertEqual(result.stderr, "")

    def test_missing_wheels_cli_selects_exact_normalized_pins(self):
        result, wheel_dir = self.run_isolated_cli(
            "missing-wheels",
            wheel_names=[
                *WHEELS.values(),
                "sgl_eval-0.1.20-py3-none-any.whl",
                "sgl_eval_extra-0.1.2-py3-none-any.whl",
                "sgl_eval-0.1.2.tar.gz",
                "torch-2.13.0-py3-none-any.whl",
            ],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "".join(f"{wheel_dir / name}\n" for name in WHEELS.values()),
        )
        self.assertEqual(result.stderr, "")

    def test_missing_wheels_cli_skips_preinstalled_distributions(self):
        for installed in (
            {"Sailify": "vendor-version", "SGL.Eval": "0.1.2"},
            {**ADDITIONS, "sailify": "vendor-version"},
        ):
            with self.subTest(installed=installed):
                missing = list(WHEELS.values())[2:] if "SGL.Eval" in installed else []
                result, wheel_dir = self.run_isolated_cli(
                    "missing-wheels", installed, missing
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    result.stdout, "".join(f"{wheel_dir / name}\n" for name in missing)
                )
                self.assertEqual(result.stderr, "")

    def test_cli_rejects_preinstalled_evaluator_pin_conflicts(self):
        for name, version in ADDITIONS.items():
            if name == "sailify":
                continue
            with self.subTest(name=name):
                result, _ = self.run_isolated_cli(
                    "missing-wheels", {name: version + ".1"}, WHEELS.values()
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn(name, result.stderr)
                self.assertIn(version + ".1", result.stderr)
                self.assertIn(version, result.stderr)

    def test_missing_wheels_cli_rejects_invalid_candidates(self):
        for name, wheel in WHEELS.items():
            for case, candidates in (
                ("missing", []),
                ("wrong-version", [wheel.replace(ADDITIONS[name], "99.0.0")]),
                ("duplicate", [wheel, wheel.replace(".whl", "2.whl")]),
            ):
                with self.subTest(name=name, case=case):
                    wheels = [value for value in WHEELS.values() if value != wheel]
                    result, _ = self.run_isolated_cli(
                        "missing-wheels", wheel_names=wheels + candidates
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, "")
                    self.assertIn(f"{name}=={ADDITIONS[name]}", result.stderr)

    def test_snapshot_uses_first_distribution_on_python_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sites = [root / name for name in ("build", "runtime")]
            for site in sites:
                info = site / "ppu_ci_snapshot_probe-1.0.dist-info"
                info.mkdir(parents=True)
                (info / "METADATA").write_text(
                    "Name: ppu-ci-snapshot-probe\nVersion: 1.0\n"
                )
                (info / "RECORD").write_text(site.name)
            output = root / "snapshot.json"
            for order in (sites, sites[::-1]):
                with self.subTest(first=order[0].name):
                    result = subprocess.run(
                        [sys.executable, str(SCRIPT), "snapshot", str(output)],
                        env={
                            **os.environ,
                            "PYTHONPATH": os.pathsep.join(map(str, order)),
                        },
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    snapshot = json.loads(output.read_text())
                    self.assertEqual(
                        snapshot["ppu-ci-snapshot-probe"],
                        {
                            "version": "1.0",
                            "record": hashlib.sha256(
                                order[0].name.encode()
                            ).hexdigest(),
                        },
                    )


if __name__ == "__main__":
    unittest.main()

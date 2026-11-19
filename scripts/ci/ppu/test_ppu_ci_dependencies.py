import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from email.parser import Parser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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


class TestPPUCIDevSailDependencies(CustomTestCase):
    def setUp(self):
        super().setUp()
        spec = importlib.util.spec_from_file_location("ppu_dependencies", SCRIPT)
        self.checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.checker)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.wheels = Path(temporary.name).resolve() / "wheel house"
        self.runtime = self.wheels / "runtime-deps"
        self.runtime.mkdir(parents=True)
        self.root_requirements = [
            'sglang[all_sail]; extra == "dev_sail"',
            'sglang[test]; extra == "dev_sail"',
            'sglang[srt_sail]; extra == "all_sail"',
            'sglang[runtime_common]; extra == "srt_sail"',
            'sglang[runtime_base]; extra == "runtime_common"',
            'torch-for-sail; extra == "srt_sail"',
            'triton-for-sail; extra == "srt_sail"',
            'deep_gemm-for-sail>=1.1.0; extra == "srt_sail"',
            'blake3>=1.0.0; extra == "runtime_base"',
            'transformers==5.19.0; extra == "runtime_base"',
            'peft>=0.18.0,<0.19.0; extra == "test"',
            'not-for-this-extra; extra == "srt_cpu"',
            'not-for-this-platform; sys_platform == "unused"',
        ]
        self.write_root()
        self.installed = {
            name: self.message(name, version) for name, version in ADDITIONS.items()
        }
        self.installed.update(
            {
                "torch": self.message("torch", "2.13.0+ppu2.2.0"),
                "triton": self.message("triton", "3.7.0+ppu2.2.0"),
                "deep-gemm": self.message("deep-gemm", "1.1.0+ppu2.2.0"),
                "transformers": self.message(
                    "transformers", "5.12.1", ["obsolete-runtime-helper>=1"]
                ),
                "peft": self.message("peft", "0.12.0"),
            }
        )
        for name, version in ADDITIONS.items():
            self.write_wheel(name, version)
        self.write_wheel("transformers", "5.19.0")
        self.write_wheel("peft", "0.18.1", ["transformers>=5"])
        self.metadata_patch = patch.object(
            self.checker.metadata,
            "distributions",
            side_effect=lambda: [
                SimpleNamespace(metadata=m) for m in self.installed.values()
            ],
        )
        self.metadata_patch.start()
        self.addCleanup(self.metadata_patch.stop)

    def message(self, name, version, requirements=(), extras=()):
        return Parser().parsestr(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + "".join(f"Provides-Extra: {extra}\n" for extra in extras)
            + "".join(f"Requires-Dist: {req}\n" for req in requirements)
        )

    def write_wheel(self, name, version, requirements=(), extras=(), directory=None):
        directory = self.runtime if directory is None else directory
        stem = f"{name.replace('-', '_')}-{version}"
        path = directory / f"{stem}-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                f"{stem}.dist-info/METADATA",
                self.message(name, version, requirements, extras).as_string(),
            )
            archive.writestr(
                f"{stem}.dist-info/WHEEL",
                "Wheel-Version: 1.0\nGenerator: fixture\n"
                "Root-Is-Purelib: true\nTag: py3-none-any\n",
            )
            archive.writestr(f"{stem}.dist-info/RECORD", "")
        return path

    def write_root(self):
        return self.write_wheel(
            "sglang", "0.5.21", self.root_requirements, ["dev_sail"], self.wheels
        )

    def packages(self):
        return {
            name: {"version": message["Version"], "record": "vendor"}
            for name, message in self.installed.items()
        }

    def test_build_collects_blake3_and_transitive_extras_without_platform_wheels(self):
        from packaging.requirements import Requirement

        for name, version in (("transformers", "5.19.0"), ("peft", "0.18.1")):
            (self.runtime / f"{name}-{version}-py3-none-any.whl").unlink()
        available = {
            "blake3": ("1.0.8", ["typing-extensions>=4.6"]),
            "typing-extensions": ("4.16.0", []),
            "transformers": ("5.19.0", ["runtime-helper[fast]>=1", "torch"]),
            "peft": ("0.18.1", ["torch", "transformers>=5"]),
            "runtime-helper": (
                "1.2.0",
                ['fast-helper; extra == "fast"', 'not-for-this-extra; extra == "slow"'],
            ),
            "fast-helper": ("2.0.0", []),
        }

        def build_wheels(command, check):
            destination = Path(command[command.index("--wheel-dir") + 1])
            for value in command[command.index("--wheel-dir") + 2 :]:
                requirement = Requirement(value)
                name = self.checker.normalize_name(requirement.name)
                version, dependencies = available[name]
                self.write_wheel(name, version, dependencies, directory=destination)
            return subprocess.CompletedProcess(command, 0)

        with patch.object(self.checker.subprocess, "run", side_effect=build_wheels):
            self.checker.collect_dev_sail(self.wheels)
        inventory = self.checker.wheel_inventory(self.runtime)
        self.assertEqual(set(inventory), set(ADDITIONS) | set(available))
        selected = self.checker.missing_dev_sail(self.wheels, self.packages())
        self.assertEqual(set(selected), set(available))
        self.assertFalse(any(self.checker.is_platform_package(n) for n in inventory))

    def local_pip(self, source):
        run = subprocess.run

        def build(command, check):
            index = command.index("--index-url")
            command = [
                *command[:index],
                "--no-index",
                "--find-links",
                str(source),
                "--disable-pip-version-check",
                *command[index + 2 :],
            ]
            return run(command, check=check, capture_output=True, text=True)

        return patch.object(self.checker.subprocess, "run", side_effect=build)

    def test_build_reselects_numpy_after_discovering_transitive_constraints(self):
        source = self.wheels / "local index"
        source.mkdir()
        for name, version, requirements, extras in (
            ("blake3", "1.0.8", [], []),
            ("numpy", "2.5.5", [], []),
            ("numpy", "2.3.5", ["new-helper[fast]>=1"], []),
            ("numpy", "1.26.4", [], []),
            (
                "array-user",
                "1.0",
                ["numpy>=2", 'numpy>=99; sys_platform == "unused"'],
                [],
            ),
            (
                "mistral-common",
                "1.13.1",
                ['numpy>=1.25,<2.4; python_version < "99"'],
                [],
            ),
            ("new-helper", "1.0", ['fast-helper; extra == "fast"'], ["fast"]),
            ("fast-helper", "1.0", [], []),
        ):
            self.write_wheel(name, version, requirements, extras, source)
        self.root_requirements.extend(["numpy", "array-user", "mistral-common==1.13.1"])
        self.write_root()
        with self.local_pip(source):
            self.checker.collect_dev_sail(self.wheels)
        inventory = self.checker.wheel_inventory(self.runtime)
        self.assertEqual(inventory["numpy"][1]["Version"], "2.3.5")
        self.assertEqual(len(list(self.runtime.glob("numpy-*.whl"))), 1)
        self.assertIn("new-helper", inventory)
        self.assertIn("fast-helper", inventory)
        self.assertFalse(any(self.checker.is_platform_package(n) for n in inventory))
        self.write_wheel("transformers", "5.19.0", ["numpy>=2,<2.4"])
        self.installed["numpy"] = self.message("numpy", "2.3.1")
        before = self.packages()
        selected = self.checker.missing_dev_sail(self.wheels, before)
        self.assertNotIn("numpy", selected)
        after = {
            **before,
            **{n: {"version": v, "record": "new"} for n, (_, v) in selected.items()},
        }
        self.assertEqual(after["numpy"], before["numpy"])
        self.assertEqual(
            self.checker.check_snapshots(
                before, after, {n: v for n, (_, v) in selected.items()}
            ),
            0,
        )
        self.installed["numpy"] = self.message("numpy", "2.5.5")
        with self.assertRaisesRegex(
            ValueError, "Preinstalled numpy.*update the runtime image"
        ):
            self.checker.missing_dev_sail(self.wheels, self.packages())

    def test_build_conflicting_constraints_preserve_existing_inventory(self):
        source = self.wheels / "local index"
        source.mkdir()
        old = self.write_wheel("numpy", "2.5.5")
        original = old.read_bytes()
        for version in ("2.3.5", "2.5.5"):
            self.write_wheel("numpy", version, directory=source)
        self.write_wheel("blake3", "1.0.8")
        self.write_wheel("mistral-common", "1.13.1", ["numpy>=1.25,<2.4"])
        self.root_requirements.extend(["numpy>=2.5", "mistral-common==1.13.1"])
        self.write_root()
        with self.local_pip(source), self.assertRaises(subprocess.CalledProcessError):
            self.checker.collect_dev_sail(self.wheels)
        self.assertEqual(old.read_bytes(), original)
        self.assertEqual([p.name for p in self.runtime.glob("numpy-*.whl")], [old.name])
        self.assertFalse(any(p.is_dir() for p in self.runtime.iterdir()))

    def test_build_rejects_invalid_replacement_without_deleting_old_wheel(self):
        self.write_wheel("blake3", "1.0.8")
        old = self.write_wheel("numpy", "2.5.5")
        original = old.read_bytes()
        self.root_requirements.append("numpy>=2,<2.4")
        self.write_root()
        for version in (None, "1.26.4"):

            def build(command, check):
                if version is not None:
                    destination = Path(command[command.index("--wheel-dir") + 1])
                    self.write_wheel("numpy", version, directory=destination)
                return subprocess.CompletedProcess(command, 0)

            with (
                self.subTest(version=version),
                patch.object(self.checker.subprocess, "run", side_effect=build),
            ):
                with self.assertRaises(ValueError):
                    self.checker.collect_dev_sail(self.wheels)
            self.assertEqual(old.read_bytes(), original)
            self.assertEqual(len(list(self.runtime.glob("numpy-*.whl"))), 1)

    def test_build_version_cycles_fail_without_repeated_downloads(self):
        self.write_wheel("blake3", "1.0.8")
        self.write_wheel("version-loop", "2.0", ["version-loop<2"])
        self.root_requirements.append("version-loop")
        self.write_root()

        def build(command, check):
            destination = Path(command[command.index("--wheel-dir") + 1])
            older = "version-loop<2" in command
            self.write_wheel(
                "version-loop",
                "1.0" if older else "2.0",
                ["version-loop>=2" if older else "version-loop<2"],
                directory=destination,
            )
            return subprocess.CompletedProcess(command, 0)

        with patch.object(self.checker.subprocess, "run", side_effect=build):
            with self.assertRaisesRegex(ValueError, "did not converge"):
                self.checker.collect_dev_sail(self.wheels)
        self.assertEqual(len(list(self.runtime.glob("version_loop-*.whl"))), 1)

    def test_runtime_upgrades_use_new_metadata_and_preserve_other_packages(self):
        """Replacing old runtime packages must traverse the new dependency graph."""
        self.write_wheel("blake3", "1.0.8", ["typing-extensions>=4.6"])
        self.write_wheel("typing-extensions", "4.16.0")
        self.write_wheel("runtime-helper", "1.2.0")
        self.write_wheel("transformers", "5.19.0", ["runtime-helper>=1"])
        self.installed["sailify"] = self.message("sailify", "vendor-version")
        before = self.packages()
        selected = self.checker.missing_dev_sail(self.wheels, before)
        additions = {n: v for n, (_, v) in selected.items()}
        after = {
            **before,
            **{n: {"version": v, "record": "new"} for n, v in additions.items()},
        }
        self.assertEqual(
            set(selected),
            {"blake3", "typing-extensions", "runtime-helper", "transformers", "peft"},
        )
        self.assertEqual(additions["transformers"], "5.19.0")
        self.assertEqual(additions["peft"], "0.18.1")
        self.assertEqual(after["sailify"], before["sailify"])
        self.assertEqual(self.checker.check_snapshots(before, after, additions), 0)
        for name in ("torch", "triton", "transformers", "peft", "blake3"):
            with self.subTest(name=name):
                changed = {**after, name: {"version": "999", "record": "new"}}
                self.assertNotEqual(
                    self.checker.check_snapshots(before, changed, additions), 0
                )

    def test_matching_runtime_versions_are_not_reinstalled(self):
        self.write_wheel("blake3", "1.0.8")
        self.installed["transformers"] = self.message("transformers", "5.19.0")
        self.installed["peft"] = self.message("peft", "0.18.1", ["transformers>=5"])
        selected = self.checker.missing_dev_sail(self.wheels, self.packages())
        self.assertEqual(set(selected), {"blake3"})
        before = self.packages()
        for name in ("transformers", "peft"):
            with self.subTest(name=name):
                after = {**before, name: {**before[name], "record": "reinstalled"}}
                self.assertNotEqual(self.checker.check_snapshots(before, after, {}), 0)
                self.assertNotEqual(
                    self.checker.check_snapshots(
                        before, after, {name: before[name]["version"]}
                    ),
                    0,
                )

    def test_upgrade_candidates_and_transitive_constraints_are_enforced(self):
        self.write_wheel("blake3", "1.0.8")
        self.installed["obsolete-runtime-helper"] = self.message(
            "obsolete-runtime-helper", "1.0"
        )
        for name, good_version, bad_version in (
            ("transformers", "5.19.0", "5.12.1"),
            ("peft", "0.18.1", "0.19.0"),
        ):
            good_path = self.runtime / f"{name}-{good_version}-py3-none-any.whl"
            good_path.unlink()
            with self.subTest(name=name, case="missing"):
                with self.assertRaisesRegex(ValueError, f"missing {name}"):
                    self.checker.missing_dev_sail(self.wheels, self.packages())
            bad_path = self.write_wheel(name, bad_version)
            with self.subTest(name=name, case="incompatible"):
                with self.assertRaisesRegex(ValueError, "does not satisfy"):
                    self.checker.missing_dev_sail(self.wheels, self.packages())
            bad_path.unlink()
            self.write_wheel(name, good_version)
        for requirement, name, version in (
            ("runtime-helper>=2", "runtime-helper", "1.0"),
            ("torch>=99", "torch", "2.13.0+ppu2.2.0"),
        ):
            with self.subTest(requirement=requirement):
                self.write_wheel("transformers", "5.19.0", [requirement])
                self.installed[name] = self.message(name, version)
                with self.assertRaisesRegex(
                    ValueError, "does not satisfy.*runtime image"
                ):
                    self.checker.missing_dev_sail(self.wheels, self.packages())
        self.write_wheel("transformers", "5.19.0")
        self.write_wheel("peft", "0.18.1", ["transformers<5.19"])
        with self.assertRaisesRegex(ValueError, "does not satisfy transformers"):
            self.checker.missing_dev_sail(self.wheels, self.packages())

    def test_snapshot_rejects_unplanned_or_incomplete_upgrades(self):
        before = self.packages()
        for name, version in (("transformers", "5.19.0"), ("peft", "0.18.1")):
            after = {**before, name: {"version": version, "record": "upgraded"}}
            with self.subTest(name=name):
                self.assertEqual(
                    self.checker.check_snapshots(before, after, {name: version}), 0
                )
                self.assertNotEqual(self.checker.check_snapshots(before, after, {}), 0)
                self.assertNotEqual(
                    self.checker.check_snapshots(before, before, {name: version}), 0
                )
                removed = {n: v for n, v in before.items() if n != name}
                self.assertNotEqual(
                    self.checker.check_snapshots(before, removed, {name: version}), 0
                )
                self.assertNotEqual(
                    self.checker.check_snapshots(before, after, {name: "999"}), 0
                )
        after = {**before, "sailify": {"version": "99", "record": "replaced"}}
        self.assertNotEqual(
            self.checker.check_snapshots(before, after, {"sailify": "99"}), 0
        )

    def test_runtime_fails_closed_on_missing_or_incompatible_offline_dependencies(self):
        self.write_wheel("runtime-helper", "1.2.0")
        for version, expected in (
            (None, "missing blake3"),
            ("0.9.0", "does not satisfy"),
        ):
            with self.subTest(version=version):
                if version is not None:
                    self.write_wheel("blake3", version)
                with self.assertRaisesRegex(ValueError, expected):
                    self.checker.missing_dev_sail(self.wheels, self.packages())
        self.write_wheel("blake3", "1.0.8")
        with self.assertRaisesRegex(ValueError, "Multiple wheels for blake3"):
            self.checker.missing_dev_sail(self.wheels, self.packages())

    def test_platform_packages_cannot_be_added_even_through_dependency_metadata(self):
        self.write_wheel("blake3", "1.0.8", ["nvidia-cublas-cu13"])
        self.write_wheel("runtime-helper", "1.2.0")
        with self.assertRaisesRegex(ValueError, "must preinstall nvidia-cublas-cu13"):
            self.checker.missing_dev_sail(self.wheels, self.packages())
        for name in ("torch", "nvidia-cublas-cu13", "torch-for-sail", "sglang"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "platform or built packages"),
            ):
                self.checker.check_snapshots({}, {}, {name: "1.0"})
        self.write_wheel("blake3", "1.0.8")
        self.installed.pop("torch")
        with self.assertRaisesRegex(ValueError, "must preinstall torch"):
            self.checker.missing_dev_sail(self.wheels, self.packages())

    def test_offline_install_upgrades_real_environment_and_preserves_other_packages(
        self,
    ):
        self.write_wheel("blake3", "1.0.8")
        self.write_wheel("runtime-helper", "1.2.0")
        self.write_wheel("transformers", "5.19.0", ["runtime-helper>=1"])
        initial_dir = self.wheels / "initial wheels"
        initial_dir.mkdir()
        initial = [
            self.write_wheel(name, message["Version"], directory=initial_dir)
            for name, message in self.installed.items()
        ]
        for wheel in [*initial, *self.runtime.glob("*.whl")]:
            with zipfile.ZipFile(wheel, "a") as archive:
                archive.writestr(
                    wheel.name.split("-", 1)[0] + ".py", "VALUE = 'offline-fixture'\n"
                )
        target = self.wheels / "isolated environment"
        result = subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(target)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        python = target / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )

        def install(paths):
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "--python",
                    str(python),
                    "install",
                    "--disable-pip-version-check",
                    "--upgrade",
                    "--no-deps",
                    "--no-index",
                    *map(str, paths),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        def snapshot(name):
            output = self.wheels / name
            result = subprocess.run(
                [str(python), "-I", str(SCRIPT), "snapshot", str(output)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return json.loads(output.read_text())

        install(initial)
        before = snapshot("before.json")
        self.assertEqual(before["transformers"]["version"], "5.12.1")
        self.assertEqual(before["peft"]["version"], "0.12.0")
        selected = self.checker.missing_dev_sail(self.wheels, before)
        install(path for path, _ in selected.values())
        after = snapshot("after.json")
        additions = {n: v for n, (_, v) in selected.items()}
        self.assertEqual(
            additions,
            {
                "transformers": "5.19.0",
                "peft": "0.18.1",
                "runtime-helper": "1.2.0",
                "blake3": "1.0.8",
            },
        )
        self.assertEqual(self.checker.check_snapshots(before, after, additions), 0)
        for name in before:
            if name not in ("transformers", "peft"):
                self.assertEqual(after[name], before[name])
        self.assertEqual(after["transformers"]["version"], "5.19.0")
        self.assertEqual(after["peft"]["version"], "0.18.1")
        site = target / (
            "Lib/site-packages"
            if sys.platform == "win32"
            else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        )
        for name, version in (("transformers", "5.19.0"), ("peft", "0.18.1")):
            infos = list(site.glob(f"{name}-*.dist-info"))
            self.assertEqual([p.name for p in infos], [f"{name}-{version}.dist-info"])

    def test_runtime_shell_installs_only_selected_wheels_and_propagates_plan_failure(
        self,
    ):
        bash = shutil.which("bash")
        if bash is None and sys.platform == "win32":
            bash = str(Path(os.environ["ProgramFiles"]) / "Git/bin/bash.exe")
        if bash is None:
            self.skipTest("bash is required for the runtime shell integration test")
        source = SCRIPT.with_name("build_and_test.sh").read_text()
        marker = "python3 -m pip uninstall -y sgl-kernel\n"
        block = marker + source.split(marker, 1)[1].split("python3 - <<'PY'", 1)[0]
        script = (
            "set -euo pipefail\n"
            "python3() {\n"
            '  printf \'%s\\t\' "$@" >> "$FIXTURE_LOG"\n'
            "  printf '\\n' >> \"$FIXTURE_LOG\"\n"
            '  if [[ "$1" == "$checker" && "$2" == missing-dev-sail ]]; then\n'
            "    printf '%s' \"$FIXTURE_MISSING\"\n"
            '    return "$FIXTURE_PLAN_EXIT"\n'
            "  fi\n"
            "}\n" + block
        )
        selected_path = str(self.runtime / "blake3-1.0.8-py3-none-any.whl")
        log = self.wheels / "commands.log"
        for missing, code in (("", 0), (selected_path + "\n", 0), ("", 19)):
            with self.subTest(missing=missing, code=code):
                log.write_text("")
                result = subprocess.run(
                    [bash, "-c", script],
                    env={
                        **os.environ,
                        "FIXTURE_LOG": str(log),
                        "FIXTURE_MISSING": missing,
                        "FIXTURE_PLAN_EXIT": str(code),
                        "checker": str(SCRIPT),
                        "wheel_dir": str(self.wheels),
                        "artifacts": str(self.wheels),
                        "additions": str(self.wheels / "additions.json"),
                    },
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                calls = [
                    line.rstrip("\t").split("\t")
                    for line in log.read_text().splitlines()
                ]
                dependency_installs = [
                    args
                    for args in calls
                    if args[:3] == ["-m", "pip", "install"]
                    and "--force-reinstall" not in args
                ]
                self.assertEqual(
                    len(dependency_installs), int(bool(missing) and code == 0)
                )
                if dependency_installs:
                    self.assertEqual(dependency_installs[0][-1], selected_path)
                    self.assertIn("--no-deps", dependency_installs[0])
                    self.assertIn("--no-index", dependency_installs[0])
                    self.assertIn("--upgrade", dependency_installs[0])
                checks = [
                    args for args in calls if len(args) > 1 and args[1] == "check"
                ]
                self.assertEqual(len(checks), int(code == 0))
                if checks:
                    self.assertEqual(
                        checks[0][-2:],
                        ["--additions", str(self.wheels / "additions.json")],
                    )

    def test_runtime_summary_accepts_only_nonempty_fully_passing_suites(self):
        bash = shutil.which("bash")
        if bash is None and sys.platform == "win32":
            bash = str(Path(os.environ["ProgramFiles"]) / "Git/bin/bash.exe")
        if bash is None:
            self.skipTest("bash is required for the runtime summary test")
        source = SCRIPT.with_name("build_and_test.sh").read_text()
        command = source.split("# Require a non-empty suite", 1)[1].splitlines()[1]
        log = self.wheels / "sanity.log"
        for summary, accepted in (
            ("Test Summary: 5/5 passed", True),
            ("Test Summary: 13/13 passed", True),
            ("Test Summary: 12/13 passed", False),
            ("Test Summary: 0/0 passed", False),
            ("Test Summary: 13/130 passed", False),
            ("", False),
        ):
            with self.subTest(summary=summary):
                log.write_text(summary + "\n")
                result = subprocess.run(
                    [bash, "-c", command],
                    env={**os.environ, "artifacts": str(self.wheels)},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0 if accepted else 1)

    def test_wheel_metadata_and_extra_selection_fail_closed(self):
        root = self.write_root()
        self.write_wheel(
            "sglang", "0.5.21", self.root_requirements, directory=self.wheels
        )
        with self.assertRaisesRegex(ValueError, "does not provide dev_sail"):
            self.checker.missing_dev_sail(self.wheels, self.packages())
        with zipfile.ZipFile(root, "w") as archive:
            archive.writestr(
                "sglang-0.5.21.dist-info/METADATA",
                self.message("another-package", "0.5.21").as_string(),
            )
        with self.assertRaisesRegex(ValueError, "metadata disagree"):
            self.checker.missing_dev_sail(self.wheels, self.packages())

    def test_missing_dev_sail_cli_records_verified_additions_and_upgrades(self):
        self.write_wheel("blake3", "1.0.8")
        self.write_wheel("runtime-helper", "1.2.0")
        output = self.wheels / "additions.json"
        with (
            patch.object(self.checker, "snapshot", return_value=self.packages()),
            patch.object(
                sys,
                "argv",
                [str(SCRIPT), "missing-dev-sail", str(self.wheels), str(output)],
            ),
        ):
            self.assertEqual(self.checker.main(), 0)
        self.assertEqual(
            json.loads(output.read_text()),
            {"blake3": "1.0.8", "transformers": "5.19.0", "peft": "0.18.1"},
        )
        before, after = self.packages(), self.packages()
        for name, version in json.loads(output.read_text()).items():
            after[name] = {"version": version, "record": "added"}
        after["unrelated-package"] = {"version": "1.0", "record": "added"}
        self.assertNotEqual(
            self.checker.check_snapshots(before, after, json.loads(output.read_text())),
            0,
        )


if __name__ == "__main__":
    unittest.main()

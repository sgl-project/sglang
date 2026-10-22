"""Cheap PR execution contracts: no torch, model, or accelerator required."""

import ast
import copy
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

ROOT = Path(__file__).resolve().parents[4]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load("ppu_runner", "python/sglang/test/ci/pr_runner.py")
REPORT = load("ppu_report", "scripts/ci/ppu/pr_report.py")
ACTION_REVISION = "06c8ee7cadaae2086db4977e571b19f461653f16"
MODEL = load("ppu_model", "python/sglang/test/ppu_pr_utils.py")


class StrictRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_fixture(self, source, **options):
        file = self.root / "test_fixture.py"
        file.write_text(source)
        directory = self.root / "results"
        result = RUNNER.run_pytest_files(
            [file], directory, options.pop("timeout", 15), **options
        )
        report = json.loads((directory / "summary.json").read_text())
        return result, report

    def test_pass_and_per_file_evidence(self):
        result, report = self.run_fixture("def test_ok():\n    assert 2 + 2 == 4\n")
        self.assertEqual(result, 0)
        self.assertTrue(report["success"])
        record = report["files"][0]
        self.assertEqual(record["executed"], 1)
        self.assertTrue((self.root / "results" / record["junit"]).is_file())
        self.assertGreater(record["elapsed"], 0)

    def test_failure_retains_counts(self):
        result, report = self.run_fixture("def test_bad():\n    assert False\n")
        self.assertEqual(result, 1)
        self.assertFalse(report["success"])
        self.assertEqual(report["files"][0]["executed"], 1)
        self.assertEqual(report["files"][0]["failures"], 1)

    def test_empty_selection(self):
        self.assertEqual(RUNNER.run_pytest_files([], self.root, 1), 1)
        self.assertFalse(
            json.loads((self.root / "summary.json").read_text())["success"]
        )

    def test_duplicate_selection(self):
        self.assertEqual(RUNNER.run_pytest_files([__file__, __file__], self.root, 1), 1)

    def test_zero_collected(self):
        result, report = self.run_fixture("value = 1\n")
        self.assertEqual(result, 1)
        self.assertEqual(report["files"][0]["executed"], 0)

    def test_all_skipped_is_failure_even_if_allowed(self):
        result, report = self.run_fixture(
            "import pytest\n@pytest.mark.skip(reason='platform')\ndef test_skip(): pass\n",
            allowed_skips=("test_fixture.test_skip",),
        )
        self.assertEqual(result, 1)
        self.assertEqual(report["files"][0]["skipped"], 1)

    def test_explicit_partial_skip_allowlist(self):
        source = "import pytest\ndef test_ok(): pass\n@pytest.mark.skip(reason='platform')\ndef test_skip(): pass\n"
        result, report = self.run_fixture(
            source, allowed_skips=("test_fixture.test_skip",)
        )
        self.assertEqual(result, 0)
        self.assertEqual(report["files"][0]["executed"], 1)
        self.assertEqual(report["files"][0]["skipped"], 1)

    def test_unexpected_partial_skip(self):
        result, _ = self.run_fixture(
            "import pytest\ndef test_ok(): pass\n@pytest.mark.skip(reason='platform')\ndef test_skip(): pass\n"
        )
        self.assertEqual(result, 1)

    def test_collection_error(self):
        result, report = self.run_fixture("import missing_ppu_fixture_dependency\n")
        self.assertEqual(result, 1)
        self.assertGreater(report["files"][0]["failures"], 0)

    def test_continue_after_collection_error_stays_red(self):
        broken = self.root / "test_broken.py"
        healthy = self.root / "test_healthy.py"
        broken.write_text("import missing_ppu_fixture_dependency\n")
        healthy.write_text("def test_ok(): pass\n")
        directory = self.root / "results"
        result = RUNNER.run_pytest_files(
            [broken, healthy], directory, 15, continue_on_error=True
        )
        report = json.loads((directory / "summary.json").read_text())
        self.assertEqual(result, 1)
        self.assertFalse(report["success"])
        self.assertEqual(len(report["files"]), 2)
        self.assertFalse(report["files"][0]["passed"])
        self.assertTrue(report["files"][1]["passed"])

    def test_timeout(self):
        result, report = self.run_fixture(
            "import time\ndef test_wait(): time.sleep(60)\n", timeout=1
        )
        self.assertEqual(result, 1)
        self.assertIn("timed out", report["files"][0]["error"])
        self.assertLess(report["elapsed"], 5)

    def test_suite_deadline(self):
        result, report = self.run_fixture("def test_ok(): pass\n", suite_timeout=0)
        self.assertEqual(result, 1)
        self.assertIn("budget exhausted", report["files"][0]["error"])

    def test_missing_report_is_failure(self):
        result, report = self.run_fixture("import os\nos._exit(0)\n")
        self.assertEqual(result, 1)
        self.assertIn("error", report["files"][0])

    def test_stale_report_rejected(self):
        self.run_fixture("def test_ok(): pass\n")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_fixture("def test_ok(): pass\n")
        self.assertFalse(
            json.loads((self.root / "results/summary.json").read_text())["success"]
        )

    def test_xml_declarations_rejected(self):
        path = self.root / "report.xml"
        for text in ('<!DOCTYPE x [<!ENTITY y "hi">]><testsuite/>', "<testsuite/>\x00"):
            path.write_text(text)
            with self.assertRaises(ValueError):
                RUNNER.inspect_junit(path)

    def test_secrets_are_sanitized(self):
        with patch.dict(
            os.environ, {"PPU_ARTIFACTORY_PASSWORD": "private-fixture-value"}
        ):
            cleaned = RUNNER.sanitize(
                "https://" + "user:pass" + "@host password=private-fixture-value"
            )
        self.assertNotIn("private-fixture-value", cleaned)
        self.assertNotIn("user:pass", cleaned)

    def test_cancellation_kills_descendants_and_stays_red(self):
        # The child inherits pytest's session, just like the serving fixtures.
        heartbeat = self.root / "heartbeat"
        child = (
            "import pathlib,time\np=pathlib.Path(%r)\nwhile True:\n p.write_text(str(time.time())); time.sleep(.05)\n"
            % str(heartbeat)
        )
        fixture = self.root / "test_cancel.py"
        fixture.write_text(
            "import subprocess,sys,time\ndef test_child():\n subprocess.Popen([sys.executable, '-c', %r])\n time.sleep(60)\n"
            % child
        )
        driver = "import importlib.util,sys\ns=importlib.util.spec_from_file_location('r',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nsys.exit(m.run_pytest_files([sys.argv[2]],sys.argv[3],60))\n"
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                driver,
                str(ROOT / "python/sglang/test/ci/pr_runner.py"),
                str(fixture),
                str(self.root / "cancel"),
            ],
            stdout=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 10
            while not heartbeat.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(heartbeat.exists())
            process.send_signal(signal.SIGTERM)
            self.assertEqual(process.wait(timeout=5), 1)
            before = heartbeat.read_text()
            time.sleep(0.2)
            self.assertEqual(heartbeat.read_text(), before)
            report = json.loads((self.root / "cancel/summary.json").read_text())
            self.assertFalse(report["success"])
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()

    def test_mandatory_evidence_is_revalidated(self):
        suite = REPORT.SUITES["cpu"][0]
        file = self.root / "test_ok.py"
        file.write_text("def test_ok(): pass\n")
        self.assertEqual(RUNNER.run_pytest_files([file], self.root / suite, 15), 0)
        metadata = {
            "worker.json": {"exit_code": 0, "elapsed": 1},
            "setup.json": {"lane": "cpu", "setup_seconds": 1},
            "environment.json": {
                key: "fixture"
                for key in ("commit", "python", "torch", "checkout_import", "backend")
            },
        }
        for name, value in metadata.items():
            (self.root / name).write_text(json.dumps(value))
        self.assertEqual(REPORT.verify(self.root, "cpu")["executed"], 1)
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "export"
            REPORT.export_evidence(self.root, destination)
            self.assertEqual(REPORT.verify(destination, "cpu")["executed"], 1)
            (destination / "worker.json").write_text('{"exit_code":1,"elapsed":1}')
            with self.assertRaises(ValueError):
                REPORT.verify(destination, "cpu")
            (destination / "worker.json").unlink()
            with self.assertRaises(FileNotFoundError):
                REPORT.verify(destination, "cpu")
        for name in metadata:
            path = self.root / name
            original = path.read_text()
            path.write_text("{}")
            with self.assertRaises(ValueError):
                REPORT.verify(self.root, "cpu")
            path.write_text(original)
        xml = next((self.root / suite).glob("*.xml"))
        xml.write_text(
            '<testsuite><testcase name="bad"><failure/></testcase></testsuite>'
        )
        with self.assertRaises(ValueError):
            REPORT.verify(self.root, "cpu")


class EvidenceExportTest(unittest.TestCase):
    def test_read_only_source_redaction_and_writable_summary(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(
                os.environ,
                {
                    "PPU_ARTIFACTORY_PASSWORD": "export-fixture-value",
                    "GITHUB_STEP_SUMMARY": "",
                },
            ),
        ):
            root = Path(temp)
            source = root / "nas"
            source.mkdir()
            nested = source / "suite"
            nested.mkdir()
            files = {
                source / "aot.json": '{"aot_seconds":600}',
                source / "setup.json": '{"setup_seconds":626}',
                source / "setup.log": "https://" + "user:pass" + "@host export-fixture-value",
                source
                / "environment-pods.timing.tsv": "2026-01-01T00:00:00Z\t2026-01-01T00:00:03Z\t2026-01-01T00:00:11Z\n",
                source / "environment-pods.status": "export-fixture-value",
                nested / "case.xml": '<testsuite><testcase name="ok"/></testsuite>',
                nested / "case.log": "password=export-fixture-value",
                source / "ignored.bin": "export-fixture-value",
            }
            for path, text in files.items():
                path.write_text(text)
                path.chmod(0o444)
            nested.chmod(0o555)
            source.chmod(0o555)
            before = {p: (p.read_bytes(), p.stat()) for p in files}
            try:
                destination = root / "export"
                REPORT.export_evidence(source, destination)
                self.assertFalse((destination / "ignored.bin").exists())
                for path in destination.rglob("*"):
                    if path.is_file():
                        text = path.read_text()
                        self.assertNotIn("export-fixture-value", text)
                        self.assertNotIn("user:pass", text)
                        self.assertTrue(path.stat().st_mode & 0o200)
                        self.assertEqual(path.stat().st_uid, os.getuid())
                result = REPORT.summarize(destination, "smoke")
                self.assertEqual(result["aot_seconds"], 600)
                self.assertEqual(result["scheduling_seconds"], 3)
                self.assertTrue((destination / "timing-summary.json").is_file())
                self.assertFalse((source / "timing-summary.json").exists())
                for path, (content, metadata) in before.items():
                    self.assertEqual(path.read_bytes(), content)
                    self.assertEqual(path.stat().st_mode, metadata.st_mode)
                    self.assertEqual(path.stat().st_mtime_ns, metadata.st_mtime_ns)
            finally:
                source.chmod(0o755)
                nested.chmod(0o755)

    def test_links_and_special_files_never_publish_partial_export(self):
        for kind in ("file-link", "directory-link", "broken-link", "hard-link", "fifo"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "nas"
                source.mkdir()
                (source / "a.log").write_text("safe diagnostic")
                outside = root / "outside.log"
                outside.write_text("private diagnostic")
                entry = source / "z.log"
                if kind == "file-link":
                    entry.symlink_to(outside)
                elif kind == "directory-link":
                    entry.symlink_to(root, target_is_directory=True)
                elif kind == "broken-link":
                    entry.symlink_to(root / "missing")
                elif kind == "hard-link":
                    os.link(outside, entry)
                else:
                    os.mkfifo(entry)
                destination = root / "export"
                with self.assertRaises((ValueError, OSError)):
                    REPORT.export_evidence(source, destination)
                self.assertFalse(destination.exists())
                self.assertFalse(list(root.glob(".ppu-export-*")))
                self.assertEqual(outside.read_text(), "private diagnostic")

    def test_missing_empty_nested_or_existing_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "nas"
            destination = root / "export"
            with self.assertRaises(FileNotFoundError):
                REPORT.export_evidence(source, destination)
            source.mkdir()
            with self.assertRaisesRegex(ValueError, "No supported"):
                REPORT.export_evidence(source, destination)
            (source / "setup.log").write_text("safe diagnostic")
            for target in (source, source / "nested", root):
                with self.assertRaises(ValueError):
                    REPORT.export_evidence(source, target)
            link = root / "source-link"
            link.symlink_to(source, target_is_directory=True)
            with self.assertRaises(OSError):
                REPORT.export_evidence(link, destination)
            self.assertFalse(destination.exists())

    def test_sanitizer_or_read_failure_cannot_publish(self):
        for failure in ("sanitize", "read"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "nas"
                source.mkdir()
                (source / "a.log").write_text("safe diagnostic")
                (source / "z.log").write_text("bad diagnostic")
                destination = root / "export"
                sanitize = REPORT.runner.sanitize
                open_file = os.open

                def clean(text):
                    if text == "bad diagnostic":
                        raise ValueError("sanitizer fixture failure")
                    return sanitize(text)

                def read_file(path, *args, **kwargs):
                    if path == "z.log":
                        raise PermissionError("read fixture failure")
                    return open_file(path, *args, **kwargs)

                target, side_effect = (
                    ("runner.sanitize", clean)
                    if failure == "sanitize"
                    else ("os.open", read_file)
                )
                with (
                    patch.object(
                        REPORT.runner if failure == "sanitize" else REPORT.os,
                        target.split(".")[-1],
                        side_effect=side_effect,
                    ),
                    self.assertRaises((ValueError, PermissionError)),
                ):
                    REPORT.export_evidence(source, destination)
                self.assertFalse(destination.exists())
                self.assertFalse(list(root.glob(".ppu-export-*")))


class FinalStatusTest(unittest.TestCase):
    def needs(self, lanes):
        needs = {
            "check-changes": {
                "result": "success",
                "outputs": {
                    key: str(key in lanes).lower()
                    for key in ("cpu", "ppu_single", "ppu_kernel", "ppu_multi")
                },
            },
            "pr-gate": {"result": "success" if lanes else "skipped"},
        }
        for job, lane in REPORT.JOBS.items():
            needs[job] = {
                "result": "success" if lane in lanes else "skipped",
                "outputs": {"evidence_ok": "true"},
            }
        return needs

    def test_only_explicitly_unselected_jobs_may_skip(self):
        for lanes in (
            set(),
            {"cpu"},
            {"cpu", "ppu_single"},
            {"cpu", "ppu_single", "ppu_kernel", "ppu_multi"},
        ):
            needs = self.needs(lanes)
            self.assertTrue(REPORT.aggregate(needs))
            for job, record in needs.items():
                for status in ("failure", "cancelled", "skipped"):
                    if record["result"] == status:
                        continue
                    changed = copy.deepcopy(needs)
                    changed[job]["result"] = status
                    self.assertFalse(REPORT.aggregate(changed), (lanes, job, status))

    def test_cpu_only_requires_evidence_and_skips_hardware_and_gate(self):
        for lanes in (set(), {"cpu"}, {"cpu", "ppu_single", "ppu_kernel", "ppu_multi"}):
            needs = self.needs(lanes)
            needs["pr-gate"]["result"] = "skipped"
            for job, lane in REPORT.JOBS.items():
                if lane != "cpu":
                    needs[job]["result"] = "skipped"
            self.assertTrue(REPORT.aggregate(needs, cpu_only=True))
            if lanes:
                self.assertFalse(REPORT.aggregate(needs))
            for job in needs:
                for status in ("success", "failure", "cancelled", "skipped"):
                    if needs[job]["result"] == status:
                        continue
                    changed = copy.deepcopy(needs)
                    changed[job]["result"] = status
                    self.assertFalse(
                        REPORT.aggregate(changed, cpu_only=True), (lanes, job, status)
                    )
            if lanes:
                for evidence in ("false", "", None):
                    changed = copy.deepcopy(needs)
                    changed["stage-a-test-cpu-ppu"]["outputs"]["evidence_ok"] = evidence
                    self.assertFalse(REPORT.aggregate(changed, cpu_only=True))
            for lane in needs["check-changes"]["outputs"]:
                changed = copy.deepcopy(needs)
                changed["check-changes"]["outputs"].pop(lane)
                self.assertFalse(REPORT.aggregate(changed, cpu_only=True))
        self.assertFalse(REPORT.aggregate(self.needs({"ppu_multi"}), cpu_only=True))

    def test_cpu_only_finish_uses_strict_aggregate_cli(self):
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/pr-test-ppu.yml").read_text()
        )
        finish = workflow["jobs"]["pr-test-ppu-finish"]["steps"][-1]
        self.assertEqual(finish["env"]["CPU_ONLY"], "${{ inputs.cpu_only }}")
        self.assertEqual(finish["env"]["NEEDS_JSON"], "${{ toJson(needs) }}")
        for lanes in (set(), {"cpu"}):
            needs = self.needs(lanes)
            needs["pr-gate"]["result"] = "skipped"
            for cpu_only in ("true", "false", ""):
                with self.subTest(lanes=lanes, cpu_only=cpu_only):
                    result = subprocess.run(
                        ["bash", "-e", "-c", finish["run"]],
                        cwd=ROOT,
                        env={
                            **os.environ,
                            "NEEDS_JSON": json.dumps(needs),
                            "CPU_ONLY": cpu_only,
                        },
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertEqual(
                        result.returncode,
                        0 if not lanes or cpu_only == "true" else 1,
                        result.stderr,
                    )

    def test_missing_evidence_or_selection_is_red(self):
        needs = self.needs({"cpu", "ppu_single"})
        for job in (
            "stage-a-test-cpu-ppu",
            "stage-a-test-1-gpu-ppu",
            "stage-b-test-1-gpu-ppu",
        ):
            changed = copy.deepcopy(needs)
            changed[job]["outputs"] = {}
            self.assertFalse(REPORT.aggregate(changed))
        needs["check-changes"]["outputs"].pop("ppu_multi")
        self.assertFalse(REPORT.aggregate(needs))
        self.assertFalse(REPORT.aggregate(self.needs({"ppu_multi"})))


class CpuShortlistContractTest(unittest.TestCase):
    def test_registration_inventory_and_runtime_ownership(self):
        registry = load("shortlist_registry", "python/sglang/test/ci/ci_register.py")
        files = [
            str(path)
            for path in (ROOT / "test/registered").rglob("*.py")
            if path.name not in ("conftest.py", "__init__.py")
        ]
        registrations = registry.collect_tests(files, sanity_check=True)
        cpu = [r for r in registrations if r.effective_suite == "stage-a-test-cpu-ppu"]
        self.assertEqual(
            {
                str(Path(r.filename).relative_to(ROOT / "test/registered/ppu"))
                for r in cpu
            },
            {
                "test_ppu_constraints.py",
                "test_ppu_environment_fingerprint_unit.py",
                "accuracy_eval/test_ppu_accuracy_eval_unit.py",
                "pd_perf_eval/test_ppu_pd_perf_eval_unit.py",
                "pr/test_cpu_eval.py",
            },
        )
        self.assertEqual(len(cpu), 5)
        for registration in cpu:
            self.assertEqual(registration.backend, registry.HWBackend.CPU)
            self.assertIsNone(registration.disabled)
            self.assertFalse(registration.nightly)
        body = [
            r for r in registrations if r.effective_suite == "stage-b-test-1-gpu-ppu"
        ]
        self.assertEqual(
            {Path(r.filename).name for r in body},
            {
                "test_ppu_fa3_eval.py",
                "test_serving.py",
                "test_cpu_pool_hooks.py",
                "test_eval_runtime.py",
            },
        )
        self.assertEqual(len(body), 4)
        self.assertTrue(all(r.disabled is None and not r.nightly for r in body))
        self.assertFalse(
            (ROOT / "test/registered/ppu/pr/test_cpu_dispatch.py").exists()
        )
        for name in (
            "test_cpu_pool_hooks.py",
            "test_eval_runtime.py",
        ):
            owners = [r for r in registrations if Path(r.filename).name == name]
            self.assertEqual(len(owners), 1)
            self.assertEqual(owners[0].backend, registry.HWBackend.PPU)
            self.assertEqual(owners[0].effective_suite, "stage-b-test-1-gpu-ppu")
            self.assertIsNone(owners[0].disabled)

    def test_evaluator_adapters_partition_all_original_classes(self):
        def exports(adapter):
            tree = ast.parse((ROOT / "test/registered/ppu/pr" / adapter).read_text())
            return {
                alias.name
                for node in tree.body
                if isinstance(node, ast.ImportFrom)
                and (node.module or "").startswith("registered.ppu.")
                for alias in node.names
            }

        cpu = exports("test_cpu_eval.py")
        runtime = exports("test_eval_runtime.py")
        self.assertEqual(cpu, {"TestPPUAnswerEval", "TestPPUPerfEval"})
        self.assertEqual(
            runtime, {"TestPPUAnswerMultiNodeExchange", "TestPPUPerfMultiNodeExchange"}
        )
        self.assertFalse(cpu & runtime)
        original = set()
        for name in ("answer", "perf"):
            path = (
                ROOT / f"test/registered/ppu/{name}_eval/test_ppu_{name}_eval_unit.py"
            )
            original.update(
                node.name
                for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name.startswith("Test")
            )
        self.assertEqual(cpu | runtime, original)


class LegacyAdapterImportContractTest(unittest.TestCase):
    def setUp(self):
        from _pytest.pathlib import ImportMode, import_path

        self.enterContext(patch.dict(sys.modules))
        self.enterContext(patch.object(sys, "path", sys.path.copy()))
        for name in tuple(sys.modules):
            if name == "ppu" or name.startswith(("ppu.", "_ppu_pr_legacy_")):
                sys.modules.pop(name)
        conftest = import_path(
            ROOT / "test/registered/ppu/conftest.py",
            mode=ImportMode.prepend,
            root=ROOT,
            consider_namespace_packages=False,
        )
        self.assertEqual(conftest.__name__, "ppu.conftest")
        self.assertEqual(
            Path(sys.modules["ppu"].__file__),
            ROOT / "test/registered/ppu/__init__.py",
        )
        self.helper = sys.modules["ppu"]
        self.assertIsNone(importlib.util.find_spec("ppu.unit"))

    def test_real_dsa_tests_load_and_execute_despite_shadowed_package(self):
        relative = "unit/test_dsa_prefill_graph_int8.py"
        module = self.helper.load_legacy_ppu_module(relative)
        self.assertEqual(Path(module.__file__), ROOT / "test/ppu" / relative)
        self.assertIs(sys.modules[module.__name__], module)
        self.assertIs(self.helper.load_legacy_ppu_module(relative), module)
        suite = unittest.TestLoader().loadTestsFromTestCase(
            module.TestDsaPrefillGraphInt8
        )
        result = unittest.TestResult()
        suite.run(result)
        self.assertTrue(result.wasSuccessful(), (result.errors, result.failures))
        self.assertEqual(result.testsRun, 3)
        self.assertEqual(result.skipped, [])

    def test_adapter_paths_and_exports_keep_original_objects(self):
        from importlib.machinery import SourceFileLoader

        inventory = {
            "test_cpu_dispatch.py.disabled": {
                "unit/test_deepseek_v4_dspark_quant_metadata.py": (
                    "TestDequantSupported",
                    "TestDsparkV4QuantMetadata",
                    "TestFusedCommitKvProjSupport",
                ),
                "unit/test_dsa_prefill_graph_int8.py": ("TestDsaPrefillGraphInt8",),
            },
            "test_cpu_pool_hooks.py": {
                "module/test_ppu_deepep_hooks.py": ("TestPPUDeepEPHooks",),
                "test_kda_replayssm_pool.py": ("test_replayssm_pool_and_budget",),
            },
            "test_kernels.py": {
                "unit/test_topk_prefill_bf16_alignment.py": ("test_topk_alignment",),
            },
        }
        original_exec = SourceFileLoader.exec_module
        for adapter, sources in inventory.items():
            with self.subTest(adapter=adapter):
                expected = {
                    ROOT / "test/ppu" / path: names for path, names in sources.items()
                }
                seen, exports = set(), {}

                def record_source(loader, module):
                    source = Path(module.__file__)
                    if source not in expected:
                        return original_exec(loader, module)
                    self.assertIs(sys.modules[module.__name__], module)
                    seen.add(source)
                    definitions = {
                        node.name
                        for node in ast.parse(source.read_text()).body
                        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                    }
                    for name in expected[source]:
                        self.assertIn(name, definitions)
                        exports[name] = object()
                        setattr(module, name, exports[name])

                path = ROOT / "test/registered/ppu/pr" / adapter
                bootstrap = []
                for node in ast.parse(path.read_text()).body:
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                        call = ast.unparse(node.value.func)
                        if call == "register_ppu_ci":
                            break
                        if call == "sys.path.insert":
                            bootstrap.append(node)
                            continue
                    if bootstrap:
                        bootstrap.append(node)
                self.assertTrue(bootstrap)
                namespace = {"sys": sys, "Path": Path, "__file__": str(path)}
                # Only intercept the SDK-dependent source execution boundary;
                # exercise the real adapter bootstrap and path-based loader.
                with patch.object(SourceFileLoader, "exec_module", record_source):
                    exec(
                        compile(
                            ast.Module(body=bootstrap, type_ignores=[]),
                            str(path),
                            "exec",
                        ),
                        namespace,
                    )
                self.assertEqual(seen, set(expected))
                for name, value in exports.items():
                    self.assertIs(namespace[name], value)
                self.assertEqual(
                    {name for name in namespace if name.startswith(("Test", "test_"))},
                    set(exports),
                )

    def test_failed_source_import_is_not_cached_or_silenced(self):
        from importlib.machinery import SourceFileLoader

        relative = "unit/test_dsa_prefill_graph_int8.py"
        before = {name for name in sys.modules if name.startswith("_ppu_pr_legacy_")}
        with patch.object(
            SourceFileLoader,
            "exec_module",
            side_effect=ModuleNotFoundError("missing SDK"),
        ):
            with self.assertRaisesRegex(ModuleNotFoundError, "missing SDK"):
                self.helper.load_legacy_ppu_module(relative)
        self.assertEqual(
            {name for name in sys.modules if name.startswith("_ppu_pr_legacy_")}, before
        )
        self.assertTrue(
            hasattr(
                self.helper.load_legacy_ppu_module(relative), "TestDsaPrefillGraphInt8"
            )
        )
        with self.assertRaises(FileNotFoundError):
            self.helper.load_legacy_ppu_module("unit/missing_ci_fixture.py")
        with self.assertRaises(ValueError):
            self.helper.load_legacy_ppu_module("../registered/ppu/__init__.py")


class SetupContractTest(unittest.TestCase):
    def hardware_jobs(self):
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/pr-test-ppu.yml").read_text()
        )
        self.assertEqual(workflow["defaults"]["run"]["shell"], "bash")
        return {
            name: workflow["jobs"][name]
            for name in REPORT.JOBS
            if name != "stage-a-test-cpu-ppu"
        }

    def test_hardware_jobs_never_load_checkout_local_actions(self):
        self.assertFalse((ROOT / ".github/actions/ppu-pr-suite/action.yml").exists())
        self.assertFalse((ROOT / "scripts/ci/ppu/prepare_pr_action.py").exists())
        for name, job in self.hardware_jobs().items():
            with self.subTest(job=name):
                for step in job["steps"]:
                    self.assertFalse(step.get("uses", "").startswith("./"))
                    self.assertNotIn(".ppu-pr-action", json.dumps(step))
                    self.assertNotIn("prepare_pr_action.py", json.dumps(step))
                    if "run" in step:
                        subprocess.run(
                            ["bash", "-n"], input=step["run"], text=True, check=True
                        )
                prepare = next(
                    s
                    for s in job["steps"]
                    if s.get("name") == "Prepare worker evidence"
                )
                worker = next(s for s in job["steps"] if s.get("id") == "worker")
                self.assertLess(job["steps"].index(prepare), job["steps"].index(worker))
                env = prepare["env"]
                self.assertEqual(env["PPU_ENV_HASHED_NAME"], "true")
                self.assertEqual(env["PPU_ENV_JOB"], worker["with"]["job_suffix"])
                self.assertEqual(env["PPU_RUNNER_NAME"], "${{ runner.name }}")
                self.assertEqual(env["MATRIX_JSON"], "${{ toJSON(matrix) }}")
                self.assertEqual(env["PPU_ENV_NAMESPACE"], worker["with"]["namespace"])
                self.assertEqual(env["PPU_ENV_PR_TIMINGS"], "true")
                self.assertEqual(
                    int(env["PPU_ENV_OBSERVER_LIMIT"]),
                    worker["with"]["timeout_minutes"] * 60,
                )
                self.assertIn("observe_pod_environment.sh start", prepare["run"])
                for digest, image in (
                    ("", "image:tag"),
                    ("sha256:bad", "image@sha256:bad"),
                    ("sha256:" + "a" * 64, "image:tag"),
                ):
                    # Invalid image identity must fail before any NAS writes.
                    result = subprocess.run(
                        [
                            "bash",
                            "--noprofile",
                            "--norc",
                            "-eo",
                            "pipefail",
                            "-c",
                            prepare["run"],
                        ],
                        env={
                            **os.environ,
                            "PPU_BASE_IMAGE_DIGEST": digest,
                            "PPU_BASE_IMAGE_REF": image,
                        },
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertNotEqual(result.returncode, 0)

    def test_observer_matches_pinned_remote_action_naming(self):
        checkout = os.environ.get("PPU_ACTION_CONTRACT_SOURCE")
        if checkout:
            directory = Path(checkout)
            self.assertEqual(
                subprocess.check_output(
                    ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
                ).strip(),
                ACTION_REVISION,
            )
        names = set()
        for job in self.hardware_jobs():
            for runner_name, matrix in (
                ("runner-a", "null"),
                ("runner-b", ""),
                ("runner-b", '{"shard": 1}'),
            ):
                with (
                    self.subTest(job=job, runner=runner_name, matrix=matrix),
                    tempfile.TemporaryDirectory() as temp,
                ):
                    directory = Path(temp)
                    normalized = "" if matrix == "null" else matrix
                    digest = hashlib.md5(
                        f"{runner_name}\n{job}\n{job}\n{normalized}".encode(),
                        usedforsecurity=False,
                    ).hexdigest()[:6]
                    prefix = f"ppu-flytiger-eco-39610521197-2-{job}"[:45].rstrip("-")
                    expected = f"{prefix}-{digest}"
                    self.assertNotIn(expected, names)
                    names.add(expected)
                    env = {
                        **os.environ,
                        "GITHUB_REPOSITORY_OWNER": "flytiger-eco",
                        "GITHUB_RUN_ID": "39610521197",
                        "GITHUB_RUN_ATTEMPT": "2",
                        "GITHUB_JOB_NAME": job,
                        "INPUT_JOB_SUFFIX": job,
                        "PPU_RUNNER_NAME": runner_name,
                        "MATRIX_JSON": matrix,
                        "PPU_ENV_HASHED_NAME": "true",
                        "PPU_ENV_JOB": job,
                        "PPU_ENV_JOB_NAME": "",
                        "PPU_ENV_NAMESPACE": "ppu-sched",
                        "PPU_ENV_PR_TIMINGS": "true",
                        "PPU_ENV_EXPECTED_PODS": "1",
                        "PPU_ENV_OBSERVER_LIMIT": "2",
                        "PPU_ENV_OBSERVER_INTERVAL": "0.1",
                        "PATH": str(directory) + os.pathsep + os.environ["PATH"],
                    }
                    if checkout:
                        actual = subprocess.check_output(
                            ["bash", str(Path(checkout) / "scripts/job-name.sh")],
                            env=env,
                            text=True,
                        ).strip()
                        self.assertEqual(actual, expected)
                    kubectl = directory / "kubectl"
                    kubectl.write_text(
                        "#!/bin/sh\n"
                        f'case "$*" in *"-n ppu-sched -l ppu-job={expected}"*) ;; *) exit 9;; esac\n'
                        'case "$*" in *creationTimestamp*) printf "2026-01-01T00:00:00Z\\t2026-01-01T00:00:03Z\\t2026-01-01T00:00:11Z\\n";;\n'
                        f'*) printf "{expected}-worker-0\\tuid0\\tnode0\\timage@sha256:aaa\\tcontainerd://sha256:aaa\\n";; esac\n'
                    )
                    kubectl.chmod(0o755)
                    output = directory / "pods.tsv"
                    subprocess.run(
                        [
                            "bash",
                            str(ROOT / "scripts/ci/ppu/observe_pod_environment.sh"),
                            "watch",
                            str(output),
                        ],
                        env=env,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    self.assertEqual(
                        output.with_suffix(".status").read_text().strip(), "ok"
                    )
                    self.assertIn(expected + "-worker-0", output.read_text())
                    self.assertEqual(
                        len(
                            output.with_suffix(".timing.tsv")
                            .read_text()
                            .strip()
                            .split("\t")
                        ),
                        3,
                    )

    def test_worker_failure_and_evidence_failure_cannot_publish_success(self):
        for name, job in self.hardware_jobs().items():
            export = next(s for s in job["steps"] if s.get("id") == "export")
            evidence = next(s for s in job["steps"] if s.get("id") == "evidence")
            for outcome, export_rc, summary_rc, verify_rc in (
                ("success", 0, 0, 0),
                ("failure", 0, 0, 0),
                ("cancelled", 0, 0, 0),
                ("skipped", 0, 0, 0),
                ("success", 0, 0, 1),
                ("success", 0, 1, 0),
                ("success", 1, 0, 0),
            ):
                with (
                    self.subTest(
                        job=name,
                        outcome=outcome,
                        export_rc=export_rc,
                        summary_rc=summary_rc,
                        verify_rc=verify_rc,
                    ),
                    tempfile.TemporaryDirectory() as temp,
                ):
                    directory = Path(temp)
                    stub = directory / "python3"
                    stub.write_text(
                        f"#!{sys.executable}\nimport os,sys\nfrom pathlib import Path\n"
                        "with Path(os.environ['CALL_LOG']).open('a') as f: f.write(' '.join(sys.argv[1:]) + '\\n')\n"
                        "sys.exit(int(os.environ[{'export':'EXPORT_RC','summarize':'SUMMARY_RC','verify':'VERIFY_RC'}[sys.argv[2]]]))\n"
                    )
                    stub.chmod(0o755)
                    output = directory / "output"
                    log = directory / "calls"
                    env = {
                        **os.environ,
                        "PATH": str(directory) + os.pathsep + os.environ["PATH"],
                        "GITHUB_RUN_ID": "123",
                        "GITHUB_RUN_ATTEMPT": "2",
                        "GITHUB_JOB": name,
                        "LANE": evidence["env"]["LANE"],
                        "KERNEL": (
                            "true" if evidence["env"]["LANE"] == "body" else "false"
                        ),
                        "WORKER_OUTCOME": outcome,
                        "RUNNER_TEMP": str(directory),
                        "EXPORT_RC": str(export_rc),
                        "SUMMARY_RC": str(summary_rc),
                        "VERIFY_RC": str(verify_rc),
                        "GITHUB_OUTPUT": str(output),
                        "CALL_LOG": str(log),
                    }

                    def run_step(step):
                        step_output = output.with_name(step["id"] + "-output")
                        result = subprocess.run(
                            [
                                "bash",
                                "--noprofile",
                                "--norc",
                                "-eo",
                                "pipefail",
                                "-c",
                                step["run"],
                            ],
                            env={**env, "GITHUB_OUTPUT": str(step_output)},
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        # Container-hook runners need not retain outputs of failed steps.
                        outputs = (
                            dict(
                                line.split("=", 1)
                                for line in step_output.read_text().splitlines()
                            )
                            if result.returncode == 0 and step_output.exists()
                            else {}
                        )
                        return result, outputs

                    result, outputs = run_step(export)
                    self.assertEqual(result.returncode, export_rc, result.stderr)
                    if not export_rc:
                        env["PPU_ARTIFACT_DIR"] = outputs["artifact_directory"]
                        result, validation_outputs = run_step(evidence)
                        outputs.update(validation_outputs)
                    valid = outcome == "success" and not (
                        export_rc or summary_rc or verify_rc
                    )
                    self.assertEqual(result.returncode == 0, valid, result.stderr)
                    self.assertEqual(outputs.get("evidence_ok") == "true", valid)
                    self.assertEqual(outputs.get("sanitized") == "true", export_rc == 0)
                    calls = log.read_text().splitlines()
                    modes = ["export"]
                    if not export_rc:
                        modes.append("summarize")
                        if not summary_rc:
                            modes.append("verify")
                        artifact = outputs["artifact_directory"]
                        self.assertTrue(
                            artifact.startswith(str(directory) + "/ppu-evidence.")
                        )
                        self.assertTrue(artifact.endswith("/sanitized"))
                        self.assertIn("--destination " + artifact, calls[0])
                        for call in calls[1:]:
                            self.assertIn("--directory " + artifact, call)
                            self.assertNotIn("/wl_nas/", call)
                    else:
                        self.assertNotIn("artifact_directory", outputs)
                    self.assertEqual([line.split()[1] for line in calls], modes)
                    if modes[-1] == "verify":
                        self.assertEqual(
                            "--kernel" in calls[-1], env["KERNEL"] == "true"
                        )

    def test_summary_separates_observed_and_unknown_timings(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""}),
        ):
            root = Path(directory)
            result = REPORT.summarize(root, "smoke")
            self.assertIsNone(result["scheduling_seconds"])
            self.assertEqual(result["aot_cache"], "unknown")
            (root / "setup.json").write_text('{"setup_seconds": 40}')
            (root / "aot.json").write_text('{"aot_seconds": 30}')
            (root / "setup.log").write_text(
                "Wheel cache key: abc123\nPPU sgl-kernel installed from cache (no compile)"
            )
            (root / "environment-pods.timing.tsv").write_text(
                "2026-01-01T00:00:00Z\t2026-01-01T00:00:03Z\t2026-01-01T00:00:11Z\n"
            )
            result = REPORT.summarize(root, "smoke")
            self.assertEqual(result["scheduling_seconds"], 3)
            self.assertEqual(result["pod_startup_seconds"], 8)
            self.assertEqual(result["setup_seconds"], 40)
            self.assertEqual(result["aot_seconds"], 30)
            self.assertEqual(result["aot_cache"], "cache hit")
            self.assertIsNone(result["github_queue_seconds"])

    def test_gsm8k_sampler_has_no_retries_and_retains_errors(self):
        tree = ast.parse(
            (ROOT / "test/registered/ppu/test_ppu_fa3_eval.py").read_text()
        )
        sampler_node = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "_StrictCompletionSampler"
        )
        namespace = {}
        exec(
            compile(
                ast.Module(body=[sampler_node], type_ignores=[]),
                "<sampler contract>",
                "exec",
            ),
            namespace,
        )
        client = Mock()
        client.completions.create.return_value.choices = [Mock(text="42")]
        sampler = namespace["_StrictCompletionSampler"](client, "local-model")
        self.assertEqual(sampler([{"content": "question"}]), "42")
        self.assertEqual(client.completions.create.call_args.kwargs["max_tokens"], 512)
        client.completions.create.side_effect = TimeoutError("request expired")
        with self.assertRaises(TimeoutError):
            sampler([{"content": "question"}])
        self.assertEqual(client.completions.create.call_count, 2)
        self.assertEqual(len(sampler.errors), 1)
        self.assertEqual(len(sampler.completed), 1)

    def test_remote_action_keeps_scoped_destinations_and_strict_evidence(self):
        checkout = os.environ.get("PPU_ACTION_CONTRACT_SOURCE")
        contract = (
            yaml.safe_load((Path(checkout) / "action.yml").read_text())
            if checkout
            else None
        )
        for name, job in self.hardware_jobs().items():
            with self.subTest(job=name):
                worker = next(s for s in job["steps"] if s.get("id") == "worker")
                self.assertEqual(
                    worker["uses"],
                    f"flytiger-eco/ppu-distributed-action@{ACTION_REVISION}",
                )
                self.assertNotIn("continue-on-error", worker)
                inputs = worker["with"]
                if contract:
                    self.assertLessEqual(set(inputs), set(contract["inputs"]))
                self.assertEqual(inputs["image"], "${{ env.PPU_BASE_IMAGE_REF }}")
                self.assertEqual(inputs["node_selector"], "board-type=OAM-810E")
                self.assertEqual(inputs["namespace"], "ppu-sched")
                self.assertEqual(inputs["cleanup_policy"], "always")
                self.assertEqual(inputs["job_suffix"], "${{ github.job }}")
                self.assertEqual(inputs["nnodes"], 0)
                self.assertEqual(inputs["nproc_per_node"], 2 if "2-gpu" in name else 1)
                self.assertEqual(
                    yaml.safe_load(inputs["container_options"]),
                    {"host_ipc": True, "shm_size": "8Gi"},
                )
                mounts = inputs["host_volumes"].split(",")
                self.assertEqual(len(mounts), 2)
                self.assertTrue(all(m.startswith("/wl_nas/pr-ci/") for m in mounts))
                self.assertEqual(
                    [m.split(":")[1] for m in mounts],
                    ["/ppu-evidence", "/ppu-pr-cache"],
                )
                evidence = next(s for s in job["steps"] if s.get("id") == "evidence")
                lane = evidence["env"]["LANE"]
                self.assertEqual(
                    inputs["command"],
                    f"bash /workspace/source/scripts/ci/ppu/run_pr_suite.sh {lane}",
                )
                self.assertTrue(mounts[1].endswith(f"/{lane}:/ppu-pr-cache"))
                export = next(s for s in job["steps"] if s.get("id") == "export")
                self.assertEqual(export["if"], "always()")
                self.assertNotIn("continue-on-error", export)
                self.assertLess(
                    job["steps"].index(export), job["steps"].index(evidence)
                )
                self.assertEqual(
                    evidence["if"], "always() && steps.export.outcome == 'success'"
                )
                self.assertEqual(
                    evidence["env"]["PPU_ARTIFACT_DIR"],
                    "${{ steps.export.outputs.artifact_directory }}",
                )
                self.assertNotIn("continue-on-error", evidence)
                self.assertEqual(
                    evidence["env"]["WORKER_OUTCOME"], "${{ steps.worker.outcome }}"
                )
                self.assertEqual(
                    job["outputs"]["evidence_ok"],
                    "${{ steps.evidence.outputs.evidence_ok }}",
                )
                kernel = (
                    "${{ needs.check-changes.outputs.ppu_kernel }}"
                    if lane == "body"
                    else "false"
                )
                self.assertEqual(evidence["env"]["KERNEL"], kernel)
                self.assertIn(f"PPU_RUN_KERNEL={kernel},", inputs["extra_env"])
                for secret in ("PPU_ARTIFACTORY_USER", "PPU_ARTIFACTORY_PASSWORD"):
                    self.assertIn(
                        f"{secret}=${{{{ secrets.{secret} }}}}", inputs["extra_env"]
                    )
                    self.assertEqual(
                        evidence["env"][secret], f"${{{{ secrets.{secret} }}}}"
                    )
                    self.assertEqual(
                        export["env"][secret], f"${{{{ secrets.{secret} }}}}"
                    )
                upload = next(
                    s
                    for s in job["steps"]
                    if s.get("uses") == "actions/upload-artifact@v4"
                )
                self.assertEqual(
                    upload["if"],
                    "always() && steps.export.outcome == 'success' && steps.export.outputs.sanitized == 'true'",
                )
                self.assertNotIn("continue-on-error", upload)
                self.assertEqual(upload["with"]["if-no-files-found"], "error")
                self.assertEqual(
                    upload["with"]["path"],
                    "${{ steps.export.outputs.artifact_directory }}",
                )
                self.assertEqual(upload["with"]["retention-days"], 14)

    def test_compact_json_server_flag_is_opt_in(self):
        flag = "--constrained-json-disable-any-whitespace"
        for bounded in (True, False):
            for compact in (None, False, True):
                with self.subTest(bounded=bounded, compact=compact):
                    process = Mock(pid=123, poll=Mock(return_value=None))
                    cleanup = Mock()
                    http = Mock(get=Mock(return_value=Mock(status_code=200)))
                    options = {} if compact is None else {"compact_json": compact}
                    with (
                        patch.dict(
                            sys.modules,
                            {
                                "sglang.srt.utils": Mock(kill_process_tree=cleanup),
                                "requests": http,
                            },
                        ),
                        patch.object(MODEL, "free_port", return_value=12345),
                        patch.object(MODEL, "small_model", return_value="/model"),
                        patch.object(
                            MODEL.subprocess, "Popen", return_value=process
                        ) as launch,
                    ):
                        with MODEL.server(bounded=bounded, **options) as base:
                            self.assertEqual(base, "http://127.0.0.1:12345")
                        launch.assert_called_once()
                        args = launch.call_args.args[0]
                        self.assertEqual(args.count(flag), int(compact is True))
                        self.assertEqual("--grammar-backend" in args, bounded)
                        self.assertEqual("--sampling-backend" in args, bounded)
                        self.assertEqual(
                            args[args.index("--attention-backend") + 1], "fa3"
                        )
                        self.assertEqual(
                            launch.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1"
                        )
                        cleanup.assert_called_once_with(process.pid)
                        process.wait.assert_called_once_with(timeout=15)

    def test_serving_fixture_opts_into_compact_json(self):
        from contextlib import nullcontext

        tree = ast.parse((ROOT / "test/registered/ppu/pr/test_serving.py").read_text())
        fixture = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "base"
        )
        fixture.decorator_list = []
        launch = Mock(return_value=nullcontext("http://fixture"))
        namespace = {"server": launch}
        exec(
            compile(
                ast.Module(body=[fixture], type_ignores=[]), "<serving fixture>", "exec"
            ),
            namespace,
        )
        instance = namespace["base"]()
        self.assertEqual(next(instance), "http://fixture")
        with self.assertRaises(StopIteration):
            next(instance)
        launch.assert_called_once_with(compact_json=True)

    def test_local_model_missing_files_fail_without_download(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"PPU_SMALL_MODEL_PATH": directory}),
        ):
            with self.assertRaises(FileNotFoundError):
                MODEL.small_model()
            for name in (
                "config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "model.safetensors",
            ):
                (Path(directory) / name).write_text("{}")
            self.assertEqual(MODEL.small_model(), str(Path(directory).resolve()))
            self.assertTrue(MODEL.model_identity()["metadata_sha256"])
            (Path(directory) / "model.safetensors.index.json").write_text(
                '{"weight_map":{"weight":"missing.safetensors"}}'
            )
            with self.assertRaises(FileNotFoundError):
                MODEL.small_model()

    def test_legacy_runner_keeps_default_execution_contract(self):
        # Execute the unchanged legacy branch with mocked infrastructure boundaries;
        # this cheap contract test does not claim runtime import coverage.
        tree = ast.parse((ROOT / "python/sglang/test/ci/ci_utils.py").read_text())
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "run_unittest_files"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                function,
            ],
            type_ignores=[],
        )
        fake = Mock(returncode=0)
        registry = type("CIRegistry", (), {})
        namespace = dict(
            CIRegistry=registry,
            cuda_coredump=Mock(is_enabled=lambda: False),
            time=time,
            os=os,
            subprocess=Mock(Popen=Mock(return_value=fake)),
            logger=Mock(),
            json=json,
            run_with_timeout=lambda f, args, kwargs, timeout: f(*args, **kwargs),
            _repo_relative_path=lambda p: p,
        )
        exec(
            compile(ast.fix_missing_locations(module), "<legacy contract>", "exec"),
            namespace,
        )
        run = namespace["run_unittest_files"]
        self.assertEqual(run([]), 0)
        self.assertEqual(run([], require_nonempty=True), 1)
        self.assertEqual(
            run(
                [type("File", (), {"name": "fixture.py", "estimated_time": 1})()],
                timeout_per_file=1,
            ),
            0,
        )
        namespace["subprocess"].Popen.assert_called_once_with(
            ["python3", os.path.join(os.getcwd(), "fixture.py"), "-f"],
            stdout=None,
            stderr=None,
        )
        with self.assertRaises(ValueError):
            run(
                [],
                pytest_mode=True,
                enable_retry=True,
                junit_dir="fixture",
                timeout_per_file=1,
            )


if __name__ == "__main__":
    unittest.main()

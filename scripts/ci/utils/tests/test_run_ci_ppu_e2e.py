"""Hardware-free contracts for command-selected PPU E2E execution."""

import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci/ppu"))
import pr_e2e_report as report
import pr_e2e_worker as worker
import select_pr_e2e as selector
from test_run_ci_ppu_commands import MemoryGitHub


class FakeGitHub(MemoryGitHub):
    def __init__(self, pr, events=None, permission="write"):
        super().__init__(pr)
        tokens = sorted({label["name"] for label in pr["labels"]} & selector.TOKENS)
        self.selected_request = self.accept(tokens) if tokens else None
        self.history = (
            events
            if events is not None
            else [
                {
                    "id": index,
                    "event": "labeled",
                    "label": label,
                    "actor": {"login": "maintainer", "type": "User"},
                }
                for index, label in enumerate(pr["labels"])
            ]
        )
        self.permission = permission

    def get(self, path):
        if path.startswith("pulls/"):
            return self.pr
        if path.startswith("collaborators/"):
            return {"permission": self.permission}
        raise AssertionError(path)

    def events(self, number):
        return self.history


def pull_request(labels):
    return {
        "number": 123,
        "head": {"sha": "a" * 40},
        "base": {"sha": "b" * 40},
        "labels": [{"name": label} for label in labels],
        "draft": False,
        "state": "open",
    }


def selection_state(labels):
    api = FakeGitHub(pull_request(labels))
    return {**selector.snapshot(api.pr), "request": api.selected_request}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class E2ESelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = selector.load_catalog()

    def setUp(self):
        self.environment = mock.patch.dict(
            os.environ,
            {
                "GITHUB_SHA": "c" * 40,
                "GITHUB_RUN_ID": "42",
                "GITHUB_RUN_ATTEMPT": "1",
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def plan(self, labels=("run-ci", "test-bf16")):
        return selector.make_plan(
            self.catalog, selection_state(labels), "c" * 40, "42", "1"
        )

    def selected(self, *labels):
        entries, errors = selector.select(self.catalog, labels)
        self.assertEqual(errors, [])
        return {entry["id"] for entry in entries}

    def test_inventory_and_pinned_import_boundary(self):
        self.assertEqual(len(self.catalog["entries"]), 68)
        imported = [e for e in self.catalog["entries"] if e["family"] == "deepseek"]
        self.assertEqual(len(imported), 11)
        self.assertEqual(
            {e["source"]["revision"] for e in imported},
            {"107913533169e97ff4139664110bed4e523c68b2"},
        )
        self.assertEqual(sum(bool(e["blocked"]) for e in imported), 1)
        for entry in self.catalog["entries"]:
            self.assertGreaterEqual(
                entry["worker_minutes"] * 60 - entry["timeout_seconds"], 600
            )
            self.assertGreaterEqual(entry["job_minutes"] - entry["worker_minutes"], 5)
            self.assertEqual(entry["cards_per_node"], 8)

    def test_registration_discovery_matches_top_level_runner_contract(self):
        registration = 'register_ppu_ci(est_time=1, suite="fixture", nightly=True)\n'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tests = root / "test/registered"
            tests.mkdir(parents=True)
            (tests / "test_real.py").write_text(registration)
            (tests / "test_nested.py").write_text("def nested():\n    " + registration)
            (tests / "conftest.py").write_text(registration)
            self.assertEqual(
                selector.registered_files(root),
                {"fixture": ["test/registered/test_real.py"]},
            )

    def test_exact_labels_and_authorization_do_not_select_cases(self):
        for labels in (
            (),
            ("run-ci",),
            ("test-unknown",),
            ("TEST-BF16",),
            ("test-bf16; echo unsafe",),
        ):
            with self.subTest(labels=labels):
                self.assertEqual(self.selected(*labels), set())
        self.assertEqual(
            self.selected("test-bf16"), self.selected("run-ci", "test-bf16")
        )

    def test_union_not_intersection_and_overlap_deduplicated(self):
        qwen = self.selected("test-qwen")
        glm = self.selected("test-glm")
        self.assertTrue(qwen.isdisjoint(glm))
        self.assertEqual(self.selected("test-qwen", "test-glm"), qwen | glm)
        self.assertEqual(self.selected("test-qwen", "test-bf16", "test-qwen"), qwen)

    def test_formats_are_checkpoint_metadata_not_runtime_dtype(self):
        for label, formats in selector.FORMATS.items():
            entries, errors = selector.select(self.catalog, [label])
            if entries:
                self.assertTrue(all(e["format"] in formats for e in entries))
                blocked = {
                    f"{e['id']}: {e['blocked']}" for e in entries if e["blocked"]
                }
                self.assertEqual(set(errors), blocked)
            else:
                # A format whose cases left the catalog must fail closed,
                # never silently widen to an unrequested execution.
                self.assertEqual(errors, ["Selectors matched no execution."])
        entries, errors = selector.select(self.catalog, ["test-bf16"])
        self.assertEqual(errors, [])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["family"], "qwen")

    def test_defaults_and_full_accuracy_dependencies(self):
        entries, errors = selector.select(
            self.catalog, ["test-qwen", "test-accuracy-full"]
        )
        self.assertEqual(errors, [])
        by_id = {e["id"]: e for e in entries}
        self.assertEqual(len(by_id), len(entries))
        full = [e for e in entries if e["mode"] == "full"]
        self.assertTrue(full)
        for entry in full:
            self.assertEqual(len(entry["dependencies"]), 1)
            smoke = by_id[entry["dependencies"][0]]
            self.assertEqual(smoke["mode"], "smoke")
            self.assertEqual(
                selector.read_json(ROOT / entry["config"])["model"],
                selector.read_json(ROOT / smoke["config"])["model"],
            )
        defaults, errors = selector.select(self.catalog, ["test-qwen"])
        self.assertEqual(errors, [])
        self.assertTrue(
            all(
                e["nodes"] == 1 and (e["category"] == "answer" or e["mode"] == "smoke")
                for e in defaults
            )
        )

    def test_explicit_distributed_and_pd_expansion(self):
        entries, errors = selector.select(
            self.catalog, ["test-qwen", "test-perf", "test-multinode", "test-pd"]
        )
        self.assertEqual(errors, [])
        distributed = [e for e in entries if e["nodes"] > 1]
        self.assertEqual({e["category"] for e in distributed}, {"answer", "perf", "pd"})
        self.assertTrue(all(selector.phase(e) == "distributed" for e in distributed))
        pd, errors = selector.select(self.catalog, ["test-glm", "test-pd"])
        self.assertEqual(errors, [])
        self.assertTrue(any(e["category"] == "pd" for e in pd))
        self.assertFalse(any(e["category"] == "perf" for e in pd))

    def test_unsupported_or_blocked_requests_fail_closed(self):
        for labels in (
            ["test-perf"],
            ["test-functional"],
            ["test-deepseek"],
            ["test-bf16", "test-pd"],
            ["test-deepseek", "test-pd"],
            ["test-all"],
        ):
            with self.subTest(labels=labels):
                _, errors = selector.select(self.catalog, labels)
                self.assertTrue(errors)
                with self.assertRaises(ValueError):
                    selector.validate_plan(self.plan(labels))

    def test_plan_and_current_attempt_cpu_identity(self):
        plan = self.plan()
        selector.validate_plan(plan)
        selector.check_cpu(plan, "true", selector.cpu_identity(plan))
        for evidence, identity in (
            ("false", selector.cpu_identity(plan)),
            ("true", "42:0:" + "c" * 40),
            ("true", ""),
        ):
            with self.subTest(evidence=evidence, identity=identity), self.assertRaises(
                ValueError
            ):
                selector.check_cpu(plan, evidence, identity)
        for key in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
            with self.subTest(key=key), mock.patch.dict(
                os.environ, {key: "stale"}
            ), self.assertRaises(ValueError):
                selector.validate_plan(plan)

    def test_resealed_plan_cannot_change_reviewed_entries(self):
        original = self.plan()
        for field, value in (
            ("nodes", 4),
            ("config_digest", "f" * 64),
            ("files", []),
            ("worker_minutes", 999),
        ):
            plan = copy.deepcopy(original)
            plan["entries"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                selector.validate_plan(selector.seal(plan))
        plan = copy.deepcopy(original)
        plan["entries"] = []
        with self.assertRaises(ValueError):
            selector.validate_plan(selector.seal(plan))

    def test_catalog_rejects_duplicate_empty_and_wrong_resources(self):
        for field, value in (
            ("files", []),
            ("nodes", 2),
            ("cards_per_node", 1),
            ("environment", {"UNREVIEWED": "1"}),
        ):
            catalog = copy.deepcopy(self.catalog)
            catalog["entries"][0][field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "catalog.json"
                write_json(path, catalog)
                with self.assertRaises(ValueError):
                    selector.load_catalog(path)
        catalog = copy.deepcopy(self.catalog)
        catalog["entries"].append(catalog["entries"][0])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.json"
            write_json(path, catalog)
            with self.assertRaises(ValueError):
                selector.load_catalog(path)

    def test_run_ci_label_provenance_and_permissions(self):
        pr = pull_request(["run-ci"])
        state = selector.snapshot(pr)
        for permission in ("write", "maintain", "admin"):
            self.assertEqual(
                selector.label_authorization(
                    FakeGitHub(pr, permission=permission), state
                ),
                [],
            )
        for permission in ("read", "triage", "none"):
            self.assertTrue(
                selector.label_authorization(
                    FakeGitHub(pr, permission=permission), state
                )
            )
        api = FakeGitHub(pr)
        api.history[-1]["actor"]["type"] = "Bot"
        self.assertTrue(selector.label_authorization(api, state))
        api.history += [{"id": 10, "event": "unlabeled", "label": {"name": "run-ci"}}]
        self.assertTrue(selector.label_authorization(api, state))
        api.history += [
            {
                "id": 11,
                "event": "labeled",
                "label": {"name": "run-ci"},
                "actor": {"login": "maintainer", "type": "User"},
            }
        ]
        self.assertEqual(selector.label_authorization(api, state), [])
        with mock.patch.object(
            api, "get", side_effect=RuntimeError("API unavailable")
        ), self.assertRaises(RuntimeError):
            selector.label_authorization(api, state)

    def test_live_changes_reject_stale_plans_and_missing_authorization(self):
        plan = self.plan()
        selector.check_live(plan, FakeGitHub(pull_request(["run-ci", "test-bf16"])))
        for key, value in (
            ("draft", True),
            ("state", "closed"),
            ("head", {"sha": "new"}),
            ("base", {"sha": "new"}),
            ("labels", []),
        ):
            pr = pull_request(["run-ci", "test-bf16"])
            pr[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                selector.check_live(plan, FakeGitHub(pr))
        unauthorized = self.plan(["test-bf16"])
        api = FakeGitHub(pull_request(["test-bf16"]))
        selector.check_live(unauthorized, api, require_authorized=False)
        with self.assertRaises(ValueError):
            selector.check_live(unauthorized, api)

    def test_new_command_does_not_interrupt_pinned_workers_but_supersedes_gate(self):
        api = FakeGitHub(pull_request(["run-ci", "test-bf16"]))
        plan = self.plan()
        latest = api.accept(("test-qwen",), 20)
        selector.check_live(plan, api)
        with self.assertRaisesRegex(ValueError, "superseded"):
            selector.check_live(plan, api, require_latest=True)
        current = selector.make_plan(
            self.catalog,
            {**selector.snapshot(api.pr), "request": latest},
            "c" * 40,
            "42",
            "1",
        )
        self.assertNotEqual(plan["plan_hash"], current["plan_hash"])
        with self.assertRaises(ValueError):
            selector.check_cpu(current, "true", selector.cpu_identity(plan))
        api.notes[0]["body"] = "/clear-ppu-e2e"
        with self.assertRaises(ValueError):
            selector.check_live(plan, api)

    def test_frozen_event_rerun_uses_live_authorization_and_fresh_cpu_identity(self):
        api = FakeGitHub(pull_request(["run-ci", "test-bf16"]))
        self.assert_plan_outputs(
            api, pull_request([]), requested="true", authorized="true"
        )

    def test_baseline_and_clear_preserve_existing_bot_run_ci_behavior(self):
        api = FakeGitHub(pull_request(["run-ci"]))
        api.history[0]["actor"]["type"] = "Bot"
        for clear in (False, True):
            if clear:
                api.selected_request = api.accept((), 20)
            with self.subTest(clear=clear):
                plan = self.assert_plan_outputs(
                    api, api.pr, requested="false", authorized="true"
                )
                with mock.patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "2"}):
                    selector.check_live(plan, api, require_latest=True)

    def test_coverage_labels_alone_never_select_e2e(self):
        api = FakeGitHub(pull_request(["run-ci"]))
        api.pr["labels"].append({"name": "test-qwen"})
        self.assert_plan_outputs(api, api.pr, requested="false", authorized="true")

    def assert_plan_outputs(self, api, original, **expected):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            event = directory / "event.json"
            plan_path = directory / "selection.json"
            output = directory / "outputs"
            write_json(event, {"pull_request": original})
            with mock.patch.dict(
                os.environ,
                {
                    "GITHUB_EVENT_NAME": "pull_request",
                    "GITHUB_EVENT_PATH": str(event),
                    "GITHUB_OUTPUT": str(output),
                    "GITHUB_STEP_SUMMARY": "",
                    "GITHUB_RUN_ATTEMPT": "2",
                },
            ), mock.patch.object(
                selector, "GitHub", return_value=api
            ), mock.patch.object(
                sys, "argv", ["selector", "plan", "--plan", str(plan_path)]
            ), self.assertRaises(
                SystemExit
            ) as caught:
                selector.main()
            self.assertFalse(caught.exception.code)
            outputs = dict(
                line.split("=", 1) for line in output.read_text().splitlines()
            )
            for name, value in expected.items():
                self.assertEqual(outputs[name], value)
            plan = selector.read_json(plan_path)
            self.assertEqual(plan["attempt"], "2")
            with mock.patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "2"}):
                selector.check_cpu(plan, "true", selector.cpu_identity(plan))
            return plan

    def test_stale_event_writes_failed_selection_plan(self):
        live = pull_request(["run-ci", "test-bf16"])
        stale = copy.deepcopy(live)
        stale["head"]["sha"] = "old-head"
        stale["base"]["sha"] = "old-base"
        with tempfile.TemporaryDirectory() as temporary:
            event, output = (
                Path(temporary) / "event.json",
                Path(temporary) / "selection.json",
            )
            write_json(event, {"pull_request": stale})
            with mock.patch.dict(
                os.environ,
                {
                    "GITHUB_EVENT_NAME": "pull_request",
                    "GITHUB_EVENT_PATH": str(event),
                    "GITHUB_OUTPUT": "",
                    "GITHUB_STEP_SUMMARY": "",
                },
            ), mock.patch.object(
                selector, "GitHub", return_value=FakeGitHub(live)
            ), mock.patch.object(
                sys, "argv", ["selector", "plan", "--plan", str(output)]
            ):
                with self.assertRaises(SystemExit) as caught:
                    selector.main()
            self.assertTrue(caught.exception.code)
            plan = selector.read_json(output)
            self.assertIn("stale", " ".join(plan["errors"]))
            self.assertEqual(plan["head_sha"], "old-head")
            self.assertEqual(plan["base_sha"], "old-base")


class E2EEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = selector.load_catalog()

    def fixture(self, directory, category="answer", nodes=1):
        entry = copy.deepcopy(
            next(
                e
                for e in self.catalog["entries"]
                if e["category"] == category
                and e["nodes"] == nodes
                and not e["blocked"]
            )
        )
        config = selector.read_json(ROOT / entry["config"])
        entry["config_digest"] = selector.digest(config)
        plan = {
            "entries": [entry],
            "plan_hash": "d" * 64,
            "tested_sha": "c" * 40,
            "run_id": "42",
            "attempt": "1",
        }
        for rank in range(nodes):
            rank_dir = directory / f"rank-{rank}"
            write_json(
                rank_dir / "worker.json",
                {
                    **report.identity(plan, entry),
                    "rank": rank,
                    "exit_code": 0,
                    "elapsed": 20,
                },
            )
            write_json(
                rank_dir / "environment.json",
                {
                    "commit": plan["tested_sha"],
                    "python": "3.12",
                    "torch": "2.11",
                    "checkout_import": "/workspace/source/python/sglang/__init__.py",
                    "digest": "sha256:" + "e" * 64,
                    "image": "registry/worker@sha256:" + "e" * 64,
                    "devices": ["PPU"] * len(config["hardware"]["visible_devices"]),
                    "setup_seconds": 10,
                    "model": {
                        "path": config["model"]["path"],
                        "metadata_digest": "a" * 64,
                        "weights": ["model.safetensors"],
                    },
                },
            )
            (rank_dir / "worker.log").write_text("worker completed")
            records = []
            selected = ["/workspace/source/" + name for name in entry["files"]]
            for index, path in enumerate(selected):
                xml, log = f"file-{index}.xml", f"file-{index}.log"
                pytest = rank_dir / "pytest"
                pytest.mkdir(exist_ok=True)
                (pytest / xml).write_text(
                    '<testsuite><testcase classname="fixture" name="executed"/></testsuite>'
                )
                (pytest / log).write_text("one passing test")
                records.append(
                    {
                        "file": path,
                        "junit": xml,
                        "log": log,
                        "passed": True,
                        "returncode": 0,
                        "elapsed": 2,
                        "collected": 1,
                        "executed": 1,
                        "skipped": 0,
                        "failures": 0,
                    }
                )
            write_json(
                rank_dir / "pytest/summary.json",
                {
                    "success": True,
                    "selected": selected,
                    "allowed_skips": [],
                    "files": records,
                },
            )
        evaluation = {
            "provenance": {
                "test_config_id": entry["id"],
                "test_config_sha256": entry["config_digest"],
            }
        }
        if category == "answer":
            dataset = selector.read_json(
                ROOT
                / "test/registered/ppu/answer_eval"
                / config["evaluation"]["dataset"]
            )
            evaluation.update(
                {
                    "summary": {
                        "total": len(dataset["cases"]),
                        "failed": 1,
                        "gate": {"regressed": False},
                    },
                    "cases": [{"case_id": c["id"]} for c in dataset["cases"]],
                }
            )
        else:
            measurements = (
                [{"id": "accuracy", "status": "measured"}]
                if category == "accuracy"
                else [
                    {"id": m["id"], "status": "measured"}
                    for m in config["workload"]["measurements"]
                ]
            )
            evaluation.update(
                {
                    "test_id": entry["id"],
                    "config_digest": entry["config_digest"],
                    "measurements": measurements,
                    "summary": {
                        "total": len(measurements),
                        "measured": len(measurements),
                        "failed": 0,
                        "verdict": "passed",
                    },
                }
            )
        write_json(directory / "results/result.json", evaluation)
        cases = "".join(
            f'<testcase name="evaluation-{i}"/>'
            for i in range(evaluation["summary"]["total"])
        )
        (directory / "results/junit.xml").write_text(
            "<testsuite>" + cases + "</testsuite>"
        )
        return plan, entry

    def test_all_adapter_report_shapes_and_every_rank(self):
        for category, nodes in (
            ("answer", 1),
            ("answer", 2),
            ("accuracy", 1),
            ("perf", 1),
            ("perf", 4),
            ("pd", 2),
        ):
            with self.subTest(
                category=category, nodes=nodes
            ), tempfile.TemporaryDirectory() as temporary:
                plan, entry = self.fixture(Path(temporary), category, nodes)
                terminal = report.verify(temporary, plan, entry)
                report.verify_terminal(terminal, plan, entry)
                self.assertEqual(len(terminal["ranks"]), nodes)
                if category in ("perf", "pd"):
                    self.assertIn("measurement-only", terminal["evaluation"]["quality"])
                if nodes > 1:
                    self.assertEqual(terminal["ranks"][1]["role"], "serving peer")

    def test_missing_failed_stale_duplicate_and_unpinned_worker_evidence(self):
        mutations = [
            ("rank-0/worker.json", lambda r: r.update(exit_code=1)),
            ("rank-0/worker.json", lambda r: r.update(attempt="0")),
            ("rank-0/environment.json", lambda r: r.update(digest="latest")),
            (
                "rank-0/environment.json",
                lambda r: r.update(checkout_import="/installed/sglang.py"),
            ),
            ("rank-0/pytest/summary.json", lambda r: r.update(selected=[])),
            ("rank-0/pytest/summary.json", lambda r: r["files"].append(r["files"][0])),
            ("rank-0/pytest/summary.json", lambda r: r["files"][0].update(executed=2)),
            (
                "rank-0/pytest/summary.json",
                lambda r: r.update(allowed_skips=["fixture.executed"]),
            ),
            (
                "results/result.json",
                lambda r: r["summary"]["gate"].update(regressed=True),
            ),
            (
                "results/result.json",
                lambda r: r["provenance"].update(test_config_sha256="stale"),
            ),
            ("results/result.json", lambda r: r["cases"].pop()),
        ]
        for path, mutate in mutations:
            with self.subTest(
                path=path, mutate=mutate
            ), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                plan, entry = self.fixture(directory)
                payload = selector.read_json(directory / path)
                mutate(payload)
                write_json(directory / path, payload)
                with self.assertRaises(ValueError):
                    report.verify(directory, plan, entry)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan, entry = self.fixture(directory, nodes=2)
            (directory / "rank-1").rename(directory / "missing-rank")
            with self.assertRaises(ValueError):
                report.verify(directory, plan, entry)

    def test_skipped_zero_failed_and_malformed_junit_are_rejected(self):
        for xml in (
            "<testsuite/>",
            "<testsuite><testcase><skipped/></testcase></testsuite>",
            "<testsuite><testcase><failure/></testcase></testsuite>",
            "invalid XML",
        ):
            with self.subTest(xml=xml), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                _, entry = self.fixture(directory)
                (directory / "rank-0/pytest/file-0.xml").write_text(xml)
                with self.assertRaises((ValueError, report.runner.ET.ParseError)):
                    report.verify_pytest(directory / "rank-0/pytest", entry)

    def test_aggregate_rejects_missing_extra_duplicate_and_stale_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan, entry = self.fixture(directory)
            terminal = report.verify(directory, plan, entry)
            results = directory / "terminals"
            with self.assertRaises(ValueError):
                report.aggregate(results, plan)
            write_json(results / "first/entry.json", terminal)
            self.assertEqual(len(report.aggregate(results, plan)), 1)
            write_json(results / "second/entry.json", terminal)
            with self.assertRaises(ValueError):
                report.aggregate(results, plan)
            terminal["attempt"] = "0"
            with self.assertRaises(ValueError):
                report.verify_terminal(terminal, plan, entry)

    def test_measurement_status_and_evaluator_junit_are_mandatory(self):
        for category in ("accuracy", "perf", "pd"):
            for status in ("failed", "unmeasured", None):
                with self.subTest(
                    category=category, status=status
                ), tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    _, entry = self.fixture(
                        directory, category, 2 if category == "pd" else 1
                    )
                    payload = selector.read_json(directory / "results/result.json")
                    payload["measurements"][0]["status"] = status
                    write_json(directory / "results/result.json", payload)
                    with self.assertRaises(ValueError):
                        report.verify_evaluation(directory, entry)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, entry = self.fixture(directory, "perf")
            (directory / "results/junit.xml").write_text("<testsuite/>")
            with self.assertRaises(ValueError):
                report.verify_evaluation(directory, entry)

    def test_checkpoint_mismatch_between_ranks_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan, entry = self.fixture(directory, nodes=2)
            path = directory / "rank-1/environment.json"
            environment = selector.read_json(path)
            environment["model"]["metadata_digest"] = "b" * 64
            write_json(path, environment)
            with self.assertRaisesRegex(ValueError, "different checkpoints"):
                report.verify(directory, plan, entry)

    def test_unknown_timings_are_not_zero(self):
        values = report.orchestration_metrics(
            {
                "duration_seconds": "3.5",
                "queue_duration_seconds": "nan",
                "execution_ppu_card_hours": "-1",
            }
        )
        self.assertEqual(values["duration_seconds"], 3.5)
        self.assertIsNone(values["queue_duration_seconds"])
        self.assertIsNone(values["execution_ppu_card_hours"])
        self.assertIsNone(values["pod_startup_seconds"])

    def test_full_accuracy_requires_successful_same_plan_smoke(self):
        full = next(e for e in self.catalog["entries"] if e["mode"] == "full")
        smoke = next(
            e for e in self.catalog["entries"] if e["id"] == full["dependencies"][0]
        )
        plan = {
            "entries": [dict(e, config_digest="a" * 64) for e in (full, smoke)],
            "run_id": "42",
            "attempt": "1",
            "tested_sha": "c" * 40,
            "plan_hash": "d" * 64,
        }
        smoke = plan["entries"][1]
        terminal = {
            **report.identity(plan, smoke),
            "success": True,
            "ranks": [{"rank": 0}],
            "evaluation": {"count": 1},
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan_path = directory / "plan.json"
            write_json(plan_path, plan)
            for state in ("missing", "failed", "stale", "passed", "duplicate"):
                if state != "missing":
                    value = copy.deepcopy(terminal)
                    value["success"] = state != "failed"
                    if state == "stale":
                        value["attempt"] = "0"
                    write_json(directory / "results/one/entry.json", value)
                    if state == "duplicate":
                        write_json(directory / "results/two/entry.json", value)
                with self.subTest(state=state), mock.patch.object(
                    report, "validate_plan"
                ), mock.patch.object(
                    sys,
                    "argv",
                    [
                        "report",
                        "dependencies",
                        "--plan",
                        str(plan_path),
                        "--entry",
                        full["id"],
                        "--directory",
                        str(directory / "results"),
                    ],
                ):
                    if state == "passed":
                        report.main()
                    else:
                        with self.assertRaises(ValueError):
                            report.main()

    def test_baseline_aggregation_includes_e2e_on_docs_only_requests(self):
        import pr_report

        needs = {
            "check-changes": {
                "result": "success",
                "outputs": {
                    "cpu": "true",
                    "ppu_single": "false",
                    "ppu_kernel": "false",
                    "ppu_multi": "false",
                },
            },
            "pr-gate": {"result": "success"},
            "stage-a-test-cpu-ppu": {
                "result": "success",
                "outputs": {"evidence_ok": "true"},
            },
            "stage-a-test-1-gpu-ppu": {"result": "skipped"},
            "stage-b-test-1-gpu-ppu": {"result": "skipped"},
            "stage-c-test-2-gpu-ppu": {"result": "skipped"},
        }
        self.assertFalse(pr_report.aggregate(needs, e2e_requested=True))
        needs["e2e"] = {"result": "success", "outputs": {"evidence_ok": "true"}}
        self.assertTrue(pr_report.aggregate(needs, e2e_requested=True))
        needs["e2e"]["result"] = "skipped"
        self.assertFalse(pr_report.aggregate(needs, e2e_requested=True))
        needs["pr-gate"]["result"] = "skipped"
        self.assertTrue(pr_report.aggregate(needs, cpu_only=True, e2e_requested=True))

    def test_outer_gate_never_accepts_skipped_full_ci_for_selected_work(self):
        plan = selector.make_plan(
            self.catalog,
            selection_state(["test-bf16", "run-ci"]),
            "c" * 40,
            "42",
            "1",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.json"
            write_json(path, plan)
            for result, evidence in (
                ("success", "true"),
                ("skipped", "true"),
                ("failure", "true"),
                ("success", "false"),
            ):
                needs = {
                    "e2e-selection": {"result": "success"},
                    "ci": {"result": result, "outputs": {"e2e_evidence_ok": evidence}},
                }
                with self.subTest(result=result, evidence=evidence), mock.patch.object(
                    report, "check_live"
                ), mock.patch.object(report, "GitHub"), mock.patch.object(
                    report, "validate_plan"
                ), mock.patch.dict(
                    os.environ, {"NEEDS_JSON": json.dumps(needs)}
                ), mock.patch.object(
                    sys, "argv", ["report", "outer", "--plan", str(path)]
                ):
                    if result == "success" and evidence == "true":
                        report.main()
                    else:
                        with self.assertRaises(ValueError):
                            report.main()


class E2EWorkerTest(unittest.TestCase):
    def test_adapter_dispatch_and_timeout_leave_terminal_failure(self):
        for category, nodes, adapter in (
            ("answer", 1, "run_answer_suite_board.sh"),
            ("accuracy", 1, "run_accuracy_suite_board.sh"),
            ("perf", 1, "run_perf_suite_board.sh"),
            ("answer", 2, "run_pr_e2e_multinode.sh"),
            ("pd", 2, "run_pd_perf_suite_node.sh"),
        ):
            for timeout in (False, True):
                with self.subTest(
                    category=category, nodes=nodes, timeout=timeout
                ), tempfile.TemporaryDirectory() as temporary:
                    entry = {
                        "id": "fixture",
                        "category": category,
                        "nodes": nodes,
                        "blocked": "",
                        "environment": {},
                        "config": "fixture.json",
                        "config_digest": "digest",
                        "suite": "fixture-suite",
                        "timeout_seconds": 30,
                        "worker_minutes": 20,
                    }
                    plan = {
                        "plan_hash": "hash",
                        "tested_sha": "sha",
                        "run_id": "42",
                        "attempt": "1",
                    }
                    config = {
                        "model": {"path": "/staged/checkpoint"},
                        "hardware": {"visible_devices": [0]},
                        "evaluation": {"dataset_dir": "/staged/data"},
                    }
                    process = mock.Mock(pid=12345)
                    process.wait.side_effect = (
                        [subprocess.TimeoutExpired("worker", 60), -9, -9]
                        if timeout
                        else [0, 0]
                    )
                    with mock.patch.object(
                        worker,
                        "context",
                        return_value=(plan, entry, 0, Path(temporary)),
                    ), mock.patch.object(
                        worker, "read_json", return_value=config
                    ), mock.patch.object(
                        Path, "is_file", return_value=True
                    ), mock.patch.object(
                        Path, "is_dir", return_value=True
                    ), mock.patch.object(
                        worker.subprocess, "Popen", return_value=process
                    ) as launch, mock.patch.object(
                        worker.os, "killpg"
                    ) as kill, mock.patch.object(
                        worker.signal, "signal"
                    ):
                        self.assertEqual(worker.run() == 0, not timeout)
                    command = launch.call_args.args[0]
                    self.assertEqual(Path(command[1]).name, adapter)
                    env = launch.call_args.kwargs["env"]
                    self.assertEqual(
                        env["PPU_E2E_REQUIRE_PD"], str(category == "pd").lower()
                    )
                    self.assertEqual(env["SGLANG_TEST_MAX_RETRY"], "0")
                    self.assertEqual(
                        env["PPU_INSTALL_EVIDENCE_DIR"], str(Path(temporary) / "rank-0")
                    )
                    signals = (
                        [signal.SIGTERM, signal.SIGKILL]
                        if timeout
                        else [signal.SIGKILL]
                    )
                    self.assertEqual(
                        kill.call_args_list,
                        [mock.call(process.pid, sig) for sig in signals],
                    )
                    terminal = selector.read_json(
                        Path(temporary) / "rank-0/worker.json"
                    )
                    self.assertEqual(terminal["exit_code"] == 0, not timeout)


class E2EWorkflowTest(unittest.TestCase):
    def workflow(self, name):
        return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())

    def test_only_run_ci_grants_are_relevant_label_events(self):
        ci = self.workflow("ci.yaml")
        expressions = [
            ci["concurrency"]["group"],
            ci["jobs"]["e2e-selection"]["if"],
            ci["jobs"]["e2e-gate"]["name"],
            ci["jobs"]["e2e-gate"]["if"],
        ]
        for expression in expressions:
            self.assertIn("github.event.label.name == 'run-ci'", expression)
            for token in selector.TOKENS:
                self.assertNotIn(token, expression)
        gate = ci["jobs"]["e2e-gate"]
        self.assertIn("PPU E2E Ignored Event", gate["name"])
        self.assertIn("always()", gate["if"])
        self.assertIn("ci", gate["needs"])

    def test_ignored_events_cannot_publish_the_required_e2e_check_name(self):
        template = self.workflow("ci.yaml")["jobs"]["e2e-gate"]["name"]
        script = r"""
            const {template, events} = JSON.parse(process.argv[1]);
            console.log(JSON.stringify(events.map(github => template.replace(/\$\{\{([\s\S]*?)\}\}/g,
                (_, expression) => Function('github', 'contains', 'fromJSON', `return (${expression});`)(
                    github, (items, value) => items.includes(value), JSON.parse)).trim())));
        """
        cases = [
            ("pull_request", "synchronize", "", True),
            ("pull_request", "converted_to_draft", "", True),
            ("pull_request_review", "submitted", "", False),
            ("pull_request", "labeled", "documentation", False),
            ("pull_request", "labeled", "test-unknown", False),
        ]
        cases += [
            ("pull_request", action, label, action == "labeled" and label == "run-ci")
            for action in ("labeled", "unlabeled")
            for label in selector.TOKENS | {"run-ci"}
        ]
        events = [
            {"event_name": event, "event": {"action": action, "label": {"name": label}}}
            for event, action, label, _ in cases
        ]
        completed = subprocess.run(
            [
                "node",
                "-e",
                script,
                json.dumps({"template": template, "events": events}),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        self.assertEqual(
            json.loads(completed.stdout),
            [
                "PPU E2E Gate" if active else "PPU E2E Ignored Event"
                for _, _, _, active in cases
            ],
        )
        owners = [
            p.name
            for p in (ROOT / ".github/workflows").iterdir()
            if p.suffix in {".yml", ".yaml"} and "PPU E2E Gate" in p.read_text()
        ]
        self.assertEqual(owners, ["ci.yaml"])

    def test_artifact_retention_outlives_worst_case_plan_budget(self):
        # Phases run serially with max-parallel 1, so a plan's wall time is
        # bounded by the sum of its entry job ceilings; union selectors can
        # select every unblocked entry. Plan and result artifacts must be
        # downloadable until the final aggregate, so retention must exceed
        # that worst case.
        catalog = selector.load_catalog()
        budget = sum(
            entry["job_minutes"] for entry in catalog["entries"] if not entry["blocked"]
        )
        ci = self.workflow("ci.yaml")
        entry = self.workflow("pr-test-ppu-e2e-entry.yml")
        plan_upload = next(
            step
            for job in ci["jobs"].values()
            for step in job.get("steps", [])
            if step.get("with", {}).get("name", "").startswith("ppu-e2e-plan-")
        )
        result_upload = next(
            step
            for job in entry["jobs"].values()
            for step in job.get("steps", [])
            if step.get("with", {}).get("name", "").startswith("ppu-e2e-result-")
        )
        for upload in (plan_upload, result_upload):
            self.assertGreater(upload["with"]["retention-days"] * 24 * 60, budget)

    def test_cpu_reuse_rejects_other_attempt_or_plan(self):
        cpu = self.workflow("pr-test-ppu.yml")["jobs"]["stage-a-test-cpu-ppu"]
        reuse = next(step for step in cpu["steps"] if step.get("id") == "reuse")
        for identity in (
            "42:0:sha:plan",
            "41:1:sha:plan",
            "42:1:other:plan",
            "42:1:sha:other",
        ):
            with self.subTest(
                identity=identity
            ), tempfile.TemporaryDirectory() as temporary:
                result = subprocess.run(
                    ["bash", "-e", "-c", reuse["run"]],
                    env={
                        **os.environ,
                        "CPU_RESULT": "success",
                        "CPU_EVIDENCE_OK": "true",
                        "CPU_IDENTITY": identity,
                        "GITHUB_RUN_ID": "42",
                        "GITHUB_RUN_ATTEMPT": "1",
                        "GITHUB_SHA": "sha",
                        "PLAN_HASH": "plan",
                        "GITHUB_OUTPUT": str(Path(temporary) / "output"),
                    },
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertNotEqual(result.returncode, 0)

    def test_sequential_bounded_phases_and_cpu_identity(self):
        jobs = self.workflow("pr-test-ppu-e2e.yml")["jobs"]
        self.assertEqual(jobs["extended"]["needs"], ["prepare", "default"])
        self.assertEqual(
            jobs["distributed"]["needs"], ["prepare", "default", "extended"]
        )
        for name in ("default", "extended", "distributed"):
            self.assertEqual(jobs[name]["strategy"]["max-parallel"], 1)
            self.assertFalse(jobs[name]["strategy"]["fail-fast"])
            self.assertEqual(jobs[name]["with"]["phase"], name)
        preparation = json.dumps(jobs["prepare"])
        self.assertIn("select_pr_e2e.py cpu", preparation)
        self.assertIn("inputs.cpu_identity", preparation)
        self.assertEqual(jobs["finish"]["if"], "always()")

    def test_entry_uses_pinned_action_and_strict_sanitized_uploads(self):
        workflow = self.workflow("pr-test-ppu-e2e-entry.yml")
        self.assertEqual(set(workflow["permissions"].values()), {"read"})
        job = workflow["jobs"]["execute"]
        self.assertEqual(job["runs-on"], "k8s-runner-group-cpu-flytiger")
        steps = job["steps"]
        action = next(s for s in steps if s.get("id") == "worker")
        self.assertEqual(
            action["uses"],
            "flytiger-eco/ppu-distributed-action@06c8ee7cadaae2086db4977e571b19f461653f16",
        )
        self.assertEqual(action["with"]["node_selector"], "board-type=ZW-M890P")
        self.assertEqual(action["with"]["nproc_per_node"], 8)
        self.assertEqual(action["with"]["cleanup_policy"], "always")
        self.assertIn("|| 0", action["with"]["nnodes"])
        self.assertNotIn("GH_TOKEN", action["with"]["extra_env"])
        self.assertFalse(any("continue-on-error" in s for s in steps))
        uploads = [s for s in steps if s.get("uses") == "actions/upload-artifact@v4"]
        self.assertEqual(len(uploads), 2)
        for upload in uploads:
            self.assertEqual(upload["with"]["if-no-files-found"], "error")
            self.assertNotIn("/wl_nas/", upload["with"]["path"])
        self.assertIn("pr_report.py export", json.dumps(steps))
        self.assertIn(
            "select_pr_e2e.py check", json.dumps(steps[: steps.index(action)])
        )


if __name__ == "__main__":
    unittest.main()

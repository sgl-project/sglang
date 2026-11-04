"""Lifecycle replay contracts; loopback fixtures never start a GPU server."""

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lifecycle_bench as lb
import test_router_bench as router_fixture
from analyze_lifecycle import analyze


def args_for(**changes):
    args = lb.parser().parse_args(
        [
            "--router",
            "http://router",
            "--router-control-key-env",
            "LIFECYCLE_FIXTURE_KEY",
            "--engine-a",
            "http://a",
            "--engine-b",
            "http://b",
            "--model",
            "mock-model",
            "--output",
            "unused",
            "--cases",
            "4",
            "--prefix-tokens",
            "64",
            "--blocks",
            "1",
            "--passes",
            "2",
            "--concurrency",
            "2",
            "--tool-wait-s",
            ".02",
            "--output-tokens",
            "2",
            "--poll-ms",
            "5",
            "--timeout-s",
            "1",
            "--cleanup-timeout-s",
            ".1",
            "--run-id",
            "fixture",
        ]
    )
    args.page_size = 16
    for name, value in changes.items():
        setattr(args, name, value)
    return args


class PlanningTests(unittest.TestCase):
    def test_replay_keeps_plan_and_cancellation_policy_independent(self):
        args = args_for(
            wait_jitter_s=0.2,
            prediction_error_s=0.5,
            cancel_fraction=0.3,
            route_change_fraction=0.2,
        )
        left, right = lb.replay_plan(args), lb.replay_plan(args)
        self.assertEqual(left, right)
        self.assertEqual(sum(map(len, left)), args.cases * args.passes)
        self.assertEqual(
            len({item["id"] for lane in left for item in lane}),
            args.cases * args.passes,
        )
        self.assertTrue(
            all(
                item["tool_wait_s"] >= 0 and item["predicted_wait_s"] >= 0
                for lane in left
                for item in lane
            )
        )

    def test_family_ancestry_and_salt_are_explicit_and_arm_isolated(self):
        args = args_for(branches_per_family=2, shared_prefix_fraction=0.5)
        material = [[index] * 64 for index in range(4)]
        cases = lb.cases_for_arm(args, material, {"id": "arm-a"})
        self.assertEqual(cases[0]["prefix"][:32], cases[1]["prefix"][:32])
        self.assertNotEqual(cases[0]["prefix"][32:], cases[1]["prefix"][32:])
        self.assertEqual(cases[0]["salt"], cases[1]["salt"])
        self.assertNotEqual(cases[1]["salt"], cases[2]["salt"])
        self.assertNotEqual(
            cases[0]["salt"],
            lb.cases_for_arm(args, material, {"id": "arm-b"})[0]["salt"],
        )

    def test_reused_source_salts_are_shared_only_within_one_block(self):
        args = args_for(reuse_published_sources=True)
        material = [[1] * 64] * args.cases
        first = lb.cases_for_arm(args, material, {"id": "a", "block": 0})
        second = lb.cases_for_arm(args, material, {"id": "b", "block": 0})
        later = lb.cases_for_arm(args, material, {"id": "a", "block": 1})
        self.assertEqual(first, second)
        self.assertNotEqual(first[0]["salt"], later[0]["salt"])

    def test_invalid_bounds_rejected(self):
        for changes in (
            {"arms": ["on_demand", "on_demand"]},
            {"prediction_error_s": float("nan")},
            {"control_workers": 4, "control_backlog": 2},
            {"via_router": True, "route_change_fraction": 0.1},
            {"cancel_fraction": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                lb.validate(args_for(**changes))

    def test_working_set_deduplicates_only_shared_salt_and_ancestors(self):
        args = args_for(cases=2, branches_per_family=2, shared_prefix_fraction=0.5)
        cases = lb.cases_for_arm(args, [[0] * 64, [1] * 64], {"id": "same"})
        self.assertEqual(lb.logical_working_set(cases, 16), 96)
        cases[1]["salt"] = "independent"
        self.assertEqual(lb.logical_working_set(cases, 16), 128)
        self.assertEqual(len(lb.resume_prompt(cases[0])), 65)
        self.assertEqual(lb.resume_prompt(cases[0])[:-1], cases[0]["prefix"])

    def test_long_replay_metadata_preflight_excludes_secrets(self):
        args = args_for(cases=64, passes=20, blocks=5)
        with self.assertRaisesRegex(lb.BenchError, "router-budget-file"):
            lb.metadata_budget(args)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "router.json")
            path.write_text(
                json.dumps(
                    {
                        "control_bearer": "secret-never-retain",
                        "max_operations": 1000000,
                        "limits": {"max_sessions": 1024, "max_tombstones": 1000000},
                    }
                )
            )
            args.router_budget_file = str(path)
            budget = lb.metadata_budget(args)
            self.assertNotIn("secret-never-retain", json.dumps(budget))
            self.assertEqual(budget["required"]["limits.max_tombstones"], 19200)

    def test_pressure_requires_current_unambiguous_capacity(self):
        journal = SimpleNamespace(emit=lambda *args, **kwargs: None)
        sampler = lb.PressureSampler(None, journal, 0.9)
        self.assertFalse(sampler.snapshot()["pressure"])
        sampler.latest = {"sample_monotonic": time.monotonic(), "usage": 0.95}
        self.assertTrue(sampler.snapshot()["pressure"])
        sampler.latest["sample_monotonic"] -= 10
        self.assertFalse(sampler.snapshot()["pressure"])
        sample = 'sglang:hicache_host_used_tokens{a="b"} 95\nsglang:hicache_host_total_tokens{a="b"} 100\n'
        sampler = lb.PressureSampler(
            SimpleNamespace(request=lambda *args, **kwargs: sample), journal, 0.9
        )
        sampler.thread.start()
        deadline = time.monotonic() + 1
        while not sampler.snapshot()["pressure"] and time.monotonic() < deadline:
            time.sleep(0.005)
        sampler.stop.set()
        sampler.thread.join()
        self.assertTrue(sampler.snapshot()["pressure"])

    def test_async_queue_is_bounded_and_retains_cleanup_errors(self):
        controls = lb.AsyncControls(1, 1)
        release = threading.Event()
        controls.submit(lambda: (release.wait(), {"quiescent": True})[1])
        returned = threading.Event()

        def enqueue():
            controls.submit(
                lambda: {
                    "id": "failed-cleanup",
                    "quiescent": False,
                    "errors": ["not drained"],
                }
            )
            returned.set()

        worker = threading.Thread(target=enqueue)
        worker.start()
        self.assertFalse(returned.wait(0.03))
        release.set()
        worker.join(1)
        result = controls.drain()
        self.assertEqual(result["peak_pending"], 1)
        self.assertEqual(result["errors"][0]["id"], "failed-cleanup")

    def test_resume_prediction_deducts_time_spent_reporting_tool_wait(self):
        clock, reported = [0.0], []
        args = args_for()
        holder = {}

        def event(task, name, revision, event_id, trial, **fields):
            if name == "tool_wait":
                clock[0] += 3
            elif name == "resume_expected":
                reported.append(fields["expected_wait_ms"])
                holder["preparation"].done.set()
                return {"state": "on_demand_fallback"}
            return {}

        preparation = lb.Preparation(
            args,
            SimpleNamespace(event=event, poll_ms=1),
            SimpleNamespace(http=SimpleNamespace(url="http://b")),
            {"session_id": "task"},
            {"prefix": [1], "salt": "salt"},
            {
                "trial": "prediction",
                "arm": "lifecycle",
                "tool_wait_s": 10,
                "predicted_wait_s": 10,
            },
            10,
            SimpleNamespace(emit=lambda *args, **kwargs: None),
        )
        holder["preparation"] = preparation
        with patch.object(lb.time, "monotonic", side_effect=lambda: clock[0]):
            result = preparation.run()
        self.assertEqual(reported, [7000])
        self.assertTrue(result["quiescent"])

    def test_scheduled_job_without_engine_receipt_remains_unknown(self):
        holder = {}

        def event(task, name, revision, event_id, trial, **fields):
            if name == "resume_expected":
                holder["preparation"].done.set()
                return {
                    "state": "scheduled",
                    "known_scope": [{"branch": "main", "end_token": 1}],
                    "operations": [
                        {
                            "action": "sglang.prefetch",
                            "worker_url": "http://b",
                            "engine_id": "b",
                            "owner_epoch": "epoch",
                            "result": None,
                            "ready_scope": None,
                        }
                    ],
                }
            return {}

        router = SimpleNamespace(
            event=event,
            poll_ms=1,
            command=lambda *args: {"acknowledged": True},
            query=lambda *args: {"quiescent": True, "operations": []},
        )
        preparation = lb.Preparation(
            args_for(),
            router,
            SimpleNamespace(
                http=SimpleNamespace(url="http://b"),
                cap={"engine_id": "b", "engine_epoch": "epoch"},
            ),
            {"session_id": "task"},
            {"prefix": [1], "salt": "salt"},
            {
                "trial": "scheduled",
                "arm": "lifecycle",
                "tool_wait_s": 10,
                "predicted_wait_s": 10,
            },
            time.monotonic() + 10,
            SimpleNamespace(emit=lambda *args, **kwargs: None),
        )
        holder["preparation"] = preparation
        result = preparation.run()
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["child_statuses"], ["unknown"])
        self.assertFalse(result["strict_preparation_complete"])
        self.assertEqual(result["prepared_tokens_observed"], 0)
        self.assertTrue(result["quiescent"])

    def test_block_statistics_include_incomplete_and_invalid_responses(self):
        args = args_for(min_arm_seconds=10, min_arm_resumes=3)
        signatures = {}

        def row(identity, output=1):
            return {
                "id": identity,
                "case": identity,
                "valid": True,
                "errors": [],
                "cancelled": False,
                "task_latency_ms": 50,
                "generation": {
                    "output_ids": [output],
                    "text": "text",
                    "ttft_ms": 20,
                    "latency_ms": 30,
                },
            }

        base = lb.arm_summary(
            {"block": 0, "arm": "on_demand"},
            [row("first"), row("second")],
            2,
            {},
            args,
            1,
            signatures,
        )
        candidate = lb.arm_summary(
            {"block": 0, "arm": "lifecycle"},
            [row("first", 2), row("second")],
            1,
            {},
            args,
            1,
            signatures,
        )
        self.assertEqual(candidate["valid_tasks"], 1)
        self.assertEqual(candidate["completed"], 2)
        self.assertFalse(candidate["meets_minimum_duration"])
        self.assertFalse(candidate["meets_minimum_resumes"])
        report = lb.summarize([base, candidate])
        self.assertEqual(report["paired"][0]["effective_throughput_change_percent"], 0)
        self.assertEqual(candidate["all_completed_tasks_per_s"], 2)
        self.assertEqual(candidate["effective_completed"], 1)
        self.assertEqual(
            report["paired_block_throughput_ci_percent"]["lifecycle"]["n_pairs"], 1
        )
        self.assertIsNone(report["trainer_waiting_time"])


class LifecycleHTTPTests(unittest.TestCase):
    # Reuse the existing HTTP contract fixtures without inheriting/re-running
    # the unrelated B2/B3 test methods.
    setUp = router_fixture.RouterBenchmarkTests.setUp
    tearDown = router_fixture.RouterBenchmarkTests.tearDown
    serve = router_fixture.RouterBenchmarkTests.serve

    def make_args(self, **changes):
        return args_for(
            router=self.url,
            engine_a=self.a.http.url,
            engine_b=self.b.http.url,
            output=str(Path(self.directory.name, "run")),
            **changes,
        )

    def test_full_three_arm_run_keeps_input_plan_and_outputs(self):
        args = self.make_args()
        configuration = {"fingerprint_a": {}, "fingerprint_b": {"context_length": 1024}}
        with (
            patch.dict(os.environ, {args.router_control_key_env: self.state.bearer}),
            patch.object(lb, "parity", return_value=configuration),
            patch.object(
                lb,
                "build_tokens",
                side_effect=lambda model, length, seed: ([seed % 100] * length, []),
            ),
            patch("builtins.print"),
        ):
            self.assertEqual(lb.run(args), 0)
        result = json.loads(Path(args.output, "lifecycle-summary.json").read_text())
        self.assertEqual(len(result["arms"]), 3)
        self.assertEqual(sum(arm["completed"] for arm in result["arms"]), 24)
        self.assertEqual(len(result["paired"]), 2)
        raw = Path(args.output, "lifecycle-raw.jsonl").read_text()
        self.assertNotIn(self.state.bearer, raw)
        publications = [
            json.loads(line)
            for line in raw.splitlines()
            if json.loads(line).get("kind") == "lifecycle_publication"
        ]
        self.assertEqual(len(publications), 12)
        self.assertTrue(all("initial_generation" in row for row in publications))
        self.assertTrue(all(row["initial_reference_equal"] for row in publications))
        self.assertEqual(
            sum(body.get("event") == "resume_expected" for body in self.state.calls), 8
        )
        self.assertEqual(
            sum(
                path.startswith("/flush_cache")
                for engine in self.engines
                for path, body in engine.requests
            ),
            6,
        )
        for arm in result["arms"]:
            rows = json.loads(Path(args.output, arm["id"] + ".json").read_text())[
                "rows"
            ]
            self.assertEqual(len({row["id"] for row in rows}), 8)
            self.assertTrue(all(row["output_equal"] for row in rows))
        recomputed = analyze(args.output)
        self.assertTrue(
            recomputed["verification"]["verified"], recomputed["verification"]
        )
        self.assertTrue(recomputed["acceptance_passed"])
        self.assertEqual(recomputed["paired"], result["paired"])
        # Replaying a manifest field must be verified independently from the
        # generated summary, so a later edited wait cannot silently pass.
        path = Path(args.output, result["arms"][0]["id"] + ".json")
        altered = json.loads(path.read_text())
        altered["rows"][0]["tool_wait_s"] += 1
        path.write_text(json.dumps(altered))
        self.assertFalse(analyze(args.output)["verification"]["verified"])

    def prepare_source(self, case):
        self.a.http.generate(
            lb.resume_prompt(case), case["salt"], 1, trial="source", seed=1
        )
        lb.publish(
            self.a,
            self.a.target(case["prefix"], case["salt"]),
            len(case["prefix"]),
            "source",
        )

    def test_three_arm_reuse_executes_64_source_pairs_once_per_block(self):
        args = self.make_args(
            cases=64, passes=1, blocks=2, tool_wait_s=0, reuse_published_sources=True
        )
        with (
            patch.dict(os.environ, {args.router_control_key_env: self.state.bearer}),
            patch.object(
                lb, "parity", return_value={"fingerprint_b": {"context_length": 1024}}
            ),
            patch.object(
                lb,
                "build_tokens",
                side_effect=lambda model, length, seed: ([seed % 100] * length, []),
            ),
            patch("builtins.print"),
        ):
            self.assertEqual(lb.run(args), 0)
        arms = json.loads(Path(args.output, "lifecycle-summary.json").read_text())[
            "arms"
        ]
        self.assertEqual(sum(arm["completed"] for arm in arms), 64 * 3 * 2)
        self.assertEqual(
            sum(arm["source_generation_calls"] for arm in arms), 64 * 2 * 2
        )
        self.assertEqual(
            sum(path == "/generate" for path, body in self.engines[0].requests),
            64 * 2 * 2,
        )
        self.assertEqual(
            sum(
                path.startswith("/flush_cache")
                for path, body in self.engines[0].requests
            ),
            2,
        )
        for block in range(2):
            members = [arm for arm in arms if arm["block"] == block]
            self.assertEqual(len({arm["source_publication_s"] for arm in members}), 1)
            self.assertAlmostEqual(
                sum(arm["source_publication_s"] for arm in members),
                sum(arm["source_execution_wall_s"] for arm in members),
            )
            self.assertEqual(
                sum(arm["source_reused_from"] is None for arm in members), 1
            )
            self.assertTrue(
                all(
                    arm["source_execution_wall_s"] == 0
                    for arm in members
                    if arm["source_reused_from"]
                )
            )
        checked = analyze(args.output)
        self.assertTrue(checked["acceptance_passed"], checked["verification"])
        path = Path(args.output, "lifecycle-raw.jsonl")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        reused = next(
            record
            for record in records
            if record.get("kind") == "lifecycle_publication"
            and record.get("reused_from")
        )
        self.assertTrue(reused["reference_only"])
        self.assertNotIn("publication", reused)
        reused["reused_from"] = reused["reused_from"].rsplit("-", 1)[0] + "-63"
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        self.assertFalse(analyze(args.output)["verification"]["verified"])

    def test_slow_control_cleanup_is_outside_generation_path_and_counted(self):
        args = self.make_args(via_router=True)
        self.state.pick = self.engines[1]
        self.state.delay_s = 0.5
        self.state.cleanup_unknown = True
        case = {
            "id": 0,
            "family": 0,
            "branch": 0,
            "prefix": list(range(64)),
            "salt": "async",
        }
        self.prepare_source(case)
        controls = lb.AsyncControls(2, 2)
        spec = {
            "id": "trial",
            "trial": "trial",
            "arm": "lifecycle",
            "tool_wait_s": 0.02,
            "predicted_wait_s": 0.02,
            "cancelled": False,
            "route_change": False,
        }
        started = time.monotonic()
        row = lb.resume_task(
            args, self.a, self.b, self.router, case, spec, controls, self.journal
        )
        self.assertTrue(row["valid"], row)
        self.assertFalse(row["ready_before_resume"])
        self.assertLess(time.monotonic() - started, 0.1)
        drain = controls.drain()
        self.assertTrue(drain["errors"])
        self.assertGreater(drain["wall_ms"], 30)
        self.assertLess(row["dispatch_lateness_ms"], 100)

    def test_completed_preparation_does_not_keep_polling_through_consumption(self):
        args = self.make_args()
        case = {
            "id": 0,
            "family": 0,
            "branch": 0,
            "prefix": list(range(64)),
            "salt": "ready-stop",
        }
        self.prepare_source(case)
        controls = lb.AsyncControls(1, 2)
        row = lb.resume_task(
            args,
            self.a,
            self.b,
            self.router,
            case,
            {
                "id": "ready",
                "trial": "ready",
                "arm": "lifecycle",
                "tool_wait_s": 0.2,
                "predicted_wait_s": 0.2,
                "cancelled": False,
                "route_change": False,
            },
            controls,
            self.journal,
        )
        drained = controls.drain()
        self.assertTrue(row["valid"], row)
        self.assertFalse(drained["errors"])
        self.assertTrue(row["ready_before_resume"])
        self.assertIsNotNone(row["preparation_observation_age_ms"])
        queries = [
            body
            for body in self.state.calls
            if body.get("kind") == "query" and body.get("message_id") == "prepare"
        ]
        self.assertLessEqual(len(queries), 4)

    def test_explicit_zero_cache_total_counts_full_repeated_prefill(self):
        args = self.make_args()
        case = {
            "id": 0,
            "family": 0,
            "branch": 0,
            "prefix": list(range(64)),
            "salt": "full-miss",
        }
        self.prepare_source(case)
        self.engines[1].storage.pop(case["salt"])
        generate = self.b.http.generate

        def upstream_zero_details(*args, **kwargs):
            response = generate(*args, **kwargs)
            self.assertEqual(response["meta_info"]["cached_tokens"], 0)
            response["meta_info"]["cached_tokens_details"] = None
            return response

        controls = lb.AsyncControls(1, 2)
        spec = {
            "id": "full-miss",
            "trial": "full-miss",
            "arm": "on_demand",
            "tool_wait_s": 0,
            "predicted_wait_s": 0,
            "cancelled": False,
            "route_change": False,
        }
        with patch.object(self.b.http, "generate", side_effect=upstream_zero_details):
            row = lb.resume_task(
                args, self.a, self.b, self.router, case, spec, controls, self.journal
            )
        controls.drain()
        self.assertTrue(row["valid"], row)
        self.assertEqual(row["repeated_prefill_tokens"], 64)
        self.assertEqual(row["cache"]["provenance"], "explicit_zero_total")
        self.assertFalse(row["cache"]["details_present"])

    def test_unknown_router_owner_keeps_completed_response(self):
        args = self.make_args(via_router=True)
        self.state.omit_header = True
        case = {
            "id": 0,
            "family": 0,
            "branch": 0,
            "prefix": list(range(64)),
            "salt": "missing-owner",
        }
        self.prepare_source(case)
        controls = lb.AsyncControls(1, 2)
        spec = {
            "id": "unknown",
            "trial": "unknown",
            "arm": "on_demand",
            "tool_wait_s": 0,
            "predicted_wait_s": 0,
            "cancelled": False,
            "route_change": False,
        }
        row = lb.resume_task(
            args, self.a, self.b, self.router, case, spec, controls, self.journal
        )
        controls.drain()
        self.assertFalse(row["valid"])
        self.assertIn("generation", row)
        self.assertIn("attribution", row["errors"][0])

    def test_cancelled_branch_does_not_inflate_effective_rollouts(self):
        args = self.make_args()
        case = {
            "id": 0,
            "family": 0,
            "branch": 0,
            "prefix": list(range(64)),
            "salt": "cancelled",
        }
        self.prepare_source(case)
        controls = lb.AsyncControls(1, 2)
        spec = {
            "id": "cancelled",
            "trial": "cancelled",
            "arm": "fixed_prefetch",
            "tool_wait_s": 0,
            "predicted_wait_s": 1,
            "cancelled": True,
            "route_change": False,
        }
        row = lb.resume_task(
            args, self.a, self.b, self.router, case, spec, controls, self.journal
        )
        controls.drain()
        self.assertTrue(row["valid"])
        self.assertNotIn("generation", row)
        self.assertFalse(
            any(path == "/generate" for path, body in self.engines[1].requests)
        )

    def test_failed_native_drain_remains_unknown_after_task_retirement(self):
        args = self.make_args()
        case = {
            "id": 0,
            "family": 0,
            "branch": 0,
            "prefix": list(range(64)),
            "salt": "unknown-native",
        }
        self.prepare_source(case)
        controls = lb.AsyncControls(1, 2)
        spec = {
            "id": "native-failed",
            "trial": "native-failed",
            "arm": "fixed_prefetch",
            "tool_wait_s": 0.02,
            "predicted_wait_s": 0.02,
            "cancelled": False,
            "route_change": False,
        }
        with patch.object(
            self.b, "transfer", side_effect=lb.BenchError("native drain unknown")
        ):
            row = lb.resume_task(
                args, self.a, self.b, self.router, case, spec, controls, self.journal
            )
            drained = controls.drain()
        self.assertIn("generation", row)
        self.assertFalse(drained["errors"][0]["quiescent"])
        self.assertIn("native drain unknown", drained["errors"][0]["errors"])


if __name__ == "__main__":
    unittest.main()

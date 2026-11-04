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
        self.a.http.generate(lb.resume_prompt(case), case["salt"], 1, trial="source", seed=1)
        lb.publish(
            self.a,
            self.a.target(case["prefix"], case["salt"]),
            len(case["prefix"]),
            "source",
        )

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

import json
import math
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sglang.benchmark import stress_suite
from sglang.benchmark.datasets.common import DatasetRow
from sglang.benchmark.datasets.dynamic import (
    DynamicDataset,
    fit_prompt_length,
    generate_arrival_offsets,
    load_workload_plan,
)
from sglang.benchmark.serving import (
    PhaseResultWriter,
    RequestFuncOutput,
    calculate_phase_metrics,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class DummyTokenizer:
    vocab_size = 128
    all_special_ids = []

    def decode(self, token_ids):
        return " ".join(str(token_id) for token_id in token_ids)

    def num_special_tokens_to_add(self):
        return 0

    def get_vocab(self):
        return {str(token_id): token_id for token_id in range(self.vocab_size)}

    def encode(self, text):
        return text.split()


class TestStressSuite(unittest.TestCase):
    def args(self, *extra):
        parser = stress_suite.build_parser()
        args = parser.parse_args(["--base-url", "http://127.0.0.1:30000", *extra])
        stress_suite.validate_args(parser, args)
        return args

    def test_profile_resolution_is_ordered_and_deduplicated(self):
        scenarios = stress_suite.resolve_scenarios(
            "standard", ["burst", "smoke", "burst"]
        )
        self.assertEqual([scenario.name for scenario in scenarios], ["burst", "smoke"])
        self.assertEqual(
            [scenario.name for scenario in stress_suite.resolve_scenarios("quick", [])],
            ["smoke", "steady", "burst", "recovery"],
        )

    def test_workload_has_contiguous_dynamic_phases(self):
        workload = stress_suite.build_workload(
            self.args("--profile", "quick", "--baseline-qps", "2", "--peak-qps", "8")
        )
        phases = workload["phases"]

        self.assertEqual([phase["request_rate"] for phase in phases], [1, 2, 8, 1])
        self.assertEqual(
            [phase["start_time"] for phase in phases],
            [0, 3, 23, 33],
        )
        for left, right in zip(phases, phases[1:]):
            self.assertEqual(left["start_time"] + left["duration"], right["start_time"])

    def test_total_duration_repeats_without_repeating_smoke(self):
        workload = stress_suite.build_workload(
            self.args("--profile", "quick", "--total-duration-sec", "50")
        )

        self.assertEqual(workload["duration"], 50)
        self.assertEqual(
            [phase["scenario"] for phase in workload["phases"]].count("smoke"), 1
        )
        self.assertEqual(workload["phases"][-1]["name"], "steady-c002")
        self.assertEqual(workload["phases"][-1]["duration"], 7)

    def test_explicit_smoke_can_fill_a_target_duration(self):
        workload = stress_suite.build_workload(
            self.args("--scenario", "smoke", "--total-duration-sec", "5")
        )
        self.assertEqual(workload["duration"], 5)
        self.assertEqual([phase["duration"] for phase in workload["phases"]], [3, 2])

    def test_soak_defaults_to_twenty_hours(self):
        workload = stress_suite.build_workload(self.args("--profile", "soak"))
        self.assertEqual(workload["duration"], 20 * 60 * 60)
        self.assertGreater(len(workload["phases"]), len(stress_suite.SCENARIOS))
        self.assertEqual(
            [phase["scenario"] for phase in workload["phases"]].count("smoke"), 1
        )

    def test_command_uses_one_timestamped_serving_run(self):
        args = self.args("--profile", "quick")
        workload = stress_suite.build_workload(args)
        command = stress_suite.build_command(
            args, Path("workload.json"), Path("result.jsonl"), workload
        )

        self.assertEqual(
            command[:3], [stress_suite.sys.executable, "-m", "sglang.benchmark.serving"]
        )
        self.assertEqual(command[command.index("--dataset-name") + 1], "dynamic")
        self.assertIn("--use-trace-timestamps", command)
        self.assertIn("--max-pending-requests", command)
        self.assertIn("--phase-output-file", command)
        self.assertEqual(command.count("sglang.benchmark.serving"), 1)

    def test_command_redacts_headers_and_nested_secrets(self):
        command = [
            "python",
            "--header",
            "Authorization=Bearer secret",
            "Cookie=session-secret",
            "--base-url",
            "https://user:password@example.com",
            "--extra-request-body",
            '{"metadata":{"api_key":"secret"},"temperature":0}',
        ]
        rendered = " ".join(stress_suite.redact_command(command))
        self.assertNotIn("session-secret", rendered)
        self.assertNotIn('"secret"', rendered)
        self.assertNotIn("password", rendered)
        self.assertIn("temperature", rendered)

    def test_capabilities_skip_unsupported_phases(self):
        workload = stress_suite.build_workload(self.args("--profile", "edge"))
        skipped = stress_suite.apply_capabilities(
            workload,
            {"context_length": 4096, "tool_support": False},
        )
        skipped_scenarios = {phase["scenario"] for phase in skipped}
        self.assertIn("tool_rich", skipped_scenarios)
        self.assertIn("long_context_32k", skipped_scenarios)
        self.assertNotIn("steady", skipped_scenarios)
        self.assertEqual(workload["duration"], 81)

    def test_suite_reports_skip_when_every_phase_is_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "result"
            args = self.args(
                "--scenario",
                "long_context_80k",
                "--context-length",
                "4096",
                "--output-dir",
                str(output_dir),
            )
            with (
                patch.object(
                    stress_suite,
                    "detect_capabilities",
                    return_value={"context_length": 4096, "tool_support": None},
                ),
                patch.object(stress_suite.subprocess, "run") as run,
            ):
                self.assertEqual(stress_suite.run_suite(args), 0)
            self.assertFalse(run.called)
            summary = json.loads((output_dir / "summary.json").read_text())
        self.assertEqual(summary["verdict"], "SKIP")

    def test_phase_judgement_checks_completion_and_slo(self):
        args = self.args("--max-ttft-p99-ms", "100")
        self.assertEqual(
            stress_suite.judge_phase(
                args,
                {"planned": 3, "completed": 3, "p99_ttft_ms": 90},
            ),
            ("PASS", ""),
        )
        verdict, reason = stress_suite.judge_phase(
            args,
            {"planned": 3, "completed": 2, "p99_ttft_ms": 120},
        )
        self.assertEqual(verdict, "FAIL")
        self.assertIn("completed 2/3", reason)
        self.assertIn("TTFT p99", reason)

    def test_health_judgement_enforces_failure_budget(self):
        checks = [{"ok": True}, {"ok": False}, {"ok": False}]
        self.assertEqual(
            stress_suite.judge_health(checks, 1),
            "health failed 2 times (allowed 1)",
        )
        self.assertEqual(stress_suite.judge_health(checks, 2), "")

    def test_suite_runs_all_phases_in_one_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "result"
            args = self.args(
                "--profile",
                "quick",
                "--duration-scale",
                "0.1",
                "--output-dir",
                str(output_dir),
                "--health-check-interval-sec",
                "0",
            )

            def fake_run(command, **kwargs):
                workload_path = Path(
                    command[command.index("--dynamic-workload-path") + 1]
                )
                workload = json.loads(workload_path.read_text())
                phase_metrics = []
                for phase in workload["phases"]:
                    planned = math.ceil(phase["duration"] * phase["request_rate"])
                    phase_metrics.append(
                        {
                            "name": phase["name"],
                            "planned": planned,
                            "completed": planned,
                            "request_rate": phase["request_rate"],
                            "input_len": phase["input_len"],
                            "output_len": phase["output_len"],
                            "p99_ttft_ms": 10,
                            "p99_e2e_latency_ms": 20,
                        }
                    )
                result_file = Path(command[command.index("--output-file") + 1])
                result_file.write_text(
                    json.dumps(
                        {
                            "completed": sum(
                                phase["completed"] for phase in phase_metrics
                            ),
                            "p99_ttft_ms": 10,
                            "p99_e2e_latency_ms": 20,
                            "phase_metrics": phase_metrics,
                        }
                    )
                    + "\n"
                )
                return subprocess.CompletedProcess(command, 0)

            healthy = {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "status": 200,
                "ok": True,
                "latency_ms": 1,
                "error": "",
            }
            with (
                patch.object(
                    stress_suite.subprocess, "run", side_effect=fake_run
                ) as run,
                patch.object(stress_suite, "check_health", return_value=healthy),
            ):
                self.assertEqual(stress_suite.run_suite(args), 0)

            self.assertEqual(run.call_count, 1)
            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(summary["verdict"], "PASS")
            self.assertEqual(len(summary["runs"]), 4)
            self.assertEqual(summary["health"]["total"], 1)
            progress = json.loads((output_dir / "progress.json").read_text())
            self.assertEqual(progress["status"], "completed")
            self.assertEqual(
                stress_suite.load_jsonl(output_dir / "health.jsonl"), [healthy]
            )

    def test_health_monitor_writes_atomic_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            progress_file = Path(directory) / "progress.json"
            health_file = Path(directory) / "health.jsonl"
            phase_file = Path(directory) / "phases.jsonl"
            stop = threading.Event()
            stop.set()
            checks = []
            healthy = {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "status": 200,
                "ok": True,
                "latency_ms": 1,
                "error": "",
            }
            with patch.object(stress_suite, "check_health", return_value=healthy):
                stress_suite.monitor_health(
                    stop,
                    base_url="http://127.0.0.1:30000",
                    interval=1,
                    progress_file=progress_file,
                    health_file=health_file,
                    phase_file=phase_file,
                    workload={"duration": 20, "phases": []},
                    checks=checks,
                    headers={},
                    started_at="2026-01-01T00:00:00+00:00",
                )

            progress = json.loads(progress_file.read_text())
            self.assertEqual(progress["status"], "running")
            self.assertEqual(progress["health"]["latest"], healthy)
            self.assertEqual(stress_suite.load_jsonl(health_file), [healthy])
            self.assertFalse(progress_file.with_suffix(".json.tmp").exists())

    def test_dynamic_dataset_generates_timestamped_synthetic_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "workload.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "source": "random-ids",
                        "arrival_pattern": "constant",
                        "phases": [
                            {
                                "name": "small",
                                "duration": 1,
                                "request_rate": 2,
                                "input_len": 8,
                                "output_len": 2,
                                "extra_request_body": {"temperature": 0.2},
                            },
                            {
                                "name": "large",
                                "duration": 1,
                                "request_rate": 1,
                                "input_len": 16,
                                "output_len": 4,
                            },
                        ],
                    }
                )
            )
            dataset = DynamicDataset(str(plan_path), "", seed=7)
            rows = dataset.load(DummyTokenizer())

        self.assertEqual([row.phase for row in rows], ["small", "small", "large"])
        self.assertEqual([row.timestamp for row in rows], [0, 500, 1000])
        self.assertEqual([row.prompt_len for row in rows], [8, 8, 16])
        self.assertEqual(rows[0].extra_request_body, {"temperature": 0.2})
        self.assertEqual(rows[2].extra_request_body, {})

    def test_prompt_length_is_calibrated_after_text_round_trip(self):
        prompt, length = fit_prompt_length(DummyTokenizer(), "1 2", 8)
        self.assertEqual(length, 8)
        self.assertEqual(len(DummyTokenizer().encode(prompt)), 8)

    def test_dynamic_plan_validation_and_poisson_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "invalid.json"
            plan_path.write_text(json.dumps({"phases": []}))
            with self.assertRaisesRegex(ValueError, "at least one phase"):
                load_workload_plan(str(plan_path))

        offsets = generate_arrival_offsets(
            duration=2,
            request_rate=2,
            pattern="poisson",
            rng=np.random.default_rng(1),
        )
        self.assertEqual(offsets[0], 0)
        self.assertTrue(all(left < right for left, right in zip(offsets, offsets[1:])))
        self.assertLess(offsets[-1], 2)

    def test_dynamic_dataset_reuses_shared_prefix_generator(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "workload.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "source": "generated-shared-prefix",
                        "prompt_pool_size": 2,
                        "phases": [
                            {
                                "name": "prefix",
                                "duration": 1,
                                "request_rate": 2,
                                "input_len": 8,
                                "output_len": 2,
                                "range_ratio": 0.9,
                            }
                        ],
                    }
                )
            )
            rows = DynamicDataset(str(plan_path), "", seed=7).load(DummyTokenizer())

        self.assertEqual(len(rows), 2)
        self.assertEqual({row.phase for row in rows}, {"prefix"})
        self.assertTrue(all(row.prompt for row in rows))

    def test_dynamic_dataset_reuses_sharegpt_sampler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "sharegpt.json"
            dataset_path.write_text(
                json.dumps(
                    [
                        {
                            "conversations": [
                                {"value": "synthetic question one"},
                                {"value": "synthetic answer one"},
                            ]
                        },
                        {
                            "conversations": [
                                {"value": "synthetic question two"},
                                {"value": "synthetic answer two"},
                            ]
                        },
                    ]
                )
            )
            plan_path = root / "workload.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "source": "sharegpt",
                        "prompt_pool_size": 2,
                        "phases": [
                            {
                                "name": "open_source",
                                "duration": 1,
                                "request_rate": 1,
                                "input_len": 8,
                                "output_len": 2,
                            }
                        ],
                    }
                )
            )
            rows = DynamicDataset(str(plan_path), str(dataset_path), seed=7).load(
                DummyTokenizer()
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].phase, "open_source")
        self.assertTrue(rows[0].prompt)

    def test_serving_reports_metrics_for_each_phase(self):
        requests = [
            DatasetRow(
                prompt="a",
                prompt_len=1,
                output_len=2,
                phase="low",
                phase_duration=1,
                phase_request_rate=1,
                phase_max_concurrency=2,
            ),
            DatasetRow(
                prompt="b",
                prompt_len=1,
                output_len=2,
                phase="high",
                phase_duration=1,
                phase_request_rate=4,
                phase_max_concurrency=8,
            ),
        ]
        outputs = [
            RequestFuncOutput(
                success=True,
                latency=0.2,
                ttft=0.1,
                prompt_len=1,
                cached_tokens=1,
                stream_complete=True,
                scheduled_offset_ms=0,
                dispatch_offset_ms=10,
                status_code=200,
                finish_reason="stop",
                usage_present=True,
            ),
            RequestFuncOutput(
                success=False,
                error="overloaded",
                status_code=429,
                scheduled_offset_ms=1000,
                dispatch_offset_ms=1020,
            ),
        ]

        phases = calculate_phase_metrics(requests, outputs)
        self.assertEqual(phases[0]["completed"], 1)
        self.assertEqual(phases[0]["p99_ttft_ms"], 100)
        self.assertEqual(phases[1]["completed"], 0)
        self.assertEqual(phases[1]["errors"], ["overloaded"])
        self.assertEqual(phases[1]["failure_categories"], {"http_429": 1})
        self.assertEqual(phases[0]["cache_hit_rate_pct"], 100)
        self.assertEqual(phases[0]["mean_schedule_lag_ms"], 10)
        self.assertEqual(phases[1]["max_concurrency"], 8)

    def test_phase_writer_checkpoints_completed_phase(self):
        requests = [
            DatasetRow(prompt="a", prompt_len=1, output_len=1, phase="one"),
            DatasetRow(prompt="b", prompt_len=1, output_len=1, phase="one"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phases.jsonl"
            writer = PhaseResultWriter(requests, str(path))
            writer.record(requests[0], RequestFuncOutput(success=True))
            self.assertFalse(path.exists())
            writer.record(requests[1], RequestFuncOutput(success=True))
            rows = stress_suite.load_jsonl(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["planned"], 2)

    def test_phase_writer_snapshots_selected_prometheus_metrics(self):
        request = DatasetRow(prompt="a", prompt_len=1, output_len=1, phase="one")
        response = type(
            "Response",
            (),
            {
                "text": (
                    'sglang:num_queue_reqs{model="m",rank="0"} 2\n'
                    'sglang:num_queue_reqs{model="m",rank="1"} 3\n'
                    "unrelated_metric 9\n"
                ),
                "raise_for_status": lambda self: None,
            },
        )()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("sglang.benchmark.serving.requests.get", return_value=response),
        ):
            path = Path(directory) / "phases.jsonl"
            writer = PhaseResultWriter(
                [request],
                str(path),
                "http://127.0.0.1:30000",
                ["sglang:num_queue_reqs"],
            )
            writer.record(request, RequestFuncOutput(success=True))
            row = stress_suite.load_jsonl(path)[0]
        self.assertEqual(row["prometheus_snapshot"], {"sglang:num_queue_reqs": 5.0})

    def test_repository_fixture_is_synthetic_custom_data(self):
        fixture = (
            Path(__file__).resolve().parents[3]
            / "benchmark"
            / "stress_suite"
            / "data"
            / "synthetic_trace.jsonl"
        )
        rows = [json.loads(line) for line in fixture.read_text().splitlines() if line]
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(len(row["conversations"]) == 2 for row in rows))


if __name__ == "__main__":
    unittest.main()

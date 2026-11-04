"""Offline evidence fixtures; no Engine, Router or GPU is started."""

import copy
import gzip
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze_lifecycle_costs as costs


def process_row(t, created=1, user=None):
    return {
        "monotonic": t,
        "roles": {
            "engine_b": {
                "root_pid": 10,
                "processes": [
                    {
                        "pid": 10,
                        "created": created,
                        "user_s": 2 * t if user is None else user,
                        "system_s": 0,
                    }
                ],
            },
            "router": {"root_pid": 99, "error": "process PID not found"},
        },
    }


def resource_row(t, used=40, duration=1, host=True):
    metrics = [
        "sglang:kv_used_tokens 1",
        "sglang:kv_available_tokens 20",
        "sglang:hicache_host_total_tokens 100",
    ]
    if host:
        metrics.append("sglang:hicache_host_used_tokens " + str(used))
    return {
        "monotonic": t,
        "sample_elapsed_s": duration,
        "gpu_csv": "1,uuid,A100,100,60,40,20\n",
        "http": [{"url": "http://b/metrics", "body": "\n".join(metrics)}],
    }


def write_records(path, rows):
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as out:
            out.write(text)
    else:
        path.write_text(text)


class CostAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.schedule = [
            {"id": "b0-" + name, "arm": name, "block": 0}
            for name in ("on_demand", "fixed_prefetch", "lifecycle")
        ]
        manifest = {
            "args": {"engine_b": "http://b", "prefix_tokens": 16},
            "schedule": self.schedule,
        }
        (self.root / "lifecycle-manifest.json").write_text(json.dumps(manifest))
        self.raw, self.arms = [], []
        for index, spec in enumerate(self.schedule):
            start = index * 20
            rows = [
                {
                    "generation": {"ttft_ms": value, "latency_ms": value + 10},
                    "ready_before_resume": True,
                    "selected_worker_url": "http://b",
                    "cache": {"cpu": 16, "device": 0},
                    "valid": i != 3,
                }
                for i, value in enumerate([10, 20, 30, 100])
            ]
            rows[0]["cache"] = {"cpu": 0, "device": 16}
            rows[2]["cache"] = {"cpu": 8, "device": 0, "storage": 8}
            rows[3]["selected_worker_url"] = "http://a"
            before = {
                "success": True,
                "data": {
                    "storage_reads": {
                        "storage_read_get_requested_bytes": 100,
                        "storage_read_get_seconds": 1.0,
                        "storage_read_wire_bytes": None,
                        "native_jobs_completed": 10,
                        "native_active_jobs": 2,
                        "native_waiting_whole_scope_capacity": 2,
                    }
                },
            }
            after = copy.deepcopy(before)
            after["data"]["storage_reads"].update(
                storage_read_get_requested_bytes=160,
                storage_read_get_seconds=3.0,
                native_jobs_completed=12,
                native_active_jobs=1,
                native_waiting_whole_scope_capacity=1,
            )
            arm = {
                **spec,
                "started_monotonic": start,
                "completed_monotonic": start + 10,
                "wall_s": 10,
                "rows": rows,
                "effective_completed": 3,
                "effective_tasks_per_s": 1 + index * 0.1,
                "server_before": {"b": {"max_total_num_tokens": 100}},
                "costs_before": {"http://b": before},
                "costs_after": {"http://b": after},
            }
            self.arms.append(arm)
            (self.root / (spec["id"] + ".json")).write_text(json.dumps(arm))
            for offset, response in ((-0.1, before), (10.1, after)):
                self.raw.append(
                    {
                        "kind": "http",
                        "url": "http://b",
                        "trial": spec["id"],
                        "monotonic": start + offset,
                        "request": {
                            "kind": "capacity",
                            "context": {
                                "engine_id": "b",
                                "engine_epoch": "epoch",
                                "namespace": "weights",
                            },
                        },
                        "response": response,
                    }
                )
        write_records(self.root / "lifecycle-raw.jsonl", self.raw)

    def tearDown(self):
        self.directory.cleanup()

    def test_exact_windows_cpu_coverage_ready_and_all_completed_timing(self):
        resource = self.root / "resources.jsonl"
        write_records(
            resource,
            [
                resource_row(-1, 999),
                resource_row(0),
                resource_row(2),
                resource_row(8, 100),
                resource_row(9.5, 999),
                resource_row(11, 999),
            ],
        )
        process = self.root / "process.jsonl"
        write_records(process, [process_row(t) for t in (-1, 0, 2, 8, 10, 11)])
        result = costs.analyze([self.root], [resource], [process], gap=4)
        first = result["arms"][0]
        self.assertEqual(first["resources"]["samples"], 3)
        self.assertEqual(
            first["resources"]["metrics"]["b_host_used_tokens"]["sampled_peak"], 100
        )
        self.assertEqual(
            first["resources"]["metrics"]["b_gpu_resident_or_reserved_tokens"][
                "sampled_peak"
            ],
            80,
        )
        self.assertEqual(first["resources"]["b_gpu"]["busy_percent"]["mean"], 40)
        self.assertEqual(first["resource_coverage"]["observed_seconds"], 2)
        self.assertEqual(first["resource_coverage"]["gaps_s"], [6])
        engine = first["cpu"]["engine_b"]
        self.assertEqual(engine["observed_seconds"], 4)
        self.assertEqual(engine["observed_core_seconds"], 8)
        self.assertEqual(engine["average_cores_observed"], 2)
        self.assertAlmostEqual(
            engine["observed_core_seconds_per_effective_task"], 8 / 3
        )
        self.assertEqual(engine["state"], "partial")
        self.assertIsNone(first["cpu"]["router"]["observed_core_seconds"])
        self.assertEqual(
            (
                first["ready_total"],
                first["ready_same_owner"],
                first["ready_restore_miss"],
            ),
            (4, 3, 1),
        )
        self.assertEqual(first["ready_miss_fraction"], 1 / 3)
        self.assertEqual(first["ttft_ms"]["n"], 4)
        self.assertEqual(first["latency_ms"]["n"], 4)
        self.assertEqual(len(result["paired"]), 2)
        self.assertAlmostEqual(result["paired"][0]["throughput_percent"], 10)
        self.assertEqual(result["paired"][0]["ttft_ms_p95_percent"], 0)
        self.assertTrue(all(row["small_sample"] for row in result["paired_ci"]))

    def test_epoch_verified_counter_delta_never_differences_gauges(self):
        observed = costs.storage(
            self.arms[0], {self.schedule[0]["id"]: self.raw[:2]}, "http://b"
        )
        self.assertTrue(observed["epoch_verified"])
        self.assertEqual(observed["delta"]["storage_read_get_requested_bytes"], 60)
        self.assertEqual(observed["gauges"]["native_active_jobs"], [2, 1])
        self.assertNotIn("native_active_jobs", observed["delta"])
        self.assertNotIn("native_waiting_whole_scope_capacity", observed["delta"])
        self.assertIsNone(observed["wire_bytes"])
        self.assertIsNone(observed["delta"]["storage_read_wire_bytes"])

    def test_missing_or_changed_epoch_and_counter_reset_are_unknown(self):
        key = self.schedule[0]["id"]
        changed = copy.deepcopy(self.raw[:2])
        changed[1]["request"]["context"]["engine_epoch"] = "restarted"
        for evidence in ({}, {key: changed}):
            result = costs.storage(self.arms[0], evidence, "http://b")
            self.assertFalse(result["epoch_verified"])
            self.assertTrue(all(value is None for value in result["delta"].values()))
        reset_arm = copy.deepcopy(self.arms[0])
        reset_arm["costs_after"]["http://b"]["data"]["storage_reads"][
            "native_jobs_completed"
        ] = 1
        replies = copy.deepcopy(self.raw[:2])
        replies[1]["response"] = reset_arm["costs_after"]["http://b"]
        result = costs.storage(reset_arm, {key: replies}, "http://b")
        self.assertTrue(result["counter_reset"])
        self.assertTrue(all(value is None for value in result["delta"].values()))

    def test_process_restart_pid_reuse_and_negative_cpu_are_partial(self):
        samples = [
            process_row(0, user=0),
            process_row(1, user=2),
            process_row(2, created=2, user=3),
            process_row(3, created=2, user=1),
            process_row(4, created=2, user=2),
        ]
        result = costs.cpu(samples, 0, 4, 6, 2)["engine_b"]
        self.assertEqual(result["state"], "partial")
        self.assertEqual(result["observed_core_seconds"], 3)
        self.assertEqual(result["observed_seconds"], 2)
        self.assertEqual(len(result["unknown_intervals"]), 2)
        self.assertIsNone(result["full_arm_core_seconds"])

    def test_ordered_gzip_shards_bad_tail_and_missing_host_metric(self):
        one, two = self.root / "one.jsonl", self.root / "two.jsonl.gz"
        write_records(one, [resource_row(0, host=False)])
        write_records(two, [resource_row(2, host=False)])
        good = costs.analyze([self.root], [one, two])
        self.assertFalse(good["issues"])
        self.assertEqual(good["arms"][0]["resources"]["samples"], 2)
        self.assertNotIn("b_host_fraction", good["arms"][0]["resources"]["metrics"])
        bad = costs.analyze([self.root], [two, one])
        self.assertTrue(bad["issues"])
        with one.open("a") as output:
            output.write('{"monotonic":')
        self.assertTrue(costs.analyze([self.root], [one])["issues"])

    def test_incomplete_arm_and_invalid_interval_are_explicit(self):
        path = self.root / (self.schedule[-1]["id"] + ".json")
        path.unlink()
        result = costs.analyze([self.root])
        self.assertEqual(len(result["incomplete"]), 1)
        self.assertEqual(len(result["paired"]), 1)
        arm = self.arms[0]
        arm["completed_monotonic"] += 1
        (self.root / (arm["id"] + ".json")).write_text(json.dumps(arm))
        with self.assertRaisesRegex(ValueError, "invalid arm interval"):
            costs.analyze([self.root])

    def test_cli_writes_only_json_and_paired_csv_with_tail_regression(self):
        path = self.root / (self.schedule[1]["id"] + ".json")
        arm = json.loads(path.read_text())
        for row in arm["rows"]:
            row["generation"]["ttft_ms"] *= 2
        path.write_text(json.dumps(arm))
        output = self.root / "analysis"
        with patch(
            "sys.argv",
            [
                "analyze_lifecycle_costs.py",
                "--results",
                str(self.root),
                "--output",
                str(output),
            ],
        ):
            self.assertEqual(costs.main(), 0)
        self.assertEqual(
            {file.name for file in output.iterdir()},
            {"cost-resources.json", "paired.csv"},
        )
        result = json.loads((output / "cost-resources.json").read_text())
        fixed = next(
            pair for pair in result["paired"] if pair["candidate"] == "fixed_prefetch"
        )
        self.assertEqual(fixed["ttft_ms_p95_percent"], 100)
        self.assertAlmostEqual(fixed["throughput_percent"], 10)

    def test_nested_tp1_capacity_and_ambiguous_or_missing_pool(self):
        arm = self.arms[0]
        arm["server_before"]["b"] = {
            "tp_size": 1,
            "memory_usage": None,
            "internal_states": [{"memory_usage": {"token_capacity": 100}}],
        }
        path = self.root / (arm["id"] + ".json")
        path.write_text(json.dumps(arm))
        resource = self.root / "nested-resources.jsonl"
        write_records(resource, [resource_row(0), resource_row(2)])
        observed = costs.analyze([self.root], [resource])["arms"][0]
        self.assertEqual(observed["capacity"], 100)
        self.assertEqual(
            observed["resources"]["metrics"]["b_gpu_resident_or_reserved_tokens"][
                "sampled_peak"
            ],
            80,
        )
        for server in (
            None,
            {},
            {"internal_states": []},
            {"internal_states": [{}]},
            {"max_total_num_tokens": 100, "internal_states": [{}, {}]},
            {"tp_size": 2, "max_total_num_tokens": 100},
            {
                "internal_states": [
                    {"tp_size": 2, "memory_usage": {"token_capacity": 100}}
                ]
            },
        ):
            self.assertIsNone(costs.gpu_capacity(server))

    def test_missing_null_and_nonobject_storage_snapshots_remain_unknown(self):
        for value in (None, {}, "invalid"):
            arm = copy.deepcopy(self.arms[0])
            arm["costs_after"]["http://b"]["data"]["storage_reads"] = value
            replies = copy.deepcopy(self.raw[:2])
            replies[1]["response"] = arm["costs_after"]["http://b"]
            observed = costs.storage(arm, {arm["id"]: replies}, "http://b")
            self.assertFalse(observed["available"])
            self.assertIsNone(observed["delta"])
            self.assertIn("storage_reads", observed["reason"])
        arm["costs_after"]["http://b"]["data"].pop("storage_reads")
        self.assertIsNone(costs.storage(arm, {}, "http://b")["delta"])

    def test_bad_record_shapes_and_timestamps_are_issues(self):
        path = self.root / "bad-records.jsonl"
        write_records(
            path,
            [
                [],
                1,
                None,
                {"monotonic": "bad"},
                {"monotonic": None},
                {"monotonic": True},
                {"monotonic": float("nan")},
                {"monotonic": 1},
            ],
        )
        issues = []
        self.assertEqual(
            list(costs.records([path], issues, ordered=True)), [{"monotonic": 1}]
        )
        self.assertEqual(len(issues), 7)

    def test_cpu_capture_duration_filters_boundaries_and_marks_unknown(self):
        path = self.root / "cpu-durations.jsonl"
        samples = [process_row(0), process_row(2), process_row(9.5)]
        for row in samples:
            row["sample_elapsed_s"] = 1
        write_records(path, samples)
        result = costs.analyze([self.root], process_paths=[path])["arms"][0]
        coverage = result["cpu_sample_coverage"]
        self.assertFalse(coverage["capture_duration_unknown"])
        self.assertEqual(coverage["capture_boundary_error_bounds_s"], [1, 1])
        self.assertEqual(result["cpu"]["engine_b"]["observed_seconds"], 2)
        for row in samples:
            row.pop("sample_elapsed_s")
        write_records(path, samples)
        result = costs.analyze([self.root], process_paths=[path])["arms"][0]
        self.assertTrue(result["cpu_sample_coverage"]["capture_duration_unknown"])
        self.assertIsNone(
            result["cpu_sample_coverage"]["capture_boundary_error_bounds_s"]
        )
        self.assertIn(
            "capture start", result["cpu_sample_coverage"]["clock_definition"]
        )

    def test_bootstrap_seed_survives_archive_directory_change(self):
        first = costs.analyze([self.root])["paired_ci"]
        archive = self.root / "relocated"
        archive.mkdir()
        for path in self.root.glob("*.json*"):
            shutil.copy2(path, archive / path.name)
        second = costs.analyze([archive])["paired_ci"]
        strip_path = lambda rows: [
            {key: value for key, value in row.items() if key != "cell"} for row in rows
        ]
        self.assertEqual(strip_path(first), strip_path(second))


if __name__ == "__main__":
    unittest.main()

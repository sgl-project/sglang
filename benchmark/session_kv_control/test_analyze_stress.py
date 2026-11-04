"""Offline evidence integrity, all-request accounting and sampled resources."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_stress as analyze
from stress_bench import summarize


def scope():
    return [{"coverage": {"start_token": 0, "end_token": 64, "component": "full"}}]


def operation():
    return {"ref": {"context": {"engine_epoch": "epoch"}},
            "update": {"status": "completed", "quiescent": True}, "public_result": {"ready_scope": scope()}}


def transfer():
    return {"operation": operation()}


def generation(count=2):
    return {"ttft_ms": 10., "latency_ms": 20., "output_ids": [7]*count, "text": "same",
            "started_monotonic": 100., "completed_monotonic": 100.02,
            "meta_info": {"completion_tokens": count, "cached_tokens": 64,
                          "cached_tokens_details": {"device": 0, "host": 64, "storage": 0}}}


def claim(name, active):
    return {"claim_id": name, "owner_epoch": "epoch", "state": "active" if active else "released",
            "tombstone": not active, "effective_protection": "eviction_exclusion" if active else None,
            "remaining_ttl": 90 if active else 0, "actual_scope": scope(), "location": {"tier": "host"}}


def write(path, value):
    path.write_text(json.dumps(value)+"\n")


def checksums(directory):
    write(directory/"checksums.json", {p.name: analyze.sha256(p) for p in directory.iterdir()
                                     if p.is_file() and p.name != "checksums.json"})


def fixture(directory, invalid=False):
    directory.mkdir()
    manifest = {"args": {"command": "l2", "allow_small_working_set": False, "rounds": 3,
                         "length": 64, "hot_sessions": 1, "engine_b": "http://127.0.0.1:51803"},
                "capabilities": {"b": {"engine_epoch": "epoch"}}}
    arms, raw = [], []
    for block in range(3):
        for name in ("on_demand", "pin"):
            label = f"b{block}-{name}"
            prime = {"id": label+"-prime", "phase": "prime", "valid": True, "errors": [], "generation": generation(1),
                     "cache": {"device": 0, "cpu": 64, "storage": 0}, "publication": transfer(), "demote": transfer(),
                     "started_monotonic": 100.+block*5}
            pressure = {**copy.deepcopy(prime), "phase": "pressure", "id": label+"-pressure"}
            resumed = {**copy.deepcopy(prime), "phase": "resume", "case": 0, "id": label+"-resume",
                       "generation": generation(), "repeated_prefill_tokens": 0}
            if invalid and block == 0 and name == "pin":
                pressure["valid"] = False
                pressure["errors"] = ["DEMOTE control failure"]
                pressure["demote"]["operation"]["update"].update(status="failed", quiescent=True)
            arm = {"id": label, "block": block, "arm": name, "valid": all(r["valid"] for r in (prime, pressure, resumed)),
                   "rows": [prime, pressure, resumed], "errors": [], "planned_requests": 3, "wall_ms": 1000.,
                   "pressure_wall_ms": 500., "foreground_wall_ms": 100., "cleanup": []}
            if name == "pin":
                ident = label+"-claim"
                op = operation()
                op["public_result"]["claim"] = claim(ident, True)
                raw.append({"kind": "pin_acquire", "trial": label, "operation": op})
                arm["cleanup"] = [{"claim_id": ident, "valid": True, "query": claim(ident, False)}]
            arms.append(arm)
            raw += [{"kind": "generate", "origin": row["phase"], "trial": row["id"]} for row in arm["rows"]]
            write(directory/(label+".json"), arm)
    write(directory/"manifest.json", manifest)
    write(directory/"schedule.json", [{k: arm[k] for k in ("id", "block", "arm")} for arm in arms])
    summary = summarize(arms)
    for arm in summary["arms"]:
        arm.pop("retention_mode", None)  # Backward-compatible stored summaries.
    summary["notes"] = ["older wording"]
    write(directory/"summary.json", summary)
    (directory/"raw.jsonl").write_text("".join(json.dumps(row)+"\n" for row in raw))
    checksums(directory)
    return arms


class AnalysisChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_complete_six_arm_cell_and_old_summary_schema(self):
        directory = self.root/"formal"
        fixture(directory)
        cell = analyze.analyze_cell(directory)
        self.assertEqual(cell["state"], "completed", cell)
        self.assertTrue(cell["verified"], cell["verification_errors"])
        self.assertTrue(cell["valid"])
        self.assertEqual(cell["total_generation_requests_including_source"], 18)
        self.assertEqual(cell["groups"]["pin"]["completed_requests_per_s"], 3)
        self.assertEqual(cell["groups"]["pin"]["l2_foreground_requests_per_s"], 10)
        self.assertEqual(cell["proofs"]["cleanup_tombstones"], 3)

    def test_invalid_control_rows_remain_in_throughput_and_latency(self):
        directory = self.root/"invalid"
        fixture(directory, invalid=True)
        cell = analyze.analyze_cell(directory)
        self.assertTrue(cell["verified"], cell["verification_errors"])
        self.assertFalse(cell["valid"])
        group = cell["groups"]["pin"]
        self.assertEqual(group["completed_requests"], 9)
        self.assertEqual(group["invalid_requests"], 1)
        self.assertEqual(group["failed_control_rows"], 1)
        self.assertEqual(group["pressure_ttft_ms"]["n"], 3)
        self.assertEqual(group["output_tokens"], 12)

    def test_tampered_physical_file_fails_before_parsing(self):
        directory = self.root/"checksum"
        fixture(directory)
        (directory/"b0-pin.json").write_text("not JSON")
        cell = analyze.analyze_cell(directory)
        self.assertEqual(cell["state"], "verification_failed")
        self.assertIn("checksum mismatch", cell["verification_errors"][0])

    def test_recomputation_catches_changed_latency_even_after_rehash(self):
        directory = self.root/"latency"
        fixture(directory)
        path = directory/"b0-pin.json"
        arm = analyze.read_json(path)
        arm["rows"][2]["generation"]["ttft_ms"] = 15.
        write(path, arm)
        checksums(directory)
        cell = analyze.analyze_cell(directory)
        self.assertEqual(cell["state"], "completed")
        self.assertFalse(cell["verified"])
        self.assertTrue(any("resume_ttft_ms" in error for error in cell["verification_errors"]))

    def test_actual_tier_metadata_catches_rehashed_cache_tampering(self):
        directory = self.root/"tier"
        fixture(directory)
        path = directory/"b0-pin.json"
        arm = analyze.read_json(path)
        arm["rows"][2]["cache"].update(cpu=0, storage=64)
        write(path, arm)
        checksums(directory)
        cell = analyze.analyze_cell(directory)
        self.assertFalse(cell["verified"])
        self.assertTrue(any("tier counts disagree" in error for error in cell["verification_errors"]))

    def test_failed_cache_metadata_keeps_completed_request_in_analysis(self):
        directory = self.root/"missing-cache-details"
        arms = fixture(directory)
        arm = arms[1]
        row = arm["rows"][2]
        row["generation"]["meta_info"]["cached_tokens_details"] = None
        row["cache"] = {}
        row["valid"] = arm["valid"] = False
        row["errors"] = ["cached_tokens_details missing"]
        write(directory/(arm["id"]+".json"), arm)
        write(directory/"summary.json", summarize(arms))
        checksums(directory)
        cell = analyze.analyze_cell(directory)
        self.assertEqual(cell["state"], "completed")
        self.assertFalse(cell["verified"])
        self.assertFalse(cell["valid"])
        self.assertEqual(cell["groups"]["pin"]["completed_requests"], 9)
        self.assertEqual(cell["groups"]["pin"]["resume_ttft_ms"]["n"], 3)
        self.assertTrue(any("cached_tokens_details missing" in error for error in cell["verification_errors"]))


    def test_output_text_and_ids_are_checked_independently(self):
        directory = self.root/"outputs"
        fixture(directory)
        path = directory/"b0-pin.json"
        arm = analyze.read_json(path)
        arm["rows"][2]["generation"]["text"] = "changed"
        write(path, arm)
        checksums(directory)
        cell = analyze.analyze_cell(directory)
        self.assertFalse(cell["verified"])
        self.assertTrue(any("output IDs/text differ" in error for error in cell["verification_errors"]))

    def test_valid_transfer_requires_full_ready_scope(self):
        directory = self.root/"ready"
        fixture(directory)
        path = directory/"b0-pin.json"
        arm = analyze.read_json(path)
        arm["rows"][0]["publication"]["operation"]["public_result"]["ready_scope"][0]["coverage"]["end_token"] = 32
        write(path, arm)
        checksums(directory)
        cell = analyze.analyze_cell(directory)
        self.assertFalse(cell["verified"])
        self.assertTrue(any("failed transfer proof" in error for error in cell["verification_errors"]))

    def test_raw_pin_acquisition_requires_complete_active_claim(self):
        directory = self.root/"claim"
        fixture(directory)
        path = directory/"raw.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        next(row for row in rows if row["kind"] == "pin_acquire")["operation"]["public_result"]["claim"]["remaining_ttl"] = 0
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        checksums(directory)
        cell = analyze.analyze_cell(directory)
        self.assertFalse(cell["verified"])
        self.assertTrue(any("raw PIN proof" in error for error in cell["verification_errors"]))

    def test_l3_source_publication_count_and_cost_are_separate(self):
        directory = self.root/"l3"
        directory.mkdir()
        manifest = {"args": {"command": "l3", "allow_small_working_set": False, "blocks": 3,
                             "sessions": 1, "length": 64, "engine_b": "http://127.0.0.1:51803"}}
        arms = []
        for block in range(3):
            for name in ("on_demand", "prefetch"):
                label = f"b{block}-{name}"
                row = {"id": label+"-resume", "phase": "resume", "case": 0, "generation": generation(),
                       "cache": {"device": 0, "cpu": 64, "storage": 0}, "repeated_prefill_tokens": 0,
                       "valid": True, "errors": [], "demote": transfer(), "client_cycle_ms": 500.}
                if name == "prefetch":
                    row.update(prefetch=transfer(), prepared_tokens=64, prefetch_status="completed",
                               prefetch_ready_before_resume=True, prefetch_late=False)
                    row["prefetch"]["timing"] = {"completion_observed_monotonic": 90.}
                arm = {"id": label, "block": block, "arm": name, "rows": [row], "valid": True,
                       "planned_requests": 1, "wall_ms": 1000., "errors": []}
                arms.append(arm)
                write(directory/(label+".json"), arm)
        write(directory/"manifest.json", manifest)
        write(directory/"schedule.json", [{k: a[k] for k in ("id", "block", "arm")} for a in arms])
        write(directory/"summary.json", summarize(arms))
        write(directory/"source-publication.json", {"wall_ms": 2500., "rows": [{"valid": True,
              "generation": generation(1), "cache": {"device": 0, "cpu": 64, "storage": 0}, "publication": transfer()}]})
        raw = [{"kind": "generate", "origin": "resume"} for _ in range(6)]+[{"kind": "generate", "origin": "publication"}]
        (directory/"raw.jsonl").write_text("".join(json.dumps(row)+"\n" for row in raw))
        checksums(directory)
        cell = analyze.analyze_cell(directory)
        self.assertTrue(cell["verified"], cell["verification_errors"])
        self.assertEqual(cell["total_generation_requests_including_source"], 7)
        self.assertEqual(cell["source_publication"]["wall_ms"], 2500)
        self.assertEqual(cell["groups"]["prefetch"]["output_tokens_per_s"], 2)
        self.assertEqual(cell["proofs"]["ready_before_resume_checked"], 3)

    def test_gzip_json_and_raw_with_physical_checksums(self):
        directory = self.root/"packed"
        fixture(directory)
        for path in list(directory.glob("b*-*.json"))+[directory/"raw.jsonl"]:
            with gzip.open(str(path)+".gz", "wb") as packed:
                packed.write(path.read_bytes())
            path.unlink()
        checksums(directory)
        cell = analyze.analyze_cell(directory)
        self.assertTrue(cell["verified"], cell["verification_errors"])
        self.assertIn("raw.jsonl.gz", cell["checked_files"])

    def test_no_checksum_closure_means_live_partial_not_completed(self):
        directory = self.root/"partial"
        fixture(directory)
        (directory/"checksums.json").unlink()
        self.assertEqual(analyze.analyze_cell(directory)["state"], "incomplete")

    def test_failed_setup_is_separate_and_not_silently_dropped(self):
        directory = self.root/"failed"
        directory.mkdir()
        write(directory/"failure.json", {"error": "publication failed"})
        checksums(directory)
        cell = analyze.analyze_cell(directory)
        self.assertEqual(cell["state"], "failed_setup_or_run")
        self.assertEqual(cell["failure"]["error"], "publication failed")

    def test_sampled_gpu_and_owner_metrics_use_b_and_approximate_bounds(self):
        row = {"monotonic": 102., "gpu_csv": "0, UUID0, GPU0, 81920, 65000, 90, 22\n1, UUID1, GPU1, 81920, 66000, 95, 25\n",
               "http": [{"url": "http://127.0.0.1:51801/metrics", "body": 'sglang:token_usage{rank="0"} 0.1\n'},
                        {"url": "http://127.0.0.1:51803/metrics", "body": 'sglang:token_usage{rank="0"} 0.98\nsglang:hicache_host_used_tokens 650000\nsglang:hicache_host_total_tokens 693952\n'},
                        {"url": "http://127.0.0.1:52092/metrics", "body": "master_allocated_bytes 1234\nmaster_total_capacity_bytes 9876\n"}]}
        sample = analyze.resource_sample(row, "http://127.0.0.1:51803")
        peak = analyze.resource_peaks([sample])
        self.assertEqual(peak["metrics"]["b_gpu_active_token_usage_fraction"]["sampled_peak"], .98)
        self.assertEqual(peak["metrics"]["master_allocated_bytes"]["sampled_peak"], 1234)
        self.assertEqual(peak["gpu"]["1"]["busy_percent"]["sampled_peak"], 95)
        self.assertIn("not exact", peak["definition"])

    def test_gpu_residency_uses_capacity_minus_free_not_active_tokens(self):
        row = {"monotonic": 100, "http": [{"url": "http://b/metrics", "body":
            "sglang:token_usage 0.0236\nsglang:kv_used_tokens 8192\n"
            "sglang:kv_evictable_tokens 335872\nsglang:kv_available_tokens 2880\n"}]}
        sample = analyze.resource_sample(row, "http://b", capacity=346944)
        metrics = sample["metrics"]
        self.assertEqual(metrics["b_gpu_active_tokens"], 8192)
        self.assertEqual(metrics["b_gpu_evictable_tokens"], 335872)
        self.assertEqual(metrics["b_gpu_free_tokens"], 2880)
        self.assertEqual(metrics["b_gpu_resident_or_reserved_tokens"], 344064)
        self.assertAlmostEqual(metrics["b_gpu_resident_or_reserved_fraction"], 344064/346944)
        self.assertNotIn("b_gpu_used_tokens", metrics)
        self.assertNotIn("b_gpu_resident_or_reserved_tokens", analyze.resource_sample(row, "http://b")["metrics"])

    def test_reset_cost_is_separate_and_raw_windows_do_not_overlap(self):
        directory = self.root/"resets"
        arms = fixture(directory)
        previous_raw = [json.loads(line) for line in (directory/"raw.jsonl").read_text().splitlines()]
        raw, next_start = [], 1000.
        for arm in arms:
            flush_s = 60. if arm["arm"] == "on_demand" else 10.
            arm["wall_ms"] = (150.+flush_s)*1000
            raw += [{"kind": "http", "trial": arm["id"], "path": "/flush_cache?timeout=120",
                     "url": "http://127.0.0.1:51803", "elapsed_ms": flush_s*1000,
                     "started_monotonic": next_start, "monotonic": next_start+flush_s, "status_code": 200},
                    {"kind": "http", "trial": None, "path": "/v1/kv/capabilities",
                     "url": "http://127.0.0.1:51803", "elapsed_ms": 100.,
                     "started_monotonic": next_start+flush_s, "monotonic": next_start+flush_s+.1, "status_code": 200}]
            for index, row in enumerate(arm["rows"]):
                row["started_monotonic"] = next_start+flush_s+index+1
            raw.append({"kind": "arm_result", "id": arm["id"], "monotonic": next_start+150.+flush_s})
            write(directory/(arm["id"]+".json"), arm)
            next_start += 150.+flush_s+.1
        (directory/"raw.jsonl").write_text("".join(json.dumps(row)+"\n" for row in raw+previous_raw))
        write(directory/"summary.json", summarize(arms))
        checksums(directory)
        cell = analyze.analyze_cell(directory)
        self.assertTrue(cell["verified"], cell["verification_errors"])
        base, pin = cell["groups"]["on_demand"], cell["groups"]["pin"]
        self.assertAlmostEqual(base["sum_arm_wall_s"], 630.)
        self.assertAlmostEqual(pin["sum_arm_wall_s"], 480.)
        self.assertGreater(pin["completed_requests_per_s"], base["completed_requests_per_s"])
        self.assertEqual(base["sum_work_excluding_flush_s"], 450.)
        self.assertEqual(pin["sum_work_excluding_flush_s"], 450.)
        self.assertEqual(base["work_excluding_flush_completed_requests_per_s"], pin["work_excluding_flush_completed_requests_per_s"])
        self.assertEqual(base["work_excluding_flush_output_tokens_per_s"], pin["work_excluding_flush_output_tokens_per_s"])
        self.assertTrue(all(pair["work_excluding_flush_wall_delta_ms"] == 0 for pair in cell["paired"]))
        self.assertEqual(cell["experiment_resets"][0]["capability_refresh"]["elapsed_ms"], 100.)
        windows = cell["arm_windows"]
        self.assertEqual(windows[0]["start"], 1000.)
        self.assertEqual(windows[0]["work_start"], 1060.)
        self.assertEqual(windows[0]["end"], 1210.)
        self.assertTrue(all(first["end"] < second["start"] for first, second in zip(windows, windows[1:])))
        self.assertTrue(all(window["logging_validation_gap_ms"] is None for window in windows))
        report = analyze.report_text({"cells": [cell]})
        self.assertIn("initial flush seconds", report)
        self.assertIn("capability refresh seconds (included)", report)

    def test_analysis_preserves_all_cell_states_and_negative_deltas(self):
        fixture(self.root/"good")
        fixture(self.root/"bad", invalid=True)
        partial = self.root/"partial"
        partial.mkdir()
        (partial/"raw.jsonl").write_text("")
        with patch.object(analyze, "export_plot_csv", return_value={"generated": False}):
            report = analyze.run(self.root, self.root/"analysis")
        self.assertEqual(report["counts"]["completed_formal_cells"], 2)
        self.assertEqual(report["counts"]["total_generation_requests_including_source"], 36)
        self.assertIn("incomplete", {c["state"] for c in report["cells"]})
        self.assertTrue((self.root/"analysis"/"report.md").is_file())
        self.assertIn("bad", (self.root/"analysis"/"report.md").read_text())


if __name__ == "__main__":
    unittest.main()

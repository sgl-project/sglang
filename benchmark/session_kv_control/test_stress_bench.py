"""Sustained workload timing, pairing, failure accounting and PIN ownership."""
import argparse
import concurrent.futures
import threading
import time
import unittest
from unittest.mock import patch

import stress_bench as bench


def scope(n):
    return [{"coverage": {"start_token": 0, "end_token": n, "component": "full"}}]


def operation(n):
    return {"update": {"status": "completed", "quiescent": True},
            "public_result": {"ready_scope": scope(n)}}


class QuietJournal:
    def emit(self, *args, **kwargs):
        pass


class FakeHTTP:
    def __init__(self):
        self.journal = QuietJournal()
        self.url = "mock"
        self.calls = []
        self.lock = threading.Lock()

    def generate(self, tokens, salt, count, *, trial, seed, origin):
        sent = time.monotonic()
        with self.lock:
            self.calls.append((origin, salt, sent))
        return {"ttft_ms": 10, "latency_ms": 20, "started_monotonic": sent,
                "meta_info": {"completion_tokens": count, "cached_tokens": 64,
                              "cached_tokens_details": {"device": 0, "host": 64, "storage": 0}},
                "output_ids": [7]*count, "text": "same"}


class FakeControl:
    def __init__(self):
        self.http = FakeHTTP()
        self.context = {"engine_epoch": "epoch"}
        self.claims = {}
        self.delay = 0
        self.fail_pressure = False
        self.transfers = []

    def target(self, tokens, salt):
        return {"token_ids": tokens, "cache_salt": salt}

    def transfer(self, what, target, src, dst, trial, observe=None):
        start = time.monotonic()
        self.transfers.append((what, src, dst, trial))
        if self.fail_pressure and "pressure" in trial:
            raise bench.BenchError("pressure control failed")
        if what == "prefetch":
            time.sleep(self.delay)
        op = operation(len(target["token_ids"]))
        if observe:
            observe(op)
        return {"operation": op, "timing": {"started_monotonic": start,
                "completion_observed_monotonic": time.monotonic()}}

    def residency(self, target, tier, trial):
        return {"items": [{"descriptor": {"start_token": 0, "end_token": len(target["token_ids"]), "component": "full"}}]}

    def pin(self, target, claim, mode, revision, ttl, trial):
        active = mode != "release"
        result = {"claim_id": claim, "owner_epoch": "epoch", "state": "active" if active else "released",
                  "tombstone": not active, "effective_protection": "eviction_exclusion" if active else None,
                  "remaining_ttl": ttl if active else 0, "location": {"tier": "host"},
                  "actual_scope": scope(len(target["token_ids"]))}
        self.claims[claim] = result
        op = operation(0)
        op["public_result"]["claim"] = result
        return op

    def claim(self, claim, trial):
        return self.claims[claim]


def arguments(**extra):
    values = dict(length=64, concurrency=2, seed=11, run_id="unit", timeout_s=1,
                  pressure_prefixes=3, output_tokens=2, pin_ttl_s=90,
                  tool_wait_s=.02, lead_s=.01, passes=2, retention_mode="demote")
    return argparse.Namespace(**{**values, **extra})


def case(index=0):
    return {"prefix": list(range(64)), "suffix": [4, 5], "salt": "case-"+str(index)}


class StressChecks(unittest.TestCase):
    def test_pair_order_is_reproducible_and_complete(self):
        first = bench.paired_schedule(17, 8, "prefetch")
        self.assertEqual(first, bench.paired_schedule(17, 8, "prefetch"))
        self.assertNotEqual(first, bench.paired_schedule(18, 8, "prefetch"))
        for block in range(8):
            self.assertEqual({x["arm"] for x in first if x["block"] == block}, {"on_demand", "prefetch"})

    def test_bounded_workers_preserve_all_failures_and_order(self):
        lock, active, peak = threading.Lock(), 0, 0
        def work(item):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(.003)
                if item % 4 == 0:
                    raise ValueError("injected failure")
                return {"valid": True}
            finally:
                with lock:
                    active -= 1
        rows = bench.bounded_jobs(list(range(21)), 3, work)
        self.assertEqual([x["item"] for x in rows], list(range(21)))
        self.assertEqual(sum(not x["valid"] for x in rows), 6)
        self.assertLessEqual(peak, 3)
        self.assertGreater(peak, 1)

    def test_late_prefetch_does_not_postpone_generation_deadline(self):
        control, args = FakeControl(), arguments()
        control.delay = .15
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as controls:
            row = bench.prefetch_resume(args, control, case(), "late", True, controls)
        self.assertTrue(row["valid"], row)
        self.assertLess(row["deadline_dispatch_delay_ms"], 75)
        self.assertLess(row["generation"]["started_monotonic"], row["prefetch"]["timing"]["completion_observed_monotonic"])
        self.assertTrue(row["prefetch_late"])
        self.assertFalse(row["prefetch_ready_before_resume"])
        self.assertGreater(row["client_cycle_ms"], 140)
        self.assertIn("demote", row)

    def test_real_cold_metadata_allows_zero_hits_without_tier_details(self):
        control = FakeControl()
        original = control.http.generate
        def cold(*args, **kwargs):
            reply = original(*args, **kwargs)
            reply["meta_info"].update(cached_tokens=0, cached_tokens_details=None)
            return reply
        control.http.generate = cold
        row = bench.generation_row(control, [1]*64, "cold", 1, 0, "cold-prime", "prime")
        self.assertTrue(row["valid"], row)
        self.assertEqual([row["cache"][k] for k in ("device", "cpu", "storage")], [0, 0, 0])
        def unknown(*args, **kwargs):
            reply = cold(*args, **kwargs)
            reply["meta_info"]["cached_tokens"] = 64
            return reply
        control.http.generate = unknown
        row = bench.generation_row(control, [1]*64, "unknown", 1, 0, "unknown-prime", "prime")
        self.assertFalse(row["valid"])
        self.assertIn("generation", row)

    def test_l2_hot_and_pressure_salts_isolate_both_arms_and_rounds(self):
        args = arguments()
        salts = []
        for block in (0, 1):
            for arm in ("on_demand", "pin"):
                for kind in ("hot", "pressure"):
                    salts += [bench.l2_salt(args, {"block": block, "arm": arm}, kind, index) for index in range(3)]
        self.assertEqual(len(salts), len(set(salts)))
        self.assertEqual(salts[0], bench.l2_salt(args, {"block": 0, "arm": "on_demand"}, "hot", 0))

    def test_control_failure_keeps_completed_generation(self):
        control = FakeControl()
        row = bench.generation_row(control, list(range(64)), "salt", 1, 0, "pressure", "pressure")
        control.fail_pressure = True
        row = bench.materialize(row, control, control.target(list(range(64)), "salt"), 64, "pressure")
        self.assertFalse(row["valid"])
        self.assertEqual(row["generation"]["meta_info"]["completion_tokens"], 1)
        self.assertIn("pressure control failed", row["errors"][0])

    def test_pin_cleanup_happens_after_resume_exception(self):
        control, args = FakeControl(), arguments()
        original = control.http.generate
        def generate(*a, **kw):
            if kw["origin"] == "resume":
                raise RuntimeError("generation failed after acquire")
            return original(*a, **kw)
        control.http.generate = generate
        with patch.object(bench, "flush"):
            arm = bench.l2_arm(args, control, [case(), case(1)], (case()["prefix"], [5]),
                               {"block": 0, "arm": "pin", "id": "pin-arm"}, QuietJournal())
        self.assertFalse(arm["valid"])
        self.assertEqual(len(arm["cleanup"]), 2)
        self.assertTrue(all(x["valid"] for x in arm["cleanup"]))
        self.assertTrue(all(x["tombstone"] for x in control.claims.values()))
        summary = bench.summarize([arm])["arms"][0]
        self.assertEqual(summary["planned_requests"], 7)
        self.assertEqual(summary["completed_requests"], 5)
        self.assertEqual(summary["failed_requests"], 2)

    def test_l2_pressure_salts_are_fresh_between_arms(self):
        args, control = arguments(), FakeControl()
        arms = []
        for name in ("on_demand", "pin"):
            with patch.object(bench, "flush"):
                arms.append(bench.l2_arm(args, control, [case()], (case()["prefix"], [5]),
                            {"block": 0, "arm": name, "id": name}, QuietJournal()))
        salts = [{r["cache_salt"] for r in a["rows"] if r.get("phase") == "pressure"} for a in arms]
        self.assertFalse(salts[0] & salts[1])

    def test_l3_same_client_schedule_and_no_per_request_flush(self):
        args, control = arguments(tool_wait_s=0, lead_s=0), FakeControl()
        spec = {"block": 0, "arm": "on_demand", "id": "base", "dataset_order": [3, 1, 2, 0]}
        with patch.object(bench, "flush") as flush:
            arm = bench.l3_arm(args, control, [case(i) for i in range(4)], spec, QuietJournal())
        self.assertTrue(arm["valid"], arm)
        self.assertEqual(flush.call_count, 1)
        self.assertEqual([(r["passage"], r["case"]) for r in arm["rows"] if r["client"] == 0], [(0, 3), (0, 2), (1, 3), (1, 2)])
        self.assertTrue(all("demote" in r for r in arm["rows"]))

    def test_natural_l3_preserves_gpu_cache_without_demote(self):
        args, control = arguments(tool_wait_s=0, lead_s=0, retention_mode="natural"), FakeControl()
        spec = {"block": 0, "arm": "prefetch", "id": "natural", "dataset_order": [0, 1]}
        with patch.object(bench, "flush"):
            arm = bench.l3_arm(args, control, [case(0), case(1)], spec, QuietJournal())
        self.assertTrue(arm["valid"], arm)
        self.assertEqual(len(control.transfers), 4)
        self.assertTrue(all(what == "prefetch" for what, _, _, _ in control.transfers))
        self.assertTrue(all("demote" not in row for row in arm["rows"]))

    def test_natural_l2_pressure_copies_host_without_explicit_reclamation(self):
        args, control = arguments(retention_mode="natural"), FakeControl()
        with patch.object(bench, "flush"):
            arm = bench.l2_arm(args, control, [case()], (case()["prefix"], [5]),
                               {"block": 0, "arm": "pin", "id": "natural-pin"}, QuietJournal())
        self.assertTrue(arm["valid"], arm)
        pressure = [x for x in control.transfers if "pressure" in x[3]]
        self.assertEqual(len(pressure), args.pressure_prefixes)
        self.assertTrue(all(what == "publish" and src == {"tier": "gpu"} and dst == {"tier": "host"}
                            for what, src, dst, _ in pressure))
        self.assertTrue(all("host_copy" in row and "demote" not in row
                            for row in arm["rows"] if row["phase"] == "pressure"))
        self.assertTrue(any(what == "demote" and "prime" in trial for what, _, _, trial in control.transfers))
        self.assertTrue(all(receipt["valid"] for receipt in arm["cleanup"]))

    def test_retention_mode_cli_default_and_natural_option(self):
        common = ["--engine-a", "a", "--engine-b", "b", "--model", "m", "--output", "o",
                  "--host-capacity-tokens", "100"]
        for mode in ("l2", "l3"):
            self.assertEqual(bench.parser().parse_args([mode]+common).retention_mode, "demote")
            self.assertEqual(bench.parser().parse_args([mode]+common+["--retention-mode", "natural"]).retention_mode, "natural")

    def test_summary_denominator_tier_counts_and_throughput(self):
        control = FakeControl()
        row = bench.generation_row(control, [1]*64, "salt", 2, 0, "one", "resume")
        row.update(case=0, repeated_prefill_tokens=0)
        arm = {"block": 0, "arm": "on_demand", "valid": False, "planned_requests": 4,
               "rows": [row, {"valid": False, "errors": ["broken"]}], "wall_ms": 2000, "errors": []}
        summary = bench.summarize([arm])["arms"][0]
        self.assertEqual(summary["completed_requests"], 1)
        self.assertEqual(summary["failed_requests"], 3)
        self.assertEqual(summary["unattempted_requests"], 2)
        self.assertEqual(summary["achieved_completed_requests_per_s"], .5)
        self.assertEqual(summary["achieved_output_tokens_per_s"], 1)
        self.assertEqual(summary["resume_tier_tokens"], {"device": 0, "cpu": 64, "storage": 0})

    def test_output_checks_ids_and_text_and_marks_arm_invalid(self):
        signature = {"output_ids": [1], "text": "one"}
        self.assertTrue(bench.output_equal(signature, signature))
        self.assertFalse(bench.output_equal(signature, {"output_ids": [2], "text": "one"}))
        self.assertFalse(bench.output_equal(signature, {"output_ids": [1], "text": "different"}))
        self.assertFalse(bench.output_equal({}, {}))
        arm = {"rows": [{"phase": "resume", "case": 0, "generation": {"output_ids": [2], "text": "one"},
                         "valid": True, "errors": []}], "valid": True}
        bench.validate_outputs(arm, {0: signature})
        self.assertFalse(arm["valid"])
        self.assertFalse(arm["rows"][0]["valid"])

    def test_parity_allows_only_small_automatic_pool_difference(self):
        a, b = FakeControl(), FakeControl()
        a.cap = {"engine_id": "a"}
        b.cap = {"engine_id": "b"}
        a.executor = b.executor = {}
        a.backend_namespace = b.backend_namespace = "same"
        config = {"mem_fraction_static": .8, "max_total_tokens": None, "max_total_num_tokens": 346688,
                  "enable_hierarchical_cache": True, "hicache_storage_backend": "mooncake"}
        metadata = {"a": {"server": config}, "b": {"server": {**config, "max_total_num_tokens": 346944}}}
        checked = bench.parity(a, b, metadata, .8)
        self.assertEqual(checked["automatic_pool_difference"], [346688, 346944])
        metadata["b"]["server"]["max_total_num_tokens"] = 350688
        with self.assertRaisesRegex(bench.BenchError, "configuration mismatch"):
            bench.parity(a, b, metadata, .8)
        metadata["b"]["server"] = {**config, "max_total_tokens": 32768}
        with self.assertRaisesRegex(bench.BenchError, "configuration mismatch"):
            bench.parity(a, b, metadata, .8)


if __name__ == "__main__":
    unittest.main()

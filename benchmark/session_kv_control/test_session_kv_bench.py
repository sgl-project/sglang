"""Accounting, real-recording replay, and HTTP lifecycle regression checks."""
import argparse
import copy
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import session_kv_bench as bench
from bench_inputs import load_trace, estimate_kv_bytes_per_token
from bench_statistics import PrefetchCostEMA, bootstrap_mean_ci


class State:
    def __init__(self, engine):
        self.engine = engine
        self.ops = {}
        self.prefetched = set()
        self.requests = []
        self.terminal_status = "completed"
        self.force_device_hit = False
        self.claims = {}
        self.host = set()
        self.prefetch_delay_s = 0
        self.pending = {}
        self.cancel_stays_unknown = False

    def operation(self, command):
        env = command["envelope"]
        action = env["actions"][0]
        target = action["payload"]["target"]
        n = len(target["token_ids"])
        salt = target["cache_salt"]
        state = "missing" if salt.endswith("-missing") else self.terminal_status
        kind = action["action_type"]
        destination = action["payload"].get("destination", action["payload"].get("dst", {}))
        if destination.get("tier") == "host" and state == "completed":
            self.host.add(salt)
        legacy_restore = kind == "sglang.copy_to" and action["payload"].get("src", {}).get("tier") == "storage"
        if (kind == "sglang.prefetch" or legacy_restore) and state == "completed":
            self.prefetched.add(salt)
        ready = [{"coverage": {"start_token": 0, "end_token": n, "component": "full"},
                  "status": state, "completed_steps": ["destination_ready"]}] if state == "completed" else []
        op = {"ref": {"context": command["context"], "message_id": env["message_id"],
                      "action_id": action["action_id"], "operation_id": env["message_id"]},
              "action_type": kind, "action_version": "1.0", "sequence": 1,
              "update": {"status": state, "quiescent": True, "completed_steps": ["destination_ready"], "target_outcomes": ready},
              "public_result": {"ready_scope": ready, "owner_epoch": "epoch"}}
        if kind == "sglang.pin":
            payload = action["payload"]
            active = payload["mode"] != "release"
            claim = {"claim_id": payload["claim_id"], "owner_epoch": "epoch", "state": "active" if active else "released",
                     "tombstone": not active, "effective_protection": "eviction_exclusion" if active else None,
                     "remaining_ttl": payload.get("ttl_seconds", 0), "actual_scope": ready,
                     "location": {"tier": "host", "engine_id": self.engine}}
            op["public_result"].update(ready_scope=[], accepted_scope=ready, claim=claim)
            self.claims[payload["claim_id"]] = claim
        self.ops[env["message_id"]] = op
        if kind == "sglang.prefetch" and self.prefetch_delay_s:
            self.pending[env["message_id"]] = (time.monotonic()+self.prefetch_delay_s, copy.deepcopy(op), salt)
            op["update"].update(status="running", quiescent=False, completed_steps=[], target_outcomes=[])
            op["public_result"]["ready_scope"] = []
            self.prefetched.discard(salt)
            self.host.discard(salt)
        return op

    def query(self, message):
        if message in self.pending and time.monotonic() >= self.pending[message][0]:
            _, operation, salt = self.pending.pop(message)
            self.ops[message] = operation
            self.host.add(salt)
            self.prefetched.add(salt)
        return self.ops[message]


def handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def reply(self, value, code=200):
            body = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/model_info":
                self.reply({"model_path": "mock-model"})
            elif self.path == "/server_info":
                self.reply({"model_path": "mock-model", "random_seed": 0, "page_size": 1})
            elif self.path == "/get_hicache_l3_cache_layout":
                self.reply({"enabled": False})
            elif self.path == "/v1/kv/capabilities":
                self.reply({"success": True, "data": {"enabled": True, "engine_id": state.engine,
                    "engine_epoch": "epoch", "namespace": "weights", "executor": {
                        "actions": ["sglang.copy_to", "sglang.move_to"], "layout_id": "layout"},
                    "public_protocol": {"actions": {"sglang.prefetch": ["1.0"], "sglang.demote": ["1.0"], "sglang.pin": ["1.0"]},
                        "paths": {"sglang.prefetch": [["storage", "host"]], "sglang.demote": [["gpu", "host"]]},
                        "backend_namespace": "mock-storage-domain", "backend_namespace_selection": True}}})
            else:
                self.reply({}, 404)

        def do_POST(self):
            command = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append((self.path, command))
            if self.path == "/generate":
                count = command["sampling_params"]["max_new_tokens"]
                source = count == 1 or command["cache_salt"].startswith("warmup-")
                hit = 0 if source else 64
                cache = {"device": 0, "host": 0, "storage": 0}
                cache["host" if command["cache_salt"] in state.prefetched else "storage"] = hit
                if not source and state.force_device_hit:
                    cache = {"device": hit, "host": 0, "storage": 0}
                payload = {"text": "test", "output_ids": [123]*count, "meta_info": {
                    "completion_tokens": count, "prompt_tokens": len(command["input_ids"]),
                    "cached_tokens": hit, "cached_tokens_details": cache,
                    "finish_reason": {"type": "length", "length": count}}}
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"text":"","meta_info":{"completion_tokens":0}}\n\n')
                self.wfile.flush()
                time.sleep(.005)
                self.wfile.write(b"data: " + json.dumps(payload).encode()+b"\n\ndata: [DONE]\n\n")
                self.wfile.flush()
                return
            if command["kind"] == "submit":
                data = {"results": [state.operation(command)]}
            elif command["kind"] == "residency":
                target = command["target"]
                present = target["cache_salt"] in state.host and command["location"]["tier"] == "host"
                data = {"items": [{"descriptor": {"start_token": 0, "end_token": len(target["token_ids"]), "component": "full"}}] if present else [], "missing": []}
            elif command["kind"] == "ack":
                data = {"acknowledged": True}
            elif command["kind"] == "query":
                data = {"results": [state.query(command["message_id"])]}
            elif command["kind"] == "cancel":
                data = state.ops[command["ref"]["message_id"]]
                if bench.status(data) in bench.ACTIVE:
                    data["update"].update(status="cancelling" if state.cancel_stays_unknown else "cancelled", quiescent=not state.cancel_stays_unknown)
                    if not state.cancel_stays_unknown:
                        state.pending.pop(command["ref"]["message_id"], None)
            elif command["kind"] == "claim":
                data = state.claims[command["claim_id"]]
            else:
                self.reply({}, 400)
                return
            self.reply({"success": True, "data": data})
    return Handler


class BenchmarkChecks(unittest.TestCase):
    def test_local_trace_provenance_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "trace.jsonl")
            record = {"id": "application-step", "token_prefix": list(range(64)), "tool_suffix": [1, 2], "tool_wait_s": .2,
                      "declared_origin": "user_supplied"}
            source.write_text(json.dumps(record)+"\n")
            first = load_trace(source, directory)[0]
            self.assertEqual(first["provenance"]["source_line"], 1)
            self.assertEqual(first["provenance"]["source_file"], str(source.resolve()))
            self.assertFalse(first["provenance"]["origin_independently_verified"])
            record["tool_wait_s"] = .3
            source.write_text(json.dumps(record)+"\n")
            self.assertNotEqual(first["provenance"]["source_file_sha256"], load_trace(source, directory)[0]["provenance"]["source_file_sha256"])
            source.write_text(json.dumps(record)+"\n"+json.dumps(record)+"\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_trace(source, directory)
            record["model"] = "/different/model"
            source.write_text(json.dumps(record)+"\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                load_trace(source, directory)

    def test_ema_separates_contended_costs_and_paired_bootstrap(self):
        cost = PrefetchCostEMA(alpha=.5, initial_ms=100, safety=2)
        self.assertEqual(cost.choose("scope", 1)["lead_s"], .2)
        cost.observe("scope", 400, 1000, contaminated=True)
        self.assertEqual(cost.choose("scope", 1)["sample_n"], 0)
        cost.observe("scope", 200, 1000)
        cost.observe("scope", 100, 1000)
        choice = cost.choose("scope", .2)
        self.assertEqual(choice["predicted_completion_ms"], 150)
        self.assertTrue(choice["window_insufficient"])
        self.assertEqual(choice["lead_s"], .2)
        ci = bootstrap_mean_ci([-10, -20, 30], seed=17, resamples=1000)
        self.assertEqual(ci, bootstrap_mean_ci([-10, -20, 30], seed=17, resamples=1000))
        self.assertEqual(ci["n_pairs"], 3)

    def test_pin_protection_is_not_transfer_readiness(self):
        scope = [{"coverage": {"start_token": 0, "end_token": 64, "component": "full"}}]
        claim = {"claim_id": "claim", "owner_epoch": "epoch", "state": "active", "tombstone": False,
                 "effective_protection": "eviction_exclusion", "remaining_ttl": 10,
                 "actual_scope": scope, "location": {"tier": "host"}}
        op = {"update": {"status": "completed", "quiescent": True},
              "public_result": {"ready_scope": [], "accepted_scope": scope, "claim": claim}}
        bench.require_pin_success(op, 64, "claim", "epoch")
        claim["remaining_ttl"] = 0
        with self.assertRaises(bench.BenchError):
            bench.require_pin_success(op, 64, "claim", "epoch")
        with self.assertRaises(bench.BenchError):
            bench.require_pin_released(claim, "claim", "epoch")
        claim.update(state="released", tombstone=True, effective_protection=None)
        bench.require_pin_released(claim, "claim", "epoch")
    def test_accounting_and_scope_union(self):
        counts = bench.cache_details({"cached_tokens": 64, "cached_tokens_details": {"device": 0, "host": 32, "storage": 32}})
        self.assertEqual(counts["cpu"], 32)
        self.assertEqual(bench.cache_details({"cached_tokens": 0})["storage"], 0)
        with self.assertRaises(bench.BenchError):
            bench.cache_details({"cached_tokens": 64})
        with self.assertRaises(bench.BenchError):
            bench.cache_details({"cached_tokens": 64, "cached_tokens_details": {"device": 0, "host": 64, "storage": 64}})
        spans = [{"coverage": {"start_token": a, "end_token": b, "component": c}}
                 for a, b, c in [(0, 32, "full"), (16, 64, "full"), (0, 64, "full"), (0, 128, "swa")]]
        self.assertEqual(bench.intervals_size(spans), 64)

    def test_required_cache_details_accept_only_explicit_integer_zero(self):
        for meta in ({"cached_tokens": 0},
                     {"cached_tokens": 0, "cached_tokens_details": None}):
            counts = bench.cache_details(meta, require=True)
            self.assertEqual([counts[t] for t in ("device", "cpu", "storage")], [0, 0, 0])
            self.assertFalse(counts["details_present"])
            self.assertEqual(counts["provenance"], "explicit_zero_total")
        for meta in ({}, {"cached_tokens": None}, {"cached_tokens": False},
                     {"cached_tokens": True}, {"cached_tokens": 0.0},
                     {"cached_tokens": "0"}, {"cached_tokens": 1},
                     {"cached_tokens": -1}):
            with self.subTest(meta=meta), self.assertRaises(bench.BenchError):
                bench.cache_details(meta, require=True)

    def test_optional_cache_details_preserve_legacy_zero_behavior(self):
        expected = {"device": 0, "cpu": 0, "storage": 0, "details_present": False}
        for meta in ({}, {"cached_tokens": 0}, {"cached_tokens": False},
                     {"cached_tokens": 0.0}):
            self.assertEqual(bench.cache_details(meta), expected)

    def test_sse_truncation_and_comments(self):
        events = list(bench.sse_objects([b":keepalive\n", b'data: {"text":\n', b'data: "x"}\n', b"\n", b"data: [DONE]\n", b"\n"]))
        self.assertEqual(events, [{"text": "x"}])
        with self.assertRaises(bench.BenchError):
            list(bench.sse_objects([b'data: {"text":"x"}']))

    def test_replay_recorded_real_sse(self):
        recording = Path(__file__).with_name("real-parser-smoke.jsonl")
        if not recording.exists():
            self.skipTest("real SGLang parser recording not bundled")
        records = [json.loads(x) for x in recording.read_text().splitlines()]
        generation = next(x for x in records if x["kind"] == "generate")
        wire = [line for frame in generation["events"]
                for line in ["data: " + json.dumps(frame["event"]), ""]] + ["data: [DONE]", ""]
        events = list(bench.sse_objects(wire))
        self.assertEqual(events[-1]["meta_info"]["completion_tokens"], 3)
        self.assertEqual(len(events[-1]["output_ids"]), 3)

    def test_failed_trials_remain_in_primary_statistics(self):
        def row(group, valid, latency):
            return {"kind": "trial", "id": group, "pair": "pair", "group": group,
                "prefix_tokens": 64, "tool_wait_s": .05, "lead_s": .05,
                "valid": valid, "invalid_reasons": [] if valid else ["control failed"],
                "resume": {"ttft_ms": latency, "latency_ms": latency+10},
                "source": {"latency_ms": 10}, "complete_cycle_ms": latency+20,
                "resume_cache": {"device": 0, "cpu": 0, "storage": 64}, "repeated_prefill_tokens": 0}
        report = bench.summarize([row("on_demand", True, 10), row("prefetch", False, 20)])
        self.assertEqual(report["attempts"], 2)
        self.assertEqual(len(report["failures"]), 1)
        prefetch = next(v for k, v in report["groups"].items() if "prefetch" in k)
        self.assertEqual(prefetch["resumed_ttft_ms"]["p50"], 20)
        pair = next(iter(report["paired"].values()))
        self.assertEqual(pair["ttft_delta_ms"]["p50"], 10)
        self.assertEqual(pair["valid_pairs"], 0)

    def test_two_http_engines_and_negative_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = bench.Journal(Path(directory, "raw.jsonl"))
            servers, controls, states = [], [], []
            for name in ("a", "b"):
                state = State(name)
                server = ThreadingHTTPServer(("127.0.0.1", 0), handler(state))
                threading.Thread(target=server.serve_forever, daemon=True).start()
                servers.append(server)
                states.append(state)
                http = bench.HTTP(f"http://127.0.0.1:{server.server_port}", journal, 2)
                controls.append(bench.Control(http, "caller", 1, 2))
            args = argparse.Namespace(run_id="test", seed=1729, pin_ttl_s=30,
                timeout_s=2, storage_backend="mooncake", cold_mode="salt", output_tokens=2)
            try:
                for group in ("on_demand", "prefetch", "demote", "pin_demote", "pin_demote_prefetch", "unused", "missing", "cancel", "route_change"):
                    spec = {"id": group, "pair": "pair", "prefix_tokens": 64,
                            "tool_wait_s": .03, "lead_s": .03, "group": group}
                    result = bench.one_trial(args, *controls, (list(range(64)), [64, 65]), spec, journal)
                    self.assertTrue(result["valid"], result)
                    if group == "prefetch":
                        self.assertEqual(result["resume_cache"]["cpu"], 64)
                        self.assertTrue(result["prefetch_ready_by_resume"])
                        self.assertGreater(result["resume"]["ttft_ms"], 4)
                    elif group == "unused":
                        self.assertEqual(result["unused_prepared_tokens"], 64)
                    elif group == "missing":
                        self.assertEqual(result["prefetch_outcome_status"], "missing")
                # A baseline GPU hit is retained and rejected.
                states[1].force_device_hit = True
                spec = {"id": "bad-baseline", "pair": "pair", "prefix_tokens": 64,
                        "tool_wait_s": .01, "lead_s": .01, "group": "on_demand"}
                bad = bench.one_trial(args, *controls, (list(range(64)), [64]), spec, journal)
                self.assertFalse(bad["valid"])
                self.assertEqual(bad["resume_cache"]["device"], 64)
                self.assertIn("zero device/CPU", " ".join(bad["invalid_reasons"]))
                states[1].force_device_hit = False
                # Legacy fallback follows capabilities and uses src/dst fields.
                controls[1].public["actions"].pop("sglang.prefetch")
                spec.update(id="legacy-prefetch", group="prefetch")
                legacy = bench.one_trial(args, *controls, (list(range(64)), [64]), spec, journal)
                self.assertTrue(legacy["valid"], legacy)
                self.assertEqual(legacy["background"]["prefetch"]["action"], "sglang.copy_to")
                self.assertTrue(any(c.get("kind") == "ack" for _, c in states[0].requests))
                release = [c for _, c in states[0].requests if c.get("kind") == "submit"
                           and c["envelope"]["actions"][0]["payload"].get("mode") == "release"]
                self.assertEqual(len(release), 2)
                states[1].prefetch_delay_s = .08
                controls[1].public["actions"]["sglang.prefetch"] = ["1.0"]
                spec.update(id="late-control", group="prefetch", tool_wait_s=.01, lead_s=.01)
                late = bench.one_trial(args, *controls, (list(range(64)), [64]), spec, journal)
                self.assertTrue(late["valid"], late)
                self.assertFalse(late["prefetch_ready_before_resume"])
                self.assertTrue(late["prefetch_usage_ambiguous"])
                self.assertIsNone(late["unused_prepared_tokens"])
                timing = late["background"]["prefetch"]["timing"]
                self.assertGreater(timing["completion_elapsed_ms"], 50)
                self.assertGreaterEqual(timing["completion_interval_ms"][1], timing["completion_interval_ms"][0])
                states[1].prefetch_delay_s = 10
                states[1].cancel_stays_unknown = True
                controls[1].timeout = .01
                with self.assertRaisesRegex(bench.BenchError, "non-quiescent"):
                    controls[1].transfer("prefetch", controls[1].target(list(range(64)), "unknown-drain"),
                        {"tier": "storage", "backend": "mooncake"}, {"tier": "host"}, "unknown-drain")
                controls[1].timeout = 2
                states[1].prefetch_delay_s = 0
                states[1].cancel_stays_unknown = False
                run_args = argparse.Namespace(engine_a=controls[0].http.url, engine_b=controls[1].http.url,
                    model="mock-model", output=str(Path(directory, "full-run")), seed=1729, run_id="mock-run",
                    prefix_lengths=[64], tool_waits=[.03], lead_times=[.03], repetitions=1,
                    warmup_repetitions=1, groups="on_demand,prefetch", negative="", output_tokens=2,
                    storage_backend="mooncake", cold_mode="salt", poll_ms=1, timeout_s=2,
                    pin_ttl_s=30, api_key_env=None, router=None, reference_url=None)
                with patch.object(bench, "build_tokens", return_value=(list(range(64)), [64])), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(bench.run(run_args), 0)
                summary = json.loads(Path(run_args.output, "summary.json").read_text())
                self.assertEqual(summary["attempts"], 2)
                self.assertEqual(summary["valid_attempts"], 2)
                self.assertTrue(Path(run_args.output, "manifest.json").exists())
                trace = Path(directory, "input.jsonl")
                trace.write_text(json.dumps({"id": "trace-step", "token_prefix": list(range(64)), "tool_suffix": [64],
                    "tool_wait_s": .05, "declared_origin": "synthetic_fixture"})+"\n")
                profile = Path(directory, "bandwidth.json")
                profile.write_text(json.dumps({"link": "mock", "bandwidth_mbps": 1000, "source": "unverified_fixture"}))
                run_args.output = str(Path(directory, "trace-run"))
                run_args.run_id = "trace-run"
                run_args.trace_jsonl = str(trace)
                run_args.lead_policy = "ema"
                run_args.ema_initial_ms = 5
                run_args.ema_safety_factor = 1.5
                run_args.cost_history_jsonl = str(Path(directory, "full-run", "raw.jsonl"))
                run_args.background_concurrency = 2
                run_args.background_prefix_tokens = 16
                run_args.background_output_tokens = 3
                run_args.host_pressure_prefixes = 1
                run_args.host_pressure_tokens = 64
                run_args.bandwidth_profile_json = str(profile)
                # Restore public path after explicit legacy coverage above.
                controls[1].public["actions"]["sglang.prefetch"] = ["1.0"]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(bench.run(run_args), 0)
                report = json.loads(Path(run_args.output, "summary.json").read_text())
                self.assertEqual(report["valid_attempts"], 2)
                raw = [json.loads(x) for x in Path(run_args.output, "raw.jsonl").read_text().splitlines()]
                attempts = [x for x in raw if x["kind"] == "trial"]
                self.assertEqual({x["input_provenance"]["source_line"] for x in attempts}, {1})
                self.assertTrue(all(x["background_load"]["started_requests"] >= 2 for x in attempts))
                self.assertTrue(all(x["host_pressure"] for x in attempts))
                self.assertTrue(all(x["lead_selection"]["lead_s"] <= .05 for x in attempts))
                manifest = json.loads(Path(run_args.output, "manifest.json").read_text())
                self.assertFalse(manifest["load"]["bandwidth_profile"]["applied_by_harness"])
            finally:
                for server in servers:
                    server.shutdown()
                    server.server_close()
                journal.file.close()


if __name__ == "__main__":
    unittest.main()

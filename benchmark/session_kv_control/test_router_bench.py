"""Loopback HTTP contract tests; no SGLang/Router process or GPU is launched."""
import argparse
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.request
from unittest.mock import patch

import router_bench as rb
from session_kv_bench import BenchError, Control, HTTP, Journal


class Engine:
    def __init__(self, name, storage):
        self.name, self.storage = name, storage
        self.epoch, self.local, self.requests = 0, {}, []

    def caps(self):
        return {"enabled": True, "engine_id": self.name, "engine_epoch": f"epoch-{self.name}-{self.epoch}",
            "namespace": "weights", "weight_version": "default", "cache_salt_encoding": "test-encoding",
            "executor": {"layout_id": "full:p64", "components": ["full"], "actions": ["sglang.copy_to"],
                "directions": {"copy_to": [["gpu","storage"],["storage","host"]], "page_alignment": 64}},
            "public_protocol": {"actions": {"sglang.prefetch": ["1.0"]}, "paths": {"sglang.prefetch": [["storage","host"]]},
                "prefix": {"page_size": 64}, "backend_namespace": "shared-storage", "backend_namespace_selection": True}}


def engine_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def reply(self, value, code=200, content_type="application/json"):
            raw = value.encode() if isinstance(value, str) else json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/v1/kv/capabilities":
                self.reply({"success": True, "data": state.caps()})
            elif self.path == "/model_info":
                self.reply({"model_path": "mock-model", "tokenizer_path": "mock-model"})
            elif self.path == "/server_info":
                self.reply({"model_path": "mock-model", "random_seed": 0, "page_size": 64, "context_length": 1024})
            elif self.path == "/get_hicache_l3_cache_layout":
                self.reply({"enabled": False})
            else:
                self.reply({}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append((self.path, copy.deepcopy(body)))
            if self.path.startswith("/flush_cache"):
                state.local.clear()
                state.epoch += 1
                self.reply("Cache flushed.\n", content_type="text/plain")
                return
            if self.path == "/generate":
                if "kv_context" in body:
                    self.reply({"error": "Router-only metadata leaked into Engine"}, 400)
                    return
                count, salt = body["sampling_params"]["max_new_tokens"], body["cache_salt"]
                cached = state.local.get(salt)
                tiers = {"device": 0, "host": 0, "storage": 0}
                if cached:
                    tiers[cached[0]] = len(cached[1])
                elif salt in state.storage:
                    tiers["storage"] = len(state.storage[salt])
                event = {"text": "output", "output_ids": [77]*count, "meta_info": {
                    "completion_tokens": count, "finish_reason": {"type": "length", "length": count},
                    "prompt_tokens": len(body["input_ids"]), "cached_tokens": sum(tiers.values()), "cached_tokens_details": tiers}}
                state.local[salt] = ("device", body["input_ids"])
                self.reply("data: "+json.dumps(event)+"\n\ndata: [DONE]\n\n", content_type="text/event-stream")
                return
            kind = body.get("kind")
            if kind == "residency":
                target, tier = body["target"], body["location"]["tier"]
                entry = state.local.get(target["cache_salt"])
                items = [{"descriptor": {"start_token": 0, "end_token": len(entry[1]), "component": "full"}}] if entry and entry[0] == ("device" if tier == "gpu" else tier) else []
                value = {"items": items, "missing": []}
            elif kind == "submit":
                action = body["envelope"]["actions"][0]
                payload, target = action["payload"], action["payload"]["target"]
                salt = target["cache_salt"]
                dst = payload.get("dst", payload.get("destination"))["tier"]
                if dst == "storage":
                    state.storage[salt] = target["token_ids"]
                else:
                    if salt not in state.storage:
                        self.reply({"success": False, "error": {"code": "missing"}}, 404)
                        return
                    state.local[salt] = ("host", target["token_ids"])
                scope = [{"coverage": {"start_token": 0, "end_token": len(target["token_ids"]), "component": "full"}, "completed_steps": ["destination_ready"]}]
                value = {"results": [{"ref": {"context": body["context"], "message_id": body["envelope"]["message_id"], "operation_id": "op", "action_id": action["action_id"]},
                    "update": {"status": "completed", "quiescent": True, "target_outcomes": scope}, "public_result": {"ready_scope": scope}}]}
            elif kind == "ack":
                value = {"acknowledged": True}
            else:
                self.reply({}, 400)
                return
            self.reply({"success": True, "data": value})
    return Handler


class RouterState:
    def __init__(self, a, b, bearer):
        self.a, self.b, self.bearer = a, b, bearer
        self.tasks, self.calls, self.forwarded = {}, [], []
        self.pick = a
        self.delay_s = .005
        self.omit_header = False
        self.wrong_header = False
        self.wrong_epoch = False
        self.cleanup_unknown = False
        self.only_one_healthy = False
        self.fallback = False

    def view(self, task):
        record = self.tasks[task["session_id"]]
        if not record.get("released") and not record.get("prepared") and time.monotonic() >= record["due"]:
            engine = self.b
            engine.local[record["salt"]] = ("host", record["tokens"])
            record["prepared"] = True
        ready = record.get("prepared") and not record.get("released")
        return {"snapshot_revision": 1, "known_scope": [{"branch": "main", "end_token": len(record["tokens"])}],
            "missing_scope": [], "admission_gaps": [], "materialization_required": [],
            "quiescent": not self.cleanup_unknown if record.get("released") else bool(record.get("prepared")),
            "operations": [{"worker_url": self.b.url, "engine_id": self.b.name,
                "owner_epoch": "wrong-epoch" if self.wrong_epoch else self.b.caps()["engine_epoch"],
                "action": "sglang.prefetch", "error": None,
                "quiescent": not self.cleanup_unknown if record.get("released") else bool(record.get("prepared")),
                "result": {"update": {"status": "completed" if record.get("prepared") else "running"}},
                "ready_scope": {"start_token": 0, "end_token": len(record["tokens"]), "tier": "host", "provenance": "engine_scoped_query"} if ready else None}]}


def router_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def reply(self, value, code=200, content_type="application/json"):
            raw = value.encode() if isinstance(value, str) else json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def authorized(self):
            if self.headers.get("Authorization") != "Bearer "+state.bearer:
                self.reply({"success": False, "error": {"code": "unauthorized"}}, 401)
                return False
            return True

        def do_GET(self):
            if self.path == "/metrics":
                engines = [state.a] if state.only_one_healthy else [state.a, state.b]
                self.reply("\n".join(f'sgl_router_worker_health{{worker_url="{e.url}"}} 1' for e in engines)+"\n", content_type="text/plain")
            elif self.path == "/v1/kv/capabilities" and self.authorized():
                self.reply({"success": True, "data": {"enabled": True, "lifecycle": "explicit", "caller_id": "configured-test-domain"}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/generate":
                state.forwarded.append(copy.deepcopy(body))
                if not isinstance(body.pop("kv_context", None), dict):
                    self.reply({"error": "kv_context required for this test"}, 400)
                    return
                worker = state.pick
                request = urllib.request.Request(worker.url+"/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=2) as response:
                    raw = response.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                if not state.omit_header:
                    self.send_header("x-sgl-worker-url", "http://unverified-owner" if state.wrong_header else worker.url)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if not self.authorized():
                return
            state.calls.append(copy.deepcopy(body))
            task = body["task"]
            event, key = body.get("event"), task["session_id"]
            if body["kind"] == "session_event":
                if event == "task_started":
                    state.tasks[key] = {}
                    value = {"state": "active"}
                elif event == "request_observed":
                    actual = [x for path, x in state.a.requests if path == "/generate" and x["cache_salt"] == body["cache_salt"]]
                    if not actual or actual[-1]["input_ids"] != body["input_ids"]:
                        self.reply({"success": False, "error": {"code": "unobserved_input"}}, 400)
                        return
                    state.tasks[key].update(tokens=body["input_ids"], salt=body["cache_salt"])
                    value = {"state": "associated", "residency": "unknown"}
                elif event == "tool_wait":
                    value = {"state": "skipped"}
                elif event == "resume_expected":
                    state.tasks[key]["due"] = time.monotonic()+state.delay_s
                    value = {"state": "on_demand_fallback", "estimated_ms": 999} if state.fallback else state.view(task)
                elif event == "task_finished":
                    state.tasks[key]["released"] = True
                    value = {"state": "cleanup_requested", "quiescent": False}
                else:
                    self.reply({}, 400)
                    return
            elif body["kind"] == "query":
                value = state.view(task)
            elif body["kind"] == "cancel":
                state.tasks[key]["released"] = True
                value = state.view(task)
            elif body["kind"] == "ack":
                if state.cleanup_unknown:
                    self.reply({"success": False, "error": {"code": "not_quiescent"}}, 400)
                    return
                value = {"acknowledged": True}
            else:
                self.reply({}, 400)
                return
            self.reply({"success": True, "data": value})
    return Handler


class RouterBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.journal = Journal(Path(self.directory.name, "raw.jsonl"))
        self.servers = []
        storage = {}
        self.engines = [Engine(name, storage) for name in ("a", "b")]
        for engine in self.engines:
            engine.url = self.serve(engine_handler(engine))
        self.state = RouterState(*self.engines, "fixture-control-credential")
        self.url = self.serve(router_handler(self.state))
        self.a, self.b = [Control(HTTP(e.url, self.journal, 1), "independent-test-client", 5, 1) for e in self.engines]
        self.router = rb.Router(HTTP(self.url, self.journal, 1, self.state.bearer), HTTP(self.url, self.journal, 1), 5)
        self.args = argparse.Namespace(run_id="test", timeout_s=1, cleanup_timeout_s=.05, seed=1729, output_tokens=2, storage_backend="mooncake")
        self.case = {"id": "fixture", "prefix": list(range(64)), "suffix": [64], "provenance": {"declared_origin": "synthetic_fixture"}}

    def serve(self, handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.servers.append(server)
        return f"http://127.0.0.1:{server.server_port}"

    def tearDown(self):
        for server in reversed(self.servers):
            server.shutdown()
            server.server_close()
        self.journal.file.close()
        self.directory.cleanup()

    def run_trial(self, group="b3_router", wait=.08, name=None):
        spec = {"id": name or group, "pair": "pair", "group": group, "prefix_tokens": 64, "tool_wait_s": wait}
        return rb.trial(self.args, self.a, self.b, self.router, self.case, spec, self.journal)

    def test_actual_worker_and_frozen_input_not_preparation_owner(self):
        self.state.pick = self.engines[0]
        row = self.run_trial()
        self.assertTrue(row["valid"], row)
        self.assertEqual(row["selected_worker_url"], self.a.http.url)
        self.assertEqual(row["resume_cache"]["storage"], 64)
        self.assertEqual(row["resume_cache"]["device"], 0)
        self.assertTrue(row["route_changed_from_prepared_b"])
        self.assertTrue(row["cleanup"]["quiescent"])
        self.assertEqual(row["preparation_usage"], "unused_on_b_route_changed")
        observed = next(c for c in self.state.calls if c.get("event") == "request_observed")
        self.assertEqual(observed["input_ids"], self.case["prefix"])
        self.assertFalse(any("caller_id" in c for c in self.state.calls))
        self.assertIn("kv_context", self.state.forwarded[0])
        self.assertTrue(all("kv_context" not in body for e in self.engines for path, body in e.requests if path == "/generate"))
        self.assertNotIn(self.state.bearer, Path(self.directory.name, "raw.jsonl").read_text())

    def test_ready_host_reuse_and_paired_all_completed_statistics(self):
        self.state.pick = self.engines[1]
        b2 = self.run_trial("b2_direct_b")
        b3 = self.run_trial("b3_router")
        self.assertTrue(b2["valid"], b2)
        self.assertTrue(b3["valid"], b3)
        self.assertEqual(b3["resume_cache"]["cpu"], 64)
        self.assertTrue(b3["prefetch_ready_before_resume"])
        b3["valid"] = False
        b3["invalid_reasons"] = ["fixture failed attribution"]
        summary = rb.report(self.directory.name, [b2, b3])
        self.assertEqual(summary["groups"]["b3_router"]["completed"], 1)
        self.assertEqual(summary["paired_all_completed_mean_delta_ci"]["n_pairs"], 1)
        self.assertEqual(len(summary["invalid"]), 1)

    def test_formal_deadline_does_not_wait_for_preparation(self):
        self.state.delay_s = .5
        self.state.pick = self.engines[1]
        row = self.run_trial(wait=.03)
        self.assertTrue(row["valid"], row)
        self.assertFalse(row["prefetch_ready_before_resume"])
        self.assertLess(row["formal_dispatch_lateness_ms"], 250)
        self.assertEqual(row["resume_cache"]["storage"], 64)
        self.assertEqual(row["preparation_usage"], "ambiguous_or_policy_fallback")

    def test_missing_or_wrong_routing_header_preserves_completed_invalid(self):
        for attribute in ("omit_header", "wrong_header"):
            setattr(self.state, attribute, True)
            row = self.run_trial(name=attribute)
            self.assertFalse(row["valid"])
            self.assertIn("resume", row)
            self.assertTrue(row["cleanup"]["quiescent"])
            setattr(self.state, attribute, False)

    def test_child_identity_and_cleanup_failure_are_not_success(self):
        self.state.wrong_epoch = True
        row = self.run_trial(name="wrong-epoch")
        self.assertFalse(row["valid"])
        self.assertTrue(any("epoch" in x for x in row["invalid_reasons"]))
        self.state.wrong_epoch = False
        self.state.cleanup_unknown = True
        row = self.run_trial(name="unknown-drain")
        self.assertFalse(row["valid"])
        self.assertFalse(row["cleanup"]["quiescent"])
        self.assertNotIn("ack", row["cleanup"])

    def test_two_healthy_workers_and_explicit_authentication(self):
        self.state.only_one_healthy = True
        with self.assertRaisesRegex(BenchError, "two expected healthy"):
            self.router.topology([self.a.http.url, self.b.http.url], "health-failure")
        with self.assertRaises(BenchError):
            rb.Router(HTTP(self.url, self.journal, 1, "wrong-fixture-key"), HTTP(self.url, self.journal, 1))

    def test_policy_fallback_does_not_claim_preparation_or_ack_unknown_intent(self):
        self.state.fallback = True
        self.state.pick = self.engines[1]
        row = self.run_trial(name="fallback")
        self.assertTrue(row["valid"], row)
        self.assertEqual(row["preparation_decision"]["state"], "on_demand_fallback")
        self.assertFalse(row["prefetch_ready_before_resume"])
        self.assertEqual(row["prepared_tokens_observed"], 0)
        self.assertEqual(row["resume_cache"]["storage"], 64)
        self.assertNotIn("ack", row["cleanup"])

    def test_full_cli_run_trace_schedule_and_negative(self):
        trace = Path(self.directory.name, "trace.jsonl")
        trace.write_text(json.dumps({"id": "trace-fixture", "token_prefix": list(range(64)), "tool_suffix": [64],
                                    "tool_wait_s": .08, "declared_origin": "synthetic_fixture"})+"\n")
        args = argparse.Namespace(router_mode="b2-b3", router=self.url, engine_a=self.a.http.url, engine_b=self.b.http.url,
            router_control_key_env="ROUTER_BENCH_FIXTURE_KEY", router_api_key_env=None, engine_key_env=None,
            model="mock-model", output=str(Path(self.directory.name, "run")), trace_jsonl=str(trace),
            prefix_lengths=[64], tool_wait_s=.08, repetitions=1, warmup_repetitions=0, negative="route_change",
            output_tokens=2, poll_ms=5, timeout_s=1, cleanup_timeout_s=.1, storage_backend="mooncake", seed=1729, run_id="full-run")
        with patch.dict(os.environ, {args.router_control_key_env: self.state.bearer}):
            self.assertEqual(rb.run(args), 0)
        summary = json.loads(Path(args.output, "router-summary.json").read_text())
        self.assertEqual(summary["attempts"], 3)
        self.assertEqual(summary["groups"]["route_change"]["attempts"], 1)
        self.assertEqual(summary["groups"]["b3_router"]["route_changed"], 1)
        raw = Path(args.output, "router-raw.jsonl").read_text()
        self.assertNotIn(self.state.bearer, raw)
        self.assertIn("x-sgl-worker-url", raw)


if __name__ == "__main__":
    unittest.main()

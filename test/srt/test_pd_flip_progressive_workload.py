import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "scripts"
    / "playground"
    / "disaggregation"
    / "pd_flip_progressive_workload.py"
)
MATRIX_SCRIPT = SCRIPT.with_name("pd_flip_progressive_matrix.py")


def load_module():
    spec = importlib.util.spec_from_file_location("pd_flip_progressive_workload", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_matrix_module():
    spec = importlib.util.spec_from_file_location("pd_flip_progressive_matrix", MATRIX_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    requests = []

    def do_GET(self):
        active = next(
            (payload[1]["rid"] for payload in self.__class__.requests if payload[1].get("stream")),
            "not-started",
        )
        if self.path.endswith("/pd_flip/runtime_role/status"):
            body = {"status": {"running_requests": [{"rid": active}]}}
        else:
            body = {
                "session_id": "session-1",
                "request_measurements": [
                    {
                        "rid": active,
                        "stitch_mode": "partial_prefix_stitch",
                        "final_owner": "target",
                        "output_boundary": 2,
                    }
                ],
            }
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.__class__.requests.append((self.headers.get("authorization"), payload))
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for token in ("alpha", "beta"):
                chunk = {"choices": [{"delta": {"content": token}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            )
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            content = "alphabeta" if payload.get("max_tokens") == 4 else "warm"
            body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *_):
        return


class IncompleteSSEHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')

    def log_message(self, *_):
        return


class AuthGetHandler(BaseHTTPRequestHandler):
    authorization = None

    def do_GET(self):
        self.__class__.authorization = self.headers.get("authorization")
        body = b"{}"
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        return


def test_mode_prompt_strategies_are_distinct_and_do_not_claim_hits():
    module = load_module()
    full = module.build_prompt_plan("full", seed=7)
    partial = module.build_prompt_plan("partial", seed=7)
    zero = module.build_prompt_plan("zero", seed=7)

    assert full.warmup_prompts == (full.active_prompt,)
    assert partial.warmup_prompts and partial.active_prompt.startswith(
        partial.warmup_prompts[0]
    )
    assert len(partial.warmup_prompts[0]) < len(partial.active_prompt)
    assert zero.warmup_prompts == ()
    assert zero.active_prompt not in {full.active_prompt, partial.active_prompt}
    assert "expected_stitch_mode" not in full.__dict__


def test_incomplete_sse_is_a_hard_error():
    module = load_module()
    server = ThreadingHTTPServer(("127.0.0.1", 0), IncompleteSSEHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = module.OpenAIClient(
            f"http://127.0.0.1:{server.server_port}", "secret", 2
        )
        with pytest.raises(RuntimeError, match="incomplete SSE stream"):
            client.stream(module.request_payload("m", "p", 1, True, "rid-1"))
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_measurement_client_sends_bearer_auth():
    from scripts.playground.disaggregation.pd_flip_migration_measure import HttpClient

    server = ThreadingHTTPServer(("127.0.0.1", 0), AuthGetHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        HttpClient(timeout_seconds=2, api_key="secret").get_json(
            f"http://127.0.0.1:{server.server_port}/status"
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert AuthGetHandler.authorization == "Bearer secret"


def test_matrix_enumerates_all_modes_and_paths_and_rejects_empty_reset(monkeypatch):
    module = load_matrix_module()
    calls = []
    monkeypatch.setenv("MATRIX_KEY", "secret")
    monkeypatch.setattr(
        module, "run_case", lambda args, mode, path: calls.append((path, mode))
    )
    args = type(
        "Args",
        (),
        {"reset_store_cmd": "reset", "admin_api_key_env": "MATRIX_KEY"},
    )()
    assert module.run(args) == 0
    assert calls == [
        (path, mode) for path in ("recovery", "commit") for mode in ("full", "partial", "zero")
    ]
    args.reset_store_cmd = "  "
    with pytest.raises(ValueError, match="non-empty"):
        module.run(args)


def test_router_placement_drains_non_source_then_restores(monkeypatch):
    module = load_matrix_module()
    calls = []
    workers = [
        {"worker_id": "w1", "url": "http://node1", "role": "decode", "draining": False},
        {"worker_id": "w2", "url": "http://node2", "role": "decode", "draining": False},
        {"worker_id": "w3", "url": "http://node3", "role": "decode", "draining": True},
    ]
    monkeypatch.setattr(module, "router_get_workers", lambda *_: workers)
    monkeypatch.setattr(
        module,
        "router_set_drain",
        lambda *args: calls.append((args[-2], args[-1])),
    )
    placement = module.prepare_router_placement(
        "http://router", "secret", "http://node1", "http://node2", "http://node3", 2
    )
    assert calls == [("w1", True), ("w3", True), ("w2", False)]
    module.release_targets(placement)
    assert calls[-2:] == [("w1", False), ("w3", False)]
    module.restore_router_placement(placement)
    assert calls[-3:] == [("w1", False), ("w2", False), ("w3", True)]


def test_router_placement_rolls_back_partial_failure(monkeypatch):
    module = load_matrix_module()
    workers = [
        {"worker_id": "w1", "url": "http://node1", "role": "decode", "draining": False},
        {"worker_id": "w2", "url": "http://node2", "role": "decode", "draining": False},
        {"worker_id": "w3", "url": "http://node3", "role": "decode", "draining": False},
    ]
    calls = []
    monkeypatch.setattr(module, "router_get_workers", lambda *_: workers)

    def mutate(*args):
        calls.append((args[-2], args[-1]))
        if len(calls) == 2:
            raise RuntimeError("target drain failed")

    monkeypatch.setattr(module, "router_set_drain", mutate)
    with pytest.raises(RuntimeError, match="target drain failed"):
        module.prepare_router_placement(
            "http://router", "secret", "http://node1", "http://node2", "http://node3", 2
        )
    assert calls[-1] == ("w1", False)


def test_pressure_timeline_contracts():
    module = load_matrix_module()
    module.validate_pressure_timeline(
        {
            "pressure_start": 1.0,
            "controller_start": 2.0,
            "first_batch_target_active": 3.0,
            "pressure_end": 4.0,
            "controller_end": 5.0,
        },
        "recovery",
    )
    module.validate_pressure_timeline(
        {
            "pressure_start": 1.0,
            "controller_start": 2.0,
            "controller_end": 3.0,
            "pressure_end": 3.0,
        },
        "commit",
    )
    with pytest.raises(RuntimeError, match="first activation"):
        module.validate_pressure_timeline(
            {
                "pressure_start": 1.0,
                "controller_start": 2.0,
                "pressure_end": 2.5,
                "first_batch_target_active": 3.0,
                "controller_end": 4.0,
            },
            "recovery",
        )


def test_controller_wait_fails_if_sidecar_exits_early(tmp_path):
    module = load_matrix_module()

    class Process:
        def __init__(self, return_code):
            self.return_code = return_code

        def poll(self):
            return self.return_code

    log = tmp_path / "measurement.log"
    log.write_text("collector crashed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="collector crashed"):
        module.wait_for_process(
            Process(None),
            1,
            0,
            [("measurement", Process(7), log)],
        )


def test_join_uses_session_and_rid_and_rejects_competing_targets():
    from scripts.playground.disaggregation.pd_flip_migration_measure import (
        join_request_migration,
    )

    request = {"request_id": "rid-1", "migration_session_id": "s1"}
    source = {"rid": "rid-1", "session_id": "s1", "final_owner": "source"}
    target = {"rid": "rid-1", "session_id": "s1", "final_owner": "target"}
    joined = join_request_migration([request], [source, target])
    assert joined[0]["worker_final_owner"] == "target"
    with pytest.raises(RuntimeError, match="competing target"):
        join_request_migration([request], [target, dict(target, node="duplicate")])
    with pytest.raises(RuntimeError, match="ambiguous migration session"):
        join_request_migration(
            [{"request_id": "rid-1"}],
            [target, dict(target, session_id="s2")],
        )


def test_workload_proves_rid_migration_and_control_output(tmp_path, monkeypatch):
    module = load_module()
    FakeOpenAIHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output_dir = tmp_path / "workload"
    done_marker = tmp_path / "controller.done"
    ready_marker = tmp_path / "workload.ready"
    pressure_stop = tmp_path / "pressure.stop"
    pressure_started = tmp_path / "pressure.started"
    pressure_ended = tmp_path / "pressure.ended"
    done_marker.touch()
    monkeypatch.setenv("PD_FLIP_TEST_KEY", "secret")
    try:
        exit_code = module.main(
            [
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--source-url",
                f"http://127.0.0.1:{server.server_port}",
                "--target-url",
                f"http://127.0.0.1:{server.server_port}",
                "--admin-api-key-env",
                "PD_FLIP_TEST_KEY",
                "--ready-marker",
                str(ready_marker),
                "--controller-done-marker",
                str(done_marker),
                "--pressure-stop-marker",
                str(pressure_stop),
                "--pressure-started-marker",
                str(pressure_started),
                "--pressure-ended-marker",
                str(pressure_ended),
                "--model",
                "test-model",
                "--mode",
                "partial",
                "--decision-path",
                "recovery",
                "--output-dir",
                str(output_dir),
                "--pressure-requests",
                "1",
                "--pressure-interval-seconds",
                "0",
                "--long-max-tokens",
                "4",
            ]
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert exit_code == 0
    metrics = [
        json.loads(line)
        for line in (output_dir / "request_metrics.jsonl").read_text().splitlines()
    ]
    errors = (output_dir / "errors.jsonl").read_text()
    config = json.loads((output_dir / "workload_config.json").read_text())
    active = next(row for row in metrics if row["prompt_kind"] == "active_long")
    assert active["output_chunks"] == [
        {"chunk_ordinal": 0, "text": "alpha"},
        {"chunk_ordinal": 1, "text": "beta"},
    ]
    assert active["start_monotonic"] <= active["end_monotonic"]
    assert active["control_exact_match"] is True
    assert active["migration_measurement"]["final_owner"] == "target"
    assert json.loads(ready_marker.read_text())["rid"] == active["request_id"]
    assert active["mode"] == "partial"
    assert errors == ""
    assert config["decision_path"] == "recovery"
    assert all(auth == "Bearer secret" for auth, _ in FakeOpenAIHandler.requests)
    assert all(payload.get("rid") for _, payload in FakeOpenAIHandler.requests)

    events = tmp_path / "migration_events.jsonl"
    midpoint = (active["start_monotonic"] + active["end_monotonic"]) / 2
    event_rows = [
        {
            "event_type": "migration_status",
            "ts_wall": 1.0,
            "ts_mono": midpoint,
            "node": "node1",
            "ok": True,
            "status": {
                "session_id": "session-1",
                "role": "source",
                "state": "source_started",
                "request_measurements": [],
            },
        },
        {
            "event_type": "migration_status",
            "ts_wall": 2.0,
            "ts_mono": active["end_monotonic"],
            "node": "node2",
            "ok": True,
            "status": {
                "session_id": "session-1",
                "role": "target",
                "state": "active",
                "transferred_reqs": 1,
                "pending_reqs": 0,
                "failed_reqs": 0,
                "request_measurements": [active["migration_measurement"]],
            },
        },
    ]
    events.write_text(
        "".join(json.dumps(row) + "\n" for row in event_rows), encoding="utf-8"
    )
    controller_log = tmp_path / "controller.log"
    controller_log.write_text('{"state":"safe","reason":"smoke"}\n', encoding="utf-8")
    summary = tmp_path / "summary"
    from scripts.playground.disaggregation.pd_flip_migration_measure import write_outputs

    write_outputs(
        events_path=events,
        output_dir=summary,
        controller_log=controller_log,
        request_metrics_path=output_dir / "request_metrics.jsonl",
        errors_path=output_dir / "errors.jsonl",
    )
    for name in (
        "migration_status_samples.csv",
        "migration_request_samples.jsonl",
        "controller_state_trace.csv",
        "request_impact_by_stage.csv",
        "migration_link_summary.json",
        "request_migration_join.jsonl",
    ):
        assert (summary / name).exists(), name
    impact_rows = list(
        __import__("csv").DictReader(
            (summary / "request_impact_by_stage.csv").open(encoding="utf-8")
        )
    )
    assert any(row["active_during_migration"] == "True" for row in impact_rows)
    joined = [
        json.loads(line)
        for line in (summary / "request_migration_join.jsonl").read_text().splitlines()
    ]
    active_join = next(row for row in joined if row["request_id"] == active["request_id"])
    assert active_join["migration_measurement_found"] is True
    assert active_join["ttft_slo_s"] > 0
    assert active_join["worker_rid"] == active["request_id"]

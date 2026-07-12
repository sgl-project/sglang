import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "scripts"
    / "playground"
    / "disaggregation"
    / "pd_flip_progressive_workload.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("pd_flip_progressive_workload", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    requests = []

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
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            body = json.dumps({"choices": [{"message": {"content": "warm"}}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
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


def test_workload_writes_stream_sequences_ledgers_and_summary_inputs(tmp_path):
    module = load_module()
    FakeOpenAIHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output_dir = tmp_path / "workload"
    try:
        exit_code = module.main(
            [
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--api-key",
                "secret",
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
    assert active["output_sequences"] == [
        {"seq": 0, "text": "alpha"},
        {"seq": 1, "text": "beta"},
    ]
    assert active["mode"] == "partial"
    assert errors == ""
    assert config["decision_path"] == "recovery"
    assert all(auth == "Bearer secret" for auth, _ in FakeOpenAIHandler.requests)

    events = tmp_path / "migration_events.jsonl"
    events.write_text(
        json.dumps(
            {
                "event_type": "migration_status",
                "ts_wall": 1.0,
                "ts_mono": 1.0,
                "node": "node3",
                "ok": True,
                "status": {
                    "session_id": "s",
                    "state": "active",
                    "request_measurements": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
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
    ):
        assert (summary / name).exists(), name

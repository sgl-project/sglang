//! A real stdlib HTTP subprocess for exercising the public runner without a model.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde_json::{Value, json};
use sglang_parity::runner::PreparedResponse;
use sglang_parity::{
    CaptureMode, HttpCase, HttpObservation, HttpSuite, ResponsePolicy, RunConfig, Violation,
};

pub struct Fixture {
    pub directory: tempfile::TempDir,
    pub config: RunConfig,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if std::thread::panicking() {
            // Startup failures need their subprocess traceback, not just the
            // runner's exit-status summary, to diagnose port and timing races.
            self.directory.disable_cleanup(true);
            eprintln!(
                "retained failed fixture: {}",
                self.directory.path().display()
            );
        }
    }
}

impl Fixture {
    pub fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source");
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let lock = if cfg!(target_os = "macos") {
            "rust/sglang-parity/environments/mlx.lock"
        } else {
            "rust/sglang-parity/environments/cuda.lock"
        };
        for relative in [
            "python/pyproject.toml",
            "python/pyproject_other.toml",
            "rust/sglang-parity/environments/profiles.json",
            "rust/sglang-parity/environments/probe.py",
            lock,
        ] {
            let destination = source.join(relative);
            fs::create_dir_all(destination.parent().unwrap()).unwrap();
            fs::copy(repository.join(relative), destination).unwrap();
        }
        fs::write(source.join("python/fixture-version.txt"), "initial HEAD").unwrap();
        git(&source, &["init", "--quiet"]);
        git(&source, &["add", "."]);
        git(&source, &["commit", "--quiet", "-m", "fixture source"]);
        let executable = directory.path().join("fixture-python");
        fs::write(&executable, SERVER).unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o755)).unwrap();
        // Allocate in a child so concurrent parent forks cannot inherit the socket.
        let output = Command::new("python3")
            .args([
                "-c",
                "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])",
            ])
            .output()
            .unwrap();
        assert!(output.status.success(), "{:?}", output);
        let port: u16 = String::from_utf8(output.stdout)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let config = serde_json::from_value(json!({
            "server": {
                "python": executable,
                "model": "fixture-model",
                "args": ["--attention-backend", "fixture"],
                "env": {
                    "PARITY_FIXTURE_LOG": directory.path().join("lifecycle.jsonl"),
                    "PARITY_FIXTURE_SHARED": "same-for-both"
                },
                "port": port
            },
            "environment": {
                "source_root": source,
                "cache_dir": directory.path().join("cache"),
                "setup_timeout_secs": 15
            },
            "startup_timeout_secs": 15,
            "request_timeout_secs": 5,
            "shutdown_timeout_secs": 1,
            "output_dir": directory.path().join("output")
        }))
        .unwrap();
        Self { directory, config }
    }

    pub fn lifecycle(&self) -> Vec<Value> {
        fs::read_to_string(self.directory.path().join("lifecycle.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    pub fn preparations(&self) -> Vec<Value> {
        fs::read_to_string(self.directory.path().join("preparation.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    pub fn source(&self) -> PathBuf {
        self.directory.path().join("source")
    }

    pub fn only_run_directory(&self) -> PathBuf {
        let entries = fs::read_dir(&self.config.output_dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 1);
        entries.into_iter().next().unwrap()
    }
}

pub fn git(source: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .args([
            "-c",
            "user.name=Parity fixture",
            "-c",
            "user.email=parity-fixture@localhost",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
        ])
        .arg("-C")
        .arg(source)
        .args(args)
        .output()
        .unwrap();
    assert!(output.status.success(), "git {args:?}: {output:?}");
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

pub fn read_json(path: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

pub fn case(name: &str, path: &str, behavior: &str) -> Value {
    json!({
        "name": name,
        "method": "POST",
        "path": path,
        "body": {
            "value": {"items": [1, null, "你好"], "keep": true},
            "behavior": behavior,
            "opaque": {"null": null, "list": [true, "preserve"]}
        },
        "expect_status": 200,
        "capture": if path == "/sse" { "sse" } else { "json" },
        "comparison_scope": "root"
    })
}

pub fn suite(cases: Vec<Value>) -> HttpSuite {
    serde_json::from_value(json!({
        "name": "fixture",
        "response_implementation": "tests::EchoPolicy",
        "output_mode": "fixture_final",
        "comparison": {
            "base": "exact_json",
            "per_result_value_exceptions": [
                {"path": "/trace", "require": "non_empty_string", "reason": "Request identity."},
                {"path": "/duration", "require": "non_negative_number", "reason": "Elapsed time."}
            ]
        },
        "cases": cases
    }))
    .unwrap()
}

/// A deliberately different protocol proves the core does not depend on generate.
pub struct EchoPolicy;

impl ResponsePolicy for EchoPolicy {
    fn prepare(
        &self,
        case: &HttpCase,
        observation: &HttpObservation,
    ) -> Result<PreparedResponse, Vec<Violation>> {
        if case.body["behavior"] == "empty_rejection" {
            return Err(vec![]);
        }
        let value = match case.capture {
            CaptureMode::Json => observation
                .json
                .clone()
                .ok_or_else(|| vec![Violation::new("", "expected JSON response")])?,
            CaptureMode::Sse => {
                if observation.events.len() != 2 || observation.events[1].data != "[FIN]" {
                    return Err(vec![Violation::new(
                        "/events",
                        "expected a snapshot followed by [FIN]",
                    )]);
                }
                serde_json::from_str(&observation.events[0].data).map_err(|error| {
                    vec![Violation::new(
                        "/events/0",
                        format!("invalid snapshot JSON: {error}"),
                    )]
                })?
            }
        };
        let required_key = if case.expect_status >= 400 {
            "error"
        } else {
            "value"
        };
        if !value.is_object() || value.get(required_key).is_none() {
            return Err(vec![Violation::new(
                "",
                format!("expected an object containing {required_key}"),
            )]);
        }
        Ok(PreparedResponse {
            equivalence: None,
            assertions: Vec::new(),
            value,
            origins: if case.capture == CaptureMode::Sse {
                [(String::new(), vec![0])].into()
            } else {
                Default::default()
            },
        })
    }
}

pub async fn wait_until(mut condition: impl FnMut() -> bool) -> bool {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    while !condition() {
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    true
}

pub async fn assert_process_stopped(pid: i32) {
    let stopped = wait_until(|| {
        let output = Command::new("ps")
            .args(["-o", "stat=", "-p", &pid.to_string()])
            .output()
            .unwrap();
        let state = String::from_utf8_lossy(&output.stdout);
        // Descendants killed with their parent may briefly await OS reaping.
        state.trim().is_empty() || state.trim().starts_with('Z')
    })
    .await;
    assert!(stopped, "fixture process {pid} survived cleanup");
}

const SERVER: &str = r#"#!/usr/bin/env python3
import http.server
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

trace_path = os.environ['PARITY_FIXTURE_LOG']
if len(sys.argv) > 1 and Path(sys.argv[1]).name == 'probe.py':
    if '--library-paths' in sys.argv:
        Path(sys.argv[-1]).write_text('[]')
        sys.exit(0)
    arguments = dict(zip(sys.argv[2::2], sys.argv[3::2]))
    source = Path(arguments['--source'])
    failed = os.environ.get('PARITY_FIXTURE_PROBE_FAIL') == '1'
    result = {
        'status': 'failed' if failed else 'passed',
        'python_version': arguments['--python-version'],
        'dependencies_only': False,
        'lock': str(Path(arguments['--lock']).resolve()),
        'python_executable': arguments['--python'],
        'source': str(source.resolve()),
        'packages': {'fixture': '1.0'},
        'backend': {'name': arguments['--backend'], 'device': 'fixture'},
        'rust_extension': {'path': 'fixture', 'fingerprint': 'fixture', 'sha256': 'fixture'},
    }
    if failed:
        result['error'] = 'deliberate fixture probe failure'
    Path(arguments['--output']).write_text(json.dumps(result))
    with open(Path(trace_path).with_name('preparation.jsonl'), 'a') as stream:
        stream.write(json.dumps(result) + '\n')
    if failed:
        print(result['error'], flush=True)
    sys.exit(17 if failed else 0)

side = os.environ['SGLANG_RUST_SERVER']
worker = None

def record(kind, **fields):
    fields.update(kind=kind, implementation=side, pid=os.getpid())
    with open(trace_path, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(fields, ensure_ascii=False) + '\n')

def terminate(signum, frame):
    if worker is not None:
        worker.terminate()
        worker.wait(timeout=2)
    record('stop')
    raise SystemExit(0)

signal.signal(signal.SIGTERM, terminate)
if os.environ.get('PARITY_FIXTURE_EARLY_EXIT') == '1':
    record('early_exit')
    sys.exit(17)

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200 if self.path == '/health_generate' else 404)
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_POST(self):
        raw = self.rfile.read(int(self.headers['Content-Length']))
        body = json.loads(raw)
        self.server.requests += 1
        count = self.server.requests
        record('request', path=self.path, raw=raw.decode('utf-8'), body=body)
        behavior = body['behavior']
        if behavior == 'wait_for_release':
            while not os.path.exists(os.environ['PARITY_FIXTURE_RELEASE']):
                time.sleep(0.01)
        payload = {
            'value': body['value'],
            'trace': side + ':' + str(count),
            'duration': count / 10,
            'unknown': {'keep': [1, None, {'nested': True}]},
        }
        if behavior == 'different' and side == '1':
            payload['value']['items'][1] = 'changed'
            payload['unexpected'] = True
        if behavior == 'unstable':
            payload['value'] = count
        if behavior in ('invalid', 'http_error'):
            payload = {'error': 'deliberate fixture failure'}
        encoded = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        if behavior == 'malformed_json':
            encoded = b'{"value":'
        if behavior == 'artifact_io_failure':
            # Block the next atomic artifact replacement before sending a valid response.
            output = os.path.join(os.path.dirname(trace_path), 'output')
            run_dir, = os.listdir(output)
            os.mkdir(os.path.join(output, run_dir, 'python', 'blocked', '1', 'final.json'))
        stream_response = self.path == '/sse' or body.get('stream', False)
        if stream_response:
            encoded = b'event: snapshot\ndata: ' + encoded + b'\n\n'
            if behavior == 'malformed_stream':
                encoded += b'data: unfinished'
            elif behavior not in ('truncate', 'timeout'):
                encoded += b'data: [FIN]\n\n'
        self.send_response(400 if behavior in ('http_error', 'wrong_status') else 200)
        content_type = 'text/event-stream; charset=utf-8' if stream_response else 'application/json'
        if behavior == 'wrong_type':
            content_type = 'text/plain'
        if behavior != 'missing_type':
            self.send_header('Content-Type', content_type)
        extra = 1000 if behavior in ('truncate', 'timeout') else 0
        self.send_header('Content-Length', str(len(encoded) + extra))
        self.end_headers()
        try:
            self.wfile.write(encoded)
            self.wfile.flush()
            if behavior == 'timeout':
                time.sleep(60)
            if behavior == 'truncate':
                self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            pass

port = int(sys.argv[sys.argv.index('--port') + 1])
server = http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler)
server.requests = 0
if os.environ.get('PARITY_FIXTURE_CHILD') == '1':
    worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
record('start', argv=sys.argv[1:], deterministic=os.environ.get('SGLANG_ENABLE_DETERMINISTIC_INFERENCE'),
       shared=os.environ.get('PARITY_FIXTURE_SHARED'), worker=None if worker is None else worker.pid,
       python_path=os.environ['PYTHONPATH'],
       source_version=Path(os.environ['PYTHONPATH'], 'fixture-version.txt').read_text())
print('fixture server ready for ' + side, flush=True)
server.serve_forever()
"#;

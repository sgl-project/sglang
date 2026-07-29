// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Near-E2E gap replay test for KV events.
//!
//! This exercises the production Rust path end-to-end without a model server:
//! `/server_info` discovery -> live ZMQ SUB -> pump gap detection -> replay
//! DEALER -> `HashTree` update. The live publisher deliberately sends seq 1
//! and seq 3 only; the replay ROUTER returns seq 2.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{routing::get, Json, Router};
use bytes::Bytes;
use serde_json::{json, Value};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::net::TcpListener;
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::oneshot;
use zeromq::{Endpoint, RouterSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

use sgl_router::policies::kv_events::{KvEventIndex, KvWorkerId};
use sgl_router::server::metrics::MetricsRegistry;

use super::zmq_helpers::{
    build_multipart, encode_block_stored_event, encode_event_batch, make_pub_bound,
};

#[tokio::test]
async fn kv_event_index_discovers_worker_and_replays_live_sequence_gap() {
    let (mut publisher, live_port) = make_pub_bound().await;
    let replay_payload = block_stored_payload(20);
    let (replay_port, replay_server) = spawn_replay_router(2, vec![(2, replay_payload)]).await;
    let (worker_url, shutdown_http) = spawn_fake_worker(live_port, replay_port).await;

    let index = KvEventIndex::new();
    let metrics = MetricsRegistry::new();
    index.attach_metrics(Arc::clone(&metrics));
    index.add_worker(&worker_url, None).await;
    assert_eq!(
        index.known_worker_count(),
        1,
        "worker should be discovered from fake /server_info"
    );

    let worker = KvWorkerId {
        url: worker_url.clone(),
        dp_rank: 0,
    };
    let seq1 = block_stored_payload(10);
    let seq3 = block_stored_payload(30);

    // PUB/SUB has an async subscription handshake. Publish seq=1 until the
    // tree proves the real subscriber + pump path is live; then publish seq=3.
    let start = Instant::now();
    loop {
        publisher
            .send(build_multipart(1, seq1.clone()))
            .await
            .expect("publish seq=1");
        if tree_has_block(&index, &worker, 10) {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "seq=1 never reached the HashTree through the live subscriber"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    // Now deliberately skip live seq=2 and send seq=3. The replay ROUTER must
    // be asked for seq=2, and the tree should end up with both replayed block
    // 20 and live block 30.
    let start = Instant::now();
    loop {
        publisher
            .send(build_multipart(3, seq3.clone()))
            .await
            .expect("publish seq=3");
        if tree_has_block(&index, &worker, 20)
            && tree_has_block(&index, &worker, 30)
            && metrics
                .render()
                .contains(r#"sgl_router_kv_event_gaps_total{outcome="replayed"} 1"#)
        {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "gap replay did not apply replayed seq=2 and live seq=3"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    replay_server.await.expect("replay server task");
    index.shutdown().await;
    let _ = shutdown_http.send(());
}

#[tokio::test]
async fn kv_event_index_replays_gap_from_python_pyzmq_harness_when_available() {
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_string());
    if !python_has_pyzmq(&python).await {
        eprintln!("skipping Python pyzmq harness test: `{python}` cannot import zmq");
        return;
    }

    let mut harness = PythonHarness::spawn(&python).await;
    let index = KvEventIndex::new();
    let metrics = MetricsRegistry::new();
    index.attach_metrics(Arc::clone(&metrics));
    index.add_worker(&harness.worker_url, None).await;
    assert_eq!(
        index.known_worker_count(),
        1,
        "worker should be discovered from Python /server_info"
    );

    let worker = KvWorkerId {
        url: harness.worker_url.clone(),
        dp_rank: 0,
    };

    let start = Instant::now();
    loop {
        harness.command("publish 1 10").await;
        if tree_has_block(&index, &worker, 10) {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "Python live seq=1 never reached the HashTree"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    harness.command("buffer 2 20").await;
    let start = Instant::now();
    loop {
        // Send seq=3 live-only so the Python replay ROUTER returns exactly the
        // missing seq=2 batch, matching the synthetic Rust near-E2E above.
        harness.command("live 3 30").await;
        if tree_has_block(&index, &worker, 20)
            && tree_has_block(&index, &worker, 30)
            && metrics
                .render()
                .contains(r#"sgl_router_kv_event_gaps_total{outcome="replayed"} 1"#)
        {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "Python harness gap replay did not apply seq=2 and live seq=3"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    index.shutdown().await;
    harness.stop().await;
}

fn block_stored_payload(block_hash: i64) -> Vec<u8> {
    encode_event_batch(
        0.0,
        vec![encode_block_stored_event(&[block_hash], None, &[], 64)],
        Some(0),
    )
}

fn tree_has_block(index: &KvEventIndex, worker: &KvWorkerId, block_hash: i64) -> bool {
    let matched = index.tree().match_prefix(None, &[block_hash]);
    matched.matched_blocks == 1 && matched.workers.contains(worker)
}

async fn spawn_fake_worker(live_port: u16, replay_port: u16) -> (String, oneshot::Sender<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fake worker HTTP");
    let http_port = listener.local_addr().expect("local addr").port();
    let body = Arc::new(json!({
        "kv_events": {
            "publisher": "zmq",
            "endpoint_host": "127.0.0.1",
            "endpoint_port_base": live_port,
            "topic": "",
            "block_size": 64,
            "dp_size": 1,
            "replay_endpoint_host": "127.0.0.1",
            "replay_endpoint_port_base": replay_port,
            "replay_buffer_steps": 16,
        }
    }));
    let app = Router::new().route(
        "/server_info",
        get(move || {
            let body = Arc::clone(&body);
            async move { Json::<Value>((*body).clone()) }
        }),
    );
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });
    (format!("http://127.0.0.1:{http_port}"), tx)
}

async fn spawn_replay_router(
    expected_start_seq: i64,
    replies: Vec<(i64, Vec<u8>)>,
) -> (u16, tokio::task::JoinHandle<()>) {
    let mut router = RouterSocket::new();
    let endpoint = router
        .bind("tcp://127.0.0.1:0")
        .await
        .expect("bind replay ROUTER");
    let port = match endpoint {
        Endpoint::Tcp(_, port) => port,
        other => panic!("unexpected replay endpoint: {other:?}"),
    };
    let handle = tokio::spawn(async move {
        let request = router.recv().await.expect("receive replay request");
        assert_eq!(request.len(), 3);
        let identity = request.get(0).expect("identity").clone();
        assert!(request.get(1).expect("delimiter").is_empty());
        assert_eq!(
            request.get(2).expect("start seq").as_ref(),
            &expected_start_seq.to_be_bytes()
        );
        for (seq, payload) in replies {
            router
                .send(replay_reply(identity.clone(), seq, payload))
                .await
                .expect("send replay batch");
        }
        router
            .send(replay_reply(identity, -1, Vec::new()))
            .await
            .expect("send replay END");
    });
    (port, handle)
}

fn replay_reply(identity: Bytes, seq: i64, payload: Vec<u8>) -> ZmqMessage {
    let mut msg = ZmqMessage::from(identity);
    msg.push_back(Bytes::new());
    msg.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
    msg.push_back(Bytes::from(payload));
    msg
}

async fn python_has_pyzmq(python: &str) -> bool {
    let status = Command::new(python)
        .arg("-c")
        .arg("import zmq")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;
    matches!(status, Ok(status) if status.success())
}

struct PythonHarness {
    child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
    worker_url: String,
}

impl PythonHarness {
    async fn spawn(python: &str) -> Self {
        let mut child = Command::new(python)
            .arg("tests/scripts/kv_replay_gap_py_worker.py")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn Python pyzmq harness");
        let stdin = child.stdin.take().expect("Python harness stdin");
        let stdout = child.stdout.take().expect("Python harness stdout");
        let mut stdout = BufReader::new(stdout).lines();
        let ready = tokio::time::timeout(Duration::from_secs(5), stdout.next_line())
            .await
            .expect("Python harness READY timeout")
            .expect("read Python harness READY")
            .expect("Python harness exited before READY");
        let body = ready
            .strip_prefix("READY ")
            .unwrap_or_else(|| panic!("unexpected Python harness READY line: {ready}"));
        let ready_json: Value = serde_json::from_str(body).expect("parse Python harness READY");
        let worker_url = ready_json
            .get("worker_url")
            .and_then(Value::as_str)
            .expect("Python harness READY worker_url")
            .to_string();

        Self {
            child,
            stdin,
            stdout,
            worker_url,
        }
    }

    async fn command(&mut self, command: &str) {
        self.stdin
            .write_all(command.as_bytes())
            .await
            .expect("write Python harness command");
        self.stdin
            .write_all(b"\n")
            .await
            .expect("write Python harness newline");
        self.stdin
            .flush()
            .await
            .expect("flush Python harness command");
        let response = tokio::time::timeout(Duration::from_secs(5), self.stdout.next_line())
            .await
            .unwrap_or_else(|_| panic!("Python harness command timed out: {command}"))
            .expect("read Python harness command response")
            .unwrap_or_else(|| panic!("Python harness exited after command: {command}"));
        assert!(
            response.starts_with("OK "),
            "Python harness rejected `{command}`: {response}"
        );
    }

    async fn stop(mut self) {
        self.command("stop").await;
        let status = tokio::time::timeout(Duration::from_secs(5), self.child.wait())
            .await
            .expect("Python harness stop timeout")
            .expect("wait for Python harness");
        assert!(status.success(), "Python harness exited with {status}");
    }
}

impl Drop for PythonHarness {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

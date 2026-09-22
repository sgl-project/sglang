// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Exercise the shipped binary, including CLI parsing, discovery and policy wiring.

use crate::common::mock_worker::MockWorker;
use serde_json::json;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;

#[tokio::test]
async fn reorg_launch_routes_each_supported_policy_and_enforces_admission() {
    let worker = MockWorker::start(vec![]).await;
    let full = MockWorker::start(vec![]).await;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();
    for policy in ["power_of_two", "cache_aware", "session_aware"] {
        let config = tempfile::NamedTempFile::new().unwrap();
        // The first bucket rejects every pick. Successful forwarding proves
        // that the configured bucket fallback and second policy were installed.
        std::fs::write(config.path(), json!({"buckets": [
            {"id": "full", "rank": 0, "max_input_tokens": 1000, "groups": {"mode": "plain", "plain": {
                "policy": policy, "worker_ids": [full.url], "admission": {"max_inflight_requests": 0}
            }}},
            {"id": "available", "rank": 1, "max_input_tokens": 1000,
             "groups": {"mode": "plain", "plain": {"policy": policy, "worker_ids": [worker.url]}}}
        ]}).to_string()).unwrap();
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let mut process = Command::new(env!("CARGO_BIN_EXE_sgl-router"))
            .args([
                "--model-id",
                "tiny",
                "--tokenizer-path",
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/fixtures/tiny_tokenizer.json"
                ),
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--worker-urls",
                &worker.url,
                &full.url,
                "--reorg-config",
                config.path().to_str().unwrap(),
                "--shutdown-drain-secs",
                "0",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let base = format!("http://127.0.0.1:{port}");
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                if let Some(status) = process.try_wait().unwrap() {
                    use tokio::io::AsyncReadExt;
                    let mut error = String::new();
                    process
                        .stderr
                        .take()
                        .unwrap()
                        .read_to_string(&mut error)
                        .await
                        .unwrap();
                    panic!("router exited {status}: {error}");
                }
                if client
                    .get(format!("{base}/readyz"))
                    .send()
                    .await
                    .is_ok_and(|r| r.status().is_success())
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .expect("router becomes ready");
        let response = client
            .post(format!("{base}/v1/chat/completions"))
            .header("x-session-id", "launch-test")
            .json(&json!({"model": "tiny", "messages": [{"role": "user", "content": "hello"}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            reqwest::StatusCode::OK,
            "{policy}: {}",
            response.text().await.unwrap()
        );
        assert!(
            full.captured.lock().unwrap().last_body.is_none(),
            "{policy} bypassed admission"
        );
        let rejected = client.post(format!("{base}/v1/chat/completions"))
            .json(&json!({"model": "tiny", "messages": [{"role": "user", "content": "hello ".repeat(1100)}]}))
            .send().await.unwrap();
        assert_eq!(
            rejected.status(),
            reqwest::StatusCode::BAD_REQUEST,
            "{policy}: {}",
            rejected.text().await.unwrap()
        );
        process.kill().await.unwrap();
        process.wait().await.unwrap();
    }
}

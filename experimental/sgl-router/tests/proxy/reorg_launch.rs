// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::common::mock_worker::MockWorker;
use serde_json::json;
use std::{process::Stdio, time::Duration};
use tokio::process::Command;

/// A free port below the ephemeral range, which the `127.0.0.1:0` listeners of
/// parallel tests never receive, so none of them can take it before the router
/// binds it or answer this test's polls in its place.
fn free_low_port() -> u16 {
    let ephemeral_start = std::fs::read_to_string("/proc/sys/net/ipv4/ip_local_port_range")
        .ok()
        .and_then(|range| range.split_whitespace().next()?.parse().ok())
        .unwrap_or(32_768);
    (1024..ephemeral_start)
        .rev()
        .find(|&port| std::net::TcpListener::bind(("127.0.0.1", port)).is_ok())
        .expect("a free port below the ephemeral range")
}

#[tokio::test]
async fn existing_policy_flags_launch_reorg_routing() {
    let worker = MockWorker::start(vec![]).await;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();
    for policy in ["power_of_two", "cache_aware", "session_aware"] {
        let port = free_low_port();
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
                "--chat-routing",
                "reorg",
                "--policy",
                policy,
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let base = format!("http://127.0.0.1:{port}");
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                assert!(process.try_wait().unwrap().is_none(), "{policy} exited");
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
        .unwrap();
        let response = client
            .post(format!("{base}/v1/chat/completions"))
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
        process.kill().await.unwrap();
        process.wait().await.unwrap();
    }
}

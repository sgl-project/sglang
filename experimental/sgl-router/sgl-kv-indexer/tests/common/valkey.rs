// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! A Valkey to test against: `KV_INDEXER_TEST_VALKEY_URL` if set, else a
//! `valkey-server` (or `redis-server`) from `PATH` spawned on a unix socket
//! in a temp dir that is removed on drop. `None` when neither is available.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use sgl_kv_indexer::{ValkeyConfig, ValkeyKvIndexerBackend};

pub struct ValkeyServer {
    pub url: String,
    child: Option<Child>,
    dir: Option<PathBuf>,
}

impl ValkeyServer {
    pub fn start() -> Option<Self> {
        Self::start_with(&[])
    }

    /// `extra_args` are appended to the server command line (ignored for an
    /// external server named by the environment).
    pub fn start_with(extra_args: &[&str]) -> Option<Self> {
        if let Ok(url) = std::env::var("KV_INDEXER_TEST_VALKEY_URL") {
            return Some(Self {
                url,
                child: None,
                dir: None,
            });
        }
        let binary = ["valkey-server", "redis-server"].into_iter().find(|name| {
            Command::new(name)
                .arg("--version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok()
        })?;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("sgl-kv-indexer-test-{nanos}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let socket = dir.join("valkey.sock");
        let child = Command::new(binary)
            .args(["--port", "0", "--unixsocket"])
            .arg(&socket)
            .args(["--save", "", "--appendonly", "no", "--loglevel", "warning"])
            .args(extra_args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn valkey-server");
        let deadline = Instant::now() + Duration::from_secs(5);
        while !socket.exists() {
            assert!(
                Instant::now() < deadline,
                "valkey-server did not open its socket"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        Some(Self {
            url: format!("valkey+unix://{}", socket.display()),
            child: Some(child),
            dir: Some(dir),
        })
    }

    pub fn config(&self, prefix: &str) -> ValkeyConfig {
        ValkeyConfig::new(self.url.clone()).with_key_prefix(prefix)
    }

    pub async fn backend(&self, prefix: &str) -> ValkeyKvIndexerBackend {
        ValkeyKvIndexerBackend::connect(self.config(prefix))
            .await
            .expect("connect to test valkey")
    }

    /// A raw client connection for assertions on the keyspace.
    pub async fn raw(&self) -> redis::aio::MultiplexedConnection {
        redis::Client::open(self.url.as_str())
            .unwrap()
            .get_multiplexed_async_connection()
            .await
            .expect("raw connection to test valkey")
    }
}

impl Drop for ValkeyServer {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(dir) = self.dir.take() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

/// Unique per test so tests sharing one external server never see each other.
pub fn fresh_prefix() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{{t{nanos}}}:")
}

// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use sgl_router::config::StaticFileDiscoveryConfig;
use sgl_router::discovery::{DiscoveryEvent, WorkerMode};
use std::time::Duration;
use tokio::sync::mpsc;

// We expose a test-only entrypoint that bypasses the dispatcher.
// (The dispatcher exists in spawn_discovery() but goes through Config.)
// For unit tests we call src/discovery/static_file.rs::spawn directly
// via a wrapper.

#[tokio::test]
async fn loads_initial_file_emits_added_events() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workers.toml");
    tokio::fs::write(
        &path,
        r#"
[[workers]]
id = "w1"
url = "http://x:30000"
mode = "plain"
model_ids = ["m1"]

[[workers]]
id = "w2"
url = "http://y:30000"
mode = "plain"
model_ids = ["m1"]
"#,
    )
    .await
    .unwrap();

    let (tx, mut rx) = mpsc::channel(16);
    let cfg = StaticFileDiscoveryConfig {
        path: path.to_string_lossy().into_owned(),
        poll_interval_ms: 50,
    };
    let _h = sgl_router::discovery::static_file::spawn(cfg, tx)
        .await
        .unwrap();

    let mut seen_ids = std::collections::HashSet::new();
    for _ in 0..2 {
        let event = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        match event {
            DiscoveryEvent::Added(spec) => {
                assert_eq!(spec.mode, WorkerMode::Plain);
                seen_ids.insert(spec.id.0.clone());
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }
    assert_eq!(seen_ids, ["w1".to_string(), "w2".to_string()].into());
}

#[tokio::test]
async fn add_worker_to_file_emits_added_event() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workers.toml");
    tokio::fs::write(
        &path,
        r#"
[[workers]]
id = "w1"
url = "http://x:30000"
mode = "plain"
model_ids = ["m1"]
"#,
    )
    .await
    .unwrap();

    let (tx, mut rx) = mpsc::channel(16);
    let cfg = StaticFileDiscoveryConfig {
        path: path.to_string_lossy().into_owned(),
        poll_interval_ms: 50,
    };
    let _h = sgl_router::discovery::static_file::spawn(cfg, tx)
        .await
        .unwrap();

    // First Added (initial load).
    let _ = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .unwrap()
        .unwrap();

    // Edit the file — add w2.
    tokio::fs::write(
        &path,
        r#"
[[workers]]
id = "w1"
url = "http://x:30000"
mode = "plain"
model_ids = ["m1"]

[[workers]]
id = "w2"
url = "http://y:30000"
mode = "plain"
model_ids = ["m1"]
"#,
    )
    .await
    .unwrap();

    // Expect an Added for w2 within poll interval + safety margin.
    let event = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let e = rx.recv().await.unwrap();
            if let DiscoveryEvent::Added(s) = &e {
                if s.id.0 == "w2" {
                    return e;
                }
            }
        }
    })
    .await
    .unwrap();

    match event {
        DiscoveryEvent::Added(spec) => assert_eq!(spec.id.0, "w2"),
        other => panic!("unexpected: {other:?}"),
    }
}

#[tokio::test]
async fn remove_worker_from_file_emits_removed_event() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workers.toml");
    tokio::fs::write(
        &path,
        r#"
[[workers]]
id = "w1"
url = "http://x:30000"
mode = "plain"
model_ids = ["m1"]

[[workers]]
id = "w2"
url = "http://y:30000"
mode = "plain"
model_ids = ["m1"]
"#,
    )
    .await
    .unwrap();

    let (tx, mut rx) = mpsc::channel(16);
    let cfg = StaticFileDiscoveryConfig {
        path: path.to_string_lossy().into_owned(),
        poll_interval_ms: 50,
    };
    let _h = sgl_router::discovery::static_file::spawn(cfg, tx)
        .await
        .unwrap();

    // Drain initial Added×2.
    let _ = rx.recv().await.unwrap();
    let _ = rx.recv().await.unwrap();

    // Remove w2 from file.
    tokio::fs::write(
        &path,
        r#"
[[workers]]
id = "w1"
url = "http://x:30000"
mode = "plain"
model_ids = ["m1"]
"#,
    )
    .await
    .unwrap();

    let event = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let e = rx.recv().await.unwrap();
            if let DiscoveryEvent::Removed { id } = &e {
                if id.0 == "w2" {
                    return e;
                }
            }
        }
    })
    .await
    .unwrap();

    assert!(matches!(event, DiscoveryEvent::Removed { .. }));
}

#[tokio::test]
async fn invalid_toml_does_not_panic_or_terminate_watcher() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workers.toml");
    tokio::fs::write(
        &path,
        r#"
[[workers]]
id = "w1"
url = "http://x:30000"
mode = "plain"
model_ids = ["m1"]
"#,
    )
    .await
    .unwrap();

    let (tx, mut rx) = mpsc::channel(16);
    let cfg = StaticFileDiscoveryConfig {
        path: path.to_string_lossy().into_owned(),
        poll_interval_ms: 50,
    };
    let _h = sgl_router::discovery::static_file::spawn(cfg, tx)
        .await
        .unwrap();
    let _ = rx.recv().await.unwrap();

    // Write garbage.
    tokio::fs::write(&path, "not valid toml { } [[")
        .await
        .unwrap();

    // Wait — no events should arrive (parser drops the change),
    // and the watcher must still be alive.
    let elapsed = tokio::time::timeout(Duration::from_millis(500), rx.recv()).await;
    assert!(elapsed.is_err(), "no events expected, got: {elapsed:?}");

    // Repair the file.
    tokio::fs::write(
        &path,
        r#"
[[workers]]
id = "w1"
url = "http://x:30000"
mode = "plain"
model_ids = ["m1"]

[[workers]]
id = "w2"
url = "http://y:30000"
mode = "plain"
model_ids = ["m1"]
"#,
    )
    .await
    .unwrap();

    // w2 should now arrive — watcher is still alive.
    let event = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let e = rx.recv().await.unwrap();
            if let DiscoveryEvent::Added(s) = &e {
                if s.id.0 == "w2" {
                    return e;
                }
            }
        }
    })
    .await
    .unwrap();
    assert!(matches!(event, DiscoveryEvent::Added(_)));
}

/// Editors and config-management tooling (vim, k8s ConfigMap mounts,
/// `mv`-based deploys) often replace a file via `rename(tmp, target)`
/// rather than overwriting in place.  The watcher must pick up the new
/// content from a rename, otherwise rolling a worker pool via atomic
/// swap would silently leave the router on the old topology.
#[tokio::test]
async fn atomic_rename_replacement_is_picked_up_by_watcher() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workers.toml");
    tokio::fs::write(
        &path,
        r#"
[[workers]]
id = "w1"
url = "http://x:30000"
mode = "plain"
model_ids = ["m1"]
"#,
    )
    .await
    .unwrap();

    let (tx, mut rx) = mpsc::channel(16);
    let cfg = StaticFileDiscoveryConfig {
        path: path.to_string_lossy().into_owned(),
        poll_interval_ms: 50,
    };
    let _h = sgl_router::discovery::static_file::spawn(cfg, tx)
        .await
        .unwrap();
    // Drain the initial Added for w1.
    let _ = rx.recv().await.unwrap();

    // Atomically swap in a new file containing w1+w2 via rename.
    let tmp = dir.path().join("workers.toml.new");
    tokio::fs::write(
        &tmp,
        r#"
[[workers]]
id = "w1"
url = "http://x:30000"
mode = "plain"
model_ids = ["m1"]

[[workers]]
id = "w2"
url = "http://y:30000"
mode = "plain"
model_ids = ["m1"]
"#,
    )
    .await
    .unwrap();
    tokio::fs::rename(&tmp, &path).await.unwrap();

    let event = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let e = rx.recv().await.unwrap();
            if let DiscoveryEvent::Added(s) = &e {
                if s.id.0 == "w2" {
                    return e;
                }
            }
        }
    })
    .await
    .expect("watcher must observe the renamed file's new content");
    assert!(matches!(event, DiscoveryEvent::Added(_)));
}

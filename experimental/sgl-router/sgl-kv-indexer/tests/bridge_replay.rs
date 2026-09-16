// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! The bridge must recover what the worker still buffers: on connect it asks
//! the replay socket for everything after its last sequence, a gap in the live
//! stream triggers a bounded replay, and a sequence reset (worker restart)
//! clears the worker's placements before new ones are applied.

#[allow(dead_code)]
#[path = "common/net.rs"]
mod test_net;
#[allow(dead_code)]
#[path = "common/valkey.rs"]
mod test_valkey;
#[allow(dead_code)]
#[path = "common/zmq.rs"]
mod test_zmq;

use std::sync::Arc;
use std::time::{Duration, Instant};

use sgl_kv_indexer::bridge::{run_bridge_until, BridgeConfig, Sink};
use sgl_kv_indexer::pb::{MatchExternalKvRequest, TierType};
use sgl_kv_indexer::{
    server_builder, InMemoryKvIndexerBackend, KvIndexerBackend, KvIndexerService,
    DEFAULT_STREAM_MAXLEN,
};
use test_net::free_addr;
use test_valkey::{fresh_prefix, ValkeyServer};
use test_zmq::{batch, removed, stored, FakePublisher};

const TOPIC: &str = "kv-events";

/// An in-memory indexer over real gRPC, returning the backend for assertions.
async fn start_indexer() -> (Arc<InMemoryKvIndexerBackend>, String) {
    let backend = Arc::new(InMemoryKvIndexerBackend::new());
    let shared: Arc<dyn KvIndexerBackend> = backend.clone();
    let svc = KvIndexerService::new(shared).into_server();
    let addr = free_addr();
    tokio::spawn(async move {
        server_builder()
            .add_service(svc)
            .serve(addr)
            .await
            .expect("server serve");
    });
    let endpoint = format!("http://{addr}");
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if tokio::net::TcpStream::connect(addr).await.is_ok() {
            break;
        }
        assert!(Instant::now() < deadline, "indexer never listened");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    (backend, endpoint)
}

fn config(publisher: &FakePublisher, indexer: &str, replay: bool) -> BridgeConfig {
    BridgeConfig {
        worker_id: "worker-0".to_string(),
        worker_address: "http://127.0.0.1:30000".to_string(),
        event_endpoint: publisher.pub_endpoint.clone(),
        event_topic: TOPIC.to_string(),
        indexer_endpoint: indexer.to_string(),
        clear_tiers: vec![TierType::TierHbm as i32, TierType::TierDram as i32],
        cache_spec: None,
        sink: Sink::Grpc,
        valkey: None,
        heartbeat_ttl: None,
        replay_endpoint: replay.then(|| publisher.replay_endpoint.clone()),
        stream_maxlen: DEFAULT_STREAM_MAXLEN,
    }
}

struct RunningBridge {
    stop: tokio::sync::oneshot::Sender<()>,
    task: tokio::task::JoinHandle<()>,
}

async fn start_bridge(config: BridgeConfig) -> RunningBridge {
    let (stop, rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        run_bridge_until(config, async move {
            let _ = rx.await;
        })
        .await
        .expect("bridge");
    });
    // ZMQ SUB is a slow joiner: give the subscription time to reach the PUB.
    tokio::time::sleep(Duration::from_millis(300)).await;
    RunningBridge { stop, task }
}

impl RunningBridge {
    async fn stop(self) {
        let _ = self.stop.send(());
        let _ = self.task.await;
    }
}

async fn held(backend: &InMemoryKvIndexerBackend, hashes: &[i64]) -> Vec<i64> {
    let resp = backend
        .match_external_kv(MatchExternalKvRequest {
            hashes: hashes.to_vec(),
            count_as_hit: false,
        })
        .await
        .unwrap();
    let mut held: Vec<i64> = resp
        .matches
        .iter()
        .flat_map(|node| node.hashes_by_tier.iter().flat_map(|t| t.hashes.clone()))
        .collect();
    held.sort();
    held.dedup();
    held
}

async fn wait_for(backend: &InMemoryKvIndexerBackend, probe: &[i64], expected: &[i64], what: &str) {
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        let got = held(backend, probe).await;
        if got == expected {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{what}: expected {expected:?}, got {got:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn replay_on_connect_recovers_the_worker_buffer() {
    let (backend, indexer) = start_indexer().await;
    let publisher = FakePublisher::bind(TOPIC).await;
    // Published before any bridge existed: only the replay buffer knows them.
    publisher.buffer_only(0, batch(vec![stored(&[1, 2], None)]));
    publisher.buffer_only(1, batch(vec![stored(&[3], Some(2))]));
    publisher.buffer_only(2, batch(vec![removed(&[1])]));

    let bridge = start_bridge(config(&publisher, &indexer, true)).await;
    wait_for(&backend, &[1, 2, 3], &[2, 3], "replay on connect").await;
    bridge.stop().await;
}

#[tokio::test]
async fn without_replay_endpoint_the_buffer_is_not_consulted() {
    let (backend, indexer) = start_indexer().await;
    let publisher = FakePublisher::bind(TOPIC).await;
    publisher.buffer_only(0, batch(vec![stored(&[1, 2], None)]));

    let bridge = start_bridge(config(&publisher, &indexer, false)).await;
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(held(&backend, &[1, 2]).await.is_empty());
    bridge.stop().await;
}

#[tokio::test]
async fn a_gap_in_the_live_stream_is_filled_from_the_buffer() {
    let (backend, indexer) = start_indexer().await;
    let mut publisher = FakePublisher::bind(TOPIC).await;
    let bridge = start_bridge(config(&publisher, &indexer, true)).await;

    publisher.publish(0, batch(vec![stored(&[1], None)])).await;
    wait_for(&backend, &[1], &[1], "first live batch").await;
    // Two batches the SUB never saw, then a live one that exposes the gap.
    publisher.buffer_only(1, batch(vec![stored(&[2], Some(1))]));
    publisher.buffer_only(2, batch(vec![stored(&[3], Some(2))]));
    publisher
        .publish(3, batch(vec![stored(&[4], Some(3))]))
        .await;
    wait_for(&backend, &[1, 2, 3, 4], &[1, 2, 3, 4], "gap replay").await;
    bridge.stop().await;
}

#[tokio::test]
async fn a_sequence_reset_clears_the_worker_before_new_events() {
    let (backend, indexer) = start_indexer().await;
    let mut publisher = FakePublisher::bind(TOPIC).await;
    let bridge = start_bridge(config(&publisher, &indexer, false)).await;

    publisher
        .publish(10, batch(vec![stored(&[1, 2], None)]))
        .await;
    publisher
        .publish(11, batch(vec![stored(&[3], Some(2))]))
        .await;
    wait_for(&backend, &[1, 2, 3], &[1, 2, 3], "before restart").await;
    // The worker restarted: its publisher counts from zero and its cache is empty.
    publisher.publish(0, batch(vec![stored(&[7], None)])).await;
    wait_for(&backend, &[1, 2, 3, 7], &[7], "after restart").await;
    bridge.stop().await;
}

/// A restarted bridge must resume from its checkpoint and ask the worker only
/// for what it missed; without the checkpoint it would replay the whole buffer
/// and briefly re-report blocks that later events removed.
#[tokio::test]
async fn restarted_bridge_resumes_from_its_valkey_checkpoint() {
    let Some(server) = ValkeyServer::start() else {
        eprintln!("skipping: no valkey-server on PATH and KV_INDEXER_TEST_VALKEY_URL unset");
        return;
    };
    let (backend, indexer) = start_indexer().await;
    let mut publisher = FakePublisher::bind(TOPIC).await;
    let mut cfg = config(&publisher, &indexer, true);
    cfg.valkey = Some(server.config(&fresh_prefix()));
    cfg.heartbeat_ttl = None;

    let first = start_bridge(cfg.clone()).await;
    publisher
        .publish(0, batch(vec![stored(&[1, 2], None)]))
        .await;
    publisher
        .publish(1, batch(vec![stored(&[3], Some(2))]))
        .await;
    wait_for(&backend, &[1, 2, 3], &[1, 2, 3], "live before restart").await;
    first.stop().await;

    // Missed while the bridge was down.
    publisher.buffer_only(2, batch(vec![stored(&[4], Some(3))]));
    let second = start_bridge(cfg).await;
    wait_for(
        &backend,
        &[1, 2, 3, 4],
        &[1, 2, 3, 4],
        "replay after restart",
    )
    .await;
    second.stop().await;

    let requests = publisher.replay_requests.lock().unwrap().clone();
    assert_eq!(
        requests.first(),
        Some(&0),
        "a fresh bridge starts from the beginning"
    );
    assert_eq!(
        requests.last(),
        Some(&2),
        "the restarted bridge resumed after its checkpoint"
    );
}

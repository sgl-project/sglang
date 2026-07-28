// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end reliability tests for the bridge's sequence-gap handling. A fake
//! SGLang publisher (ZMQ PUB for live events + ROUTER for the replay buffer)
//! emits batches with deliberate holes in the `seq` stream; a capturing
//! in-memory gRPC indexer records the `seq` and incarnation of every applied
//! batch.
//!
//! Both branches of the gap contract are covered, in this order:
//!   * recoverable -- the bridge pulls the missing batches from the replay
//!     endpoint (DEALER -> ROUTER) and applies everything exactly once in
//!     monotonic order, keeping its incarnation;
//!   * unrecoverable -- once the replay buffer can no longer close the gap the
//!     bridge retires the incarnation, which is how the indexer is told to wipe
//!     placements it can no longer reconstruct. This is the destructive path,
//!     asserted in `bridge_recovers_seq_gap_via_replay`.
//!
//! No Redis required: the capturing backend implements `KvIndexerBackend`
//! directly, so this runs in the default `cargo test`.

#[path = "common/net.rs"]
mod test_net;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use tonic::transport::Server;
use tonic::Status;
use zeromq::{PubSocket, RouterSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

use sgl_kv_indexer::bridge::{run_bridge, run_bridge_until, BridgeConfig};
use sgl_kv_indexer::pb::kv_indexer_server::KvIndexerServer;
use sgl_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, GetExternalKvHitCountsRequest,
    GetExternalKvHitCountsResponse, MatchExternalKvRequest, MatchExternalKvResponse,
};
use sgl_kv_indexer::{KvIndexerBackend, KvIndexerService};
use test_net::free_addr;

/// Sequence-tagged batches the fake publisher has emitted, shared between the
/// test body and the replay responder.
type ReplayBuffer = Arc<Mutex<Vec<(u64, Vec<u8>)>>>;

/// gRPC backend that just records the seq of every applied batch, in order.
#[derive(Clone, Default)]
struct CapturingBackend {
    seqs: Arc<Mutex<Vec<u64>>>,
    incarnations: Arc<Mutex<Vec<String>>>,
    last_seq: Arc<Mutex<Option<u64>>>,
}

#[tonic::async_trait]
impl KvIndexerBackend for CapturingBackend {
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        let checkpoint = if request.actions.is_empty() {
            *self.last_seq.lock().unwrap()
        } else {
            self.seqs.lock().unwrap().push(request.seq);
            self.incarnations
                .lock()
                .unwrap()
                .push(request.incarnation.clone());
            *self.last_seq.lock().unwrap() = Some(request.seq);
            Some(request.seq)
        };
        Ok(ApplyExternalKvBatchResponse {
            last_applied_seq: checkpoint.unwrap_or_default(),
            duplicate: false,
            has_applied_seq: checkpoint.is_some(),
        })
    }

    async fn match_external_kv(
        &self,
        _request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        Ok(MatchExternalKvResponse { matches: vec![] })
    }

    async fn get_external_kv_hit_counts(
        &self,
        _request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        Ok(GetExternalKvHitCountsResponse { entries: vec![] })
    }
}

/// Encodes a minimal SGLang `KVEventBatch` = [ts, [events]] carrying one
/// `BlockStored` (7 fields; [1]=block_hashes, [6]=medium) so the bridge decodes
/// it into a single REPORT action.
fn stored_payload(hash: i64, medium: &str) -> Vec<u8> {
    use rmpv::Value;
    let event = Value::Array(vec![
        Value::from("BlockStored"),
        Value::Array(vec![Value::from(hash)]),
        Value::Nil,
        Value::Nil,
        Value::Nil,
        Value::Nil,
        Value::from(medium),
    ]);
    let batch = Value::Array(vec![Value::from(0u64), Value::Array(vec![event])]);
    let mut buf = Vec::new();
    rmpv::encode::write_value(&mut buf, &batch).expect("encode msgpack");
    buf
}

/// One live PUB frame: [seq(8B BE u64), payload]. Empty topic => the bridge's
/// empty subscription receives it.
fn pub_frame(seq: u64, payload: Vec<u8>) -> ZmqMessage {
    let mut m = ZmqMessage::from(Bytes::copy_from_slice(&seq.to_be_bytes()));
    m.push_back(Bytes::from(payload));
    m
}

/// One ROUTER reply routed to `peer`: [peer, b"", seq(8B BE i64), payload].
/// After ROUTER pops `peer`, the bridge's DEALER sees [b"", seq, payload].
fn reply_frame(peer: Bytes, seq: i64, payload: Vec<u8>) -> ZmqMessage {
    let mut m = ZmqMessage::from(peer);
    m.push_back(Bytes::new());
    m.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
    m.push_back(Bytes::from(payload));
    m
}

#[tokio::test]
async fn bridge_recovers_seq_gap_via_replay() {
    // --- capturing gRPC indexer ---
    let backend = CapturingBackend::default();
    let seqs = backend.seqs.clone();
    let incarnations = backend.incarnations.clone();
    let grpc_addr = free_addr();
    let svc = KvIndexerServer::new(KvIndexerService::new(backend));
    tokio::spawn(async move {
        Server::builder()
            .add_service(svc)
            .serve(grpc_addr)
            .await
            .expect("grpc serve");
    });

    // --- fake SGLang: PUB (live) + ROUTER (replay buffer) ---
    let mut publisher = PubSocket::new();
    let pub_ep = publisher.bind("tcp://127.0.0.1:0").await.expect("bind pub");
    let mut router = RouterSocket::new();
    let router_ep = router.bind("tcp://127.0.0.1:0").await.expect("bind router");

    // ROUTER responder: first answer the bridge's liveness probe (u64::MAX),
    // then stream buffered missing batches (2, 3) for the real gap request.
    tokio::spawn(async move {
        loop {
            let req = router.recv().await.expect("router recv").into_vec();
            let peer = req[0].clone();
            let start = u64::from_be_bytes(req[2].as_ref().try_into().unwrap());
            if start == u64::MAX {
                router
                    .send(reply_frame(peer, -1, Vec::new()))
                    .await
                    .expect("router send probe terminator");
                continue;
            }
            let replay_seqs: &[i64] = match start {
                // Deliberately incomplete: seq 2 is absent, so a later live
                // seq 3 must not be committed past the unresolved gap.
                1 => &[1],
                2 => &[2, 3],
                other => panic!("unexpected replay start {other}"),
            };
            for &seq in replay_seqs {
                router
                    .send(reply_frame(
                        peer.clone(),
                        seq,
                        stored_payload(1000 + seq, "GPU"),
                    ))
                    .await
                    .expect("router send batch");
            }
            router
                .send(reply_frame(peer, -1, Vec::new()))
                .await
                .expect("router send terminator");
        }
    });

    // --- bridge under test ---
    let config = BridgeConfig {
        worker_id: "worker-test".to_string(),
        worker_address: String::new(),
        event_endpoint: pub_ep.to_string(),
        event_replay_endpoint: Some(router_ep.to_string()),
        event_topic: String::new(),
        indexer_endpoint: format!("http://{grpc_addr}"),
        clear_tiers: vec![],
        heartbeat_interval: None,
        incarnation: "replay-test".to_string(),
        incarnation_path: None,
    };
    tokio::spawn(async move {
        let _ = run_bridge(config).await;
    });

    // Let the bridge's SUB connect/subscribe and gRPC client connect before we
    // publish (PUB/SUB has no handshake; early sends would be dropped).
    tokio::time::sleep(Duration::from_millis(1200)).await;

    // Live stream with a hole: 0, 1, then jump to 4 (2 and 3 are "missed").
    publisher
        .send(pub_frame(0, stored_payload(1000, "GPU")))
        .await
        .expect("pub 0");
    tokio::time::sleep(Duration::from_millis(100)).await;
    publisher
        .send(pub_frame(1, stored_payload(1001, "GPU")))
        .await
        .expect("pub 1");
    tokio::time::sleep(Duration::from_millis(100)).await;
    publisher
        .send(pub_frame(4, stored_payload(1004, "GPU")))
        .await
        .expect("pub 4");

    // Wait until all five seqs are applied (or time out).
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        {
            let got = seqs.lock().unwrap();
            if got.len() >= 5 {
                break;
            }
        }
        if std::time::Instant::now() > deadline {
            panic!(
                "timed out; applied seqs so far: {:?}",
                *seqs.lock().unwrap()
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let got = seqs.lock().unwrap().clone();
    assert_eq!(
        got,
        vec![0, 1, 2, 3, 4],
        "bridge must apply every batch exactly once in monotonic order (gap 2,3 recovered via replay)"
    );

    publisher
        .send(pub_frame(0, vec![0xc1]))
        .await
        .expect("pub malformed rollback");
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        seqs.lock().unwrap().len(),
        5,
        "malformed rollback must not reset sequence/incarnation state"
    );

    // A worker-owned publisher restart resets SGLang's sequence generator. The
    // still-running bridge must rotate incarnation instead of discarding the
    // new stream forever as stale.
    publisher
        .send(pub_frame(0, stored_payload(2000, "GPU")))
        .await
        .expect("pub restarted seq 0");
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while seqs.lock().unwrap().len() < 6 {
        assert!(
            std::time::Instant::now() <= deadline,
            "timed out waiting for restarted publisher batch"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(seqs.lock().unwrap().as_slice(), &[0, 1, 2, 3, 4, 0]);
    {
        let incarnations = incarnations.lock().unwrap();
        assert!(incarnations[..5]
            .iter()
            .all(|incarnation| incarnation == "replay-test"));
        assert_ne!(
            incarnations[5], "replay-test",
            "publisher restart must rotate the worker incarnation"
        );
    }

    // The replay endpoint can only return seq 1 for the 1..3 gap. Recovery is
    // best effort: the bridge commits the recoverable prefix, then retires the
    // incarnation so the indexer wipes what can no longer be reconstructed, and
    // resyncs from the live stream. Advancing past seq 2 under the *same*
    // incarnation would silently keep stale placements alive.
    publisher
        .send(pub_frame(3, stored_payload(3003, "GPU")))
        .await
        .expect("pub incomplete-gap seq 3");
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while seqs.lock().unwrap().len() < 8 {
        assert!(
            std::time::Instant::now() <= deadline,
            "timed out waiting for resync after an unrecoverable gap: {:?}",
            *seqs.lock().unwrap()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        seqs.lock().unwrap().as_slice(),
        &[0, 1, 2, 3, 4, 0, 1, 3],
        "bridge must recover the replayable prefix and then resync past an unrecoverable gap"
    );
    {
        let incarnations = incarnations.lock().unwrap();
        assert_ne!(
            incarnations[7], incarnations[6],
            "an unrecoverable gap must retire the worker incarnation"
        );
    }
}

#[tokio::test]
async fn bridge_process_restart_resumes_from_durable_checkpoint() {
    let backend = CapturingBackend::default();
    *backend.last_seq.lock().unwrap() = Some(1);
    let seqs = backend.seqs.clone();
    let incarnations = backend.incarnations.clone();
    let grpc_addr = free_addr();
    let svc = KvIndexerServer::new(KvIndexerService::new(backend));
    tokio::spawn(async move {
        Server::builder()
            .add_service(svc)
            .serve(grpc_addr)
            .await
            .expect("grpc serve");
    });

    let mut publisher = PubSocket::new();
    let pub_ep = publisher.bind("tcp://127.0.0.1:0").await.expect("bind pub");
    let mut router = RouterSocket::new();
    let router_ep = router.bind("tcp://127.0.0.1:0").await.expect("bind router");
    tokio::spawn(async move {
        loop {
            let req = router.recv().await.expect("router recv").into_vec();
            let peer = req[0].clone();
            let start = u64::from_be_bytes(req[2].as_ref().try_into().unwrap());
            if start == u64::MAX {
                router
                    .send(reply_frame(peer, -1, Vec::new()))
                    .await
                    .expect("router send probe");
                continue;
            }
            assert_eq!(start, 2, "bridge must resume after durable seq 1");
            for seq in [2_i64, 3] {
                router
                    .send(reply_frame(
                        peer.clone(),
                        seq,
                        stored_payload(4000 + seq, "GPU"),
                    ))
                    .await
                    .expect("router send replay");
            }
            router
                .send(reply_frame(peer, -1, Vec::new()))
                .await
                .expect("router send terminator");
        }
    });

    let config = BridgeConfig {
        worker_id: "restart-worker".to_string(),
        worker_address: String::new(),
        event_endpoint: pub_ep.to_string(),
        event_replay_endpoint: Some(router_ep.to_string()),
        event_topic: String::new(),
        indexer_endpoint: format!("http://{grpc_addr}"),
        clear_tiers: vec![],
        heartbeat_interval: None,
        incarnation: "stable-publisher".to_string(),
        incarnation_path: None,
    };
    tokio::spawn(async move {
        let _ = run_bridge(config).await;
    });

    tokio::time::sleep(Duration::from_millis(1200)).await;
    publisher
        .send(pub_frame(4, stored_payload(4004, "GPU")))
        .await
        .expect("publish post-restart live batch");

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while seqs.lock().unwrap().len() < 3 {
        assert!(
            std::time::Instant::now() <= deadline,
            "timed out waiting for restart replay: {:?}",
            *seqs.lock().unwrap()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(seqs.lock().unwrap().as_slice(), &[2, 3, 4]);
    assert!(incarnations
        .lock()
        .unwrap()
        .iter()
        .all(|value| value == "stable-publisher"));
}

/// A `SIGTERM`-style shutdown must return promptly and leave the indexer's
/// checkpoint intact, so the replacement process replays what was published
/// while nothing was listening rather than skipping it.
#[tokio::test]
async fn graceful_shutdown_keeps_the_replay_checkpoint_usable() {
    let backend = CapturingBackend::default();
    let seqs = backend.seqs.clone();
    let grpc_addr = free_addr();
    let svc = KvIndexerServer::new(KvIndexerService::new(backend));
    tokio::spawn(async move {
        Server::builder()
            .add_service(svc)
            .serve(grpc_addr)
            .await
            .expect("grpc serve");
    });

    let mut publisher = PubSocket::new();
    let pub_ep = publisher.bind("tcp://127.0.0.1:0").await.expect("bind pub");
    let mut router = RouterSocket::new();
    let router_ep = router.bind("tcp://127.0.0.1:0").await.expect("bind router");

    // Stands in for SGLang's replay buffer: every batch the publisher has
    // emitted, served from the requested sequence onwards.
    let buffer: ReplayBuffer = Arc::new(Mutex::new(Vec::new()));
    let replay = buffer.clone();
    tokio::spawn(async move {
        loop {
            let req = router.recv().await.expect("router recv").into_vec();
            let peer = req[0].clone();
            let start = u64::from_be_bytes(req[2].as_ref().try_into().unwrap());
            if start != u64::MAX {
                let pending: Vec<(u64, Vec<u8>)> = replay
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(seq, _)| *seq >= start)
                    .cloned()
                    .collect();
                for (seq, payload) in pending {
                    router
                        .send(reply_frame(peer.clone(), seq as i64, payload))
                        .await
                        .expect("router send replay");
                }
            }
            router
                .send(reply_frame(peer, -1, Vec::new()))
                .await
                .expect("router send terminator");
        }
    });

    let config = BridgeConfig {
        worker_id: "shutdown-worker".to_string(),
        worker_address: String::new(),
        event_endpoint: pub_ep.to_string(),
        event_replay_endpoint: Some(router_ep.to_string()),
        event_topic: String::new(),
        indexer_endpoint: format!("http://{grpc_addr}"),
        clear_tiers: vec![],
        heartbeat_interval: None,
        incarnation: "stable-publisher".to_string(),
        incarnation_path: None,
    };

    let stop = Arc::new(AtomicBool::new(false));
    let signalled = stop.clone();
    let first = tokio::spawn(run_bridge_until(config.clone(), async move {
        while !signalled.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }));

    // PUB drops messages sent before the subscriber has finished connecting.
    tokio::time::sleep(Duration::from_millis(1200)).await;
    for seq in [0_u64, 1] {
        let payload = stored_payload(5000 + seq as i64, "GPU");
        buffer.lock().unwrap().push((seq, payload.clone()));
        publisher
            .send(pub_frame(seq, payload))
            .await
            .expect("publish live batch");
    }
    await_seqs(&seqs, 2).await;

    stop.store(true, Ordering::Relaxed);
    let stopped = tokio::time::timeout(Duration::from_secs(5), first)
        .await
        .expect("bridge must return promptly after a shutdown signal")
        .expect("bridge task must not panic");
    assert!(stopped.is_ok(), "clean shutdown must not surface an error");

    // Published into the void: the first bridge is gone and the second has not
    // started, so these only exist in the replay buffer.
    for seq in [2_u64, 3] {
        let payload = stored_payload(5000 + seq as i64, "GPU");
        buffer.lock().unwrap().push((seq, payload.clone()));
        publisher
            .send(pub_frame(seq, payload))
            .await
            .expect("publish during downtime");
    }

    tokio::spawn(run_bridge_until(config, std::future::pending()));
    tokio::time::sleep(Duration::from_millis(1200)).await;
    let payload = stored_payload(5004, "GPU");
    buffer.lock().unwrap().push((4, payload.clone()));
    publisher
        .send(pub_frame(4, payload))
        .await
        .expect("publish after restart");

    await_seqs(&seqs, 5).await;
    assert_eq!(seqs.lock().unwrap().as_slice(), &[0, 1, 2, 3, 4]);
}

async fn await_seqs(seqs: &Arc<Mutex<Vec<u64>>>, want: usize) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while seqs.lock().unwrap().len() < want {
        assert!(
            std::time::Instant::now() <= deadline,
            "timed out waiting for {want} applied batches: {:?}",
            *seqs.lock().unwrap()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

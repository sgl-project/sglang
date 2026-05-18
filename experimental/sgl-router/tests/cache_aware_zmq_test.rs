// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! E2E test for the cache-aware-zmq policy.
//!
//! Drives a real ZMQ PUB socket → `KvEventIndex` subscriber pipeline →
//! `HashTree` → `CacheAwareZmqPolicy::select`. Verifies that an event
//! published by one worker's PUB causes subsequent selection to route
//! to that worker (cache-aware affinity).
//!
//! API constraint: the subscriber registry builds endpoints as
//! `tcp://{host}:{port_base + dp_rank}` where `port_base` is in the
//! per-worker `EventConfig`. Both mock workers below share
//! `127.0.0.1` as host, so both subscribe to the same PUB socket and
//! both end up indexed in the tree. The tiebreak (lowest active_load)
//! picks the worker we want; same shape as the SMG version of this
//! test.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use rmp::encode as mp;
use zeromq::{Endpoint, PubSocket, Socket, SocketSend, ZmqMessage};

use sgl_router::config::CacheAwareConfig;
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::cache_aware_zmq::CacheAwareZmqPolicy;
use sgl_router::policies::kv_events::{compute_block_hashes, discovery::EventConfig, KvEventIndex};
use sgl_router::policies::{Policy, SelectionContext};
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::Worker;

/// Bind a PUB socket to an OS-assigned 127.0.0.1 port. Returns
/// `(socket, port)`.
async fn make_pub_bound() -> (PubSocket, u16) {
    let mut sock = PubSocket::new();
    let endpoint = sock
        .bind("tcp://127.0.0.1:0")
        .await
        .expect("bind PUB socket");
    let port = match endpoint {
        Endpoint::Tcp(_, p) => p,
        other => panic!("unexpected endpoint: {other:?}"),
    };
    (sock, port)
}

/// Encode a single `BlockStored` event in the same wire format msgspec
/// emits. Layout: `["BlockStored", block_hashes, parent, token_ids,
/// block_size, lora_id, medium]`.
fn encode_block_stored_event(
    block_hashes: &[i64],
    parent: Option<i64>,
    token_ids: &[u32],
    block_size: u32,
) -> Vec<u8> {
    let mut buf = Vec::new();
    mp::write_array_len(&mut buf, 7).unwrap();
    mp::write_str(&mut buf, "BlockStored").unwrap();
    mp::write_array_len(&mut buf, block_hashes.len() as u32).unwrap();
    for v in block_hashes {
        mp::write_sint(&mut buf, *v).unwrap();
    }
    match parent {
        Some(v) => {
            mp::write_sint(&mut buf, v).unwrap();
        }
        None => mp::write_nil(&mut buf).unwrap(),
    }
    mp::write_array_len(&mut buf, token_ids.len() as u32).unwrap();
    for v in token_ids {
        mp::write_uint(&mut buf, *v as u64).unwrap();
    }
    mp::write_uint(&mut buf, block_size as u64).unwrap();
    mp::write_nil(&mut buf).unwrap(); // lora_id
    mp::write_str(&mut buf, "GPU").unwrap();
    buf
}

/// Wrap one or more pre-encoded events into a KVEventBatch with
/// timestamp + optional dp-rank.
fn encode_event_batch(ts: f64, events: Vec<Vec<u8>>, attn_dp_rank: Option<u32>) -> Vec<u8> {
    let mut buf = Vec::new();
    mp::write_array_len(&mut buf, 3).unwrap();
    mp::write_f64(&mut buf, ts).unwrap();
    mp::write_array_len(&mut buf, events.len() as u32).unwrap();
    for ev in events {
        buf.extend_from_slice(&ev);
    }
    match attn_dp_rank {
        Some(v) => {
            mp::write_uint(&mut buf, v as u64).unwrap();
        }
        None => mp::write_nil(&mut buf).unwrap(),
    }
    buf
}

fn build_multipart(seq: i64, payload: Vec<u8>) -> ZmqMessage {
    let mut msg = ZmqMessage::from(Bytes::new());
    msg.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
    msg.push_back(Bytes::from(payload));
    msg
}

fn build_worker(url: &str, model: &str) -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(url.into()),
        url: url.into(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId(model.into())],
    }))
}

/// E2E: real PUB socket publishes a `BlockStored` for worker A's
/// hash chain. The `CacheAwareZmqPolicy`'s shared `KvEventIndex`
/// receives it, applies it to the tree, and the next `select` call
/// picks worker A.
///
/// Both workers share `127.0.0.1` as host so both subscribers connect
/// to the same PUB and both get indexed under their KvWorkerIds — the
/// same shape as the SMG e2e test. We tie-break on min-load: worker B
/// is bumped above worker A so the matched-worker pick prefers A.
#[tokio::test]
async fn zmq_indexer_routes_to_publishing_worker_e2e() {
    let model_id = ModelId("tiny".into());

    // 1. Tokenizer registry — use the in-tree tiny fixture.
    let cfg = sgl_router::config::Config {
        server: sgl_router::config::ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: Default::default(),
        models: vec![sgl_router::config::ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: sgl_router::config::PolicyKind::CacheAwareZmq,
            circuit_breaker: None,
            cache_aware: None,
        }],
        discovery: sgl_router::config::DiscoveryConfig {
            backend: sgl_router::config::DiscoveryBackend::StaticFile(
                sgl_router::config::StaticFileDiscoveryConfig {
                    path: "/tmp/x.toml".into(),
                    poll_interval_ms: 200,
                },
            ),
        },
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());

    // 2. Bind a real PUB socket on an OS-assigned port.
    let (mut pub_a, port) = make_pub_bound().await;

    // 3. Compute the hash chain for the routing prompt.
    let text = "hello world hello world hello world";
    let tok = tokenizers.get("tiny").unwrap();
    let token_ids = sgl_router::tokenizer::adapter::encode(&tok, text).unwrap();
    let block_size = 4u32;
    let hashes = compute_block_hashes(&token_ids, block_size as usize);
    assert!(!hashes.is_empty(), "tiny tokenizer must yield ≥1 block");

    // 4. Build the KvEventIndex + policy. The policy holds an
    //    Arc<HashTree> that the index also owns; events the index
    //    receives mutate the same tree the policy reads.
    let kv_index = KvEventIndex::new();
    let policy = CacheAwareZmqPolicy::new(
        CacheAwareConfig {
            cache_threshold: 0.0,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            block_size,
        },
        kv_index.tree(),
        Arc::clone(&tokenizers),
    );

    // 5. Register two workers. They share `127.0.0.1` so both
    //    subscribers connect to the same PUB; preresolved EventConfig
    //    points at the bound port.
    let url_a = "http://127.0.0.1:30000";
    let url_b = "http://127.0.0.1:30001";
    let preresolved = EventConfig {
        host: "127.0.0.1".to_string(),
        port_base: port,
        topic: String::new(),
        block_size,
        dp_size: 1,
    };
    kv_index.add_worker(url_a, Some(preresolved.clone())).await;
    kv_index.add_worker(url_b, Some(preresolved)).await;

    // SUB sockets take a moment to handshake. The polling loop below
    // soaks up any extra latency; this is just a publish-before-SUB
    // guard.
    tokio::time::sleep(Duration::from_millis(150)).await;

    // 6. Publish a BlockStored event for the routing prompt's chain.
    let event_bytes = encode_block_stored_event(&hashes, None, &token_ids, block_size);
    let payload = encode_event_batch(0.0, vec![event_bytes], Some(0));
    pub_a
        .send(build_multipart(1, payload))
        .await
        .expect("send block-stored event");

    // 7. Bump worker B's load so the tie-break picks A among matched
    //    workers. The bump stays below balance_abs_threshold so the
    //    imbalance fast-path does not skip cache-aware selection.
    //    Bind the guards to a Vec held for the rest of the test scope
    //    so the counter stays > 0 through the polling loop.
    let w_a = build_worker(url_a, "tiny");
    let w_b = build_worker(url_b, "tiny");
    let _b_load: Vec<_> = (0..3).map(|_| w_b.load_guard()).collect();
    let workers = vec![Arc::clone(&w_a), Arc::clone(&w_b)];

    // 8. Drive select until the event has been applied. The pipeline is
    //    asynchronous (publish → SUB recv → mpsc → pump → tree); a
    //    polling loop is less flaky than a fixed sleep.
    let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
    let ctx = SelectionContext::new(&model_id, Some(&body));

    let start = std::time::Instant::now();
    let mut chose_a = false;
    while start.elapsed() < Duration::from_secs(3) {
        if let Some(w) = policy.select(&workers, &ctx) {
            if w.url == url_a {
                chose_a = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        chose_a,
        "policy did not route to publishing worker A within timeout",
    );

    // 9. Shutdown cleanly.
    let r = tokio::time::timeout(Duration::from_secs(2), kv_index.shutdown()).await;
    assert!(r.is_ok(), "kv_index shutdown should not hang");
}

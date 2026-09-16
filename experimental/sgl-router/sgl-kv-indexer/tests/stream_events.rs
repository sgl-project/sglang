// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! The Valkey event stream must deliver every batch to the index in order,
//! survive the death of the consumer that held the lease, drop poison entries
//! instead of stalling, and let an in-memory indexer rebuild from the retained
//! window. Skips when no Valkey is available.

#[allow(dead_code)]
#[path = "common/kv.rs"]
mod test_kv;
#[allow(dead_code)]
#[path = "common/valkey.rs"]
mod test_valkey;

use std::sync::Arc;
use std::time::{Duration, Instant};

use sgl_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ExternalKvActionType, MatchExternalKvRequest,
    MatchExternalKvResponse,
};
use sgl_kv_indexer::{
    InMemoryKvIndexerBackend, KvIndexerBackend, StreamConsumer, StreamConsumerConfig, StreamSink,
    DEFAULT_STREAM_MAXLEN,
};
use test_kv::{action, action_with_parent, apply_request, hbm};
use test_valkey::{fresh_prefix, ValkeyServer};
use tokio::sync::watch;

macro_rules! require_valkey {
    () => {
        match ValkeyServer::start() {
            Some(server) => server,
            None => {
                eprintln!(
                    "skipping: no valkey-server on PATH and KV_INDEXER_TEST_VALKEY_URL unset"
                );
                return;
            }
        }
    };
}

const LEASE: Duration = Duration::from_millis(400);

fn normalize(resp: &MatchExternalKvResponse) -> Vec<(String, i32, Vec<i64>)> {
    let mut out: Vec<(String, i32, Vec<i64>)> = resp
        .matches
        .iter()
        .flat_map(|node| {
            node.hashes_by_tier.iter().map(|tier| {
                let mut hashes = tier.hashes.clone();
                hashes.sort();
                (node.worker_id.clone(), tier.tier, hashes)
            })
        })
        .collect();
    out.sort();
    out
}

async fn placements(
    backend: &dyn KvIndexerBackend,
    hashes: &[i64],
) -> Vec<(String, i32, Vec<i64>)> {
    let resp = backend
        .match_external_kv(MatchExternalKvRequest {
            hashes: hashes.to_vec(),
            count_as_hit: false,
        })
        .await
        .unwrap();
    normalize(&resp)
}

/// Polls until the backend's placements equal the reference's, or panics.
async fn wait_for_parity(
    actual: &dyn KvIndexerBackend,
    reference: &dyn KvIndexerBackend,
    hashes: &[i64],
    what: &str,
) {
    let deadline = Instant::now() + Duration::from_secs(8);
    let expected = placements(reference, hashes).await;
    loop {
        let got = placements(actual, hashes).await;
        if got == expected {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{what}: stream consumer never converged\n expected {expected:?}\n got      {got:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

struct RunningConsumer {
    stop: watch::Sender<bool>,
    task: tokio::task::JoinHandle<()>,
}

impl RunningConsumer {
    async fn stop(self) {
        let _ = self.stop.send(true);
        let _ = self.task.await;
    }
}

async fn start_consumer<B: KvIndexerBackend + Clone>(
    server: &ValkeyServer,
    prefix: &str,
    config: StreamConsumerConfig,
    backend: B,
) -> RunningConsumer {
    let consumer = StreamConsumer::connect(&server.config(prefix), config, backend)
        .await
        .unwrap();
    let (stop, mut rx) = watch::channel(false);
    let task = tokio::spawn(async move {
        let shutdown = async move {
            while !*rx.borrow() {
                if rx.changed().await.is_err() {
                    break;
                }
            }
        };
        consumer.run(shutdown).await.unwrap();
    });
    RunningConsumer { stop, task }
}

fn shared(name: &str) -> StreamConsumerConfig {
    let mut config = StreamConsumerConfig::shared(name);
    config.lease_ttl = Some(LEASE);
    config
}

fn report(
    worker: &str,
    seq: u64,
    parent: Option<i64>,
    hashes: &[i64],
) -> ApplyExternalKvBatchRequest {
    apply_request(
        worker,
        &format!("http://{worker}"),
        seq,
        vec![action_with_parent(
            ExternalKvActionType::ActionReport,
            hbm(),
            parent,
            hashes,
        )],
    )
}

fn revoke(worker: &str, seq: u64, hashes: &[i64]) -> ApplyExternalKvBatchRequest {
    apply_request(
        worker,
        &format!("http://{worker}"),
        seq,
        vec![action(ExternalKvActionType::ActionRevoke, hbm(), hashes)],
    )
}

#[tokio::test]
async fn sink_to_leased_consumer_matches_direct_apply() {
    let server = require_valkey!();
    let prefix = fresh_prefix();
    let sink = StreamSink::connect(&server.config(&prefix), DEFAULT_STREAM_MAXLEN)
        .await
        .unwrap();
    let valkey = server.backend(&prefix).await;
    let reference = InMemoryKvIndexerBackend::new();
    let consumer = start_consumer(&server, &prefix, shared("c1"), valkey.clone()).await;

    let requests = vec![
        report("w0", 1, None, &[1, 2, 3, 4]),
        report("w1", 1, None, &[1, 2]),
        revoke("w0", 2, &[4]),
        report("w1", 2, Some(2), &[3]),
        revoke("w1", 3, &[1]),
    ];
    for request in &requests {
        reference
            .apply_external_kv_batch(request.clone())
            .await
            .unwrap();
        sink.publish(request).await.unwrap();
    }
    wait_for_parity(&valkey, &reference, &[1, 2, 3, 4], "roundtrip").await;
    consumer.stop().await;
}

#[tokio::test]
async fn standby_consumer_takes_over_when_holder_stops() {
    let server = require_valkey!();
    let prefix = fresh_prefix();
    let sink = StreamSink::connect(&server.config(&prefix), DEFAULT_STREAM_MAXLEN)
        .await
        .unwrap();
    let valkey = server.backend(&prefix).await;
    let reference = InMemoryKvIndexerBackend::new();

    let holder = start_consumer(&server, &prefix, shared("holder"), valkey.clone()).await;
    // Give the first consumer the lease before the standby starts polling.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let standby = start_consumer(&server, &prefix, shared("standby"), valkey.clone()).await;

    let first = report("w0", 1, None, &[10, 11, 12]);
    reference
        .apply_external_kv_batch(first.clone())
        .await
        .unwrap();
    sink.publish(&first).await.unwrap();
    wait_for_parity(&valkey, &reference, &[10, 11, 12], "before failover").await;

    holder.stop().await;
    let second = revoke("w0", 2, &[12]);
    let third = report("w1", 1, None, &[10]);
    for request in [&second, &third] {
        reference
            .apply_external_kv_batch(request.clone())
            .await
            .unwrap();
        sink.publish(request).await.unwrap();
    }
    wait_for_parity(&valkey, &reference, &[10, 11, 12], "after failover").await;
    standby.stop().await;
}

#[tokio::test]
async fn pending_entries_of_a_dead_consumer_are_reclaimed() {
    let server = require_valkey!();
    let prefix = fresh_prefix();
    let sink = StreamSink::connect(&server.config(&prefix), DEFAULT_STREAM_MAXLEN)
        .await
        .unwrap();
    let valkey = server.backend(&prefix).await;
    let reference = InMemoryKvIndexerBackend::new();
    let stream = format!("{prefix}events");

    // A consumer that read two entries and died before acknowledging them.
    let mut raw = server.raw().await;
    let _: () = redis::cmd("XGROUP")
        .arg("CREATE")
        .arg(&stream)
        .arg("indexers")
        .arg("$")
        .arg("MKSTREAM")
        .query_async(&mut raw)
        .await
        .unwrap();
    let requests = vec![report("w0", 1, None, &[20, 21]), revoke("w0", 2, &[21])];
    for request in &requests {
        reference
            .apply_external_kv_batch(request.clone())
            .await
            .unwrap();
        sink.publish(request).await.unwrap();
    }
    let _: redis::Value = redis::cmd("XREADGROUP")
        .arg("GROUP")
        .arg("indexers")
        .arg("dead")
        .arg("COUNT")
        .arg(10)
        .arg("STREAMS")
        .arg(&stream)
        .arg(">")
        .query_async(&mut raw)
        .await
        .unwrap();
    let pending_before: redis::Value = redis::cmd("XPENDING")
        .arg(&stream)
        .arg("indexers")
        .query_async(&mut raw)
        .await
        .unwrap();
    assert!(
        matches!(pending_before, redis::Value::Array(ref parts) if parts.first() == Some(&redis::Value::Int(2)))
    );

    let consumer = start_consumer(&server, &prefix, shared("alive"), valkey.clone()).await;
    wait_for_parity(&valkey, &reference, &[20, 21], "reclaimed pending").await;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let pending: redis::Value = redis::cmd("XPENDING")
            .arg(&stream)
            .arg("indexers")
            .query_async(&mut raw)
            .await
            .unwrap();
        if matches!(pending, redis::Value::Array(ref parts) if parts.first() == Some(&redis::Value::Int(0)))
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "reclaimed entries were never acknowledged"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    consumer.stop().await;
}

#[tokio::test]
async fn poison_entry_is_dropped_and_the_stream_continues() {
    let server = require_valkey!();
    let prefix = fresh_prefix();
    let sink = StreamSink::connect(&server.config(&prefix), DEFAULT_STREAM_MAXLEN)
        .await
        .unwrap();
    let valkey = server.backend(&prefix).await;
    let reference = InMemoryKvIndexerBackend::new();
    let stream = format!("{prefix}events");
    let consumer = start_consumer(&server, &prefix, shared("c1"), valkey.clone()).await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let mut raw = server.raw().await;
    let _: String = redis::cmd("XADD")
        .arg(&stream)
        .arg("*")
        .arg("w")
        .arg("w0")
        .arg("s")
        .arg(1)
        .arg("b")
        .arg(b"not a protobuf".as_slice())
        .query_async(&mut raw)
        .await
        .unwrap();
    // A batch the backend rejects (own parent) is poison too.
    let self_parent = report("w0", 2, Some(30), &[30]);
    sink.publish(&self_parent).await.unwrap();
    let valid = report("w0", 3, None, &[31, 32]);
    reference
        .apply_external_kv_batch(valid.clone())
        .await
        .unwrap();
    sink.publish(&valid).await.unwrap();

    wait_for_parity(&valkey, &reference, &[30, 31, 32], "after poison").await;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let pending: redis::Value = redis::cmd("XPENDING")
            .arg(&stream)
            .arg("indexers")
            .query_async(&mut raw)
            .await
            .unwrap();
        if matches!(pending, redis::Value::Array(ref parts) if parts.first() == Some(&redis::Value::Int(0)))
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "poison entries were never acknowledged"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    consumer.stop().await;
}

#[tokio::test]
async fn private_group_rebuilds_an_in_memory_index_from_the_beginning() {
    let server = require_valkey!();
    let prefix = fresh_prefix();
    let sink = StreamSink::connect(&server.config(&prefix), DEFAULT_STREAM_MAXLEN)
        .await
        .unwrap();
    let reference = InMemoryKvIndexerBackend::new();
    let requests = vec![
        report("w0", 1, None, &[40, 41, 42]),
        report("w1", 1, None, &[40]),
        revoke("w0", 2, &[42]),
    ];
    for request in &requests {
        reference
            .apply_external_kv_batch(request.clone())
            .await
            .unwrap();
        sink.publish(request).await.unwrap();
    }

    // Started after every entry was published: nothing is live, all is history.
    let rebuilt = Arc::new(InMemoryKvIndexerBackend::new());
    let shared: Arc<dyn KvIndexerBackend> = rebuilt.clone();
    let consumer = start_consumer(
        &server,
        &prefix,
        StreamConsumerConfig::private("memory-a"),
        shared,
    )
    .await;
    wait_for_parity(rebuilt.as_ref(), &reference, &[40, 41, 42], "rebuild").await;
    consumer.stop().await;

    // The private group is gone once the consumer exits.
    let mut raw = server.raw().await;
    let groups: redis::Value = redis::cmd("XINFO")
        .arg("GROUPS")
        .arg(format!("{prefix}events"))
        .query_async(&mut raw)
        .await
        .unwrap();
    let names = format!("{groups:?}");
    assert!(
        !names.contains("rebuild-memory-a"),
        "private group leaked: {names}"
    );
}

/// The lease is the only thing serializing applies, so the central invariant is
/// that a consumer without it applies nothing. Each consumer gets its own
/// backend, which is the only way to see who applied what.
#[tokio::test]
async fn a_consumer_without_the_lease_applies_nothing() {
    let server = require_valkey!();
    let prefix = fresh_prefix();
    let sink = StreamSink::connect(&server.config(&prefix), DEFAULT_STREAM_MAXLEN)
        .await
        .unwrap();
    let holder_backend = Arc::new(InMemoryKvIndexerBackend::new());
    let standby_backend = Arc::new(InMemoryKvIndexerBackend::new());

    let holder = start_consumer(
        &server,
        &prefix,
        shared("holder"),
        Arc::clone(&holder_backend) as Arc<dyn KvIndexerBackend>,
    )
    .await;
    // Let the holder take the lease before the standby starts polling for it.
    tokio::time::sleep(Duration::from_millis(150)).await;
    let standby = start_consumer(
        &server,
        &prefix,
        shared("standby"),
        Arc::clone(&standby_backend) as Arc<dyn KvIndexerBackend>,
    )
    .await;

    let reference = InMemoryKvIndexerBackend::new();
    for request in [report("w0", 1, None, &[70, 71, 72]), revoke("w0", 2, &[72])] {
        reference
            .apply_external_kv_batch(request.clone())
            .await
            .unwrap();
        sink.publish(&request).await.unwrap();
    }
    wait_for_parity(
        holder_backend.as_ref(),
        &reference,
        &[70, 71, 72],
        "holder applied",
    )
    .await;

    // Several lease periods: a standby that ever applied would show it by now.
    tokio::time::sleep(LEASE * 3).await;
    assert!(
        placements(standby_backend.as_ref(), &[70, 71, 72])
            .await
            .is_empty(),
        "the standby consumer applied entries without holding the lease"
    );
    holder.stop().await;
    standby.stop().await;
}

/// Applies are idempotent but not commutative, so a takeover must drain what the
/// dead holder left pending BEFORE reading anything new: a REPORT stuck in the
/// dead consumer's pending list, applied after the REVOKE that followed it,
/// leaves the block present forever.
#[tokio::test]
async fn a_takeover_applies_pending_entries_before_newer_ones() {
    let server = require_valkey!();
    let prefix = fresh_prefix();
    let sink = StreamSink::connect(&server.config(&prefix), DEFAULT_STREAM_MAXLEN)
        .await
        .unwrap();
    let valkey = server.backend(&prefix).await;
    let stream = format!("{prefix}events");
    let mut raw = server.raw().await;
    let _: () = redis::cmd("XGROUP")
        .arg("CREATE")
        .arg(&stream)
        .arg("indexers")
        .arg("$")
        .arg("MKSTREAM")
        .query_async(&mut raw)
        .await
        .unwrap();

    // A consumer that read the REPORT and died before acknowledging it.
    let stored = report("w0", 1, None, &[80, 81]);
    sink.publish(&stored).await.unwrap();
    let _: redis::Value = redis::cmd("XREADGROUP")
        .arg("GROUP")
        .arg("indexers")
        .arg("dead")
        .arg("COUNT")
        .arg(10)
        .arg("STREAMS")
        .arg(&stream)
        .arg(">")
        .query_async(&mut raw)
        .await
        .unwrap();
    // Then the revoke that must land after it.
    let revoked = revoke("w0", 2, &[80, 81]);
    sink.publish(&revoked).await.unwrap();

    let reference = InMemoryKvIndexerBackend::new();
    for request in [stored, revoked] {
        reference.apply_external_kv_batch(request).await.unwrap();
    }
    assert!(
        placements(&reference, &[80, 81]).await.is_empty(),
        "the reference order leaves both blocks revoked"
    );

    let consumer = start_consumer(&server, &prefix, shared("fresh"), valkey.clone()).await;
    wait_for_parity(&valkey, &reference, &[80, 81], "takeover ordering").await;
    // Give a late claim of the pending REPORT a chance to resurrect the blocks.
    tokio::time::sleep(LEASE * 3).await;
    assert!(
        placements(&valkey, &[80, 81]).await.is_empty(),
        "a REPORT claimed after the REVOKE resurrected the blocks"
    );
    consumer.stop().await;
}

/// A flushed or restored Valkey loses the consumer group. Without recreating it
/// the consumer retries NOGROUP forever and silently stops applying, so the index
/// freezes while the fleet keeps publishing.
#[tokio::test]
async fn a_group_that_disappears_is_recreated_and_consumption_continues() {
    let server = require_valkey!();
    let prefix = fresh_prefix();
    let sink = StreamSink::connect(&server.config(&prefix), DEFAULT_STREAM_MAXLEN)
        .await
        .unwrap();
    let valkey = server.backend(&prefix).await;
    let reference = InMemoryKvIndexerBackend::new();
    let consumer = start_consumer(&server, &prefix, shared("c1"), valkey.clone()).await;

    let first = report("w0", 1, None, &[90, 91]);
    reference
        .apply_external_kv_batch(first.clone())
        .await
        .unwrap();
    sink.publish(&first).await.unwrap();
    wait_for_parity(&valkey, &reference, &[90, 91], "before the group vanishes").await;

    let mut raw = server.raw().await;
    let _: i64 = redis::cmd("XGROUP")
        .arg("DESTROY")
        .arg(format!("{prefix}events"))
        .arg("indexers")
        .query_async(&mut raw)
        .await
        .unwrap();

    let second = report("w1", 1, None, &[92]);
    reference
        .apply_external_kv_batch(second.clone())
        .await
        .unwrap();
    sink.publish(&second).await.unwrap();
    wait_for_parity(
        &valkey,
        &reference,
        &[90, 91, 92],
        "after the group was recreated",
    )
    .await;
    consumer.stop().await;
}

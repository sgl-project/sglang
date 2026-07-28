// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Failure-boundary tests for placement/reverse-index convergence.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::*;

/// Delegates to a real Redis connection but fails the first selected
/// reverse-index command, after the placement Lua script has committed.
struct FailOnceConn {
    inner: SingleConn,
    command: &'static [u8],
    failed: AtomicBool,
}

/// Simulates a newer incarnation winning after an old batch has passed
/// TOUCH_META/SEQ_CHECK but before that old batch commits its sequence.
struct ChangeGenerationBeforeCommitConn {
    inner: SingleConn,
    invokes: AtomicUsize,
}

#[tonic::async_trait]
impl RedisConn for ChangeGenerationBeforeCommitConn {
    async fn query(&self, cmd: redis::Cmd) -> redis::RedisResult<redis::Value> {
        self.inner.query(cmd).await
    }

    async fn invoke(
        &self,
        script: &super::scripts::RedisScript,
        keys: Vec<String>,
        args: Vec<String>,
    ) -> redis::RedisResult<redis::Value> {
        // Report apply invokes TOUCH_META, SEQ_CHECK, PLACEMENT_SET, SEQ_COMMIT.
        if self.invokes.fetch_add(1, Ordering::SeqCst) == 3 {
            let mut cmd = redis::cmd("HINCRBY");
            cmd.arg(&keys[0]).arg("generation").arg(1);
            self.inner.query(cmd).await?;
        }
        self.inner.invoke(script, keys, args).await
    }
}

#[tonic::async_trait]
impl RedisConn for FailOnceConn {
    async fn query(&self, cmd: redis::Cmd) -> redis::RedisResult<redis::Value> {
        let is_target = {
            let mut args = cmd.args_iter();
            matches!(
                args.next(),
                Some(redis::Arg::Simple(name)) if name.eq_ignore_ascii_case(self.command)
            )
        };
        if is_target && !self.failed.swap(true, Ordering::SeqCst) {
            return Err(redis::RedisError::from((
                redis::ErrorKind::IoError,
                "injected reverse-index failure",
            )));
        }
        self.inner.query(cmd).await
    }

    async fn invoke(
        &self,
        script: &super::scripts::RedisScript,
        keys: Vec<String>,
        args: Vec<String>,
    ) -> redis::RedisResult<redis::Value> {
        self.inner.invoke(script, keys, args).await
    }
}

async fn backend(test: &str, command: &'static [u8]) -> Option<RedisKvIndexerBackend> {
    let inner = test_connection(test).await?;
    Some(test_backend(
        test,
        FailOnceConn {
            inner,
            command,
            failed: AtomicBool::new(false),
        },
    ))
}

async fn generation_race_backend(test: &str) -> Option<RedisKvIndexerBackend> {
    let inner = test_connection(test).await?;
    Some(test_backend(
        test,
        ChangeGenerationBeforeCommitConn {
            inner,
            invokes: AtomicUsize::new(0),
        },
    ))
}

async fn test_connection(test: &str) -> Option<SingleConn> {
    let Ok(url) = std::env::var("KV_INDEXER_REDIS_URL") else {
        // Same contract as tests/common/require.rs, which the integration
        // suites use; this module lives in the library and cannot share it.
        if std::env::var("KV_INDEXER_REQUIRE_REDIS").is_ok_and(|v| v == "1") {
            panic!("{test} requires a store but KV_INDEXER_REDIS_URL is not set");
        }
        eprintln!("skipping {test}: KV_INDEXER_REDIS_URL is not set");
        return None;
    };
    Some(SingleConn::connect(&url).await.expect("connect redis"))
}

fn test_backend(test: &str, conn: impl RedisConn) -> RedisKvIndexerBackend {
    RedisKvIndexerBackend::new(Arc::new(conn), format!("fault:{test}:{}", now_ms()))
}

fn batch(worker: &str, seq: u64, kind: ExternalKvActionType) -> ApplyExternalKvBatchRequest {
    ApplyExternalKvBatchRequest {
        worker_id: worker.to_string(),
        seq,
        actions: vec![crate::pb::ExternalKvAction {
            r#type: kind as i32,
            tier: crate::pb::TierType::TierHbm as i32,
            hashes: vec!["hash-a".to_string()],
        }],
        worker_address: String::new(),
        incarnation: String::new(),
    }
}

fn batch_inc(
    worker: &str,
    seq: u64,
    incarnation: &str,
    kind: ExternalKvActionType,
) -> ApplyExternalKvBatchRequest {
    ApplyExternalKvBatchRequest {
        incarnation: incarnation.to_string(),
        ..batch(worker, seq, kind)
    }
}

fn report(worker: &str, seq: u64, incarnation: &str) -> ApplyExternalKvBatchRequest {
    batch_inc(worker, seq, incarnation, ExternalKvActionType::ActionReport)
}

fn heartbeat(worker: &str, incarnation: &str) -> ApplyExternalKvBatchRequest {
    ApplyExternalKvBatchRequest {
        worker_id: worker.to_string(),
        seq: 0,
        actions: Vec::new(),
        worker_address: String::new(),
        incarnation: incarnation.to_string(),
    }
}

async fn matched_workers(backend: &RedisKvIndexerBackend, hash: &str) -> Vec<String> {
    backend
        .do_match(MatchExternalKvRequest {
            hashes: vec![hash.to_string()],
            count_as_hit: false,
        })
        .await
        .expect("match")
        .matches
        .into_iter()
        .map(|matched| matched.worker_id)
        .collect()
}

async fn reverse_hashes(backend: &RedisKvIndexerBackend, worker: &str) -> Vec<String> {
    let mut cmd = redis::cmd("SMEMBERS");
    cmd.arg(worker_blocks_key(&backend.ns, worker));
    let value = backend.conn.query(cmd).await.expect("read reverse index");
    Vec::<String>::from_redis_value(&value)
        .expect("decode reverse index")
        .into_iter()
        .map(|member| parse_reverse_member(&member).1.to_string())
        .collect()
}

async fn meta_field(backend: &RedisKvIndexerBackend, worker: &str, field: &str) -> Option<String> {
    let mut cmd = redis::cmd("HGET");
    cmd.arg(worker_meta_key(&backend.ns, worker)).arg(field);
    let value = backend.conn.query(cmd).await.expect("read meta field");
    Option::<String>::from_redis_value(&value).expect("decode meta field")
}

async fn assert_meta(
    backend: &RedisKvIndexerBackend,
    worker: &str,
    field: &str,
    expected: Option<&str>,
) {
    assert_eq!(
        meta_field(backend, worker, field).await.as_deref(),
        expected
    );
}

async fn assert_reverse(backend: &RedisKvIndexerBackend, worker: &str, expected: &[&str]) {
    let actual = reverse_hashes(backend, worker).await;
    assert_eq!(
        actual.iter().map(String::as_str).collect::<Vec<_>>(),
        expected
    );
}

async fn apply_ok(backend: &RedisKvIndexerBackend, request: ApplyExternalKvBatchRequest) {
    backend.apply(request).await.expect("apply");
}

#[tokio::test]
async fn report_replay_repairs_missing_reverse_index_after_sadd_failure() {
    let Some(backend) = backend("report_repair", b"SADD").await else {
        return;
    };
    let request = report("worker-0", 1, "");

    // PLACEMENT_SET commits, then SADD fails.
    assert!(backend.apply(request.clone()).await.is_err());
    assert_reverse(&backend, "worker-0", &[]).await;

    // Replay must run SADD although placement is already set.
    apply_ok(&backend, request).await;
    assert_reverse(&backend, "worker-0", &["hash-a"]).await;

    apply_ok(
        &backend,
        batch("worker-0", 2, ExternalKvActionType::ActionRevoke),
    )
    .await;
}

#[tokio::test]
async fn incarnation_reset_hides_orphan_placement_missing_from_reverse_index() {
    let Some(backend) = backend("orphan_generation", b"SADD").await else {
        return;
    };
    let worker = "worker-orphan";

    // Generation A writes placement, then the injected reverse SADD fails.
    assert!(backend.apply(report(worker, 1, "inc-a")).await.is_err());
    assert_reverse(&backend, worker, &[]).await;

    // Simulate losing the bridge's volatile pending batch. Generation B reset
    // cannot discover the orphan through reverse, but generation filtering must
    // still make it permanently unroutable.
    apply_ok(&backend, heartbeat(worker, "inc-b")).await;
    assert!(matched_workers(&backend, "hash-a").await.is_empty());
    assert_meta(&backend, worker, "generation", Some("1")).await;
}

#[tokio::test]
async fn retired_incarnation_cannot_roll_current_generation_back() {
    let Some(backend) = backend("retired_incarnation", b"NEVER").await else {
        return;
    };
    let worker = "worker-fenced";
    apply_ok(&backend, report(worker, 1, "inc-a")).await;
    apply_ok(&backend, heartbeat(worker, "inc-b")).await;

    let error = backend
        .apply(report(worker, 2, "inc-a"))
        .await
        .expect_err("retired incarnation must be fenced");
    assert_eq!(error.code(), tonic::Code::FailedPrecondition);
    assert_meta(&backend, worker, "incarnation", Some("inc-b")).await;
}

#[tokio::test]
async fn empty_incarnation_uses_retirable_legacy_token() {
    let Some(backend) = backend("legacy_incarnation", b"NEVER").await else {
        return;
    };
    let worker = "worker-legacy";
    apply_ok(&backend, report(worker, 1, "")).await;
    apply_ok(&backend, heartbeat(worker, "inc-b")).await;
    let error = backend
        .apply(report(worker, 2, ""))
        .await
        .expect_err("legacy token must be retired");
    assert_eq!(error.code(), tonic::Code::FailedPrecondition);
}

#[tokio::test]
async fn in_flight_old_generation_cannot_commit_sequence_after_restart() {
    let Some(backend) = generation_race_backend("generation_commit_fence").await else {
        return;
    };
    let worker = "worker-race";
    let error = backend
        .apply(report(worker, 99, "inc-a"))
        .await
        .expect_err("generation change before commit must fence old batch");
    assert_eq!(error.code(), tonic::Code::FailedPrecondition);
    assert_meta(&backend, worker, "seq", None).await;
    assert!(
        matched_workers(&backend, "hash-a").await.is_empty(),
        "old-generation placement written by the in-flight batch must be filtered"
    );
}

#[tokio::test]
async fn late_concurrent_reset_cannot_delete_current_generation_state() {
    let Some(backend) = backend("late_reset", b"NEVER").await else {
        return;
    };
    let worker = "worker-reset-race";
    apply_ok(&backend, report(worker, 9, "inc-a")).await;
    let meta = worker_meta_key(&backend.ns, worker);
    let touch = backend
        .touch_meta(&meta, "", "inc-b")
        .await
        .expect("touch generation B");
    assert!(touch.reset_needed);
    backend
        .reset_worker(worker, &meta, touch.generation)
        .await
        .expect("first reset");
    apply_ok(&backend, report(worker, 1, "inc-b")).await;

    // Simulate another resetter that started before the first one finished but
    // reaches cleanup after current-generation data has already committed.
    backend
        .reset_worker(worker, &meta, touch.generation)
        .await
        .expect("late reset");
    assert_eq!(matched_workers(&backend, "hash-a").await, vec![worker]);
    assert_meta(&backend, worker, "seq", Some("1")).await;
}

#[tokio::test]
async fn in_flight_old_placement_mutation_cannot_clobber_new_generation() {
    let Some(backend) = backend("placement_generation_fence", b"NEVER").await else {
        return;
    };
    let worker = "worker-placement-race";
    apply_ok(&backend, report(worker, 9, "inc-a")).await;
    apply_ok(&backend, heartbeat(worker, "inc-b")).await;
    apply_ok(&backend, report(worker, 1, "inc-b")).await;

    assert!(backend
        .report_one(
            worker,
            "hash-a",
            tier_bit(crate::pb::TierType::TierHbm as i32),
            0
        )
        .await
        .is_err());
    assert!(backend
        .revoke_one(
            worker,
            "hash-a",
            tier_bit(crate::pb::TierType::TierHbm as i32),
            0
        )
        .await
        .is_err());
    assert_eq!(matched_workers(&backend, "hash-a").await, vec![worker]);
}

#[tokio::test]
async fn revoke_replay_repairs_stale_reverse_index_after_srem_failure() {
    let Some(backend) = backend("revoke_repair", b"SREM").await else {
        return;
    };
    apply_ok(&backend, report("worker-0", 1, "")).await;

    let request = batch("worker-0", 2, ExternalKvActionType::ActionRevoke);

    // PLACEMENT_CLEAR commits, then SREM fails.
    assert!(backend.apply(request.clone()).await.is_err());
    assert_reverse(&backend, "worker-0", &["hash-a"]).await;

    // Replay sees placement already absent but must retry SREM.
    apply_ok(&backend, request).await;
    assert_reverse(&backend, "worker-0", &[]).await;
}

#[tokio::test]
async fn restart_reset_retried_after_mid_reset_failure() {
    // P1 regression: a worker restart bumps the durable incarnation and sets a
    // `reset_pending` flag BEFORE the (non-atomic) reset runs. If the reset fails
    // partway, the flag must survive so the next apply retries and finishes it —
    // otherwise the incarnation already looks current and the stale state sticks.
    let Some(backend) = backend("reset_retry", b"SREM").await else {
        return;
    };

    // Incarnation "A" reports hash-a.
    apply_ok(&backend, report("worker-0", 1, "A")).await;
    assert_reverse(&backend, "worker-0", &["hash-a"]).await;

    // Incarnation "B" (a restart): reset_pending is set, but the reverse-index
    // SREM inside reset_worker fails once, aborting the reset mid-flight.
    let restart = report("worker-0", 1, "B");
    assert!(backend.apply(restart.clone()).await.is_err());
    assert_meta(&backend, "worker-0", "reset_pending", Some("1")).await;

    // Retry: incarnation is already "B" (unchanged), yet the lingering
    // reset_pending must still drive an idempotent reset to completion.
    apply_ok(&backend, restart).await;
    assert_meta(&backend, "worker-0", "reset_pending", None).await;
    assert_reverse(&backend, "worker-0", &["hash-a"]).await;
}

#[tokio::test]
async fn reset_window_hides_worker_from_match_until_reset_completes() {
    // P1: while a restart reset is pending, the worker's still-present placement
    // must not be routed to. Fail the reset's very first step (the reverse-index
    // SSCAN) so the stale placement/reverse entries survive with
    // reset_pending=1, then assert `match` drops the worker (even with liveness
    // TTL disabled) until a retry finishes the reset.
    let Some(backend) = backend("reset_window", b"SSCAN").await else {
        return;
    };

    // Incarnation "A" reports hash-a: matchable while healthy.
    apply_ok(&backend, report("worker-0", 1, "A")).await;
    assert_eq!(matched_workers(&backend, "hash-a").await, vec!["worker-0"]);

    // Incarnation "B": reset_pending is set, but the SSCAN fails so the reset
    // never clears the stale placement. The worker must now be hidden from match.
    let restart = report("worker-0", 1, "B");
    assert!(backend.apply(restart.clone()).await.is_err());
    assert_meta(&backend, "worker-0", "reset_pending", Some("1")).await;
    assert!(
        matched_workers(&backend, "hash-a").await.is_empty(),
        "worker with reset_pending must be dropped from match"
    );

    // Retry completes the reset and re-indexes the fresh report; routable again.
    apply_ok(&backend, restart).await;
    assert_meta(&backend, "worker-0", "reset_pending", None).await;
    assert_eq!(matched_workers(&backend, "hash-a").await, vec!["worker-0"]);
}

#[tokio::test]
async fn touch_meta_persists_meta_left_with_ttl_by_old_build() {
    // P1 (rolling upgrade): an older build set a TTL on the whole meta hash. The
    // new build must PERSIST it on first contact so `incarnation`/`seq` can no
    // longer expire (which would silently resurrect stale placements).
    let Some(backend) = backend("ttl_persist", b"UNUSED-CMD").await else {
        return;
    };
    let worker = "worker-0";
    let meta = worker_meta_key(&backend.ns, worker);

    // Simulate the legacy meta hash: incarnation + seq, with a TTL on the key.
    let mut hset = redis::cmd("HSET");
    hset.arg(&meta)
        .arg("incarnation")
        .arg("A")
        .arg("seq")
        .arg("3");
    backend.conn.query(hset).await.expect("seed legacy meta");
    let mut pexpire = redis::cmd("PEXPIRE");
    pexpire.arg(&meta).arg(60_000);
    backend.conn.query(pexpire).await.expect("seed ttl");

    // Any apply from the same incarnation (no reset) must persist the key.
    apply_ok(&backend, report(worker, 4, "A")).await;

    let mut pttl = redis::cmd("PTTL");
    pttl.arg(&meta);
    let v = backend.conn.query(pttl).await.expect("pttl");
    let ttl = i64::from_redis_value(&v).expect("decode pttl");
    assert_eq!(
        ttl, -1,
        "meta key must be persisted (no TTL) after touch_meta"
    );
    assert_meta(&backend, worker, "incarnation", Some("A")).await;
}

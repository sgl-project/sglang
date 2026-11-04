use sgl_kv_indexer::{pb::*, InMemoryKvIndexerBackend, KvIndexerBackend, SourceLimits};
use tonic::Code;
fn report(hash: i64, tier: i32) -> ExternalKvAction {
    ExternalKvAction {
        r#type: ExternalKvActionType::ActionReport as i32,
        tier,
        hashes: vec![hash],
        ..Default::default()
    }
}
fn apply(
    worker: &str,
    epoch: &str,
    seq: u64,
    chunk: u32,
    count: u32,
    hash: i64,
) -> ApplyExternalKvBatchRequest {
    ApplyExternalKvBatchRequest {
        worker_id: worker.into(),
        worker_address: format!("http://{worker}"),
        source_epoch: Some(epoch.into()),
        seq,
        chunk_index: chunk,
        chunk_count: count,
        actions: vec![report(hash, 1)],
        ..Default::default()
    }
}
async fn register(
    b: &InMemoryKvIndexerBackend,
    worker: &str,
    epoch: &str,
    previous: Option<&str>,
    tiers: Vec<i32>,
) -> Result<SourceObservation, tonic::Status> {
    Ok(b.register_source(RegisterSourceRequest {
        worker_id: worker.into(),
        source_epoch: epoch.into(),
        previous_epoch: previous.map(str::to_owned),
        tiers,
    })
    .await?
    .source
    .unwrap())
}
async fn empty(
    b: &InMemoryKvIndexerBackend,
    source: &SourceObservation,
    next_seq: u64,
) -> SourceObservation {
    b.reconcile_source(ReconcileSourceRequest {
        worker_id: source.worker_id.clone(),
        source_epoch: source.source_epoch.clone(),
        expected_source_version: source.source_version,
        expected_indexer_epoch: source.indexer_epoch.clone(),
        confirmed_empty: true,
        next_seq,
        evidence_id: "trusted owner confirms empty cache at exact publisher sequence cut".into(),
    })
    .await
    .unwrap()
    .source
    .unwrap()
}
async fn query(b: &InMemoryKvIndexerBackend, hashes: Vec<i64>) -> MatchExternalKvResponse {
    b.match_external_kv(MatchExternalKvRequest {
        hashes,
        count_as_hit: false,
    })
    .await
    .unwrap()
}
fn source<'a>(q: &'a MatchExternalKvResponse, worker: &str) -> &'a SourceObservation {
    q.observation
        .as_ref()
        .unwrap()
        .sources
        .iter()
        .find(|s| s.worker_id == worker)
        .unwrap()
}
fn hashes(q: &MatchExternalKvResponse, worker: &str, tier: i32) -> Vec<i64> {
    q.matches
        .iter()
        .filter(|m| m.worker_id == worker)
        .flat_map(|m| &m.hashes_by_tier)
        .filter(|t| t.tier == tier)
        .flat_map(|t| t.hashes.clone())
        .collect()
}

#[tokio::test]
async fn chunks_share_publisher_sequence_but_commit_individually_and_duplicate_is_noop() {
    let b = InMemoryKvIndexerBackend::new();
    let s = register(&b, "w", "e", None, vec![1, 2]).await.unwrap();
    empty(&b, &s, 10).await;
    let first = apply("w", "e", 10, 0, 2, 100);
    let response = b.apply_external_kv_batch(first.clone()).await.unwrap();
    assert!(!response.duplicate_chunk);
    let q = query(&b, vec![100, 101]).await;
    let s = source(&q, "w");
    assert!(!s.complete);
    assert_eq!(s.pending_seq, Some(10));
    assert_eq!(s.next_chunk_index, 1);
    assert_eq!(s.watermark_seq, None);
    assert_eq!(hashes(&q, "w", 1), vec![100]);
    assert!(
        b.apply_external_kv_batch(first)
            .await
            .unwrap()
            .duplicate_chunk
    );
    b.apply_external_kv_batch(apply("w", "e", 10, 1, 2, 101))
        .await
        .unwrap();
    let q = query(&b, vec![100, 101]).await;
    let s = source(&q, "w");
    assert!(s.complete);
    assert_eq!(s.watermark_seq, Some(10));
    assert_eq!(s.pending_seq, None);
    assert_eq!(hashes(&q, "w", 1), vec![100, 101]);
    assert!(q.observation.as_ref().unwrap().scope_only);
    assert!(!q.observation.unwrap().full_snapshot);
}
#[tokio::test]
async fn gap_invalidates_only_affected_worker_registered_tiers_and_stays_incomplete() {
    let b = InMemoryKvIndexerBackend::new();
    b.apply_external_kv_batch(ApplyExternalKvBatchRequest {
        worker_id: "w".into(),
        actions: vec![report(77, 2)],
        ..Default::default()
    })
    .await
    .unwrap();
    let s = register(&b, "w", "e", None, vec![1]).await.unwrap();
    empty(&b, &s, 1).await;
    b.apply_external_kv_batch(apply("w", "e", 1, 0, 1, 10))
        .await
        .unwrap();
    let other = register(&b, "other", "other-e", None, vec![1])
        .await
        .unwrap();
    empty(&b, &other, 1).await;
    b.apply_external_kv_batch(apply("other", "other-e", 1, 0, 1, 50))
        .await
        .unwrap();
    b.apply_external_kv_batch(apply("w", "e", 3, 0, 1, 30))
        .await
        .unwrap();
    let q = query(&b, vec![10, 30, 50, 77]).await;
    assert_eq!(hashes(&q, "w", 1), vec![30]);
    assert_eq!(hashes(&q, "w", 2), vec![77]);
    assert_eq!(hashes(&q, "other", 1), vec![50]);
    assert!(source(&q, "other").complete);
    assert!(!source(&q, "w").complete);
    assert_eq!(source(&q, "w").reason, "sequence_or_chunk_gap");
    assert!(
        b.apply_external_kv_batch(apply("w", "e", 1, 0, 1, 10))
            .await
            .unwrap()
            .duplicate_chunk
    );
    assert_eq!(hashes(&query(&b, vec![10, 30]).await, "w", 1), vec![30]);
}
#[tokio::test]
async fn missing_out_of_order_chunk_clears_partial_batch_and_old_batches_cannot_rollback() {
    let b = InMemoryKvIndexerBackend::new();
    let s = register(&b, "w", "e", None, vec![1]).await.unwrap();
    empty(&b, &s, 1).await;
    b.apply_external_kv_batch(apply("w", "e", 1, 0, 3, 10))
        .await
        .unwrap();
    assert_eq!(
        b.apply_external_kv_batch(apply("w", "e", 1, 2, 3, 12))
            .await
            .unwrap_err()
            .code(),
        Code::FailedPrecondition
    );
    let q = query(&b, vec![10, 12]).await;
    assert!(q.matches.is_empty());
    assert!(!source(&q, "w").complete);
    b.apply_external_kv_batch(apply("w", "e", 2, 0, 2, 20))
        .await
        .unwrap();
    assert!(
        b.apply_external_kv_batch(apply("w", "e", 1, 0, 3, 10))
            .await
            .unwrap()
            .duplicate_chunk
    );
    let q = query(&b, vec![10, 20]).await;
    assert_eq!(hashes(&q, "w", 1), vec![20]);
    assert_eq!(source(&q, "w").pending_seq, Some(2));
    b.apply_external_kv_batch(apply("w", "e", 2, 1, 2, 21))
        .await
        .unwrap();
    assert!(!source(&query(&b, vec![20, 21]).await, "w").complete);
}
#[tokio::test]
async fn conflicting_duplicate_chunk_clears_uncertain_observations() {
    let b = InMemoryKvIndexerBackend::new();
    register(&b, "w", "e", None, vec![1]).await.unwrap();
    b.apply_external_kv_batch(apply("w", "e", 1, 0, 2, 10))
        .await
        .unwrap();
    assert_eq!(
        b.apply_external_kv_batch(apply("w", "e", 1, 0, 2, 99))
            .await
            .unwrap_err()
            .code(),
        Code::DataLoss
    );
    assert!(query(&b, vec![10, 99]).await.matches.is_empty());
}
#[tokio::test]
async fn stale_epoch_apply_and_register_cannot_replace_current_owner() {
    let b = InMemoryKvIndexerBackend::new();
    register(&b, "w", "old", None, vec![1]).await.unwrap();
    b.apply_external_kv_batch(apply("w", "old", 1, 0, 1, 10))
        .await
        .unwrap();
    let current = register(&b, "w", "new", Some("old"), vec![1])
        .await
        .unwrap();
    assert_eq!(current.source_epoch, "new");
    assert!(query(&b, vec![10]).await.matches.is_empty());
    b.apply_external_kv_batch(apply("w", "new", 1, 0, 1, 20))
        .await
        .unwrap();
    assert_eq!(
        b.apply_external_kv_batch(apply("w", "old", 2, 0, 1, 99))
            .await
            .unwrap_err()
            .code(),
        Code::FailedPrecondition
    );
    assert_eq!(
        register(&b, "w", "old", Some("new"), vec![1])
            .await
            .unwrap_err()
            .code(),
        Code::FailedPrecondition
    );
    assert_eq!(
        register(&b, "w", "unknown", Some("old"), vec![1])
            .await
            .unwrap_err()
            .code(),
        Code::FailedPrecondition
    );
    assert_eq!(hashes(&query(&b, vec![20, 99]).await, "w", 1), vec![20]);
}
#[tokio::test]
async fn reset_evidence_and_cas_restore_complete_only_at_a_new_sequence_cut() {
    let b = InMemoryKvIndexerBackend::new();
    register(&b, "w", "e", None, vec![1]).await.unwrap();
    b.apply_external_kv_batch(apply("w", "e", 5, 0, 1, 10))
        .await
        .unwrap();
    let q = query(&b, vec![10]).await;
    let s = source(&q, "w").clone();
    assert!(!s.complete);
    let stale = ReconcileSourceRequest {
        worker_id: "w".into(),
        source_epoch: "e".into(),
        expected_source_version: s.source_version - 1,
        expected_indexer_epoch: s.indexer_epoch.clone(),
        confirmed_empty: true,
        next_seq: 6,
        evidence_id: "reset receipt".into(),
    };
    assert_eq!(
        b.reconcile_source(stale.clone()).await.unwrap_err().code(),
        Code::FailedPrecondition
    );
    let mut earlier = stale;
    earlier.expected_source_version = s.source_version;
    earlier.next_seq = 5;
    assert_eq!(
        b.reconcile_source(earlier).await.unwrap_err().code(),
        Code::FailedPrecondition
    );
    let restored = empty(&b, &s, 6).await;
    assert!(restored.complete);
    assert!(query(&b, vec![10]).await.matches.is_empty());
    assert!(
        b.apply_external_kv_batch(apply("w", "e", 5, 0, 1, 10))
            .await
            .unwrap()
            .duplicate_chunk
    );
    let mut clear = apply("w", "e", 8, 0, 1, 80);
    clear.actions = vec![ExternalKvAction {
        r#type: 3,
        tier: 1,
        ..Default::default()
    }];
    b.apply_external_kv_batch(clear).await.unwrap();
    assert!(!source(&query(&b, vec![10]).await, "w").complete); // ordinary clear is not recovery evidence
}
#[tokio::test]
async fn explicit_worker_death_invalidation_suspends_until_reconciled() {
    let b = InMemoryKvIndexerBackend::new();
    let s = register(&b, "w", "e", None, vec![1]).await.unwrap();
    empty(&b, &s, 1).await;
    b.apply_external_kv_batch(apply("w", "e", 1, 0, 1, 10))
        .await
        .unwrap();
    let q = query(&b, vec![10]).await;
    let s = source(&q, "w");
    let response = b
        .reconcile_source(ReconcileSourceRequest {
            worker_id: "w".into(),
            source_epoch: "e".into(),
            expected_source_version: s.source_version,
            expected_indexer_epoch: s.indexer_epoch.clone(),
            confirmed_empty: false,
            next_seq: 2,
            evidence_id: "trusted worker registry reports death".into(),
        })
        .await
        .unwrap();
    assert!(response.source.unwrap().suspended);
    assert!(query(&b, vec![10]).await.matches.is_empty());
    assert_eq!(
        b.apply_external_kv_batch(apply("w", "e", 2, 0, 1, 20))
            .await
            .unwrap_err()
            .code(),
        Code::FailedPrecondition
    );
}
#[tokio::test]
async fn failed_second_chunk_keeps_first_visible_but_no_complete_claim() {
    let b = InMemoryKvIndexerBackend::new();
    let s = register(&b, "w", "e", None, vec![1]).await.unwrap();
    empty(&b, &s, 1).await;
    b.apply_external_kv_batch(apply("w", "e", 1, 0, 2, 10))
        .await
        .unwrap();
    let mut invalid = apply("w", "e", 1, 1, 2, 11);
    invalid.actions[0].component_masks = vec![1, 1];
    assert!(b.apply_external_kv_batch(invalid).await.is_err());
    let q = query(&b, vec![10, 11]).await;
    assert_eq!(hashes(&q, "w", 1), vec![10]);
    assert_eq!(source(&q, "w").pending_seq, Some(1));
    assert!(!source(&q, "w").complete);
}
#[tokio::test]
async fn legacy_is_unknown_and_cannot_overwrite_registered_sources() {
    let b = InMemoryKvIndexerBackend::new();
    let legacy = ApplyExternalKvBatchRequest {
        worker_id: "legacy".into(),
        worker_address: "http://legacy".into(),
        actions: vec![report(10, 1)],
        ..Default::default()
    };
    b.apply_external_kv_batch(legacy.clone()).await.unwrap();
    let q = query(&b, vec![10]).await;
    let s = source(&q, "legacy");
    assert!(!s.complete);
    assert!(s.source_epoch.is_empty());
    assert_eq!(s.reason, "legacy_unknown");
    register(&b, "legacy", "e", None, vec![1]).await.unwrap();
    assert_eq!(
        b.apply_external_kv_batch(legacy).await.unwrap_err().code(),
        Code::FailedPrecondition
    );
}
#[tokio::test]
async fn restart_has_no_source_authority_and_no_implicit_recovery() {
    let b = InMemoryKvIndexerBackend::new();
    let q = query(&b, vec![10]).await;
    assert!(q.matches.is_empty());
    let m = q.observation.unwrap();
    assert!(m.scope_only);
    assert!(!m.full_snapshot);
    assert!(m.sources.is_empty());
    assert_eq!(
        b.apply_external_kv_batch(apply("w", "e", 2, 0, 1, 10))
            .await
            .unwrap_err()
            .code(),
        Code::FailedPrecondition
    );
    assert_eq!(
        register(&b, "w", "new", Some("e"), vec![1])
            .await
            .unwrap_err()
            .code(),
        Code::FailedPrecondition
    );
    let s = register(&b, "w", "new", None, vec![1]).await.unwrap();
    assert!(!s.complete);
}
#[tokio::test]
async fn bounded_sources_and_retired_epoch_tombstones_never_evict_fences() {
    let b = InMemoryKvIndexerBackend::with_source_limits(SourceLimits {
        max_sources: 1,
        max_retired_epochs: 1,
    });
    register(&b, "w", "one", None, vec![1]).await.unwrap();
    assert_eq!(
        register(&b, "other", "one", None, vec![1])
            .await
            .unwrap_err()
            .code(),
        Code::ResourceExhausted
    );
    register(&b, "w", "two", Some("one"), vec![1])
        .await
        .unwrap();
    assert_eq!(
        register(&b, "w", "three", Some("two"), vec![1])
            .await
            .unwrap_err()
            .code(),
        Code::ResourceExhausted
    );
    assert_eq!(
        register(&b, "w", "one", Some("two"), vec![1])
            .await
            .unwrap_err()
            .code(),
        Code::FailedPrecondition
    );
}
#[tokio::test]
async fn session_attribution_is_aligned_without_duplicate_physical_placements() {
    let b = InMemoryKvIndexerBackend::new();
    register(&b, "w", "e", None, vec![1]).await.unwrap();
    let mut r = apply("w", "e", 1, 0, 1, 10);
    r.actions[0].session_id = Some("session-a".into());
    b.apply_external_kv_batch(r).await.unwrap();
    let q = query(&b, vec![10]).await;
    assert_eq!(q.matches.len(), 1);
    let t = &q.matches[0].hashes_by_tier[0];
    assert_eq!(t.hashes, vec![10]);
    assert_eq!(t.session_ids, vec!["session-a"]);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gap_clear_apply_and_watermark_are_atomic_to_concurrent_readers() {
    let b = std::sync::Arc::new(InMemoryKvIndexerBackend::new());
    register(&b, "w", "e", None, vec![1]).await.unwrap();
    b.apply_external_kv_batch(apply("w", "e", 2, 0, 1, 2))
        .await
        .unwrap();
    let writer = b.clone();
    let task = tokio::spawn(async move {
        for seq in (4..=100).step_by(2) {
            writer
                .apply_external_kv_batch(apply("w", "e", seq, 0, 1, seq as i64))
                .await
                .unwrap();
            tokio::task::yield_now().await;
        }
    });
    for _ in 0..200 {
        let q = query(&b, (2..=100).collect()).await;
        let h = hashes(&q, "w", 1);
        assert_eq!(h.len(), 1);
        assert_eq!(Some(h[0] as u64), source(&q, "w").watermark_seq);
        tokio::task::yield_now().await;
    }
    task.await.unwrap();
}

#[tokio::test]
async fn explicit_gap_is_idempotent_but_reconnect_replay_cannot_preserve_complete() {
    let b = InMemoryKvIndexerBackend::new();
    let s = register(&b, "w", "e", None, vec![1]).await.unwrap();
    empty(&b, &s, 1).await;
    let mut first = apply("w", "e", 1, 0, 2, 10);
    first.observation_gap = true;
    b.apply_external_kv_batch(first.clone()).await.unwrap();
    assert!(
        b.apply_external_kv_batch(first)
            .await
            .unwrap()
            .duplicate_chunk
    );
    b.apply_external_kv_batch(apply("w", "e", 1, 1, 2, 11))
        .await
        .unwrap();
    let q = query(&b, vec![10, 11]).await;
    assert_eq!(hashes(&q, "w", 1), vec![10, 11]);
    assert!(!source(&q, "w").complete);
    let mut reconnect = apply("w", "e", 1, 0, 1, 99);
    reconnect.actions.clear();
    reconnect.observation_gap = true;
    b.apply_external_kv_batch(reconnect).await.unwrap();
    assert!(query(&b, vec![10, 11]).await.matches.is_empty());
}
fn cache_watermark(epoch: &str, generation: u64, version: u64) -> CacheWatermark {
    CacheWatermark {
        source_id: "cache:worker".into(),
        source_epoch: epoch.into(),
        generation,
        version,
    }
}
#[tokio::test]
async fn cache_watermark_stays_distinct_and_changes_require_trusted_authority() {
    let b = InMemoryKvIndexerBackend::new();
    let s = register(&b, "w", "publisher-epoch", None, vec![1])
        .await
        .unwrap();
    empty(&b, &s, 1).await;
    let mut r = apply("w", "publisher-epoch", 1, 0, 1, 10);
    r.cache_watermark = Some(cache_watermark("cache-epoch", 2, 500));
    b.apply_external_kv_batch(r).await.unwrap();
    let q = query(&b, vec![10]).await;
    let s = source(&q, "w");
    assert_eq!(s.watermark_seq, Some(1));
    assert_eq!(s.cache_watermark.as_ref().unwrap().version, 500);
    assert_eq!(s.source_epoch, "publisher-epoch");
    let mut r = apply("w", "publisher-epoch", 2, 0, 1, 20);
    r.cache_watermark = Some(cache_watermark("cache-epoch", 3, 600));
    b.apply_external_kv_batch(r).await.unwrap();
    let q = query(&b, vec![10, 20]).await;
    assert_eq!(hashes(&q, "w", 1), vec![20]);
    assert!(!source(&q, "w").complete);
    let mut r = apply("w", "publisher-epoch", 3, 0, 1, 30);
    r.cache_watermark = Some(cache_watermark("different-cache-epoch", 0, 1));
    assert_eq!(
        b.apply_external_kv_batch(r).await.unwrap_err().code(),
        Code::FailedPrecondition
    );
    let q = query(&b, vec![20, 30]).await;
    assert!(q.matches.is_empty());
    assert!(source(&q, "w").suspended);
    assert_eq!(
        source(&q, "w")
            .cache_watermark
            .as_ref()
            .unwrap()
            .source_epoch,
        "cache-epoch"
    );
}
#[tokio::test]
async fn native_watermark_does_not_advance_until_all_chunks_commit() {
    let b = InMemoryKvIndexerBackend::new();
    register(&b, "w", "e", None, vec![1]).await.unwrap();
    let mut first = apply("w", "e", 1, 0, 2, 10);
    first.cache_watermark = Some(cache_watermark("cache-e", 0, 500));
    b.apply_external_kv_batch(first.clone()).await.unwrap();
    assert!(source(&query(&b, vec![10]).await, "w")
        .cache_watermark
        .is_none());
    let mut second = apply("w", "e", 1, 1, 2, 11);
    second.cache_watermark = first.cache_watermark;
    b.apply_external_kv_batch(second).await.unwrap();
    assert_eq!(
        source(&query(&b, vec![10, 11]).await, "w")
            .cache_watermark
            .as_ref()
            .unwrap()
            .version,
        500
    );
}

#[tokio::test]
async fn invalidation_cannot_erase_sequence_fence_or_enable_old_recovery_cut() {
    let b = InMemoryKvIndexerBackend::new();
    register(&b, "w", "e", None, vec![1]).await.unwrap();
    b.apply_external_kv_batch(apply("w", "e", 10, 0, 2, 10))
        .await
        .unwrap();
    let q = query(&b, vec![10]).await;
    let s = source(&q, "w");
    let invalid = b
        .reconcile_source(ReconcileSourceRequest {
            worker_id: "w".into(),
            source_epoch: "e".into(),
            expected_source_version: s.source_version,
            expected_indexer_epoch: s.indexer_epoch.clone(),
            confirmed_empty: false,
            next_seq: 0,
            evidence_id: "worker death".into(),
        })
        .await
        .unwrap()
        .source
        .unwrap();
    let old_cut = ReconcileSourceRequest {
        worker_id: "w".into(),
        source_epoch: "e".into(),
        expected_source_version: invalid.source_version,
        expected_indexer_epoch: invalid.indexer_epoch.clone(),
        confirmed_empty: true,
        next_seq: 1,
        evidence_id: "stale reset receipt".into(),
    };
    assert_eq!(
        b.reconcile_source(old_cut).await.unwrap_err().code(),
        Code::FailedPrecondition
    );
    empty(&b, &invalid, 11).await;
    assert!(
        b.apply_external_kv_batch(apply("w", "e", 10, 0, 2, 10))
            .await
            .unwrap()
            .duplicate_chunk
    );
    assert!(query(&b, vec![10]).await.matches.is_empty());
}
#[tokio::test]
async fn bounded_chunk_coordinates_reject_before_mutation() {
    let b = InMemoryKvIndexerBackend::new();
    register(&b, "w", "e", None, vec![1]).await.unwrap();
    for (index, count) in [(0, 0), (0, 1025), (2, 2)] {
        let request = apply("w", "e", 1, index, count, 10);
        assert_eq!(
            b.apply_external_kv_batch(request).await.unwrap_err().code(),
            Code::InvalidArgument
        );
    }
    assert!(query(&b, vec![10]).await.matches.is_empty());
}

#[tokio::test]
async fn native_generation_change_clears_once_before_a_multi_chunk_batch() {
    let b = InMemoryKvIndexerBackend::new();
    register(&b, "w", "e", None, vec![1]).await.unwrap();
    let mut prior = apply("w", "e", 1, 0, 1, 1);
    prior.cache_watermark = Some(cache_watermark("cache-e", 0, 1));
    b.apply_external_kv_batch(prior).await.unwrap();
    for (chunk, hash) in [(0, 20), (1, 21)] {
        let mut r = apply("w", "e", 2, chunk, 2, hash);
        r.cache_watermark = Some(cache_watermark("cache-e", 1, 50));
        b.apply_external_kv_batch(r).await.unwrap();
    }
    let q = query(&b, vec![1, 20, 21]).await;
    assert_eq!(hashes(&q, "w", 1), vec![20, 21]);
    assert_eq!(source(&q, "w").watermark_seq, Some(2));
    assert!(!source(&q, "w").complete);
}

#[tokio::test]
async fn indexer_restart_epoch_fences_old_reset_receipts_even_when_source_version_repeats() {
    let old = InMemoryKvIndexerBackend::new();
    let old_source = register(&old, "w", "same-source-epoch", None, vec![1])
        .await
        .unwrap();
    let restarted = InMemoryKvIndexerBackend::new();
    let new_source = register(&restarted, "w", "same-source-epoch", None, vec![1])
        .await
        .unwrap();
    assert_eq!(old_source.source_version, new_source.source_version);
    assert_ne!(old_source.indexer_epoch, new_source.indexer_epoch);
    let q = query(&restarted, vec![10]).await;
    assert_eq!(
        q.observation.unwrap().indexer_epoch,
        new_source.indexer_epoch
    );
    let stale = ReconcileSourceRequest {
        worker_id: "w".into(),
        source_epoch: "same-source-epoch".into(),
        expected_source_version: old_source.source_version,
        expected_indexer_epoch: old_source.indexer_epoch,
        confirmed_empty: true,
        next_seq: 1,
        evidence_id: "old process empty-cache receipt".into(),
    };
    assert_eq!(
        restarted.reconcile_source(stale).await.unwrap_err().code(),
        Code::FailedPrecondition
    );
    assert!(!source(&query(&restarted, vec![10]).await, "w").complete);
}

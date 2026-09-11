use std::time::Duration;

use sgl_kv_indexer::pb::kv_replica_server::KvReplica;
use sgl_kv_indexer::pb::*;
use sgl_kv_indexer::replica::ReplicaService;
use tonic::Request;

fn key(worker: &str, rank: u32) -> StreamKey {
    StreamKey {
        namespace: "default".into(),
        worker_id: worker.into(),
        dp_rank: rank,
    }
}

fn cut(worker: &str, rank: u32, epoch: &str, count: u64) -> SnapshotCut {
    SnapshotCut {
        version: 2,
        stream: Some(StreamDescriptor {
            key: Some(key(worker, rank)),
            worker_address: format!("http://{worker}"),
            model: "model".into(),
            hash_schema_version: 1,
            page_size: 4,
            is_bigram: false,
            cache_spec: None,
        }),
        worker_epoch: epoch.into(),
        barrier_seq: 10,
        resume_seq: 11,
        barrier_id: "barrier".into(),
        record_count: count,
    }
}

fn block(hash: i64, tier: i32) -> PlacementBlock {
    PlacementBlock {
        block_hash: hash,
        parent_block_hash: None,
        block_size: 4,
        tier,
        component_mask: 0,
    }
}

async fn begin(s: &ReplicaService, c: SnapshotCut, previous: &str) -> StreamSession {
    let key = c.stream.as_ref().unwrap().key.clone();
    let response = s
        .begin_snapshot(Request::new(BeginSnapshotRequest {
            cut: Some(c),
            session: previous.into(),
        }))
        .await
        .unwrap()
        .into_inner();
    StreamSession {
        key,
        session: response.session,
    }
}

async fn chunk(s: &ReplicaService, owner: &StreamSession, blocks: Vec<PlacementBlock>) {
    s.snapshot_chunk(Request::new(SnapshotChunkRequest {
        owner: Some(owner.clone()),
        offset: 0,
        blocks,
    }))
    .await
    .unwrap();
}

async fn confirm(
    s: &ReplicaService,
    owner: &StreamSession,
    epoch: &str,
    next: u64,
) -> StreamProgress {
    s.confirm_stream(Request::new(ConfirmStreamRequest {
        owner: Some(owner.clone()),
        worker_epoch: epoch.into(),
        resume_seq: next,
        barrier_id: "barrier".into(),
        barrier_seq: 10,
    }))
    .await
    .unwrap()
    .into_inner()
}

async fn query(s: &ReplicaService, workers: Vec<StreamKey>) -> ReplicaPrefixResponse {
    s.match_prefix(Request::new(ReplicaPrefixRequest {
        namespace: "default".into(),
        model: "model".into(),
        hash_schema_version: 1,
        page_size: 4,
        is_bigram: false,
        hashes: vec![1, 2],
        eligible_streams: workers,
        max_blocks: 0,
    }))
    .await
    .unwrap()
    .into_inner()
}

async fn live(
    s: &ReplicaService,
    owner: &StreamSession,
    epoch: &str,
    seq: u64,
    actions: Vec<ExternalKvAction>,
) -> StreamProgress {
    s.apply_live(Request::new(LiveBatchRequest {
        owner: Some(owner.clone()),
        worker_epoch: epoch.into(),
        sequence: seq,
        actions,
    }))
    .await
    .unwrap()
    .into_inner()
}

fn revoke(hash: i64, tier: i32) -> ExternalKvAction {
    ExternalKvAction {
        r#type: 2,
        tier,
        hashes: vec![hash],
        ..Default::default()
    }
}

#[tokio::test]
async fn staging_is_hidden_and_coverage_distinguishes_no_match() {
    let s = ReplicaService::new(Duration::from_secs(10), 4);
    let a = begin(&s, cut("a", 0, "e1", 2), "").await;
    chunk(&s, &a, vec![block(1, 1), block(2, 1)]).await;
    let partial = query(&s, vec![key("a", 0)]).await;
    assert!(!partial.complete && partial.matches.is_empty());
    assert!(confirm(&s, &a, "e1", 11).await.ready);
    assert_eq!(
        query(&s, vec![key("a", 0)]).await.matches[0].matched_prefix_blocks,
        2
    );
    live(&s, &a, "e1", 11, vec![revoke(1, 1)]).await;
    let miss = query(&s, vec![key("a", 0)]).await;
    assert!(miss.complete && miss.matches.is_empty());
    assert_eq!(miss.coverage[0].watermark, 11);
    assert!(
        !query(&s, vec![key("a", 0), key("missing", 0)])
            .await
            .complete
    );
}

#[tokio::test]
async fn gap_replays_in_order_duplicates_cannot_undo_and_epoch_fences() {
    let s = ReplicaService::new(Duration::from_secs(10), 4);
    let a = begin(&s, cut("a", 0, "e1", 2), "").await;
    chunk(&s, &a, vec![block(1, 1), block(2, 1)]).await;
    confirm(&s, &a, "e1", 11).await;
    let gap = live(&s, &a, "e1", 12, vec![]).await;
    assert!(!gap.ready && !gap.needs_snapshot && gap.next_sequence == 11);
    assert!(!query(&s, vec![key("a", 0)]).await.complete);
    live(&s, &a, "e1", 11, vec![revoke(2, 1)]).await;
    live(&s, &a, "e1", 12, vec![]).await;
    assert!(confirm(&s, &a, "e1", 13).await.ready);
    live(&s, &a, "e1", 11, vec![revoke(1, 1)]).await;
    assert_eq!(
        query(&s, vec![key("a", 0)]).await.matches[0].matched_prefix_blocks,
        1
    );
    assert!(live(&s, &a, "e2", 13, vec![]).await.needs_snapshot);
    assert!(confirm(&s, &a, "e1", 13).await.needs_snapshot);
    let replacement = begin(&s, cut("a", 0, "e2", 0), &a.session).await;
    assert!(confirm(&s, &replacement, "e2", 11).await.ready);
    assert!(query(&s, vec![key("a", 0)]).await.matches.is_empty());
    assert!(s.invalidate_stream(Request::new(a)).await.is_err());
}

#[tokio::test]
async fn remove_isolated_by_worker_rank_tier_and_namespace() {
    let s = ReplicaService::new(Duration::from_secs(10), 4);
    let a = begin(&s, cut("a", 0, "e1", 2), "").await;
    chunk(&s, &a, vec![block(1, 1), block(1, 2)]).await;
    confirm(&s, &a, "e1", 11).await;
    let b = begin(&s, cut("a", 1, "rank1", 1), "").await;
    chunk(&s, &b, vec![block(1, 1)]).await;
    confirm(&s, &b, "rank1", 11).await;
    live(&s, &a, "e1", 11, vec![revoke(1, 1)]).await;
    assert_eq!(
        query(&s, vec![key("a", 0), key("a", 1)])
            .await
            .matches
            .len(),
        2
    );
    s.remove_stream(Request::new(a)).await.unwrap();
    assert_eq!(query(&s, vec![key("a", 1)]).await.matches.len(), 1);
    let mut other = cut("a", 1, "other", 1);
    other
        .stream
        .as_mut()
        .unwrap()
        .key
        .as_mut()
        .unwrap()
        .namespace = "other".into();
    let c = begin(&s, other, "").await;
    chunk(&s, &c, vec![block(2, 1)]).await;
    confirm(&s, &c, "other", 11).await;
    assert_eq!(
        query(&s, vec![key("a", 1)]).await.matches[0].matched_prefix_blocks,
        1
    );
}

#[tokio::test]
async fn invalid_barrier_chunks_and_competing_owner_cannot_publish() {
    let s = ReplicaService::new(Duration::from_secs(10), 4);
    let a = begin(&s, cut("a", 0, "e1", 1), "").await;
    assert!(s
        .begin_snapshot(Request::new(BeginSnapshotRequest {
            cut: Some(cut("a", 0, "e2", 0)),
            session: String::new()
        }))
        .await
        .is_err());
    assert!(s
        .snapshot_chunk(Request::new(SnapshotChunkRequest {
            owner: Some(a.clone()),
            offset: 0,
            blocks: vec![block(1, 1), block(1, 1)]
        }))
        .await
        .is_err());
    assert!(s
        .confirm_stream(Request::new(ConfirmStreamRequest {
            owner: Some(a.clone()),
            worker_epoch: "e1".into(),
            resume_seq: 11,
            barrier_id: "wrong".into(),
            barrier_seq: 10
        }))
        .await
        .is_err());
    assert!(!query(&s, vec![key("a", 0)]).await.complete);
    chunk(&s, &a, vec![block(1, 1)]).await;
    assert!(confirm(&s, &a, "e1", 11).await.ready);
}

#[tokio::test]
async fn lease_expiry_hides_placement_and_requires_new_snapshot() {
    let s = ReplicaService::new(Duration::from_millis(20), 4);
    let a = begin(&s, cut("a", 0, "e1", 1), "").await;
    chunk(&s, &a, vec![block(1, 1)]).await;
    confirm(&s, &a, "e1", 11).await;
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(!query(&s, vec![key("a", 0)]).await.complete);
    assert!(confirm(&s, &a, "e1", 11).await.needs_snapshot);
    let b = begin(&s, cut("a", 0, "e1", 0), "").await;
    assert!(confirm(&s, &b, "e1", 11).await.ready);
}

#[tokio::test]
async fn component_recovery_preserves_boundary_rules() {
    let s = ReplicaService::new(Duration::from_secs(10), 4);
    let mut c = cut("a", 0, "e1", 3);
    c.stream.as_mut().unwrap().cache_spec = Some(WorkerCacheSpec {
        version: 1,
        components: 5,
        full_tier_mask: 6,
        mamba_tier_mask: 6,
        ..Default::default()
    });
    let a = begin(&s, c, "").await;
    let mut blocks = vec![block(1, 1), block(2, 1), block(2, 2)];
    blocks[0].component_mask = 1;
    blocks[1].component_mask = 1;
    blocks[2].component_mask = 4;
    chunk(&s, &a, blocks).await;
    confirm(&s, &a, "e1", 11).await;
    assert_eq!(
        query(&s, vec![key("a", 0)]).await.matches[0].matched_prefix_blocks,
        2
    );
    live(&s, &a, "e1", 11, vec![revoke(2, 2)]).await;
    assert!(query(&s, vec![key("a", 0)]).await.matches.is_empty());
}

#[tokio::test]
async fn expired_stream_gc_removes_sessions_without_touching_current_owner() {
    let s = ReplicaService::new(Duration::from_millis(20), 4);
    let old = begin(&s, cut("old", 0, "e1", 1), "").await;
    chunk(&s, &old, vec![block(1, 1)]).await;
    confirm(&s, &old, "e1", 11).await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    let current = begin(&s, cut("current", 0, "e2", 1), "").await;
    chunk(&s, &current, vec![block(1, 1)]).await;
    confirm(&s, &current, "e2", 11).await;
    assert_eq!(s.reap_expired().unwrap(), 1);
    assert!(query(&s, vec![key("current", 0)]).await.complete);
    assert!(s.invalidate_stream(Request::new(old)).await.is_err());
    assert_eq!(s.reap_expired().unwrap(), 0);
}

#[tokio::test]
async fn unsupported_snapshot_schemas_cannot_replace_visible_state() {
    let s = ReplicaService::new(Duration::from_secs(10), 4);
    let owner = begin(&s, cut("a", 0, "e1", 1), "").await;
    chunk(&s, &owner, vec![block(1, 1)]).await;
    confirm(&s, &owner, "e1", 11).await;
    for version in [0, 2, 99] {
        let mut bad = cut("a", 0, "e2", 0);
        bad.stream.as_mut().unwrap().cache_spec = Some(WorkerCacheSpec {
            version,
            components: 5,
            ..Default::default()
        });
        assert!(s
            .begin_snapshot(Request::new(BeginSnapshotRequest {
                cut: Some(bad),
                session: owner.session.clone(),
            }))
            .await
            .is_err());
        let visible = query(&s, vec![key("a", 0)]).await;
        assert!(visible.complete);
        assert_eq!(visible.coverage[0].worker_epoch, "e1");
    }
}

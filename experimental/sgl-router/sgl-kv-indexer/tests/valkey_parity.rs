// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! The Valkey backend must answer exactly like the in-memory reference. Every
//! test here drives one scenario stream into both backends and compares each
//! response, so a semantic drift in either shows up as a diff, not a flake.
//!
//! Needs a `valkey-server` (or `redis-server`) binary on `PATH`, or
//! `KV_INDEXER_TEST_VALKEY_URL` pointing at a running server. Otherwise the
//! tests print a skip line and pass, so CI without Valkey stays green.

#[path = "common/id.rs"]
mod test_id;
#[path = "common/kv.rs"]
mod test_kv;

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use sgl_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ExternalKvAction, ExternalKvActionType,
    GetExternalKvHitCountsRequest, MatchExternalKvPrefixRequest, MatchExternalKvPrefixResponse,
    MatchExternalKvRequest, MatchExternalKvResponse, WorkerCacheSpec,
};
use sgl_kv_indexer::{
    InMemoryKvIndexerBackend, KvIndexerBackend, ValkeyConfig, ValkeyKvIndexerBackend,
    COMPONENT_FULL, COMPONENT_SWA,
};
use test_id::nanos;
use test_kv::{
    action, action_with_parent, apply_request, component_report, component_report_with_parent,
    dram, hbm,
};

// ---- a Valkey to test against -------------------------------------------------

struct ValkeyServer {
    url: String,
    child: Option<Child>,
    dir: Option<PathBuf>,
}

impl ValkeyServer {
    /// An external server from the environment, a freshly spawned one on a
    /// unix socket, or `None` when neither is available.
    fn start() -> Option<Self> {
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
        let dir = std::env::temp_dir().join(format!("sgl-kv-indexer-test-{}", nanos()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let socket = dir.join("valkey.sock");
        let child = Command::new(binary)
            .args(["--port", "0", "--unixsocket"])
            .arg(&socket)
            .args(["--save", "", "--appendonly", "no", "--loglevel", "warning"])
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

    async fn backend(&self, prefix: &str) -> ValkeyKvIndexerBackend {
        ValkeyKvIndexerBackend::connect(ValkeyConfig::new(self.url.clone()).with_key_prefix(prefix))
            .await
            .expect("connect to test valkey")
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
fn fresh_prefix() -> String {
    format!("{{t{}}}:", nanos())
}

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

// ---- the pair harness ---------------------------------------------------------

struct Pair {
    memory: InMemoryKvIndexerBackend,
    valkey: ValkeyKvIndexerBackend,
}

type NormalizedMatch = Vec<(String, String, Vec<(i32, Vec<(i64, u32, u32)>)>)>;

fn normalize_match(resp: &MatchExternalKvResponse) -> NormalizedMatch {
    let mut out: NormalizedMatch = resp
        .matches
        .iter()
        .map(|node| {
            let tiers = node
                .hashes_by_tier
                .iter()
                .map(|tier| {
                    let mut rows: Vec<(i64, u32, u32)> = (0..tier.hashes.len())
                        .map(|i| {
                            (
                                tier.hashes[i],
                                tier.component_masks.get(i).copied().unwrap_or(0),
                                tier.block_sizes.get(i).copied().unwrap_or(0),
                            )
                        })
                        .collect();
                    rows.sort();
                    (tier.tier, rows)
                })
                .collect();
            (node.worker_id.clone(), node.address.clone(), tiers)
        })
        .collect();
    out.sort();
    out
}

/// `blocks_read` is observability, not semantics, so it is left out.
fn normalize_prefix(resp: &MatchExternalKvPrefixResponse) -> (u32, Vec<(String, String, u32)>) {
    let mut rows: Vec<(String, String, u32)> = resp
        .matches
        .iter()
        .map(|m| {
            (
                m.worker_address.clone(),
                m.worker_id.clone(),
                m.matched_prefix_blocks,
            )
        })
        .collect();
    rows.sort();
    (resp.best_prefix_blocks, rows)
}

impl Pair {
    async fn new(server: &ValkeyServer) -> Self {
        Self {
            memory: InMemoryKvIndexerBackend::new(),
            valkey: server.backend(&fresh_prefix()).await,
        }
    }

    /// Applies to both and asserts they agree on success or on the status code.
    async fn apply(&self, req: ApplyExternalKvBatchRequest) -> Result<(), tonic::Code> {
        let expected = self.memory.apply_external_kv_batch(req.clone()).await;
        let actual = self.valkey.apply_external_kv_batch(req).await;
        match (&expected, &actual) {
            (Ok(_), Ok(_)) => Ok(()),
            (Err(e), Err(a)) => {
                assert_eq!(e.code(), a.code(), "status codes differ: {e:?} vs {a:?}");
                Err(e.code())
            }
            _ => panic!("backends disagree on acceptance: memory={expected:?} valkey={actual:?}"),
        }
    }

    async fn assert_match(&self, hashes: &[i64], count_as_hit: bool) {
        let req = MatchExternalKvRequest {
            hashes: hashes.to_vec(),
            count_as_hit,
        };
        let expected = self.memory.match_external_kv(req.clone()).await.unwrap();
        let actual = self.valkey.match_external_kv(req).await.unwrap();
        assert_eq!(
            normalize_match(&expected),
            normalize_match(&actual),
            "match differs for {hashes:?}"
        );
    }

    async fn assert_prefix(&self, hashes: &[i64], max_blocks: u32) {
        let req = MatchExternalKvPrefixRequest {
            hashes: hashes.to_vec(),
            max_blocks,
        };
        let expected = self
            .memory
            .match_external_kv_prefix(req.clone())
            .await
            .unwrap();
        let actual = self.valkey.match_external_kv_prefix(req).await.unwrap();
        assert_eq!(
            normalize_prefix(&expected),
            normalize_prefix(&actual),
            "prefix differs for {hashes:?}"
        );
    }

    async fn assert_hits(&self, hashes: &[i64]) {
        let req = GetExternalKvHitCountsRequest {
            hashes: hashes.to_vec(),
        };
        let mut expected = self
            .memory
            .get_external_kv_hit_counts(req.clone())
            .await
            .unwrap();
        let mut actual = self.valkey.get_external_kv_hit_counts(req).await.unwrap();
        expected.entries.sort_by_key(|e| e.hash);
        actual.entries.sort_by_key(|e| e.hash);
        assert_eq!(expected, actual, "hit counts differ for {hashes:?}");
    }

    /// The full comparison for one hash list: placements, prefix at a few
    /// ceilings, and hit counts.
    async fn assert_all(&self, hashes: &[i64]) {
        self.assert_match(hashes, false).await;
        for max_blocks in [0, 1, 2] {
            self.assert_prefix(hashes, max_blocks).await;
        }
        self.assert_hits(hashes).await;
    }
}

fn report(worker: &str, addr: &str, tier: i32, hashes: &[i64]) -> ApplyExternalKvBatchRequest {
    apply_request(
        worker,
        addr,
        1,
        vec![action(ExternalKvActionType::ActionReport, tier, hashes)],
    )
}

fn report_with_parent(
    worker: &str,
    addr: &str,
    tier: i32,
    parent: Option<i64>,
    hashes: &[i64],
) -> ApplyExternalKvBatchRequest {
    apply_request(
        worker,
        addr,
        1,
        vec![action_with_parent(
            ExternalKvActionType::ActionReport,
            tier,
            parent,
            hashes,
        )],
    )
}

fn revoke(worker: &str, addr: &str, tier: i32, hashes: &[i64]) -> ApplyExternalKvBatchRequest {
    apply_request(
        worker,
        addr,
        1,
        vec![action(ExternalKvActionType::ActionRevoke, tier, hashes)],
    )
}

fn clear(worker: &str, addr: &str, tier: i32) -> ApplyExternalKvBatchRequest {
    apply_request(
        worker,
        addr,
        1,
        vec![action(
            ExternalKvActionType::ActionClearAllAtTier,
            tier,
            &[],
        )],
    )
}

fn with_spec(
    mut req: ApplyExternalKvBatchRequest,
    spec: WorkerCacheSpec,
) -> ApplyExternalKvBatchRequest {
    req.cache_spec = Some(spec);
    req
}

fn full_swa_spec(window: u32) -> WorkerCacheSpec {
    WorkerCacheSpec {
        version: 1,
        components: COMPONENT_FULL | COMPONENT_SWA,
        swa_window_tokens: window,
        full_tier_mask: 1 << hbm() | 1 << dram(),
        swa_tier_mask: 1 << hbm(),
        mamba_tier_mask: 0,
    }
}

// ---- scenarios ----------------------------------------------------------------

#[tokio::test]
async fn report_revoke_clear_across_two_workers() {
    let server = require_valkey!();
    let pair = Pair::new(&server).await;
    let all = [1, 2, 3, 4, 5, 6, 7];

    pair.apply(report("a", "http://a", hbm(), &[1, 2, 3]))
        .await
        .unwrap();
    pair.apply(report("b", "http://b", hbm(), &[1, 2]))
        .await
        .unwrap();
    pair.apply(report_with_parent(
        "b",
        "http://b",
        dram(),
        Some(2),
        &[3, 4],
    ))
    .await
    .unwrap();
    pair.assert_all(&all).await;

    pair.apply(revoke("a", "http://a", hbm(), &[2]))
        .await
        .unwrap();
    pair.assert_all(&all).await;
    pair.assert_match(&[2, 2, 3], false).await;

    pair.apply(clear("b", "http://b", hbm())).await.unwrap();
    pair.assert_all(&all).await;

    // Address changes are snapshots carried on every batch.
    pair.apply(apply_request("a", "http://a-moved", 2, vec![]))
        .await
        .unwrap();
    pair.assert_all(&all).await;

    // Clearing a tier the worker never held is a no-op on both.
    pair.apply(clear("a", "http://a-moved", dram()))
        .await
        .unwrap();
    pair.apply(revoke("a", "http://a-moved", hbm(), &[99]))
        .await
        .unwrap();
    pair.assert_all(&all).await;
}

#[tokio::test]
async fn chained_reports_and_prefix_queries() {
    let server = require_valkey!();
    let pair = Pair::new(&server).await;
    let chain: Vec<i64> = (100..112).collect();

    pair.apply(report_with_parent(
        "a",
        "http://a",
        hbm(),
        None,
        &chain[..4],
    ))
    .await
    .unwrap();
    pair.apply(report_with_parent(
        "a",
        "http://a",
        hbm(),
        Some(chain[3]),
        &chain[4..8],
    ))
    .await
    .unwrap();
    pair.apply(report_with_parent(
        "b",
        "http://b",
        hbm(),
        None,
        &chain[..6],
    ))
    .await
    .unwrap();
    // A branch off block 2 held by b only.
    pair.apply(report_with_parent(
        "b",
        "http://b",
        hbm(),
        Some(chain[1]),
        &[500, 501],
    ))
    .await
    .unwrap();
    for len in [1, 3, 6, 8, 12] {
        pair.assert_prefix(&chain[..len], 0).await;
    }
    pair.assert_prefix(&[chain[0], chain[1], 500, 501], 0).await;
    pair.assert_prefix(&[chain[1], chain[2]], 0).await;
    pair.assert_all(&chain).await;

    // A hole in the middle shortens a's prefix and leaves b intact.
    pair.apply(revoke("a", "http://a", hbm(), &[chain[2]]))
        .await
        .unwrap();
    pair.assert_prefix(&chain, 0).await;
    pair.assert_prefix(&chain, 3).await;
    pair.assert_all(&chain).await;

    // Prune: revoking a leaf chain end to start drops the records; a fresh
    // report must rebuild them identically.
    pair.apply(revoke("a", "http://a", hbm(), &chain[4..8]))
        .await
        .unwrap();
    pair.assert_all(&chain).await;
    pair.apply(report_with_parent(
        "a",
        "http://a",
        hbm(),
        Some(chain[3]),
        &chain[4..8],
    ))
    .await
    .unwrap();
    pair.assert_all(&chain).await;
}

#[tokio::test]
async fn rejected_batches_leave_both_backends_untouched() {
    let server = require_valkey!();
    let pair = Pair::new(&server).await;

    pair.apply(report_with_parent("w", "old-address", hbm(), None, &[1, 2]))
        .await
        .unwrap();
    let err = pair
        .apply(apply_request(
            "w",
            "new-address",
            2,
            vec![
                action(ExternalKvActionType::ActionReport, hbm(), &[3]),
                action_with_parent(ExternalKvActionType::ActionReport, hbm(), Some(9), &[2]),
            ],
        ))
        .await
        .unwrap_err();
    assert_eq!(err, tonic::Code::InvalidArgument);
    pair.assert_all(&[1, 2, 3, 9]).await;

    // Self-parent and an in-batch cycle.
    for bad in [
        report_with_parent("w", "old-address", hbm(), Some(5), &[5]),
        report_with_parent("w", "old-address", hbm(), Some(2), &[1, 2]),
        apply_request(
            "w",
            "old-address",
            3,
            vec![
                action_with_parent(ExternalKvActionType::ActionReport, hbm(), Some(20), &[10]),
                action_with_parent(ExternalKvActionType::ActionReport, hbm(), Some(10), &[20]),
            ],
        ),
    ] {
        assert_eq!(
            pair.apply(bad).await.unwrap_err(),
            tonic::Code::InvalidArgument
        );
    }
    pair.assert_all(&[1, 2, 5, 10, 20]).await;

    // A cycle that closes through existing state: 31 hangs under 32 first,
    // then a batch tries to hang 32 under 31.
    pair.apply(report_with_parent(
        "w",
        "old-address",
        hbm(),
        Some(32),
        &[31],
    ))
    .await
    .unwrap();
    assert_eq!(
        pair.apply(report_with_parent(
            "w",
            "old-address",
            hbm(),
            Some(31),
            &[32]
        ))
        .await
        .unwrap_err(),
        tonic::Code::InvalidArgument
    );
    // The same shape one level deeper: 41 <- 42 <- 43 exists, then 43 under 41.
    pair.apply(report_with_parent(
        "w",
        "old-address",
        hbm(),
        Some(43),
        &[42, 41],
    ))
    .await
    .unwrap();
    assert_eq!(
        pair.apply(report_with_parent(
            "w",
            "old-address",
            hbm(),
            Some(41),
            &[43]
        ))
        .await
        .unwrap_err(),
        tonic::Code::InvalidArgument
    );
    pair.assert_all(&[31, 32, 41, 42, 43]).await;

    // Attaching a referenced-only root under a real parent is legal and must
    // rewrite the subtree's root so a later cycle attempt is still caught.
    pair.apply(report_with_parent(
        "w",
        "old-address",
        hbm(),
        None,
        &[50, 32],
    ))
    .await
    .unwrap();
    assert_eq!(
        pair.apply(report_with_parent(
            "w",
            "old-address",
            hbm(),
            Some(31),
            &[50]
        ))
        .await
        .unwrap_err(),
        tonic::Code::InvalidArgument
    );
    pair.assert_all(&[31, 32, 50]).await;
    pair.assert_prefix(&[50, 32, 31], 0).await;
}

#[tokio::test]
async fn component_aware_workers_and_specs() {
    let server = require_valkey!();
    let pair = Pair::new(&server).await;
    let chain: Vec<i64> = (200..206).collect();
    let both = COMPONENT_FULL | COMPONENT_SWA;
    let masks_both = vec![both; 6];
    let sizes = vec![64u32; 6];

    pair.apply(with_spec(
        apply_request(
            "s",
            "http://s",
            1,
            vec![component_report(hbm(), &chain, &masks_both, &sizes)],
        ),
        full_swa_spec(128),
    ))
    .await
    .unwrap();
    // Legacy worker on the same blocks.
    pair.apply(report_with_parent("l", "http://l", hbm(), None, &chain))
        .await
        .unwrap();
    pair.assert_all(&chain).await;

    // Partial eviction replaces the snapshot: SWA drops from block 3, so the
    // trailing-window rule shortens s's prefix.
    pair.apply(with_spec(
        apply_request(
            "s",
            "http://s",
            2,
            vec![component_report_with_parent(
                hbm(),
                Some(chain[2]),
                &chain[3..4],
                &[COMPONENT_FULL],
                &[64],
            )],
        ),
        full_swa_spec(128),
    ))
    .await
    .unwrap();
    pair.assert_all(&chain).await;

    // Components reported without a spec fail closed on both.
    pair.apply(apply_request(
        "s",
        "http://s",
        3,
        vec![component_report(
            dram(),
            &chain[..2],
            &[both, both],
            &[64, 64],
        )],
    ))
    .await
    .unwrap();
    pair.assert_all(&chain).await;

    // Restoring the spec restores the component path.
    pair.apply(with_spec(
        apply_request("s", "http://s", 4, vec![]),
        full_swa_spec(128),
    ))
    .await
    .unwrap();
    pair.assert_all(&chain).await;

    // The same block twice in one batch keeps the last snapshot.
    pair.apply(with_spec(
        apply_request(
            "s",
            "http://s",
            5,
            vec![
                component_report(hbm(), &chain[..1], &[COMPONENT_FULL], &[64]),
                component_report(hbm(), &chain[..1], &[both], &[64]),
            ],
        ),
        full_swa_spec(128),
    ))
    .await
    .unwrap();
    pair.assert_all(&chain).await;

    // An unusable spec (SWA without a window) excludes the worker on both.
    let mut broken = full_swa_spec(128);
    broken.swa_window_tokens = 0;
    pair.apply(with_spec(apply_request("s", "http://s", 6, vec![]), broken))
        .await
        .unwrap();
    pair.assert_all(&chain).await;
}

#[tokio::test]
async fn hit_counts_follow_placements() {
    let server = require_valkey!();
    let pair = Pair::new(&server).await;

    pair.apply(report("a", "http://a", hbm(), &[1, 2, 3]))
        .await
        .unwrap();
    pair.assert_match(&[1, 2, 9], true).await;
    pair.assert_match(&[1, 1, 3], true).await;
    pair.assert_match(&[2], false).await;
    pair.assert_hits(&[1, 2, 3, 9]).await;

    // Losing the last placement forgets the counter; a second holder keeps it.
    pair.apply(report_with_parent("b", "http://b", dram(), Some(2), &[3]))
        .await
        .unwrap();
    pair.apply(revoke("a", "http://a", hbm(), &[1, 3]))
        .await
        .unwrap();
    pair.assert_hits(&[1, 2, 3]).await;
    pair.assert_all(&[1, 2, 3]).await;
}

#[tokio::test]
async fn valkey_state_survives_server_restart_and_is_shared() {
    let server = require_valkey!();
    let prefix = fresh_prefix();
    let memory = InMemoryKvIndexerBackend::new();
    let chain: Vec<i64> = (300..308).collect();

    let first = server.backend(&prefix).await;
    let req = report_with_parent("a", "http://a", hbm(), None, &chain);
    memory.apply_external_kv_batch(req.clone()).await.unwrap();
    first.apply_external_kv_batch(req).await.unwrap();
    drop(first);

    // A brand new server process over the same keyspace: the index is intact.
    let second = server.backend(&prefix).await;
    let pair = Pair {
        memory,
        valkey: second,
    };
    pair.assert_all(&chain).await;

    // Two servers active-active: writes through one are read through the other.
    let third = server.backend(&prefix).await;
    let req = report_with_parent("b", "http://b", hbm(), None, &chain[..3]);
    pair.memory
        .apply_external_kv_batch(req.clone())
        .await
        .unwrap();
    third.apply_external_kv_batch(req).await.unwrap();
    pair.assert_all(&chain).await;
}

#[tokio::test]
async fn valkey_blocks_read_matches_reference_on_first_miss_and_cap() {
    let server = require_valkey!();
    let backend = server.backend(&fresh_prefix()).await;
    backend
        .apply_external_kv_batch(report_with_parent(
            "a",
            "http://a",
            hbm(),
            None,
            &[1, 2, 3, 4],
        ))
        .await
        .unwrap();
    let miss = backend
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: vec![99, 1, 2],
            max_blocks: 0,
        })
        .await
        .unwrap();
    assert_eq!(miss.blocks_read, 1);
    assert!(miss.matches.is_empty());
    let capped = backend
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: vec![1, 2, 3, 4],
            max_blocks: 2,
        })
        .await
        .unwrap();
    assert_eq!(capped.blocks_read, 2);
    assert_eq!(capped.best_prefix_blocks, 2);
}

// ---- randomized parity ----------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % bound.max(1)
    }
}

/// Random report/revoke/clear streams over a few chains and three workers, one
/// component-aware. Every step is followed by a full comparison. Deterministic
/// seeds so a failure is reproducible.
#[tokio::test]
async fn randomized_streams_stay_in_parity() {
    let server = require_valkey!();
    for seed in [
        0x5eed_1234_abcd_0001u64,
        0x0bad_cafe_f00d_0002,
        0x7777_1111_2222_0003,
    ] {
        randomized_stream(&server, seed).await;
    }
}

async fn randomized_stream(server: &ValkeyServer, seed: u64) {
    let pair = Pair::new(server).await;
    let mut rng = Lcg(seed);

    const CHAINS: i64 = 4;
    const LEN: usize = 8;
    let chain = |c: i64| -> Vec<i64> { (0..LEN as i64).map(|i| 1000 * (c + 1) + i).collect() };
    let workers = ["w0", "w1", "w2"];
    let spec = full_swa_spec(128);

    let mut queries: Vec<Vec<i64>> = (0..CHAINS).map(chain).collect();
    let mut pool: Vec<i64> = queries.iter().flatten().copied().collect();
    pool.push(7);
    pool.push(-7);
    queries.push(pool.clone());

    for step in 0..300 {
        let worker = workers[rng.next(3)];
        let component_aware = worker == "w2";
        let address = if rng.next(20) == 0 {
            String::new()
        } else {
            format!("http://{worker}")
        };
        let tier = if rng.next(3) == 0 { dram() } else { hbm() };
        let c = chain(rng.next(CHAINS as usize) as i64);
        let mut actions: Vec<ExternalKvAction> = Vec::new();
        for _ in 0..1 + rng.next(3) {
            match rng.next(10) {
                0..=4 => {
                    // A root-first chain segment, possibly continuing an
                    // earlier prefix of the same chain.
                    let start = rng.next(LEN);
                    let end = (start + 1 + rng.next(LEN - start)).min(LEN);
                    let parent = if start == 0 { None } else { Some(c[start - 1]) };
                    let hashes = &c[start..end];
                    if component_aware {
                        let masks: Vec<u32> = hashes
                            .iter()
                            .map(|_| {
                                if rng.next(4) == 0 {
                                    COMPONENT_FULL
                                } else {
                                    COMPONENT_FULL | COMPONENT_SWA
                                }
                            })
                            .collect();
                        let sizes = vec![64u32; hashes.len()];
                        actions.push(component_report_with_parent(
                            tier, parent, hashes, &masks, &sizes,
                        ));
                    } else {
                        actions.push(action_with_parent(
                            ExternalKvActionType::ActionReport,
                            tier,
                            parent,
                            hashes,
                        ));
                    }
                }
                5..=7 => {
                    let n = 1 + rng.next(3);
                    let hashes: Vec<i64> = (0..n).map(|_| pool[rng.next(pool.len())]).collect();
                    actions.push(action(ExternalKvActionType::ActionRevoke, tier, &hashes));
                }
                8 => actions.push(action(
                    ExternalKvActionType::ActionClearAllAtTier,
                    tier,
                    &[],
                )),
                _ => {
                    // A conflicting parent for a block that already has one, or
                    // a stray root report: both must be judged identically.
                    let index = 1 + rng.next(LEN - 1);
                    let parent = if rng.next(2) == 0 { None } else { Some(7) };
                    actions.push(action_with_parent(
                        ExternalKvActionType::ActionReport,
                        tier,
                        parent,
                        &c[index..index + 1],
                    ));
                }
            }
        }
        let mut req = apply_request(worker, &address, step as u64, actions);
        if component_aware && rng.next(6) != 0 {
            req.cache_spec = Some(spec);
        }
        // Captured by the harness; shown only when the test fails.
        eprintln!("seed {seed:#x} step {step}: {req:?}");
        let _ = pair.apply(req).await;

        for q in &queries {
            pair.assert_match(q, rng.next(4) == 0).await;
            pair.assert_prefix(q, rng.next(3) as u32 * 3).await;
        }
        pair.assert_hits(&pool).await;
    }
}

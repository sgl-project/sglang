// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `/metrics` endpoint — Prometheus 0.0.4 exposition.
//!
//! Returns the live snapshot of [`crate::server::metrics::MetricsRegistry`].
//! Plain-text body; charset is utf-8. We deliberately don't gate this on
//! readiness — scrapers should be able to read the metrics surface even
//! while the router is warming up so the "router started but no workers
//! discovered" failure mode is observable.

use crate::discovery::WorkerMode;
use crate::policies::kv_events::bootstrap::{BootstrapTracker, PeerRegistry};
use crate::policies::kv_events::{KvIndexMetrics, Tiers, ACCOUNTING_REASONS};
use crate::server::app_context::AppContext;
use crate::server::metrics::{escape_label, WorkerSnapshot};
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use std::sync::Arc;

/// Content-Type per Prometheus exposition format spec.
const PROMETHEUS_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

pub async fn metrics(State(ctx): State<Arc<AppContext>>) -> impl IntoResponse {
    // Sample the live registry into a snapshot for the worker gauges. These
    // are pull-on-scrape (not pushed) so removed workers stop emitting series
    // immediately; see `MetricsRegistry::render_with_workers`.
    let workers: Vec<WorkerSnapshot> = ctx
        .registry
        .all()
        .into_iter()
        .map(|w| {
            // One lock acquisition for both health + state so the two gauges
            // can't report a torn (self-contradictory) pair for one scrape.
            let cb = w.breaker.snapshot();
            WorkerSnapshot {
                worker_url: w.url.clone(),
                mode: match w.mode() {
                    WorkerMode::Plain => "plain",
                    WorkerMode::Prefill => "prefill",
                    WorkerMode::Decode => "decode",
                },
                healthy: cb.admit,
                cb_state: cb.state_code,
                // Saturating rather than `as i64`: a guard-accounting
                // underflow would wrap usize and render as a nonsensical
                // negative gauge; clamp to a large positive ceiling instead.
                inflight: i64::try_from(w.active_load()).unwrap_or(i64::MAX),
            }
        })
        .collect();
    let mut body = ctx.metrics.render_with_workers(&workers);
    // Pull-on-scrape, like the worker gauges above: the tree and the tally
    // own the numbers, so a worker that goes away stops emitting series
    // without anything having to reset a pushed counter.
    // Emitted unconditionally: without it, "no kv series" is indistinguishable
    // between the intended metadata-only mode, a broken `kv_metrics` wiring,
    // and a regressed endpoint.
    body.push_str(
        "# HELP sgl_router_kv_tree_maintained 1 when this router maintains its own cache-aware KV tree and therefore emits the sgl_router_kv_* series; 0 when placement comes from an external Indexer, where those series would be a structural zero and are omitted rather than reported as an empty tier stream.\n",
    );
    body.push_str("# TYPE sgl_router_kv_tree_maintained gauge\n");
    body.push_str(&format!(
        "sgl_router_kv_tree_maintained {}\n",
        u8::from(ctx.kv_metrics.is_some()),
    ));
    if let Some(kv) = ctx.kv_metrics.as_ref() {
        body.push_str(&render_kv_tiers(
            kv,
            ctx.block_size_oracle.get().unwrap_or(0),
        ));
    }
    if let Some(index) = ctx.kv_index.as_ref() {
        body.push_str(&render_kv_peers(&index.peers()));
        body.push_str(&render_kv_bootstrap(
            &index.bootstrap(),
            index.tree().node_count(),
        ));
    }
    (
        StatusCode::OK,
        [(CONTENT_TYPE, PROMETHEUS_CONTENT_TYPE)],
        body,
    )
}

/// Render the cache-aware bootstrap series.
///
/// These make "this replica is serving cache-blind" alertable. Without them
/// the condition is only inferable from a hit-rate dip, which arrives late and
/// names nothing.
fn render_kv_bootstrap(tracker: &BootstrapTracker, tree_nodes: usize) -> String {
    let mut out = String::new();

    // A whole-tree count, which the per-(worker, tier) block gauges cannot be
    // summed into — they count a node once per carrier and once per tier. This
    // is the number that makes a failed bootstrap legible at a glance: in the
    // incident this feature exists for, two replicas came up holding 140k and
    // 205k nodes beside a warm sibling's 8.3M, and the per-replica panel said
    // so immediately where the hit-rate dip took much longer to attribute.
    out.push_str(
        "# HELP sgl_router_kv_tree_nodes Nodes in this replica's cache-aware KV tree. Compare across replicas of one Deployment: an order-of-magnitude outlier is a replica that failed to inherit the fleet's prefixes and is routing cache-blind.\n",
    );
    out.push_str("# TYPE sgl_router_kv_tree_nodes gauge\n");
    out.push_str(&format!("sgl_router_kv_tree_nodes {tree_nodes}\n"));

    out.push_str(
        "# HELP sgl_router_kv_bootstrap_settled 1 once initial peer bootstrap has settled, which is also readiness condition 3. Latches: a later scale-up cannot drag an already-serving replica back to 503.\n",
    );
    out.push_str("# TYPE sgl_router_kv_bootstrap_settled gauge\n");
    out.push_str(&format!(
        "sgl_router_kv_bootstrap_settled {}\n",
        u8::from(tracker.settled()),
    ));

    out.push_str(
        "# HELP sgl_router_kv_bootstrap_seed_failed 1 when a sweep ended timed_out over a non-empty candidate set: siblings were there and their tree could not be pulled. With --kv-bootstrap-seed-required this holds /readyz at 503; without it the replica serves cache-blind and this is the only signal that it did.\n",
    );
    out.push_str("# TYPE sgl_router_kv_bootstrap_seed_failed gauge\n");
    out.push_str(&format!(
        "sgl_router_kv_bootstrap_seed_failed {}\n",
        u8::from(tracker.seed_failed()),
    ));

    let states = tracker.states();
    if !states.is_empty() {
        out.push_str(
            "# HELP sgl_router_kv_bootstrap_state Per-rank bootstrap state (0=pending, 1=recovered, 2=failed). A rank stuck at 0 is holding its events back and will overflow; a fleet of 2s means every rank is routing on live deltas alone.\n",
        );
        out.push_str("# TYPE sgl_router_kv_bootstrap_state gauge\n");
        for (id, state) in states {
            out.push_str(&format!(
                "sgl_router_kv_bootstrap_state{{worker_url=\"{}\",dp_rank=\"{}\"}} {}\n",
                escape_label(&id.url),
                id.dp_rank,
                state.as_metric(),
            ));
        }
    }

    // Two counters, deliberately not one: this counts FETCHES against peers,
    // the next counts RANKS. One accepted fetch can settle several ranks and
    // one rank can outlive many rejected fetches, so a shared counter would be
    // divisible by nothing.
    let peer_outcomes = tracker.peer_outcome_counts();
    if !peer_outcomes.is_empty() {
        out.push_str(
            "# HELP sgl_router_kv_peer_snapshot_total Peer snapshot fetches by outcome. A fleet pinned at unreachable with a warm tree is the per-fetch timeout being too small for the body, not a network fault.\n",
        );
        out.push_str("# TYPE sgl_router_kv_peer_snapshot_total counter\n");
        for (outcome, count) in peer_outcomes {
            out.push_str(&format!(
                "sgl_router_kv_peer_snapshot_total{{outcome=\"{outcome}\"}} {count}\n",
            ));
        }
    }

    // Recorded once per rank, at the point its verdict is final — so `warm`
    // lags the state gauge by the splice proof, and the labels sum to the
    // number of ranks that finished bootstrapping.
    let rank_outcomes = tracker.rank_outcome_counts();
    if !rank_outcomes.is_empty() {
        out.push_str(
            "# HELP sgl_router_kv_bootstrap_rank_total Ranks by final bootstrap outcome, counted once each when the verdict becomes final. warm_unwitnessed means the graft was kept without proof that the live stream joins it; a persistent gap share means peers are exporting staler than the ranks can splice.\n",
        );
        out.push_str("# TYPE sgl_router_kv_bootstrap_rank_total counter\n");
        for (outcome, count) in rank_outcomes {
            out.push_str(&format!(
                "sgl_router_kv_bootstrap_rank_total{{outcome=\"{outcome}\"}} {count}\n",
            ));
        }
    }

    // One per sweep, not per rank: this is what separates "settled cold as soon
    // as every sibling proved empty" (fleet_cold) from "burned the whole
    // deadline" (timed_out), which the rank outcomes above deliberately fold
    // into the same `abandoned` label.
    let sweep_results = tracker.sweep_result_counts();
    if !sweep_results.is_empty() {
        out.push_str(
            "# HELP sgl_router_kv_bootstrap_sweep_total Peer sweeps by terminal verdict. fleet_cold is a healthy early settle on a fleet with nothing to hand over; timed_out over real candidates is a failed seed.\n",
        );
        out.push_str("# TYPE sgl_router_kv_bootstrap_sweep_total counter\n");
        for (result, count) in sweep_results {
            out.push_str(&format!(
                "sgl_router_kv_bootstrap_sweep_total{{result=\"{result}\"}} {count}\n",
            ));
        }
    }
    out
}

/// Render the peer-discovery series.
///
/// Emitted only where peer bootstrap could run at all (this router maintains
/// its own tree), so a 0 here means "the selector matched no ready sibling",
/// never "this build has no such feature". `synced` is what separates the two
/// readings of a zero peer count that matter operationally: not yet told, vs
/// told and genuinely alone. An RBAC failure on the router's own Service shows
/// up as `synced 0` that never becomes 1.
fn render_kv_peers(peers: &PeerRegistry) -> String {
    let mut out = String::new();
    out.push_str(
        "# HELP sgl_router_kv_bootstrap_peers Ready sibling router replicas this replica could pull a cache-aware tree snapshot from. 0 with sgl_router_kv_bootstrap_peers_synced=1 means the fleet genuinely has no other ready replica; 0 with synced=0 means peer discovery has not reported yet (check RBAC for endpointslices on the router's own Service).\n",
    );
    out.push_str("# TYPE sgl_router_kv_bootstrap_peers gauge\n");
    out.push_str(&format!("sgl_router_kv_bootstrap_peers {}\n", peers.len()));
    out.push_str(
        "# HELP sgl_router_kv_bootstrap_peers_synced 1 once peer discovery has reported at least once, even with an empty result. Until then an empty peer set is not evidence of anything.\n",
    );
    out.push_str("# TYPE sgl_router_kv_bootstrap_peers_synced gauge\n");
    out.push_str(&format!(
        "sgl_router_kv_bootstrap_peers_synced {}\n",
        u8::from(peers.synced()),
    ));
    out
}

/// Render the storage-tier series: what the tree holds per worker and tier,
/// the block size to convert it to tokens, and the tagged event stream it
/// consumed.
///
/// These exist to make a router-vs-engine tier mismatch a number instead of
/// an inference. `sgl_router_kv_tree_blocks * sgl_router_kv_block_size`
/// for a worker and tier, divided by that pod's own occupancy of the tier
/// (device: `sglang_kv_used_tokens + sglang_kv_evictable_tokens`; host:
/// `sglang_hicache_host_used_tokens`; `tp_rank="0"`), is the tree's coverage
/// of the tier. About 1 means the tree mirrors the engine; about 0 means the
/// engine holds a tier that routing cannot see; a missing series means the
/// worker publishes nothing. The event counters show whether the tagged
/// stream that should feed the tree is arriving at all.
/// Emitted only when the router maintains a local tree; in metadata-only mode
/// (external Indexer) the tier and event series here would be a structural
/// zero, which the HELP text below would have the operator read as a missing
/// tier stream. See `KvEventIndex::metrics_source`.
///
/// `block_size` is 0 until the first worker reports, and a coverage panel
/// multiplies by it, so such a panel reads 0 rather than NaN on a fleet that
/// has not registered yet.
fn render_kv_tiers(kv: &KvIndexMetrics, block_size: u32) -> String {
    let mut out = String::new();

    out.push_str(
        "# HELP sgl_router_kv_block_size Tokens per KV block hash, as established from the fleet (0 until a worker reports). Multiply sgl_router_kv_tree_blocks by this to compare with the engine's token gauges.\n",
    );
    out.push_str("# TYPE sgl_router_kv_block_size gauge\n");
    out.push_str(&format!("sgl_router_kv_block_size {block_size}\n"));

    // Every tier is emitted per carrier, zeros included: a host row at 0 next
    // to a device row in the millions is the mismatch signature, and an
    // absent series cannot be told from a tier the tree never tracked.
    out.push_str(
        "# HELP sgl_router_kv_tree_blocks Blocks the cache-aware tree attributes to a worker rank, by the storage tier the worker holds them on (a block held on device and host counts under both). Times sgl_router_kv_block_size, and divided by the engine's own occupancy of that tier for the same pod (device: sglang_kv_used_tokens + sglang_kv_evictable_tokens; host: sglang_hicache_host_used_tokens; tp_rank=\"0\"), this is the tree's coverage of the tier: ~1 mirrors the engine, ~0 means the engine holds a tier routing cannot see.\n",
    );
    out.push_str("# TYPE sgl_router_kv_tree_blocks gauge\n");
    for (id, counts) in kv.tree.tier_occupancy() {
        for (slot, (_, tier)) in Tiers::SLOTS.iter().enumerate() {
            out.push_str(&format!(
                "sgl_router_kv_tree_blocks{{worker_url=\"{}\",dp_rank=\"{}\",tier=\"{}\"}} {}\n",
                escape_label(&id.url),
                id.dp_rank,
                tier,
                counts[slot],
            ));
        }
    }

    let rows = kv.tally.snapshot();
    out.push_str(
        "# HELP sgl_router_kv_events_total KV-cache events the pump consumed, by kind and the storage medium tag they carried (untagged = no medium field; unknown = a medium this build does not recognise). On a hierarchical-cache fleet block_stored/CPU_PINNED runs at about the block_removed/GPU rate; a CPU_PINNED row pinned at 0 with hicache enabled means the tier stream is not reaching the router. A nonzero block_stored/unknown row means the engine publishes a tier this build cannot rank and the tree is dropping those stores: upgrade the router.\n",
    );
    out.push_str("# TYPE sgl_router_kv_events_total counter\n");
    for r in &rows {
        out.push_str(&format!(
            "sgl_router_kv_events_total{{event=\"{}\",medium=\"{}\"}} {}\n",
            r.event, r.medium, r.events,
        ));
    }
    out.push_str(
        "# HELP sgl_router_kv_event_blocks_total Block hashes carried by the KV-cache events the pump consumed, by kind and storage medium tag. Times sgl_router_kv_block_size this is comparable to the engine's device eviction volume (block_removed/GPU) and, summed without its pool label, to sglang_hicache_backup_tokens_total (block_stored/CPU_PINNED). The two are not equal: the engine also evicts device blocks it never backed up.\n",
    );
    out.push_str("# TYPE sgl_router_kv_event_blocks_total counter\n");
    for r in &rows {
        out.push_str(&format!(
            "sgl_router_kv_event_blocks_total{{event=\"{}\",medium=\"{}\"}} {}\n",
            r.event, r.medium, r.blocks,
        ));
    }

    // A tagged removal clears only its own tier, so a batch lost in transit
    // can strand a tier bit the tree will never clear on its own. Nonzero
    // here is the explanation for tree coverage drifting above 1.
    out.push_str(
        "# HELP sgl_router_kv_event_batches_lost_total KV-event batches dropped in transit, inferred from gaps in each publisher's dense sequence number (ZMQ drops at the publisher's high-water mark). Nonzero means the tree may hold tiers a worker has already released, which shows up as sgl_router_kv_tree_blocks exceeding the engine's own occupancy of that tier.\n",
    );
    out.push_str("# TYPE sgl_router_kv_event_batches_lost_total counter\n");
    out.push_str(&format!(
        "sgl_router_kv_event_batches_lost_total {}\n",
        kv.tally.batches_lost(),
    ));

    out.push_str(
        "# HELP sgl_router_kv_tree_accounting_errors_total Times the tree's per-tier occupancy bookkeeping contradicted itself. Always 0 on a correct tree. Nonzero means sgl_router_kv_tree_blocks understates what the tree holds, and can drop a worker's series entirely — which the gauge's own HELP would have you read as a worker that publishes nothing.\n",
    );
    out.push_str("# TYPE sgl_router_kv_tree_accounting_errors_total counter\n");
    for (reason, count) in ACCOUNTING_REASONS.iter().zip(kv.tree.accounting_errors()) {
        out.push_str(&format!(
            "sgl_router_kv_tree_accounting_errors_total{{reason=\"{reason}\"}} {count}\n",
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::metrics::{RequestOutcome, WorkerModeLabel};
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    /// The bootstrap series are what make "this replica is serving
    /// cache-blind" alertable, so their names and label keys are contract.
    /// `seed_failed` in particular is the ONLY signal when the gate is off —
    /// the replica serves anyway, and nothing else says the seed did not land.
    #[tokio::test]
    async fn kv_bootstrap_series_report_state_and_verdicts() {
        use crate::policies::kv_events::bootstrap::{RankOutcome, SnapshotOutcome, SweepOutcome};
        use crate::policies::kv_events::{BootstrapState, BootstrapTracker, KvWorkerId};

        let tracker = BootstrapTracker::new(std::time::Duration::from_secs(300));
        let warm = KvWorkerId::new("http://w0:30000".into(), 0);
        let cold = KvWorkerId::new("http://w0:30000".into(), 1);
        tracker.register(&[warm.clone(), cold.clone()]);
        tracker.set(&warm, BootstrapState::Recovered);
        tracker.record_rank_outcome(RankOutcome::Warm);
        tracker.record_peer_outcome(SnapshotOutcome::Accepted, "http://peer:30000", None);
        tracker.record_sweep_result(SweepOutcome::TimedOut, 9);

        let out = render_kv_bootstrap(&tracker, 4_242);
        for want in [
            "sgl_router_kv_tree_nodes 4242\n",
            "sgl_router_kv_bootstrap_settled 0\n",
            "sgl_router_kv_bootstrap_seed_failed 1\n",
            r#"sgl_router_kv_bootstrap_state{worker_url="http://w0:30000",dp_rank="0"} 1"#,
            r#"sgl_router_kv_bootstrap_state{worker_url="http://w0:30000",dp_rank="1"} 0"#,
            r#"sgl_router_kv_peer_snapshot_total{outcome="accepted"} 1"#,
            r#"sgl_router_kv_bootstrap_rank_total{outcome="warm"} 1"#,
            r#"sgl_router_kv_bootstrap_sweep_total{result="timed_out"} 1"#,
        ] {
            assert!(out.contains(want), "missing {want:?}; got:\n{out}");
        }

        // A rank reaching a terminal state settles the tracker, and the gauge
        // must move with it rather than latch at boot.
        tracker.set(&cold, BootstrapState::Failed);
        assert!(tracker.settled());
        assert!(
            render_kv_bootstrap(&tracker, 4_242).contains("sgl_router_kv_bootstrap_settled 1\n")
        );
    }

    /// The two readings of a zero peer count an operator has to tell apart:
    /// "discovery has not reported yet" and "this replica is genuinely alone".
    /// Only `synced` separates them, so both series are contract.
    #[tokio::test]
    async fn kv_peer_series_distinguish_unsynced_from_alone() {
        let peers = PeerRegistry::new();
        let before = render_kv_peers(&peers);
        assert!(before.contains("sgl_router_kv_bootstrap_peers 0\n"));
        assert!(before.contains("sgl_router_kv_bootstrap_peers_synced 0\n"));

        peers.replace(vec![]);
        let alone = render_kv_peers(&peers);
        assert!(alone.contains("sgl_router_kv_bootstrap_peers 0\n"));
        assert!(
            alone.contains("sgl_router_kv_bootstrap_peers_synced 1\n"),
            "a synced empty set is a different fact from an unsynced one",
        );

        peers.replace(vec!["http://a:30000".into(), "http://b:30000".into()]);
        let warm = render_kv_peers(&peers);
        assert!(warm.contains("sgl_router_kv_bootstrap_peers 2\n"));
        assert!(warm.contains("sgl_router_kv_bootstrap_peers_synced 1\n"));
    }

    /// A router with no local tree cannot peer-bootstrap at all, so the series
    /// are absent rather than a confidently wrong 0. The live wiring is part
    /// of the claim, hence the full scrape.
    #[tokio::test]
    async fn kv_peer_series_follow_the_local_tree() {
        use crate::policies::kv_events::KvEventIndex;

        let scrape = |ctx: Arc<AppContext>| async move {
            let res = crate::server::app::build_router(ctx)
                .oneshot(
                    Request::builder()
                        .uri("/metrics")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let body = res.into_body().collect().await.unwrap().to_bytes();
            String::from_utf8(body.to_vec()).unwrap()
        };

        assert!(
            !scrape(Arc::new(AppContext::stub()))
                .await
                .contains("sgl_router_kv_bootstrap_peers"),
            "a router with no local tree emits no peer series at all",
        );

        let index = KvEventIndex::new();
        index.peers().replace(vec!["http://sibling:30000".into()]);
        let mut ctx = AppContext::stub();
        ctx.kv_index = index.snapshot_source();
        let body = scrape(Arc::new(ctx)).await;
        assert!(body.contains("sgl_router_kv_bootstrap_peers 1\n"));
        assert!(body.contains("sgl_router_kv_bootstrap_peers_synced 1\n"));
    }

    /// The tier series are what a coverage dashboard joins on, so their names
    /// and label keys are contract: per-worker blocks by tier with zeros
    /// emitted, the block size to convert them, and every (event, medium)
    /// cell of the tally.
    #[tokio::test]
    async fn kv_tier_series_render_per_worker_and_per_medium() {
        use crate::policies::kv_events::{EventKind, EventTally, HashTree, KvWorkerId};

        let kv = KvIndexMetrics::new(Arc::new(HashTree::new()), Arc::new(EventTally::new()));
        let w = KvWorkerId::new("http://w0:30000".into(), 0);
        kv.tree.insert_tiered(&w, None, &[1, 2, 3], Tiers::DEVICE);
        kv.tree.insert_tiered(&w, None, &[1, 2], Tiers::HOST);
        kv.tree.insert_tiered(&w, None, &[1], Tiers::EXTERNAL);
        kv.tally
            .record(EventKind::BlockStored, Some("CPU_PINNED"), 2);

        let out = render_kv_tiers(&kv, 64);
        let blocks = |tier: &str, n: u32| {
            format!(
                r#"sgl_router_kv_tree_blocks{{worker_url="http://w0:30000",dp_rank="0",tier="{tier}"}} {n}"#
            )
        };
        // Every tier is asserted, the zero rows included: those are the ones
        // a later edit to `Tiers::SLOTS` would silently drop.
        let mut want: Vec<String> = [("device", 3), ("host", 2), ("disk", 0), ("external", 1)]
            .iter()
            .map(|(t, n)| blocks(t, *n))
            .collect();
        want.extend(
            [
                "sgl_router_kv_block_size 64\n",
                r#"sgl_router_kv_events_total{event="block_stored",medium="CPU_PINNED"} 1"#,
                r#"sgl_router_kv_event_blocks_total{event="block_stored",medium="CPU_PINNED"} 2"#,
                r#"sgl_router_kv_events_total{event="block_removed",medium="GPU"} 0"#,
            ]
            .iter()
            .map(|s| (*s).to_owned()),
        );
        for w in want {
            assert!(out.contains(&w), "missing {w:?}; got:\n{out}");
        }
    }

    /// The series are pulled from the tree on every scrape, so dropping a
    /// worker must make its rows disappear rather than freeze at their last
    /// value. This is what `KvEventIndex::remove_worker` relies on when it
    /// calls `clear_worker`.
    #[tokio::test]
    async fn kv_tree_blocks_drop_with_the_worker() {
        use crate::policies::kv_events::{EventTally, HashTree, KvWorkerId};

        let kv = KvIndexMetrics::new(Arc::new(HashTree::new()), Arc::new(EventTally::new()));
        let w = KvWorkerId::new("http://w0:30000".into(), 0);
        kv.tree.insert_tiered(&w, None, &[1, 2], Tiers::HOST);
        assert!(render_kv_tiers(&kv, 64).contains("http://w0:30000"));

        kv.tree.clear_worker(&w);
        let out = render_kv_tiers(&kv, 64);
        assert!(
            !out.contains("http://w0:30000"),
            "a cleared worker must stop emitting series; got:\n{out}"
        );
        // The family itself stays declared so the scrape shape is stable.
        assert!(out.contains("# TYPE sgl_router_kv_tree_blocks gauge"));
    }

    /// The route wiring itself: `render_kv_tiers` had two direct unit tests
    /// but nothing exercised `ctx.kv_metrics`, so deleting the `if let` in the
    /// handler left the suite green while `/metrics` silently stopped emitting
    /// all four families.
    #[tokio::test]
    async fn metrics_endpoint_emits_kv_series_when_a_tree_is_maintained() {
        use crate::policies::kv_events::{EventTally, HashTree, KvWorkerId};

        let mut ctx = AppContext::stub();
        let tree = Arc::new(HashTree::new());
        tree.insert_tiered(
            &KvWorkerId::new("http://w0:30000".into(), 0),
            None,
            &[1, 2],
            Tiers::HOST,
        );
        ctx.kv_metrics = Some(KvIndexMetrics::new(tree, Arc::new(EventTally::new())));
        let ctx = Arc::new(ctx);

        let app = crate::server::app::build_router(ctx.clone());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&body).unwrap();
        assert!(body.contains("sgl_router_kv_tree_maintained 1"));
        assert!(body.contains(
            r#"sgl_router_kv_tree_blocks{worker_url="http://w0:30000",dp_rank="0",tier="host"} 2"#
        ));
        assert!(body.contains("sgl_router_kv_event_batches_lost_total 0"));
        assert!(
            body.contains(r#"sgl_router_kv_tree_accounting_errors_total{reason="underflow"} 0"#)
        );
    }

    /// The other half of the gate: with no local tree the families are absent,
    /// but the mode gauge still says so rather than leaving the operator to
    /// guess whether the endpoint regressed.
    #[tokio::test]
    async fn metrics_endpoint_reports_the_mode_when_no_tree_is_maintained() {
        let ctx = Arc::new(AppContext::stub());
        assert!(ctx.kv_metrics.is_none());
        let app = crate::server::app::build_router(ctx.clone());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&body).unwrap();
        assert!(body.contains("sgl_router_kv_tree_maintained 0"));
        assert!(
            !body.contains("sgl_router_kv_tree_blocks{"),
            "a structural zero must not be emitted as if it were a reading",
        );
    }

    #[tokio::test]
    async fn metrics_endpoint_returns_prometheus_text() {
        let ctx = Arc::new(AppContext::stub());
        let app = crate::server::app::build_router(ctx.clone());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let content_type = res
            .headers()
            .get(CONTENT_TYPE)
            .expect("content-type header")
            .to_str()
            .unwrap()
            .to_owned();
        assert!(
            content_type.starts_with("text/plain"),
            "expected text/plain, got {content_type}",
        );
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&body).unwrap();
        // Every metric family should at least carry its HELP/TYPE lines.
        assert!(body.contains("# TYPE sgl_router_requests_total counter"));
        assert!(body.contains("# TYPE sgl_router_active_load gauge"));
    }

    #[tokio::test]
    async fn metrics_endpoint_reflects_recorded_counters() {
        let ctx = Arc::new(AppContext::stub());
        ctx.metrics.record_worker_request(
            "http://w-test:30000",
            "tiny",
            WorkerModeLabel::Prefill,
            RequestOutcome::Success,
        );
        let app = crate::server::app::build_router(ctx.clone());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&body).unwrap();
        assert!(
            body.contains(r#"worker_url="http://w-test:30000""#),
            "metrics did not include the recorded worker_url; got:\n{body}",
        );
    }

    #[tokio::test]
    async fn metrics_endpoint_samples_worker_gauges_from_registry() {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};

        let ctx = Arc::new(AppContext::stub());
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("p0".into()),
                url: "http://p0:30000".into(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: None,
            })
            .unwrap();
        let app = crate::server::app::build_router(ctx.clone());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&body).unwrap();
        // Pool size reflects the registered prefill worker, and the per-worker
        // gauges are sampled (fresh breaker => healthy, closed, 0 inflight).
        assert!(
            body.contains(r#"sgl_router_workers{mode="prefill"} 1"#),
            "got:\n{body}"
        );
        assert!(body.contains(r#"sgl_router_worker_health{worker_url="http://p0:30000"} 1"#));
        assert!(body.contains(r#"sgl_router_worker_cb_state{worker_url="http://p0:30000"} 0"#));
        assert!(
            body.contains(r#"sgl_router_worker_inflight_requests{worker_url="http://p0:30000"} 0"#)
        );
    }
}

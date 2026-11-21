// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `GET /internal/member_load`: this router's member load, per worker role,
//! from the engine-reported load it already holds.
//!
//! A member is one router and its pool of engines. A sidecar beside the
//! router polls this on loopback so an inference gateway can weigh members
//! against each other. The endpoint only reads: it takes one pass over the
//! worker registry and the [`EngineLoadTable`] per request, starts no
//! polling, and changes nothing about routing. The response shape is the
//! contract in `docs/member-load.md`; change the two together.
//!
//! Load source on this tree: the [`EngineLoadTable`] fed by the engines'
//! `LoadStat` load socket (push) and, with `--load-monitor`, by
//! `GET /v1/loads` (pull). It carries running / waiting requests and KV
//! used / capacity tokens per DP rank and, from engines new enough to send
//! the [`NativeLoad`](crate::policies::engine_load::NativeLoad) tail, uncached
//! tokens waiting, `max_running_requests` and the cumulative prefill counters
//! the prefill-queue estimate is built from. Without that tail those fields
//! are `null`.

use crate::discovery::WorkerMode;
use crate::policies::engine_load::{EngineLoadTable, Freshness, WorkerNativeLoad};
use crate::server::app_context::AppContext;
use crate::workers::Worker;
use axum::extract::State;
use axum::Json;
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const CONTRACT: &str = "member-load/v1";

/// Counts are engine-supplied `u64`s; the consumer decodes them as signed
/// 64-bit, so a sum is capped there rather than refused whole downstream.
const MAX_WIRE_COUNT: u64 = i64::MAX as u64;

#[derive(Debug, Serialize)]
pub struct MemberLoadReport {
    pub contract: &'static str,
    pub observed_at_ms: u64,
    pub router: RouterInfo,
    /// `plain` / `prefill` / `decode`, each present iff the registry holds at
    /// least one registered worker (one in a model pool) of that role.
    pub roles: BTreeMap<&'static str, RoleLoad>,
}

#[derive(Debug, Serialize)]
pub struct RouterInfo {
    pub instance: &'static str,
    pub version: &'static str,
}

/// One role's aggregate. Load is summed over the role's in-service workers
/// with a fresh engine-reported load (`workers_reporting`); the rest of the
/// role contributes nothing, so read the sums against that count. In service
/// means the worker has the shape routing needs: a prefill must know its
/// bootstrap port. The circuit breaker is deliberately not consulted for
/// load: it opens on the workers failing under load, and dropping their load
/// would make a saturated member look idle.
#[derive(Debug, Default, Clone, PartialEq, Serialize)]
pub struct RoleLoad {
    /// Registered workers (in a model pool) of this role.
    pub workers_total: usize,
    /// Of those, the ones the router would send a request now: breaker
    /// allows, a prefill knows its bootstrap port, and with `--load-monitor`
    /// the load report is fresh (the chat handler's freshness gate).
    pub workers_ready: usize,
    /// Of the in-service ones, those whose every reported DP rank is fresh.
    /// It can exceed `workers_ready` while breakers are open.
    pub workers_reporting: usize,
    pub running_reqs: u64,
    pub waiting_reqs: u64,
    pub kv_used_tokens: u64,
    pub kv_capacity_tokens: u64,
    /// Mean and max of per-worker `kv_used / kv_capacity`, over reporting
    /// workers with a known (non-zero) capacity; `null` when there are none.
    pub kv_usage_mean: Option<f64>,
    pub kv_usage_max: Option<f64>,
    /// Uncached prompt tokens waiting for prefill, summed over reporting
    /// workers that send the extended load; `null` when none does.
    pub waiting_uncached_tokens: Option<u64>,
    /// Min / max over those workers of the expected wait before a request
    /// sent now starts prefill: its uncached tokens waiting ÷ its measured
    /// prefill throughput (see [`WorkerNativeLoad::est_prefill_queue_ms`]).
    /// A worker with tokens waiting but no warm throughput yet gives no
    /// estimate; `null` when no worker gives one.
    pub est_prefill_queue_ms_min: Option<f64>,
    pub est_prefill_queue_ms_max: Option<f64>,
    /// Sum of the engines' `max_running_requests` (over DP ranks, then over
    /// reporting workers that send a non-zero limit); `null` when none does.
    pub max_running_reqs: Option<u64>,
    /// Max over those workers of `running / max_running_requests` (each
    /// summed over the worker's DP ranks); `null` when none sends a limit.
    /// Can pass 1 by the engine's own accounting.
    pub running_frac_max: Option<f64>,
}

/// Wire label of a worker role, as the consumer expects it.
pub fn role_label(mode: WorkerMode) -> &'static str {
    match mode {
        WorkerMode::Plain => "plain",
        WorkerMode::Prefill => "prefill",
        WorkerMode::Decode => "decode",
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// The one engine load store: the KV-event index's table (always attached in
/// production), which the Load Monitor also feeds when enabled.
fn load_table(ctx: &AppContext) -> Option<Arc<EngineLoadTable>> {
    ctx.kv_index()
        .map(|idx| idx.engine_load())
        .or_else(|| ctx.load_monitor.as_ref().map(|lm| Arc::clone(lm.table())))
}

pub fn build_report(ctx: &AppContext) -> MemberLoadReport {
    let workers = ctx.registry.all();
    let table = load_table(ctx);
    MemberLoadReport {
        contract: CONTRACT,
        observed_at_ms: now_ms(),
        router: RouterInfo {
            instance: crate::server::app::pod_id(),
            version: crate::VERSION,
        },
        roles: aggregate_roles(
            &workers,
            table.as_deref(),
            ctx.load_monitor.is_some(),
            Instant::now(),
        ),
    }
}

/// Per-role aggregate of `workers` against the load in `table` as of `now`.
/// `freshness_gate` mirrors the chat handler: with the Load Monitor enabled a
/// worker without a fresh report is not routable. Load rows for URLs not in
/// `workers` are ignored.
pub fn aggregate_roles(
    workers: &[Arc<Worker>],
    table: Option<&EngineLoadTable>,
    freshness_gate: bool,
    now: Instant,
) -> BTreeMap<&'static str, RoleLoad> {
    #[derive(Default)]
    struct Acc {
        load: RoleLoad,
        kv_usage_sum: f64,
        kv_usage_n: usize,
    }
    let mut acc: BTreeMap<&'static str, Acc> = BTreeMap::new();
    // A worker without a model is in no pool, so nothing can route to it.
    for w in workers.iter().filter(|w| !w.model_ids.is_empty()) {
        let a = acc.entry(role_label(w.mode())).or_default();
        let r = &mut a.load;
        r.workers_total += 1;
        // Same rule as `PdResolver::resolve`: a prefill without a bootstrap
        // port stays registered but is excluded from routing.
        let in_service = w.mode() != WorkerMode::Prefill || w.bootstrap_port().is_some();
        // `Freshness::Fresh` already folds in the pull status: an unreachable
        // or zero-capacity pull reads as not fresh.
        let fresh = table.is_some_and(|t| t.worker_freshness(&w.url, now) == Freshness::Fresh);
        if in_service && w.breaker.would_allow() && (!freshness_gate || fresh) {
            r.workers_ready += 1;
        }
        if !in_service || !fresh {
            continue;
        }
        // Rows are last-write-wins gauges, so a rank written between the
        // freshness check and this read only makes the sum newer.
        let Some((load, _)) = table.and_then(|t| t.worker_aggregate(&w.url)) else {
            continue;
        };
        r.workers_reporting += 1;
        r.running_reqs = r.running_reqs.saturating_add(load.num_running_requests);
        r.waiting_reqs = r.waiting_reqs.saturating_add(load.num_waiting_requests);
        r.kv_used_tokens = r.kv_used_tokens.saturating_add(load.num_used_tokens);
        r.kv_capacity_tokens = r.kv_capacity_tokens.saturating_add(load.max_total_tokens);
        if load.max_total_tokens > 0 {
            let usage = load.num_used_tokens as f64 / load.max_total_tokens as f64;
            a.kv_usage_sum += usage;
            a.kv_usage_n += 1;
            r.kv_usage_max = Some(r.kv_usage_max.map_or(usage, |m| m.max(usage)));
        }
        if let Some(native) = table.and_then(|t| t.worker_native(&w.url)) {
            add_native(r, &native);
        }
    }
    acc.into_iter()
        .map(|(role, mut a)| {
            if a.kv_usage_n > 0 {
                a.load.kv_usage_mean = Some(a.kv_usage_sum / a.kv_usage_n as f64);
            }
            let r = &mut a.load;
            for count in [
                &mut r.running_reqs,
                &mut r.waiting_reqs,
                &mut r.kv_used_tokens,
                &mut r.kv_capacity_tokens,
            ] {
                *count = (*count).min(MAX_WIRE_COUNT);
            }
            for count in [&mut r.waiting_uncached_tokens, &mut r.max_running_reqs]
                .into_iter()
                .flatten()
            {
                *count = (*count).min(MAX_WIRE_COUNT);
            }
            (role, a.load)
        })
        .collect()
}

/// Fold one reporting worker's extended load into its role.
fn add_native(r: &mut RoleLoad, n: &WorkerNativeLoad) {
    r.waiting_uncached_tokens = Some(
        r.waiting_uncached_tokens
            .unwrap_or(0)
            .saturating_add(n.num_waiting_uncached_tokens),
    );
    if let Some(ms) = n.est_prefill_queue_ms() {
        r.est_prefill_queue_ms_min = Some(r.est_prefill_queue_ms_min.map_or(ms, |m| m.min(ms)));
        r.est_prefill_queue_ms_max = Some(r.est_prefill_queue_ms_max.map_or(ms, |m| m.max(ms)));
    }
    if let Some(frac) = n.running_frac() {
        r.max_running_reqs = Some(
            r.max_running_reqs
                .unwrap_or(0)
                .saturating_add(n.max_running_requests),
        );
        r.running_frac_max = Some(r.running_frac_max.map_or(frac, |m| m.max(frac)));
    }
}

pub async fn member_load(State(ctx): State<Arc<AppContext>>) -> Json<MemberLoadReport> {
    Json(build_report(&ctx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerSpec};
    use crate::health::circuit_breaker::CircuitBreakerConfig;
    use crate::policies::engine_load::{LoadStat, NativeLoad, PullStatus};
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use std::num::NonZeroU32;
    use std::time::Duration;
    use tower::ServiceExt;

    const ROLE_FIELDS: [&str; 14] = [
        "workers_total",
        "workers_ready",
        "workers_reporting",
        "running_reqs",
        "waiting_reqs",
        "kv_used_tokens",
        "kv_capacity_tokens",
        "kv_usage_mean",
        "kv_usage_max",
        "waiting_uncached_tokens",
        "est_prefill_queue_ms_min",
        "est_prefill_queue_ms_max",
        "max_running_reqs",
        "running_frac_max",
    ];

    fn url(id: &str) -> String {
        format!("http://{id}:1")
    }

    fn spec(id: &str, mode: WorkerMode, with_model: bool, port: Option<u16>) -> WorkerSpec {
        WorkerSpec {
            id: WorkerId(id.into()),
            url: url(id),
            mode,
            model_ids: if with_model {
                vec![ModelId("stub-model".into())]
            } else {
                Vec::new()
            },
            bootstrap_port: port,
            transfer_group: None,
        }
    }

    fn add(ctx: &AppContext, id: &str, mode: WorkerMode, with_model: bool) {
        let port = (mode == WorkerMode::Prefill).then_some(8998);
        ctx.registry.add(spec(id, mode, with_model, port)).unwrap();
    }

    fn stat(running: u64, waiting: u64, used: u64, cap: u64) -> LoadStat {
        LoadStat {
            num_running_reqs: running,
            num_waiting_reqs: waiting,
            num_tokens: used,
            max_total_num_tokens: cap,
            native: None,
        }
    }

    /// Asserts the document passes the consumer's `ParseMemberLoad` checks
    /// (radixark-service `pkg/inferencegw/member_load.go`) and carries every
    /// contract field, nulls included.
    fn assert_contract(doc: &serde_json::Value) {
        assert_eq!(doc["contract"], CONTRACT);
        assert!(doc["observed_at_ms"].as_u64().unwrap() > 0);
        assert!(doc["observed_at_ms"].as_i64().is_some());
        for (name, role) in doc["roles"].as_object().unwrap() {
            assert!(["plain", "prefill", "decode"].contains(&name.as_str()));
            let role = role.as_object().unwrap();
            let mut keys: Vec<_> = role.keys().map(String::as_str).collect();
            keys.sort_unstable();
            let mut want = ROLE_FIELDS.to_vec();
            want.sort_unstable();
            assert_eq!(keys, want, "{name}");
            for f in &ROLE_FIELDS[..7] {
                assert!(role[*f].as_i64().is_some_and(|v| v >= 0), "{name}.{f}");
            }
            let n = |f: &str| role[f].as_i64().unwrap();
            assert!(n("workers_ready") <= n("workers_total"), "{name}");
            assert!(n("workers_reporting") <= n("workers_total"), "{name}");
            for f in ["kv_usage_mean", "kv_usage_max"] {
                let v = &role[f];
                assert!(
                    v.is_null() || v.as_f64().is_some_and(|x| (0.0..=2.0).contains(&x)),
                    "{name}.{f}"
                );
            }
            for f in ["waiting_uncached_tokens", "max_running_reqs"] {
                let v = &role[f];
                assert!(
                    v.is_null() || v.as_i64().is_some_and(|x| x >= 0),
                    "{name}.{f}"
                );
            }
            for f in [
                "est_prefill_queue_ms_min",
                "est_prefill_queue_ms_max",
                "running_frac_max",
            ] {
                let v = &role[f];
                assert!(
                    v.is_null() || v.as_f64().is_some_and(|x| x >= 0.0),
                    "{name}.{f}"
                );
            }
            let (lo, hi) = (
                &role["est_prefill_queue_ms_min"],
                &role["est_prefill_queue_ms_max"],
            );
            assert_eq!(lo.is_null(), hi.is_null(), "{name}");
            if !lo.is_null() {
                assert!(lo.as_f64() <= hi.as_f64(), "{name}");
            }
            assert_eq!(
                role["max_running_reqs"].is_null(),
                role["running_frac_max"].is_null(),
                "{name}"
            );
        }
    }

    #[test]
    fn member_load_no_workers_no_roles() {
        let ctx = AppContext::stub();
        let report = build_report(&ctx);
        assert_eq!(report.contract, CONTRACT);
        assert!(report.roles.is_empty());
        let doc = serde_json::to_value(&report).unwrap();
        assert_eq!(doc["roles"], serde_json::json!({}));
        assert_eq!(doc["router"]["instance"], crate::server::app::pod_id());
        assert_eq!(doc["router"]["version"], crate::VERSION);
        assert_contract(&doc);
    }

    /// Total, ready and reporting each count something different; sums and
    /// KV usage cover only in-service workers whose every rank is fresh, and
    /// the fields this tree cannot know are null.
    #[test]
    fn member_load_mixed_pd_roles() {
        let ctx = AppContext::stub();
        add(&ctx, "p1", WorkerMode::Prefill, true);
        // No bootstrap port yet: registered, not ready, load left out.
        ctx.registry
            .add(spec("p2", WorkerMode::Prefill, true, None))
            .unwrap();
        // Registered without a model: in no pool, so not counted.
        add(&ctx, "p3", WorkerMode::Prefill, false);
        for id in ["d1", "d2", "d3", "d4", "d5"] {
            add(&ctx, id, WorkerMode::Decode, true);
        }

        let table = EngineLoadTable::with_freshness(Duration::from_secs(5));
        let t0 = Instant::now();
        let now = t0 + Duration::from_secs(10);
        // p1 has two DP ranks: summed.
        table.set(&url("p1"), 0, stat(3, 1, 150, 500), now);
        table.set(&url("p1"), 1, stat(1, 0, 50, 500), now);
        table.set(&url("p2"), 0, stat(7, 7, 700, 1000), now);
        table.set(&url("p3"), 0, stat(9, 9, 9, 9), now);
        table.set(&url("d1"), 0, stat(10, 2, 300, 1000), now);
        table.set(&url("d2"), 0, stat(20, 3, 900, 1000), now);
        // One stale rank: the whole worker is left out of every stat.
        table.set(&url("d3"), 0, stat(50, 50, 990, 1000), now);
        table.set(&url("d3"), 1, stat(50, 50, 990, 1000), t0);
        // Capacity unknown: in the sums, not in the usage stats.
        table.set(&url("d4"), 0, stat(5, 0, 100, 0), now);
        // Latest pull failed: not reporting, previous rows ignored.
        table.set(&url("d5"), 0, stat(40, 40, 400, 1000), now);
        table.set_pull_status(&url("d5"), Some(PullStatus::Unreachable("x".into())));
        // Not in the registry: ignored.
        table.set(&url("ghost"), 0, stat(1000, 1000, 1000, 1000), now);

        let roles = aggregate_roles(&ctx.registry.all(), Some(&table), false, now);
        assert_eq!(
            roles.keys().copied().collect::<Vec<_>>(),
            ["decode", "prefill"]
        );

        let p = &roles["prefill"];
        assert_eq!(
            (p.workers_total, p.workers_ready, p.workers_reporting),
            (2, 1, 1)
        );
        assert_eq!(
            (
                p.running_reqs,
                p.waiting_reqs,
                p.kv_used_tokens,
                p.kv_capacity_tokens
            ),
            (4, 1, 200, 1000)
        );
        assert_eq!((p.kv_usage_mean, p.kv_usage_max), (Some(0.2), Some(0.2)));

        let d = &roles["decode"];
        // Without the Load Monitor gate, stale / unreachable workers are
        // still routable (ready); they just do not report.
        assert_eq!(
            (d.workers_total, d.workers_ready, d.workers_reporting),
            (5, 5, 3)
        );
        assert_eq!(
            (
                d.running_reqs,
                d.waiting_reqs,
                d.kv_used_tokens,
                d.kv_capacity_tokens
            ),
            (35, 5, 1300, 2000)
        );
        assert!((d.kv_usage_mean.unwrap() - 0.6).abs() < 1e-9);
        assert_eq!(d.kv_usage_max, Some(0.9));
        // Engines without the extended load: those fields stay null.
        for r in [p, d] {
            assert_eq!(r.waiting_uncached_tokens, None);
            assert_eq!(
                (r.est_prefill_queue_ms_min, r.est_prefill_queue_ms_max),
                (None, None)
            );
            assert_eq!((r.max_running_reqs, r.running_frac_max), (None, None));
        }

        let doc = serde_json::json!({
            "contract": CONTRACT,
            "observed_at_ms": 1,
            "roles": serde_json::to_value(&roles).unwrap(),
        });
        assert_contract(&doc);

        // With the Load Monitor on, only fresh workers are routable.
        let gated = aggregate_roles(&ctx.registry.all(), Some(&table), true, now);
        assert_eq!(gated["decode"].workers_ready, 3);
        assert_eq!(gated["prefill"].workers_ready, 1);
        assert_eq!(gated["decode"].workers_reporting, 3);

        // No load store at all: counts only, sums zero, usage null.
        let bare = aggregate_roles(&ctx.registry.all(), None, false, now);
        let doc = serde_json::to_value(&bare["decode"]).unwrap();
        assert_eq!(doc["workers_total"], 5);
        assert_eq!(doc["workers_ready"], 5);
        assert_eq!(doc["workers_reporting"], 0);
        assert_eq!(doc["running_reqs"], 0);
        assert!(doc["kv_usage_mean"].is_null() && doc["kv_usage_max"].is_null());
    }

    fn native_stat(
        running: u64,
        waiting_uncached: u64,
        max_running: u64,
        tok: u64,
        busy_us: u64,
    ) -> LoadStat {
        LoadStat {
            native: Some(NativeLoad {
                num_waiting_uncached_tokens: waiting_uncached,
                num_total_tokens: 0,
                max_running_requests: max_running,
                total_prefill_uncached_tokens: tok,
                total_prefill_busy_us: busy_us,
            }),
            ..stat(running, 0, 0, 1000)
        }
    }

    /// The prefill queue estimate and the decode running fraction, from the
    /// engines' extended load: warm, cold, idle, old-engine and stale workers.
    #[test]
    fn member_load_native_signals() {
        let ctx = AppContext::stub();
        for id in ["p1", "p2", "p3", "p4"] {
            add(&ctx, id, WorkerMode::Prefill, true);
        }
        for id in ["d1", "d2", "d3", "d4", "d5"] {
            add(&ctx, id, WorkerMode::Decode, true);
        }
        let table = EngineLoadTable::with_freshness(Duration::from_secs(5));
        let t0 = Instant::now();
        let now = t0 + Duration::from_secs(10);
        // p1: two warm ranks at 10k + 20k tok/s, 6k tokens waiting -> 200 ms.
        table.set(
            &url("p1"),
            0,
            native_stat(1, 2_000, 64, 36_000_000, 3_600_000_000),
            now,
        );
        table.set(
            &url("p1"),
            1,
            native_stat(1, 4_000, 64, 72_000_000, 3_600_000_000),
            now,
        );
        // p2: warm at 10k tok/s, 30k waiting -> 3000 ms.
        table.set(
            &url("p2"),
            0,
            native_stat(1, 30_000, 64, 36_000_000, 3_600_000_000),
            now,
        );
        // p3: cold (never prefilled), nothing waiting -> 0 ms.
        table.set(&url("p3"), 0, native_stat(0, 0, 64, 0, 0), now);
        // p4: cold with 5k waiting: counted in the tokens, no estimate.
        table.set(&url("p4"), 0, native_stat(0, 5_000, 64, 100, 1_000), now);
        // d1: 2 ranks, 96 / 128 running. d2: 16 / 64.
        table.set(&url("d1"), 0, native_stat(40, 0, 64, 0, 0), now);
        table.set(&url("d1"), 1, native_stat(56, 0, 64, 0, 0), now);
        table.set(&url("d2"), 0, native_stat(16, 0, 64, 0, 0), now);
        // d3: older engine, no tail: in the counts, not the fraction.
        table.set(&url("d3"), 0, stat(60, 0, 0, 1000), now);
        // d4: limit unknown (0): no fraction, no limit.
        table.set(&url("d4"), 0, native_stat(60, 0, 0, 0, 0), now);
        // d5: stale: left out of everything.
        table.set(&url("d5"), 0, native_stat(64, 0, 64, 0, 0), t0);

        let roles = aggregate_roles(&ctx.registry.all(), Some(&table), false, now);
        let p = &roles["prefill"];
        assert_eq!(p.workers_reporting, 4);
        assert_eq!(p.waiting_uncached_tokens, Some(41_000));
        let (lo, hi) = (
            p.est_prefill_queue_ms_min.unwrap(),
            p.est_prefill_queue_ms_max.unwrap(),
        );
        assert_eq!(lo, 0.0, "the idle cold worker");
        assert!((hi - 3_000.0).abs() < 1e-6, "{hi}");
        assert_eq!(p.max_running_reqs, Some(64 * 5));

        let d = &roles["decode"];
        assert_eq!((d.workers_total, d.workers_reporting), (5, 4));
        assert_eq!(d.running_reqs, 40 + 56 + 16 + 60 + 60);
        assert_eq!(d.max_running_reqs, Some(128 + 64));
        assert_eq!(d.running_frac_max, Some(0.75));
        assert_eq!(d.waiting_uncached_tokens, Some(0));
        // Decode never prefills: nothing waiting reads 0 ms.
        assert_eq!(d.est_prefill_queue_ms_max, Some(0.0));

        let doc = serde_json::json!({
            "contract": CONTRACT,
            "observed_at_ms": 1,
            "roles": serde_json::to_value(&roles).unwrap(),
        });
        assert_contract(&doc);
    }

    /// One prefill warm with a long queue and the only other one cold with
    /// tokens waiting: the estimate is the warm one alone, never 0.
    #[test]
    fn member_load_cold_worker_does_not_read_idle() {
        let ctx = AppContext::stub();
        add(&ctx, "p1", WorkerMode::Prefill, true);
        let table = EngineLoadTable::new();
        let now = Instant::now();
        table.set(&url("p1"), 0, native_stat(0, 5_000, 64, 0, 0), now);
        let p = &aggregate_roles(&ctx.registry.all(), Some(&table), false, now)["prefill"];
        assert_eq!(p.waiting_uncached_tokens, Some(5_000));
        assert_eq!(
            (p.est_prefill_queue_ms_min, p.est_prefill_queue_ms_max),
            (None, None)
        );
        // Warm-up: 1 s of prefill at 10k tok/s, then the estimate appears.
        table.set(
            &url("p1"),
            0,
            native_stat(0, 5_000, 64, 10_000, 1_000_000),
            now,
        );
        let p = &aggregate_roles(&ctx.registry.all(), Some(&table), false, now)["prefill"];
        let ms = p.est_prefill_queue_ms_max.unwrap();
        assert!((ms - 500.0).abs() < 1e-6, "{ms}");
    }

    /// A worker whose breaker is open is not ready, but its load still
    /// counts: the breaker opens on the workers failing under load.
    #[test]
    fn member_load_breaker_open_keeps_load() {
        let ctx = AppContext::stub();
        ctx.registry
            .add_with_cb(
                spec("d1", WorkerMode::Decode, true, None),
                Some(CircuitBreakerConfig {
                    threshold: NonZeroU32::new(1).unwrap(),
                    cool_down: Duration::from_secs(30),
                }),
            )
            .unwrap();
        add(&ctx, "d2", WorkerMode::Decode, true);
        let d1 = ctx.registry.get(&WorkerId("d1".into())).unwrap();
        d1.breaker.record_failure();
        assert!(!d1.breaker.would_allow(), "breaker should be open");

        let table = EngineLoadTable::new();
        let now = Instant::now();
        table.set(&url("d1"), 0, stat(60, 40, 980, 1000), now);
        table.set(&url("d2"), 0, stat(10, 0, 200, 1000), now);
        let d = &aggregate_roles(&ctx.registry.all(), Some(&table), false, now)["decode"];
        assert_eq!(
            (d.workers_total, d.workers_ready, d.workers_reporting),
            (2, 1, 2)
        );
        assert_eq!((d.running_reqs, d.waiting_reqs), (70, 40));
        assert_eq!(d.kv_usage_max, Some(0.98));
    }

    /// Engine-supplied counts beyond the consumer's int64 are capped, not
    /// emitted as a value the consumer would refuse.
    #[test]
    fn member_load_counts_fit_the_wire() {
        let ctx = AppContext::stub();
        add(&ctx, "w1", WorkerMode::Plain, true);
        add(&ctx, "w2", WorkerMode::Plain, true);
        let table = EngineLoadTable::new();
        let now = Instant::now();
        table.set(&url("w1"), 0, stat(u64::MAX, 1, 1, 2), now);
        table.set(&url("w2"), 0, stat(u64::MAX, 1, 1, 2), now);
        let p = &aggregate_roles(&ctx.registry.all(), Some(&table), false, now)["plain"];
        assert_eq!(p.running_reqs, i64::MAX as u64);
        assert_eq!(p.waiting_reqs, 2);
    }

    /// The route serves the contract from the live load table (the KV-event
    /// index's, as in production).
    #[tokio::test]
    async fn member_load_route_serves_the_contract() {
        let ctx = AppContext::stub();
        add(&ctx, "w1", WorkerMode::Plain, true);
        add(&ctx, "w2", WorkerMode::Plain, true);
        let index = crate::policies::kv_events::KvEventIndex::new();
        index
            .engine_load()
            .set(&url("w1"), 0, stat(3, 1, 40, 100), Instant::now());
        ctx.attach_kv_index(index);
        let res = crate::server::app::build_router(Arc::new(ctx))
            .oneshot(
                Request::builder()
                    .uri("/internal/member_load")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let doc: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_contract(&doc);
        assert_eq!(doc["router"]["version"], crate::VERSION);
        assert_eq!(doc["roles"].as_object().unwrap().len(), 1);
        let plain = &doc["roles"]["plain"];
        assert_eq!(plain["workers_total"], 2);
        assert_eq!(plain["workers_ready"], 2);
        assert_eq!(plain["workers_reporting"], 1);
        assert_eq!(plain["running_reqs"], 3);
        assert_eq!(plain["waiting_reqs"], 1);
        assert_eq!(plain["kv_used_tokens"], 40);
        assert_eq!(plain["kv_capacity_tokens"], 100);
        assert_eq!(plain["kv_usage_mean"], 0.4);
        assert_eq!(plain["kv_usage_max"], 0.4);
        assert!(plain["waiting_uncached_tokens"].is_null());
        assert!(plain["max_running_reqs"].is_null());
        assert!(plain["running_frac_max"].is_null());
    }

    /// The route serves the extended fields when the engine sends them.
    #[tokio::test]
    async fn member_load_route_serves_native_fields() {
        let ctx = AppContext::stub();
        add(&ctx, "w1", WorkerMode::Plain, true);
        let index = crate::policies::kv_events::KvEventIndex::new();
        index.engine_load().set(
            &url("w1"),
            0,
            native_stat(32, 20_000, 64, 36_000_000, 3_600_000_000),
            Instant::now(),
        );
        ctx.attach_kv_index(index);
        let res = crate::server::app::build_router(Arc::new(ctx))
            .oneshot(
                Request::builder()
                    .uri("/internal/member_load")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let doc: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_contract(&doc);
        let plain = &doc["roles"]["plain"];
        assert_eq!(plain["waiting_uncached_tokens"], 20_000);
        assert_eq!(plain["est_prefill_queue_ms_min"], 2_000.0);
        assert_eq!(plain["est_prefill_queue_ms_max"], 2_000.0);
        assert_eq!(plain["max_running_reqs"], 64);
        assert_eq!(plain["running_frac_max"], 0.5);
    }

    #[test]
    fn member_load_is_an_infra_path() {
        assert!(crate::server::app::is_infra_path("/internal/member_load"));
    }
}

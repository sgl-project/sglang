// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Lightweight in-process Prometheus exposition.
//!
//! We deliberately do NOT pull in the `metrics` + `metrics-exporter-prometheus`
//! crates: the observability surface is small enough that a hand-written
//! counter + histogram + gauge family is cheaper than a new dependency, and
//! it lets us label/serialise exactly the way the convergence and PD-affinity
//! tests want.
//!
//! All operations are concurrent — counters and gauges use
//! [`std::sync::atomic`], histograms use a [`Mutex<Vec<u64>>`] over a
//! fixed bucket set. Tests sub-second; production scrapes are 15s
//! cadence. Lock contention is not a concern at these rates.
//!
//! # Metrics surface
//!
//! | Metric | Type | Labels |
//! |---|---|---|
//! | `sgl_router_requests_total` | Counter | `worker_url`, `model_id`, `mode`, `outcome` |
//! | `sgl_router_overlap_blocks` | Histogram | `model_id` |
//! | `sgl_router_active_load` | Gauge | `worker_url`, `kind` |
//! | `sgl_router_stale_requests_total` | Counter | `outcome` |
//! | `sgl_router_decode_affinity_total` | Counter | `outcome` |
//!
//! The exposition is text/plain; version=0.0.4 per the Prometheus spec.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;

/// Histogram bucket upper bounds for `sgl_router_overlap_blocks`. Chosen to
/// span 0 → ~1k blocks: blocks are 32–64 tokens each, and our `MAX_CHAT_BODY_BYTES`
/// cap (1 MiB ≈ 250 k tokens) implies an upper bound around 4–8 k blocks for
/// a maximum-length context. The `+Inf` bucket catches everything beyond
/// 1000.
const OVERLAP_BLOCKS_BUCKETS: &[f64] = &[
    0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1000.0,
];

/// Recordable outcome for a request — narrowed to a handful of variants so
/// the label cardinality stays bounded.
#[derive(Debug, Clone, Copy)]
pub enum RequestOutcome {
    Success,
    Error,
    Cancelled,
}

impl RequestOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Error => "error",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Worker dispatch mode label — narrowed to the three modes the policy
/// resolver distinguishes. The `Plain` variant covers the non-PD case.
#[derive(Debug, Clone, Copy)]
pub enum WorkerModeLabel {
    Prefill,
    Decode,
    Plain,
}

impl WorkerModeLabel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Plain => "plain",
        }
    }
}

/// Decode-affinity outcome — see `select_decode_with_affinity` for the
/// three reasons the affinity may not be honored.
#[derive(Debug, Clone, Copy)]
pub enum DecodeAffinityOutcome {
    SameHostPicked,
    FallbackBreaker,
    FallbackLoadImbalance,
}

impl DecodeAffinityOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::SameHostPicked => "same_host_picked",
            Self::FallbackBreaker => "fallback_breaker",
            Self::FallbackLoadImbalance => "fallback_load_imbalance",
        }
    }
}

/// Stale-request outcome label.
#[derive(Debug, Clone, Copy)]
pub enum StaleRequestOutcome {
    Expired,
}

impl StaleRequestOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Expired => "expired",
        }
    }
}

/// Active-load kind label — separates the two axes of per-worker load.
#[derive(Debug, Clone, Copy)]
pub enum ActiveLoadKind {
    PrefillTokens,
    DecodeBlocks,
}

impl ActiveLoadKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::PrefillTokens => "prefill_tokens",
            Self::DecodeBlocks => "decode_blocks",
        }
    }
}

/// The shared metrics registry, held on `AppContext`. Cheap to clone — all
/// internal state is `Arc`/`Atomic`/`Mutex`-protected.
#[derive(Debug, Default)]
pub struct MetricsRegistry {
    requests_total: Mutex<HashMap<RequestKey, Arc<AtomicU64>>>,
    overlap_blocks: Mutex<HashMap<String, Histogram>>,
    active_load: Mutex<HashMap<ActiveLoadKey, Arc<AtomicI64>>>,
    stale_requests_total: Mutex<HashMap<&'static str, Arc<AtomicU64>>>,
    decode_affinity_total: Mutex<HashMap<&'static str, Arc<AtomicU64>>>,
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct RequestKey {
    worker_url: String,
    model_id: String,
    mode: &'static str,
    outcome: &'static str,
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct ActiveLoadKey {
    worker_url: String,
    kind: &'static str,
}

#[derive(Debug)]
struct Histogram {
    /// One counter per bucket boundary in [`OVERLAP_BLOCKS_BUCKETS`], plus
    /// one for `+Inf`. Buckets are cumulative on render but stored as
    /// non-cumulative counts here.
    buckets: Vec<u64>,
    sum: f64,
    count: u64,
}

impl Histogram {
    fn new() -> Self {
        Self {
            buckets: vec![0; OVERLAP_BLOCKS_BUCKETS.len() + 1],
            sum: 0.0,
            count: 0,
        }
    }

    fn observe(&mut self, value: f64) {
        let mut placed = false;
        for (i, &bound) in OVERLAP_BLOCKS_BUCKETS.iter().enumerate() {
            if value <= bound {
                self.buckets[i] += 1;
                placed = true;
                break;
            }
        }
        if !placed {
            // +Inf bucket
            let last = self.buckets.len() - 1;
            self.buckets[last] += 1;
        }
        self.sum += value;
        self.count += 1;
    }
}

impl MetricsRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Bump `sgl_router_requests_total` for the given worker / model / mode / outcome.
    pub fn record_request(
        &self,
        worker_url: &str,
        model_id: &str,
        mode: WorkerModeLabel,
        outcome: RequestOutcome,
    ) {
        let key = RequestKey {
            worker_url: worker_url.to_owned(),
            model_id: model_id.to_owned(),
            mode: mode.as_str(),
            outcome: outcome.as_str(),
        };
        let mut guard = self.requests_total.lock();
        let counter = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Observe an overlap-blocks count for `sgl_router_overlap_blocks`.
    pub fn observe_overlap_blocks(&self, model_id: &str, blocks: u64) {
        let mut guard = self.overlap_blocks.lock();
        let hist = guard
            .entry(model_id.to_owned())
            .or_insert_with(Histogram::new);
        hist.observe(blocks as f64);
    }

    /// Set `sgl_router_active_load` for the given worker + kind. Replaces the
    /// previous value (gauge semantics).
    pub fn set_active_load(&self, worker_url: &str, kind: ActiveLoadKind, value: i64) {
        let key = ActiveLoadKey {
            worker_url: worker_url.to_owned(),
            kind: kind.as_str(),
        };
        let mut guard = self.active_load.lock();
        let gauge = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicI64::new(0)))
            .clone();
        drop(guard);
        gauge.store(value, Ordering::Relaxed);
    }

    /// Bump `sgl_router_stale_requests_total{outcome}`.
    pub fn record_stale_request(&self, outcome: StaleRequestOutcome) {
        let mut guard = self.stale_requests_total.lock();
        let counter = guard
            .entry(outcome.as_str())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump `sgl_router_decode_affinity_total{outcome}`.
    pub fn record_decode_affinity(&self, outcome: DecodeAffinityOutcome) {
        let mut guard = self.decode_affinity_total.lock();
        let counter = guard
            .entry(outcome.as_str())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Render the registry as a Prometheus 0.0.4 exposition-format string.
    pub fn render(&self) -> String {
        let mut out = String::new();

        // requests_total
        out.push_str(
            "# HELP sgl_router_requests_total Total chat-completions requests dispatched to a worker.\n",
        );
        out.push_str("# TYPE sgl_router_requests_total counter\n");
        let guard = self.requests_total.lock();
        // Sort for stable output — easier for tests.
        let mut entries: Vec<(&RequestKey, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| {
            (&a.0.worker_url, &a.0.model_id, a.0.mode, a.0.outcome).cmp(&(
                &b.0.worker_url,
                &b.0.model_id,
                b.0.mode,
                b.0.outcome,
            ))
        });
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_requests_total{{worker_url=\"{}\",model_id=\"{}\",mode=\"{}\",outcome=\"{}\"}} {}\n",
                escape_label(&key.worker_url),
                escape_label(&key.model_id),
                key.mode,
                key.outcome,
                value,
            ));
        }
        drop(guard);

        // overlap_blocks histogram
        out.push_str(
            "# HELP sgl_router_overlap_blocks Overlap-block count observed at cache-aware-zmq policy selection.\n",
        );
        out.push_str("# TYPE sgl_router_overlap_blocks histogram\n");
        let guard = self.overlap_blocks.lock();
        let mut models: Vec<&String> = guard.keys().collect();
        models.sort();
        for model_id in models {
            let hist = guard.get(model_id).unwrap();
            let mut cumulative: u64 = 0;
            for (i, &bound) in OVERLAP_BLOCKS_BUCKETS.iter().enumerate() {
                cumulative += hist.buckets[i];
                out.push_str(&format!(
                    "sgl_router_overlap_blocks_bucket{{model_id=\"{}\",le=\"{}\"}} {}\n",
                    escape_label(model_id),
                    bound,
                    cumulative,
                ));
            }
            cumulative += hist.buckets[OVERLAP_BLOCKS_BUCKETS.len()];
            out.push_str(&format!(
                "sgl_router_overlap_blocks_bucket{{model_id=\"{}\",le=\"+Inf\"}} {}\n",
                escape_label(model_id),
                cumulative,
            ));
            out.push_str(&format!(
                "sgl_router_overlap_blocks_sum{{model_id=\"{}\"}} {}\n",
                escape_label(model_id),
                hist.sum,
            ));
            out.push_str(&format!(
                "sgl_router_overlap_blocks_count{{model_id=\"{}\"}} {}\n",
                escape_label(model_id),
                hist.count,
            ));
        }
        drop(guard);

        // active_load gauge
        out.push_str(
            "# HELP sgl_router_active_load Per-worker active load (prefill_tokens or decode_blocks).\n",
        );
        out.push_str("# TYPE sgl_router_active_load gauge\n");
        let guard = self.active_load.lock();
        let mut entries: Vec<(&ActiveLoadKey, i64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| (&a.0.worker_url, a.0.kind).cmp(&(&b.0.worker_url, b.0.kind)));
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_active_load{{worker_url=\"{}\",kind=\"{}\"}} {}\n",
                escape_label(&key.worker_url),
                key.kind,
                value,
            ));
        }
        drop(guard);

        // stale_requests_total
        out.push_str(
            "# HELP sgl_router_stale_requests_total Total stale-request cancellations fired by the janitor.\n",
        );
        out.push_str("# TYPE sgl_router_stale_requests_total counter\n");
        let guard = self.stale_requests_total.lock();
        let mut entries: Vec<(&&str, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by_key(|e| *e.0);
        for (outcome, value) in entries {
            out.push_str(&format!(
                "sgl_router_stale_requests_total{{outcome=\"{}\"}} {}\n",
                outcome, value,
            ));
        }
        drop(guard);

        // decode_affinity_total
        out.push_str(
            "# HELP sgl_router_decode_affinity_total Decode-affinity outcomes from select_decode_with_affinity.\n",
        );
        out.push_str("# TYPE sgl_router_decode_affinity_total counter\n");
        let guard = self.decode_affinity_total.lock();
        let mut entries: Vec<(&&str, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by_key(|e| *e.0);
        for (outcome, value) in entries {
            out.push_str(&format!(
                "sgl_router_decode_affinity_total{{outcome=\"{}\"}} {}\n",
                outcome, value,
            ));
        }
        drop(guard);

        out
    }
}

/// Prometheus label-value escape rule per
/// https://prometheus.io/docs/instrumenting/exposition_formats/.
/// We only escape `\`, `"`, and newline — the three characters the
/// reference parser rejects unescaped.
fn escape_label(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str(r"\\"),
            '"' => out.push_str(r#"\""#),
            '\n' => out.push_str(r"\n"),
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_registry_renders_only_help_lines() {
        let reg = MetricsRegistry::new();
        let out = reg.render();
        // Should at least carry HELP / TYPE for every metric family.
        assert!(out.contains("# TYPE sgl_router_requests_total counter"));
        assert!(out.contains("# TYPE sgl_router_overlap_blocks histogram"));
        assert!(out.contains("# TYPE sgl_router_active_load gauge"));
        assert!(out.contains("# TYPE sgl_router_stale_requests_total counter"));
        assert!(out.contains("# TYPE sgl_router_decode_affinity_total counter"));
    }

    #[test]
    fn record_request_emits_labelled_counter_line() {
        let reg = MetricsRegistry::new();
        reg.record_request(
            "http://worker-a:30000",
            "tiny",
            WorkerModeLabel::Prefill,
            RequestOutcome::Success,
        );
        reg.record_request(
            "http://worker-a:30000",
            "tiny",
            WorkerModeLabel::Prefill,
            RequestOutcome::Success,
        );
        let out = reg.render();
        assert!(
            out.contains(r#"sgl_router_requests_total{worker_url="http://worker-a:30000",model_id="tiny",mode="prefill",outcome="success"} 2"#),
            "render did not include the expected counter line; got:\n{out}",
        );
    }

    #[test]
    fn observe_overlap_blocks_writes_buckets_and_count() {
        let reg = MetricsRegistry::new();
        reg.observe_overlap_blocks("tiny", 3);
        reg.observe_overlap_blocks("tiny", 9);
        reg.observe_overlap_blocks("tiny", 50);
        let out = reg.render();
        // 3 observations -> count=3, sum=62
        assert!(out.contains(r#"sgl_router_overlap_blocks_count{model_id="tiny"} 3"#));
        assert!(out.contains(r#"sgl_router_overlap_blocks_sum{model_id="tiny"} 62"#));
        // The le=64 bucket is cumulative: 3 is <=4, 9 is <=16, 50 is <=64.
        assert!(
            out.contains(r#"sgl_router_overlap_blocks_bucket{model_id="tiny",le="64"} 3"#),
            "bucket le=64 should be 3 (cumulative); got:\n{out}",
        );
        // The le=4 bucket should include only the 3.
        assert!(
            out.contains(r#"sgl_router_overlap_blocks_bucket{model_id="tiny",le="4"} 1"#),
            "bucket le=4 should be 1; got:\n{out}",
        );
    }

    #[test]
    fn set_active_load_gauge_overwrites() {
        let reg = MetricsRegistry::new();
        reg.set_active_load("http://w:30000", ActiveLoadKind::PrefillTokens, 100);
        reg.set_active_load("http://w:30000", ActiveLoadKind::PrefillTokens, 250);
        let out = reg.render();
        assert!(out.contains(
            r#"sgl_router_active_load{worker_url="http://w:30000",kind="prefill_tokens"} 250"#,
        ));
        // First write must NOT appear.
        assert!(!out.contains(
            r#"sgl_router_active_load{worker_url="http://w:30000",kind="prefill_tokens"} 100"#,
        ));
    }

    #[test]
    fn stale_request_counter_increments() {
        let reg = MetricsRegistry::new();
        reg.record_stale_request(StaleRequestOutcome::Expired);
        reg.record_stale_request(StaleRequestOutcome::Expired);
        reg.record_stale_request(StaleRequestOutcome::Expired);
        let out = reg.render();
        assert!(out.contains(r#"sgl_router_stale_requests_total{outcome="expired"} 3"#));
    }

    #[test]
    fn decode_affinity_counter_emits_three_outcomes() {
        let reg = MetricsRegistry::new();
        reg.record_decode_affinity(DecodeAffinityOutcome::SameHostPicked);
        reg.record_decode_affinity(DecodeAffinityOutcome::SameHostPicked);
        reg.record_decode_affinity(DecodeAffinityOutcome::FallbackBreaker);
        reg.record_decode_affinity(DecodeAffinityOutcome::FallbackLoadImbalance);
        let out = reg.render();
        assert!(out.contains(r#"sgl_router_decode_affinity_total{outcome="same_host_picked"} 2"#));
        assert!(out.contains(r#"sgl_router_decode_affinity_total{outcome="fallback_breaker"} 1"#));
        assert!(out
            .contains(r#"sgl_router_decode_affinity_total{outcome="fallback_load_imbalance"} 1"#,));
    }

    #[test]
    fn label_values_escape_quotes_and_backslashes() {
        let reg = MetricsRegistry::new();
        reg.record_request(
            r#"http://"weird":30000"#,
            r"back\slash",
            WorkerModeLabel::Plain,
            RequestOutcome::Error,
        );
        let out = reg.render();
        assert!(
            out.contains(r#"worker_url="http://\"weird\":30000""#),
            "render did not escape double-quote; got:\n{out}",
        );
        assert!(
            out.contains(r#"model_id="back\\slash""#),
            "render did not escape backslash; got:\n{out}",
        );
    }

    #[test]
    fn histogram_plus_inf_bucket_catches_overflow() {
        let reg = MetricsRegistry::new();
        // 1001 is just above the last finite bucket (1000); it should land
        // in +Inf only.
        reg.observe_overlap_blocks("m", 1001);
        let out = reg.render();
        assert!(out.contains(r#"sgl_router_overlap_blocks_bucket{model_id="m",le="1000"} 0"#));
        assert!(out.contains(r#"sgl_router_overlap_blocks_bucket{model_id="m",le="+Inf"} 1"#));
    }
}

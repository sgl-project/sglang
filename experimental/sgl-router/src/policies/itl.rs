// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-worker inter-token-latency (ITL / TPOT) signal, router-observed.
//!
//! Decode congestion is what request/queue *counts* miss: ten short requests and
//! ten 40K-context requests look identical by count but differ wildly in decode
//! pressure. ITL (milliseconds per output token) measures that pressure
//! directly. The retry path uses it as a load gate so a re-dispatch never lands
//! on a decode-congested worker (see the chat handler's retry loop).
//!
//! # How the sample is produced (router-observed)
//!
//! The SSE pump fires the `on_inter_chunk` hook **once per non-empty upstream
//! chunk after the first**, handing it the gap (ms) since the previous chunk —
//! the router-observed inter-token latency. The chat handler folds *each* such
//! per-chunk gap into the worker's EWMA (see the handler's `make_itl_hook`), so
//! a single streaming response contributes one EWMA update per token-chunk, not
//! one per-stream summary. Most engines emit one token per SSE chunk, so the
//! gap approximates per-token latency; when an engine batches multiple tokens
//! per chunk it *over*-estimates, but stays a faithful **relative** congestion
//! signal across workers, which is all the gate needs.
//!
//! Because the fold is per chunk, the EWMA updates continuously *during* a
//! stream, not only at completion — the signal is live, not per-response. A
//! sustained-slow stream contributes many high-gap samples and so pulls the
//! worker's EWMA up (and holds it there across the freshness window); that is
//! the intended congestion signal, not noise — a worker slow across many chunks
//! *is* decode-congested. It remains **streaming-only**: a non-streaming
//! (buffered) workload emits no chunks, so the table stays empty and the ITL
//! gate degrades to a no-op (retry then behaves exactly as the load-count
//! path). Engine-published ITL over the ZMQ load channel (a future step) would
//! remove the streaming-only limitation.
//!
//! # Freshness
//!
//! Entries older than [`ItlTable::FRESHNESS`] are treated as absent by
//! [`ItlTable::get_fresh`], so a worker that stopped serving doesn't pin a stale
//! ITL that would wrongly exclude (or include) it as a retry target.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// EWMA smoothing factor: `ewma = alpha*sample + (1-alpha)*ewma`. 0.3 favours
/// recent samples while still absorbing a single outlier *chunk*. (Samples are
/// per chunk, so a sustained-slow stream — many high-gap chunks — does move the
/// average; that is the intended congestion signal, not an outlier to absorb.)
const EWMA_ALPHA: f64 = 0.3;

#[derive(Debug, Clone, Copy)]
struct Entry {
    ewma_ms: f64,
    at: Instant,
}

/// Per-worker EWMA of router-observed inter-token latency (ms). Shared
/// (`Arc`) between the SSE pump (writer, via the per-chunk `on_inter_chunk`
/// hook) and the retry path (reader). Cheap `parking_lot::Mutex` around a small
/// map — the write rate is one update per streamed chunk, the read rate is one
/// snapshot per retry.
#[derive(Debug, Default)]
pub struct ItlTable {
    inner: Mutex<HashMap<String, Entry>>,
}

impl ItlTable {
    /// Entries older than this are ignored by [`Self::get_fresh`].
    pub const FRESHNESS: Duration = Duration::from_secs(60);

    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Fold one observed per-chunk inter-token-latency sample (ms) for
    /// `worker_url` into its EWMA. Called once per streamed chunk (see the chat
    /// handler's `make_itl_hook`). Non-finite or non-positive samples are
    /// ignored (a degenerate gap must not corrupt the average).
    pub fn record(&self, worker_url: &str, sample_ms: f64, now: Instant) {
        if !sample_ms.is_finite() || sample_ms <= 0.0 {
            return;
        }
        let mut map = self.inner.lock();
        let e = map.entry(worker_url.to_owned()).or_insert(Entry {
            ewma_ms: sample_ms,
            at: now,
        });
        e.ewma_ms = EWMA_ALPHA * sample_ms + (1.0 - EWMA_ALPHA) * e.ewma_ms;
        e.at = now;
    }

    /// Current smoothed ITL (ms) for `worker_url`, or `None` when there is no
    /// sample or the newest one is older than [`Self::FRESHNESS`]. `None` means
    /// "unknown" — the gate treats an unknown worker as eligible rather than
    /// excluding it, so missing data never blocks failover.
    pub fn get_fresh(&self, worker_url: &str, now: Instant) -> Option<f64> {
        let map = self.inner.lock();
        map.get(worker_url)
            .filter(|e| now.duration_since(e.at) <= Self::FRESHNESS)
            .map(|e| e.ewma_ms)
    }

    /// Snapshot of every fresh (url, ewma_ms) pair, for the metrics gauge.
    pub fn snapshot_fresh(&self, now: Instant) -> Vec<(String, f64)> {
        let map = self.inner.lock();
        map.iter()
            .filter(|(_, e)| now.duration_since(e.at) <= Self::FRESHNESS)
            .map(|(url, e)| (url.clone(), e.ewma_ms))
            .collect()
    }

    /// Drop a worker's entry (worker removed from discovery).
    pub fn forget_worker(&self, worker_url: &str) {
        self.inner.lock().remove(worker_url);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ewma_smooths_and_reads_back() {
        let t = ItlTable::new();
        let now = Instant::now();
        t.record("http://w", 100.0, now);
        assert!((t.get_fresh("http://w", now).unwrap() - 100.0).abs() < 1e-6);
        // Second sample pulls the EWMA toward it, not all the way.
        t.record("http://w", 200.0, now);
        let v = t.get_fresh("http://w", now).unwrap();
        assert!(
            v > 100.0 && v < 200.0,
            "ewma should be between samples: {v}"
        );
    }

    #[test]
    fn unknown_worker_is_none() {
        let t = ItlTable::new();
        assert!(t.get_fresh("http://absent", Instant::now()).is_none());
    }

    #[test]
    fn stale_entry_reads_as_none() {
        let t = ItlTable::new();
        let past = Instant::now();
        t.record("http://w", 50.0, past);
        let later = past + ItlTable::FRESHNESS + Duration::from_secs(1);
        assert!(
            t.get_fresh("http://w", later).is_none(),
            "an entry older than FRESHNESS must read as unknown",
        );
    }

    #[test]
    fn ignores_degenerate_samples() {
        let t = ItlTable::new();
        let now = Instant::now();
        t.record("http://w", 0.0, now);
        t.record("http://w", -5.0, now);
        t.record("http://w", f64::NAN, now);
        assert!(t.get_fresh("http://w", now).is_none());
    }

    #[test]
    fn single_slow_sample_does_not_dominate_after_recovery() {
        let t = ItlTable::new();
        let now = Instant::now();
        for _ in 0..10 {
            t.record("http://w", 20.0, now);
        }
        t.record("http://w", 500.0, now); // one slow stream
        let after_spike = t.get_fresh("http://w", now).unwrap();
        // EWMA moves toward the spike but a few healthy samples pull it back.
        for _ in 0..10 {
            t.record("http://w", 20.0, now);
        }
        let recovered = t.get_fresh("http://w", now).unwrap();
        assert!(
            recovered < after_spike && recovered < 40.0,
            "EWMA must recover after a single slow sample: spike={after_spike}, recovered={recovered}",
        );
    }
}

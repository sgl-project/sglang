//! Per-session turn-timing statistics — a router-side measurement collector.
//!
//! sgl-router already records *global* TTFT and request-duration histograms
//! (see `server/metrics.rs`), but nothing per session. This module fills that
//! gap by *collecting and reporting* per-session timing only — it does not act
//! on the data. For every session (identified by the sticky routing key, e.g.
//! the `x-session-id` header) it stamps three instants per turn —
//!
//!   * `t_dispatch` — request successfully dispatched to a worker (prefill
//!     start), **not** when the router received it, so time spent inside the
//!     router (queue / route / tokenize) is excluded;
//!   * `t_first` — the first streamed token arrived;
//!   * `t_end` — the last decode token / stream end.
//!
//! Lifecycle correctness:
//!   * **Errored / aborted turns are not recorded.** The router's TTFT first-byte
//!     hook only fires for a 2xx stream that actually produced a token, so a turn
//!     is recorded only when `t_first` was stamped. Non-2xx responses, and aborts
//!     before the first token, leave `t_first` unset and are dropped.
//!   * **Retries refresh the start.** The router does no internal retry — a
//!     failed request is resent by the client as a fresh dispatch. Since the
//!     failed attempt recorded nothing, the retry's `on_dispatch` simply
//!     re-stamps `t_dispatch` (the correct prefill start).
//!
//! Two outputs, both opt-in via `SGL_ROUTER_SESSION_STATS_DUMP`:
//!
//!   1. **`session_timestamp.json`** — the raw triples per session, written to
//!      the configured path as a whole-file snapshot (Unix seconds):
//!
//!          { "<session-id>": [[t_dispatch, t_first, t_end], ...], ... }
//!
//!      Refreshed by a periodic flush task and once more on shutdown.
//!
//!   2. **One log line per dispatch**, carrying the *previous* turn's timing `t`
//!      — each phase's share of the whole turn cycle, in [0, 1]:
//!
//!          prefill_t = prefill_ms / total_ms
//!          decode_t  = decode_ms  / total_ms
//!          acting_t  = acting_ms  / total_ms
//!
//!      where `prefill_ms = t_first - t_dispatch`, `decode_ms = t_end - t_first`,
//!      `acting_ms = total_ms - prefill_ms - decode_ms`, and
//!      `total_ms = next_t_dispatch - t_dispatch` (the full cycle incl. client
//!      acting). `total_ms` is only knowable once the next turn dispatches, so a
//!      turn's line is emitted at the start of the following turn.
//!
//! Config: `SGL_ROUTER_SESSION_STATS_DUMP` = path to `session_timestamp.json`.
//! When set, the feature (JSON dump + log lines) is enabled; unset → no-op.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dashmap::DashMap;

/// How often the periodic task rewrites `session_timestamp.json`.
const FLUSH_INTERVAL: Duration = Duration::from_secs(5);

/// One completed turn's raw timestamps, in Unix seconds:
/// `[t_dispatch, t_first, t_end]`.
type Triple = [f64; 3];

/// Shared session → triples map (behind `Arc` so the periodic flush task can
/// snapshot it independently of the collector).
type Archive = Arc<DashMap<String, Vec<Triple>>>;

/// Live per-session log state.
#[derive(Debug)]
struct SessionStat {
    /// Current turn's dispatch instant (prefill start). Re-stamped on retry.
    t_dispatch: Instant,
    /// Current turn's first-token instant. `None` until a token streams — its
    /// presence is what marks the turn as a real, successful turn.
    t_first: Option<Instant>,
    /// Derived `(prefill_ms, decode_ms)` of a completed turn awaiting its
    /// `total_ms` (computed when the next turn dispatches, then logged).
    pending: Option<(f32, f32)>,
    turns: u64,
}

/// Concurrent per-session timestamp collector.
#[derive(Debug)]
pub struct SessionStats {
    /// Live log state (in-flight turn timing per session).
    map: DashMap<String, SessionStat>,
    /// Raw-triple archive → `session_timestamp.json`. `None` when disabled.
    archive: Option<Archive>,
    /// Destination path for the JSON snapshot. `None` in tests (archive only).
    dump_path: Option<String>,
}

impl SessionStats {
    /// Build from the environment. Enabled iff `SGL_ROUTER_SESSION_STATS_DUMP`
    /// is set (to the `session_timestamp.json` path). When enabled, spawns the
    /// periodic flush task (must be called from within a tokio runtime).
    pub fn from_env() -> Arc<Self> {
        let dump_path = std::env::var("SGL_ROUTER_SESSION_STATS_DUMP")
            .ok()
            .filter(|p| !p.is_empty());
        let archive = match &dump_path {
            Some(path) => {
                tracing::info!(
                    path,
                    "session-stats: per-session timestamp collector enabled"
                );
                let archive: Archive = Arc::new(DashMap::new());
                spawn_periodic_flush(Arc::clone(&archive), path.clone());
                Some(archive)
            }
            None => None,
        };
        Arc::new(Self {
            map: DashMap::new(),
            archive,
            dump_path,
        })
    }

    /// A disabled collector (stub / when the feature is off). All hooks are
    /// cheap no-ops.
    pub fn disabled() -> Arc<Self> {
        Arc::new(Self {
            map: DashMap::new(),
            archive: None,
            dump_path: None,
        })
    }

    #[inline]
    fn enabled(&self) -> bool {
        self.archive.is_some()
    }

    /// The request for `sid` was dispatched to a worker (prefill start). Called
    /// at the dispatch site — and again on each retry, re-stamping the start.
    /// Stamps `t_dispatch`, clears `t_first`, and — if the previous turn is
    /// complete — logs its `t` now that `total_ms` is known.
    pub fn on_dispatch(&self, sid: &str, now: Instant) {
        if !self.enabled() {
            return;
        }
        let mut e = self
            .map
            .entry(sid.to_string())
            .or_insert_with(|| SessionStat {
                t_dispatch: now,
                t_first: None,
                pending: None,
                turns: 0,
            });

        // The completed prior turn's total cycle = this dispatch minus that
        // turn's dispatch (still held in `t_dispatch` until we overwrite it
        // below). Log the previous turn's `t` on every dispatch.
        if let Some((prefill_ms, decode_ms)) = e.pending.take() {
            let total_ms = dur_ms(now.saturating_duration_since(e.t_dispatch));
            // acting = client thinking / tool time = total - (prefill + decode).
            let acting_ms = (total_ms - prefill_ms - decode_ms).max(0.0);
            // Each `t` is that phase's share of the whole turn cycle, in [0, 1].
            let (prefill_t, decode_t, acting_t) = if total_ms > 0.0 {
                (
                    prefill_ms / total_ms,
                    decode_ms / total_ms,
                    acting_ms / total_ms,
                )
            } else {
                (0.0, 0.0, 0.0)
            };
            tracing::info!(
                ts = unix_millis(),
                session = sid,
                prefill_t,
                decode_t,
                acting_t,
                prefill_ms,
                decode_ms,
                acting_ms,
                total_ms,
                turn = e.turns,
                "session-stats: t"
            );
        }

        e.t_dispatch = now;
        e.t_first = None;
    }

    /// The first streamed token for `sid` arrived. Idempotent within a turn.
    /// Only fired for a 2xx stream that produced a token, so its presence marks
    /// a real, successful turn.
    pub fn on_first_token(&self, sid: &str, now: Instant) {
        if !self.enabled() {
            return;
        }
        if let Some(mut e) = self.map.get_mut(sid) {
            if e.t_first.is_none() {
                e.t_first = Some(now);
            }
        }
    }

    /// The last decode token / stream end for `sid`. Records the raw triple and
    /// marks the turn's `(prefill, decode)` pending its `total_ms` at the next
    /// dispatch. **Skipped for errored / aborted turns** — if no token ever
    /// streamed (`t_first` unset), there is no real turn to record.
    pub fn on_turn_end(&self, sid: &str, now: Instant) {
        let Some(archive) = &self.archive else {
            return;
        };
        let mut e = match self.map.get_mut(sid) {
            Some(e) => e,
            None => return,
        };
        let Some(t_first) = e.t_first else {
            // No first token → non-2xx error or abort before any token. Not a
            // recordable turn; leave state so the client's retry re-stamps.
            return;
        };
        let prefill_ms = dur_ms(t_first.saturating_duration_since(e.t_dispatch));
        let decode_ms = dur_ms(now.saturating_duration_since(t_first));
        e.pending = Some((prefill_ms, decode_ms));
        e.turns += 1;

        // Raw triple in wall-clock Unix seconds. We know `t_end`'s wall clock
        // and the Instant deltas, so back-compute `t_dispatch` / `t_first` — no
        // need to capture SystemTime at all three points.
        let end_unix = unix_secs();
        let dispatch_unix = end_unix - now.saturating_duration_since(e.t_dispatch).as_secs_f64();
        let first_unix = end_unix - now.saturating_duration_since(t_first).as_secs_f64();
        archive
            .entry(sid.to_string())
            .or_default()
            .push([dispatch_unix, first_unix, end_unix]);
    }

    /// Number of sessions in the JSON archive.
    pub fn len(&self) -> usize {
        self.archive.as_ref().map(|a| a.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Write the raw-triple archive to `session_timestamp.json` (whole-file
    /// overwrite). Called on shutdown for a final snapshot. No-op when disabled
    /// or when no path is configured (tests).
    pub fn flush_dump(&self) {
        if let (Some(archive), Some(path)) = (&self.archive, &self.dump_path) {
            write_snapshot(archive, path);
        }
    }
}

/// Serialize the archive to `path` as a whole-file JSON snapshot.
fn write_snapshot(archive: &DashMap<String, Vec<Triple>>, path: &str) {
    // Snapshot into an ordered map so the JSON is stable / diff-friendly.
    let snapshot: BTreeMap<String, Vec<Triple>> = archive
        .iter()
        .map(|kv| (kv.key().clone(), kv.value().clone()))
        .collect();
    match serde_json::to_vec(&snapshot) {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(path, &bytes) {
                tracing::warn!(path, error = %e, "session-stats: timestamp dump write failed");
            }
        }
        Err(e) => tracing::warn!(error = %e, "session-stats: timestamp dump serialize failed"),
    }
}

/// Spawn the periodic flush task that rewrites `session_timestamp.json` every
/// [`FLUSH_INTERVAL`]. Runs until the runtime is torn down at shutdown.
fn spawn_periodic_flush(archive: Archive, path: String) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(FLUSH_INTERVAL);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            ticker.tick().await;
            write_snapshot(&archive, &path);
        }
    });
}

fn dur_ms(d: Duration) -> f32 {
    d.as_secs_f32() * 1000.0
}

fn unix_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Wall-clock Unix time in milliseconds (log timestamp, ms precision).
fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Enabled collector with an in-memory archive but no file (asserts the
    /// triple/log logic without touching disk or spawning the flush task).
    fn test_stats() -> Arc<SessionStats> {
        Arc::new(SessionStats {
            map: DashMap::new(),
            archive: Some(Arc::new(DashMap::new())),
            dump_path: None,
        })
    }

    fn triples(s: &SessionStats, sid: &str) -> Vec<Triple> {
        s.archive
            .as_ref()
            .unwrap()
            .get(sid)
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    fn at(base: Instant, ms: u64) -> Instant {
        base + Duration::from_millis(ms)
    }

    #[test]
    fn records_raw_triples_per_turn() {
        let s = test_stats();
        let t0 = Instant::now();

        // Turn 1: dispatch@0, first@100, end@600.
        s.on_dispatch("sid", at(t0, 0));
        s.on_first_token("sid", at(t0, 100));
        s.on_turn_end("sid", at(t0, 600));

        // Turn 2: dispatch@1600, first@1700, end@1900.
        s.on_dispatch("sid", at(t0, 1600));
        s.on_first_token("sid", at(t0, 1700));
        s.on_turn_end("sid", at(t0, 1900));

        let ts = triples(&s, "sid");
        assert_eq!(ts.len(), 2, "two turns recorded");
        // Each triple is [t_dispatch, t_first, t_end]; assert the internal deltas.
        let [r0, f0, e0] = ts[0];
        assert!((f0 - r0 - 0.100).abs() < 0.02, "prefill≈100ms: {}", f0 - r0);
        assert!((e0 - f0 - 0.500).abs() < 0.02, "decode≈500ms: {}", e0 - f0);
        let [r1, f1, e1] = ts[1];
        assert!((f1 - r1 - 0.100).abs() < 0.02, "prefill≈100ms: {}", f1 - r1);
        assert!((e1 - f1 - 0.200).abs() < 0.02, "decode≈200ms: {}", e1 - f1);
    }

    #[test]
    fn errored_turn_without_first_token_is_not_recorded() {
        let s = test_stats();
        let t0 = Instant::now();
        // Dispatch but the stream errors before any token (t_first never set).
        s.on_dispatch("sid", at(t0, 0));
        s.on_turn_end("sid", at(t0, 50));
        assert!(
            triples(&s, "sid").is_empty(),
            "no turn recorded on error/abort"
        );

        // The client retries: fresh dispatch re-stamps the start and a real
        // (token-producing) turn is recorded.
        s.on_dispatch("sid", at(t0, 1000));
        s.on_first_token("sid", at(t0, 1100));
        s.on_turn_end("sid", at(t0, 1400));
        let ts = triples(&s, "sid");
        assert_eq!(ts.len(), 1, "only the successful retry is recorded");
        let [r, f, e] = ts[0];
        assert!((f - r - 0.100).abs() < 0.02, "prefill≈100ms: {}", f - r);
        assert!((e - f - 0.300).abs() < 0.02, "decode≈300ms: {}", e - f);
    }

    #[test]
    fn json_shape_is_session_to_triples() {
        let s = test_stats();
        let t0 = Instant::now();
        for sid in ["a", "b"] {
            s.on_dispatch(sid, at(t0, 0));
            s.on_first_token(sid, at(t0, 10));
            s.on_turn_end(sid, at(t0, 50));
        }
        let snapshot: BTreeMap<String, Vec<Triple>> = s
            .archive
            .as_ref()
            .unwrap()
            .iter()
            .map(|kv| (kv.key().clone(), kv.value().clone()))
            .collect();
        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(json.starts_with("{\"a\":[["), "shape: {json}");
        assert!(json.contains("\"b\":[["), "shape: {json}");
    }

    #[test]
    fn logs_prev_turn_on_next_dispatch() {
        let s = test_stats();
        let t0 = Instant::now();
        s.on_dispatch("sid", at(t0, 0));
        s.on_first_token("sid", at(t0, 100));
        s.on_turn_end("sid", at(t0, 600));
        assert!(s.map.get("sid").unwrap().pending.is_some());
        s.on_dispatch("sid", at(t0, 1600)); // resolves total_ms, logs, clears pending
        assert!(s.map.get("sid").unwrap().pending.is_none());
    }

    #[test]
    fn disabled_is_noop() {
        let s = SessionStats::disabled();
        let t0 = Instant::now();
        s.on_dispatch("sid", t0);
        s.on_first_token("sid", at(t0, 10));
        s.on_turn_end("sid", at(t0, 50));
        assert!(s.is_empty());
    }
}

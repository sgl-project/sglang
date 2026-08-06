//! Per-session turn timestamps — a router-side idleness signal.
//!
//! sgl-router already records *global* TTFT and request-duration histograms
//! (see `server/metrics.rs`), but nothing per session. This module fills that
//! gap: for every session (identified by the sticky routing key, e.g. the
//! `x-session-id` header) it stamps three instants per turn —
//!
//!   * `t_recv`  — the request was received / dispatched,
//!   * `t_first` — the first streamed token arrived,
//!   * `t_end`   — the last decode token / stream end.
//!
//! There are two outputs, both opt-in via `SGL_ROUTER_SESSION_STATS_DUMP`:
//!
//!   1. **`session_timestamp.json`** — the raw triples per session, written to
//!      the configured path as a whole-file snapshot:
//!
//!          { "<session-id>": [[t_recv, t_first, t_end], ...], ... }
//!
//!      (Unix seconds.) Refreshed on the sweep tick and on shutdown.
//!
//!   2. **A log line per session whenever its timing `t` changes** — carrying
//!      the three derived durations of a turn:
//!
//!          prefill_t = t_first - t_recv
//!          decode_t  = t_end   - t_first
//!          total_t   = next_t_recv - t_recv   (full turn cycle incl. acting)
//!
//!      `total_t` is only knowable once the *next* turn arrives, so a turn's
//!      line is emitted at the start of the following turn.
//!
//! Config:
//!   * `SGL_ROUTER_SESSION_STATS_DUMP` — path to `session_timestamp.json`; when
//!     set, the feature (JSON dump + log lines) is enabled. Unset → no-op.
//!   * `SGL_ROUTER_SESSION_STATS_TTL_SECS` — evict the per-session *log* state
//!     after N secs of inactivity (default 900). The JSON archive is retained.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dashmap::DashMap;

/// One completed turn's raw timestamps, in Unix seconds:
/// `[t_recv, t_first, t_end]`. `t_first` falls back to `t_recv` if no token was
/// streamed (degenerate; streaming turns always have a first token).
type Triple = [f64; 3];

/// Live per-session log state (the JSON archive is kept separately so TTL
/// sweeping the log state never drops recorded timestamps).
#[derive(Debug)]
struct SessionStat {
    /// Current turn's receive instant.
    t_recv: Instant,
    /// Current turn's first-token instant.
    t_first: Option<Instant>,
    /// Derived `(prefill_ms, decode_ms)` of a completed turn awaiting its
    /// `total_t` (computed when the next turn starts, then logged).
    pending: Option<(f32, f32)>,
    turns: u64,
    last_seen: Instant,
}

/// Concurrent per-session timestamp collector.
#[derive(Debug)]
pub struct SessionStats {
    /// Live log state, TTL-swept.
    map: DashMap<String, SessionStat>,
    ttl: Duration,
    /// Append-only raw-triple archive → `session_timestamp.json`. `None` when
    /// the feature is disabled.
    archive: Option<DashMap<String, Vec<Triple>>>,
    /// Destination path for the JSON snapshot. `None` in tests (archive only).
    dump_path: Option<String>,
}

impl SessionStats {
    /// Build from the environment. Enabled iff `SGL_ROUTER_SESSION_STATS_DUMP`
    /// is set (to the `session_timestamp.json` path).
    pub fn from_env() -> Arc<Self> {
        let ttl = Duration::from_secs(env_parse("SGL_ROUTER_SESSION_STATS_TTL_SECS", 900u64));
        let dump_path = std::env::var("SGL_ROUTER_SESSION_STATS_DUMP")
            .ok()
            .filter(|p| !p.is_empty());
        let enabled = dump_path.is_some();
        if let Some(path) = &dump_path {
            tracing::info!(
                path,
                ttl_secs = ttl.as_secs(),
                "session-stats: per-session timestamp collector enabled"
            );
        }
        Arc::new(Self {
            map: DashMap::new(),
            ttl,
            archive: enabled.then(DashMap::new),
            dump_path,
        })
    }

    /// A disabled collector (stub / when the feature is off). All hooks are
    /// cheap no-ops.
    pub fn disabled() -> Arc<Self> {
        Arc::new(Self {
            map: DashMap::new(),
            ttl: Duration::from_secs(900),
            archive: None,
            dump_path: None,
        })
    }

    #[inline]
    fn enabled(&self) -> bool {
        self.archive.is_some()
    }

    /// A request for `sid` was received. Stamps `t_recv`, and — if the previous
    /// turn is complete — logs that turn's `t` (`prefill_t` / `decode_t` /
    /// `total_t`) now that `total_t` is known. One line per received `t_recv`.
    pub fn on_turn_start(&self, sid: &str, now: Instant) {
        if !self.enabled() {
            return;
        }
        let mut e = self
            .map
            .entry(sid.to_string())
            .or_insert_with(|| SessionStat {
                t_recv: now,
                t_first: None,
                pending: None,
                turns: 0,
                last_seen: now,
            });

        // The completed prior turn's total cycle = this turn's recv minus that
        // turn's recv (still held in `t_recv` until we overwrite it below). Log
        // the previous turn's `t` on every received `t_recv`.
        if let Some((prefill_ms, decode_ms)) = e.pending.take() {
            let total_ms = dur_ms(now.saturating_duration_since(e.t_recv));
            // acting = client thinking / tool time = total - (prefill + decode).
            let acting_ms = (total_ms - prefill_ms - decode_ms).max(0.0);
            // Each `t` is that phase's share of the whole turn cycle, in [0, 1]:
            //   prefill_t = prefill_ms / total_ms
            //   decode_t  = decode_ms  / total_ms
            //   acting_t  = acting_ms  / total_ms   (== 1 - prefill_t - decode_t)
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

        e.t_recv = now;
        e.t_first = None;
        e.last_seen = now;
    }

    /// The first streamed token for `sid` arrived. Idempotent within a turn.
    pub fn on_first_token(&self, sid: &str, now: Instant) {
        if !self.enabled() {
            return;
        }
        if let Some(mut e) = self.map.get_mut(sid) {
            if e.t_first.is_none() {
                e.t_first = Some(now);
            }
            e.last_seen = now;
        }
    }

    /// The last decode token / stream end for `sid`. Records the raw triple into
    /// the JSON archive and marks the turn's `(prefill, decode)` pending its
    /// `total_t` at the next turn start.
    pub fn on_turn_end(&self, sid: &str, now: Instant) {
        if !self.enabled() {
            return;
        }
        let mut e = match self.map.get_mut(sid) {
            Some(e) => e,
            // No matching start (e.g. enabled mid-session) — ignore.
            None => return,
        };
        let t_first = e.t_first;
        let prefill_ms = t_first
            .map(|f| dur_ms(f.saturating_duration_since(e.t_recv)))
            .unwrap_or(0.0);
        let decode_ms = t_first
            .map(|f| dur_ms(now.saturating_duration_since(f)))
            .unwrap_or(0.0);
        e.pending = Some((prefill_ms, decode_ms));
        e.turns += 1;
        e.last_seen = now;

        // Raw triple in wall-clock Unix seconds. We know `t_end`'s wall clock
        // and the Instant deltas, so back-compute `t_recv` / `t_first` — no need
        // to capture SystemTime at all three points.
        if let Some(archive) = &self.archive {
            let end_unix = unix_secs();
            let recv_unix = end_unix - now.saturating_duration_since(e.t_recv).as_secs_f64();
            let first_unix = t_first
                .map(|f| end_unix - now.saturating_duration_since(f).as_secs_f64())
                .unwrap_or(recv_unix);
            archive
                .entry(sid.to_string())
                .or_default()
                .push([recv_unix, first_unix, end_unix]);
        }
    }

    /// Number of sessions in the JSON archive.
    pub fn len(&self) -> usize {
        self.archive.as_ref().map(DashMap::len).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Evict log state inactive for longer than the TTL, then refresh the JSON
    /// snapshot. Returns the number of log entries removed. Driven by
    /// `active_load::spawn_sweeper`. (The archive is never swept.)
    pub fn sweep(&self) -> usize {
        if !self.enabled() {
            return 0;
        }
        let now = Instant::now();
        let ttl = self.ttl;
        let before = self.map.len();
        self.map
            .retain(|_, e| now.saturating_duration_since(e.last_seen) <= ttl);
        let removed = before - self.map.len();
        self.flush_dump();
        removed
    }

    /// Write the raw-triple archive to `session_timestamp.json` (whole-file
    /// overwrite). No-op when disabled or when no path is configured (tests).
    pub fn flush_dump(&self) {
        let (Some(archive), Some(path)) = (&self.archive, &self.dump_path) else {
            return;
        };
        // Snapshot into an ordered map so the JSON is stable/diff-friendly.
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

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Enabled collector with an in-memory archive but no file (asserts the
    /// triple/log logic without touching disk).
    fn test_stats() -> Arc<SessionStats> {
        Arc::new(SessionStats {
            map: DashMap::new(),
            ttl: Duration::from_secs(900),
            archive: Some(DashMap::new()),
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

        // Turn 1: recv@0, first@100, end@600.
        s.on_turn_start("sid", at(t0, 0));
        s.on_first_token("sid", at(t0, 100));
        s.on_turn_end("sid", at(t0, 600));

        // Turn 2: recv@1600, first@1700, end@1900.
        s.on_turn_start("sid", at(t0, 1600));
        s.on_first_token("sid", at(t0, 1700));
        s.on_turn_end("sid", at(t0, 1900));

        let ts = triples(&s, "sid");
        assert_eq!(ts.len(), 2, "two turns recorded");
        // Each triple is [t_recv, t_first, t_end]; assert the internal deltas
        // (absolute values are wall-clock and not fixed in a test).
        let [r0, f0, e0] = ts[0];
        assert!((f0 - r0 - 0.100).abs() < 0.02, "prefill≈100ms: {}", f0 - r0);
        assert!((e0 - f0 - 0.500).abs() < 0.02, "decode≈500ms: {}", e0 - f0);
        let [r1, f1, e1] = ts[1];
        assert!((f1 - r1 - 0.100).abs() < 0.02, "prefill≈100ms: {}", f1 - r1);
        assert!((e1 - f1 - 0.200).abs() < 0.02, "decode≈200ms: {}", e1 - f1);
        // (Cross-turn cycle is validated via the log/`total_t` path, not the
        // triples: each triple's wall clock is anchored at its own `t_end`, so
        // comparing across turns would mix synthetic test instants with the real
        // clock. The within-triple deltas above are the meaningful invariant.)
    }

    #[test]
    fn json_shape_is_session_to_triples() {
        let s = test_stats();
        let t0 = Instant::now();
        s.on_turn_start("a", at(t0, 0));
        s.on_first_token("a", at(t0, 10));
        s.on_turn_end("a", at(t0, 50));
        s.on_turn_start("b", at(t0, 0));
        s.on_first_token("b", at(t0, 10));
        s.on_turn_end("b", at(t0, 50));

        let snapshot: BTreeMap<String, Vec<Triple>> = s
            .archive
            .as_ref()
            .unwrap()
            .iter()
            .map(|kv| (kv.key().clone(), kv.value().clone()))
            .collect();
        let json = serde_json::to_string(&snapshot).unwrap();
        // {"a":[[...]],"b":[[...]]}
        assert!(json.starts_with("{\"a\":[["), "shape: {json}");
        assert!(json.contains("\"b\":[["), "shape: {json}");
    }

    #[test]
    fn logs_prev_turn_t_on_each_recv() {
        let s = test_stats();
        let t0 = Instant::now();
        // Turn 1 completes → its (prefill, decode) is pending, no total yet.
        s.on_turn_start("sid", at(t0, 0));
        s.on_first_token("sid", at(t0, 100));
        s.on_turn_end("sid", at(t0, 600));
        assert!(s.map.get("sid").unwrap().pending.is_some());
        // Turn 2's t_recv → prior turn's `t` is logged (total_t = 1600 - 0) and
        // the pending slot is cleared.
        s.on_turn_start("sid", at(t0, 1600));
        assert!(
            s.map.get("sid").unwrap().pending.is_none(),
            "pending cleared after logging on the next t_recv"
        );
    }

    #[test]
    fn disabled_is_noop() {
        let s = SessionStats::disabled();
        let t0 = Instant::now();
        s.on_turn_start("sid", t0);
        s.on_first_token("sid", at(t0, 10));
        s.on_turn_end("sid", at(t0, 50));
        assert!(s.is_empty());
        assert_eq!(s.sweep(), 0);
    }
}

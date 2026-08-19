//! Per-session turn-timing statistics — a router-side measurement collector.
//!
//! sgl-router already records *global* TTFT and request-duration histograms
//! (see `server/metrics.rs`), but nothing per session. This module fills that
//! gap by *collecting and reporting* per-session timing only — it does not act
//! on the data. For every session (identified by the sticky routing key, e.g.
//! the `x-session-id` header) it stamps four instants per turn —
//!
//!   * `t_recv` — request received by the router (handler entry). This is the
//!     denominator anchor: the turn cycle is measured recv-to-recv, so it
//!     includes router-side overhead (queue / route / tokenize) **and any retry
//!     time**;
//!   * `t_dispatch` — request forwarded to a worker (prefill start). Re-stamped
//!     on each (re)dispatch, so the prefill measurement is clean even when a
//!     failed attempt or router queueing pushed dispatch well after recv;
//!   * `t_first` — the first streamed token arrived;
//!   * `t_end` — the last decode token / stream end.
//!
//! Per turn:
//!
//!     router_ms  = t_dispatch - t_recv    (router queue + retry overhead)
//!     prefill_ms = t_first    - t_dispatch
//!     decode_ms  = t_end      - t_first
//!     total_ms   = next_t_recv - t_recv   (full cycle; includes router_ms)
//!     acting_ms  = total_ms - router_ms - prefill_ms - decode_ms
//!
//! Each `*_t` is that component's share of `total_ms` (they sum to 1). `total_ms`
//! is only knowable once the next turn is received, so a turn's line is emitted
//! at the start of the following turn.
//!
//! Lifecycle correctness (abort / error / retry):
//!   * **Errored / aborted turns are not recorded.** The TTFT first-byte hook
//!     only fires for a 2xx stream that produced a token, so a turn is recorded
//!     only when `t_first` was stamped. Non-2xx responses and aborts before the
//!     first token are dropped.
//!   * **Retries keep `t_recv`, refresh `t_dispatch`.** The router does no
//!     internal retry — a failed request is resent by the client. A resend that
//!     follows a dispatched-but-tokenless attempt is treated as a retry of the
//!     same turn: `t_recv` is kept (so its router/retry time counts toward
//!     `total_ms`), while `on_dispatch` re-stamps `t_dispatch` for a clean
//!     prefill.
//!
//! Two outputs, both opt-in via `SGL_ROUTER_SESSION_STATS_DUMP`:
//!
//!   1. **`session_timestamp.json`** — the raw quads per session (Unix seconds),
//!      a whole-file snapshot refreshed by a periodic flush task and on shutdown:
//!
//!          { "<session-id>": [[t_recv, t_dispatch, t_first, t_end], ...], ... }
//!
//!   2. **One log line per turn** (`session-stats: t`) with the `*_ms` / `*_t`
//!      breakdown above.
//!
//! Config: `SGL_ROUTER_SESSION_STATS_DUMP` = path to `session_timestamp.json`.
//! When set, the feature is enabled; unset → no-op.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dashmap::DashMap;

/// How often the periodic task rewrites `session_timestamp.json`.
const FLUSH_INTERVAL: Duration = Duration::from_secs(5);

/// One completed turn's raw timestamps, in Unix seconds:
/// `[t_recv, t_dispatch, t_first, t_end]`.
type Quad = [f64; 4];

/// Shared session → quads map (behind `Arc` so the periodic flush task can
/// snapshot it independently of the collector).
type Archive = Arc<DashMap<String, Vec<Quad>>>;

/// Derived per-turn durations (ms) awaiting `total_ms` at the next recv.
#[derive(Clone, Copy, Debug)]
struct Pending {
    router_ms: f32,
    prefill_ms: f32,
    decode_ms: f32,
}

/// Live per-session state.
#[derive(Debug)]
struct SessionStat {
    /// This turn's receive instant (denominator anchor; kept across retries).
    t_recv: Instant,
    /// This turn's latest dispatch instant (prefill start; re-stamped on retry).
    t_dispatch: Instant,
    /// First-token instant. `None` until a token streams — its presence marks a
    /// real, successful turn.
    t_first: Option<Instant>,
    /// Whether the current turn has been dispatched at least once (used to tell
    /// a retry from a genuinely new turn at the next recv).
    attempted: bool,
    /// Completed turn's durations awaiting `total_ms` at the next recv.
    pending: Option<Pending>,
    turns: u64,
}

/// Concurrent per-session timestamp collector.
#[derive(Debug)]
pub struct SessionStats {
    map: DashMap<String, SessionStat>,
    /// Raw-quad archive → `session_timestamp.json`. `None` when disabled.
    archive: Option<Archive>,
    /// Destination path for the JSON snapshot. `None` in tests (archive only).
    dump_path: Option<String>,
}

impl SessionStats {
    /// Build from the environment. Enabled iff `SGL_ROUTER_SESSION_STATS_DUMP`
    /// is set. When enabled, spawns the periodic flush task (must be called from
    /// within a tokio runtime).
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

    /// A disabled collector (stub / feature off). All hooks are cheap no-ops.
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

    /// A request for `sid` was received by the router (handler entry). This is
    /// the denominator anchor. If it follows a dispatched-but-tokenless attempt
    /// it is treated as a **retry** (keep `t_recv`); otherwise it starts a new
    /// turn and logs the previous turn's `t` now that `total_ms` is known.
    pub fn on_recv(&self, sid: &str, now: Instant) {
        if !self.enabled() {
            return;
        }
        let mut e = self
            .map
            .entry(sid.to_string())
            .or_insert_with(|| SessionStat {
                t_recv: now,
                t_dispatch: now,
                t_first: None,
                attempted: false,
                pending: None,
                turns: 0,
            });

        // Retry of the current turn: the previous attempt was dispatched but
        // produced no token. Keep the original `t_recv` so its router/retry time
        // is counted in `total_ms`; the coming `on_dispatch` refreshes prefill.
        if e.attempted && e.t_first.is_none() {
            return;
        }

        // New turn → resolve and log the previous turn's cycle.
        if let Some(p) = e.pending.take() {
            let total_ms = dur_ms(now.saturating_duration_since(e.t_recv));
            let acting_ms = (total_ms - p.router_ms - p.prefill_ms - p.decode_ms).max(0.0);
            let (router_t, prefill_t, decode_t, acting_t) = if total_ms > 0.0 {
                (
                    p.router_ms / total_ms,
                    p.prefill_ms / total_ms,
                    p.decode_ms / total_ms,
                    acting_ms / total_ms,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };
            tracing::info!(
                ts = unix_millis(),
                session = sid,
                router_t,
                prefill_t,
                decode_t,
                acting_t,
                router_ms = p.router_ms,
                prefill_ms = p.prefill_ms,
                decode_ms = p.decode_ms,
                acting_ms,
                total_ms,
                turn = e.turns,
                "session-stats: t"
            );
        }

        e.t_recv = now;
        e.t_first = None;
        e.attempted = false;
    }

    /// The request for `sid` was dispatched to a worker (prefill start). Called
    /// at the dispatch site, and again on each retry — re-stamping `t_dispatch`.
    pub fn on_dispatch(&self, sid: &str, now: Instant) {
        if !self.enabled() {
            return;
        }
        if let Some(mut e) = self.map.get_mut(sid) {
            e.t_dispatch = now;
            e.t_first = None;
            e.attempted = true;
        }
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
        }
    }

    /// The last decode token / stream end for `sid`. Records the raw quad and
    /// marks the turn's durations pending `total_ms` at the next recv. **Skipped
    /// for errored / aborted turns** — if no token ever streamed (`t_first`
    /// unset) there is no real turn, and `attempted` is left set so the next
    /// recv is recognized as a retry.
    pub fn on_turn_end(&self, sid: &str, now: Instant) {
        let Some(archive) = &self.archive else {
            return;
        };
        let mut e = match self.map.get_mut(sid) {
            Some(e) => e,
            None => return,
        };
        let Some(t_first) = e.t_first else {
            return;
        };
        let router_ms = dur_ms(e.t_dispatch.saturating_duration_since(e.t_recv));
        let prefill_ms = dur_ms(t_first.saturating_duration_since(e.t_dispatch));
        let decode_ms = dur_ms(now.saturating_duration_since(t_first));
        e.pending = Some(Pending {
            router_ms,
            prefill_ms,
            decode_ms,
        });
        e.turns += 1;
        e.attempted = false;

        // Raw quad in wall-clock Unix seconds, back-computed from `t_end`.
        let end_unix = unix_secs();
        let recv_unix = end_unix - now.saturating_duration_since(e.t_recv).as_secs_f64();
        let dispatch_unix = end_unix - now.saturating_duration_since(e.t_dispatch).as_secs_f64();
        let first_unix = end_unix - now.saturating_duration_since(t_first).as_secs_f64();
        archive.entry(sid.to_string()).or_default().push([
            recv_unix,
            dispatch_unix,
            first_unix,
            end_unix,
        ]);
    }

    /// Number of sessions in the JSON archive.
    pub fn len(&self) -> usize {
        self.archive.as_ref().map(|a| a.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Write the raw-quad archive to `session_timestamp.json` (whole-file
    /// overwrite). Called on shutdown for a final snapshot. No-op when disabled.
    pub fn flush_dump(&self) {
        if let (Some(archive), Some(path)) = (&self.archive, &self.dump_path) {
            write_snapshot(archive, path);
        }
    }
}

/// Serialize the archive to `path` as a whole-file JSON snapshot.
fn write_snapshot(archive: &DashMap<String, Vec<Quad>>, path: &str) {
    let snapshot: BTreeMap<String, Vec<Quad>> = archive
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

    fn test_stats() -> Arc<SessionStats> {
        Arc::new(SessionStats {
            map: DashMap::new(),
            archive: Some(Arc::new(DashMap::new())),
            dump_path: None,
        })
    }

    fn quads(s: &SessionStats, sid: &str) -> Vec<Quad> {
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
    fn records_raw_quads_per_turn() {
        let s = test_stats();
        let t0 = Instant::now();

        // Turn 1: recv@0, dispatch@5, first@105, end@605.
        s.on_recv("sid", at(t0, 0));
        s.on_dispatch("sid", at(t0, 5));
        s.on_first_token("sid", at(t0, 105));
        s.on_turn_end("sid", at(t0, 605));

        // Turn 2: recv@1600, dispatch@1605, first@1705, end@1905.
        s.on_recv("sid", at(t0, 1600));
        s.on_dispatch("sid", at(t0, 1605));
        s.on_first_token("sid", at(t0, 1705));
        s.on_turn_end("sid", at(t0, 1905));

        let q = quads(&s, "sid");
        assert_eq!(q.len(), 2);
        let [rc, ds, fs, en] = q[0];
        assert!((ds - rc - 0.005).abs() < 0.02, "router≈5ms: {}", ds - rc);
        assert!((fs - ds - 0.100).abs() < 0.02, "prefill≈100ms: {}", fs - ds);
        assert!((en - fs - 0.500).abs() < 0.02, "decode≈500ms: {}", en - fs);
    }

    #[test]
    fn retry_keeps_recv_but_refreshes_dispatch() {
        let s = test_stats();
        let t0 = Instant::now();
        // First attempt: recv@0, dispatch@5, errors (no token).
        s.on_recv("sid", at(t0, 0));
        s.on_dispatch("sid", at(t0, 5));
        s.on_turn_end("sid", at(t0, 40)); // no first token → not recorded, attempted stays
        assert!(quads(&s, "sid").is_empty());

        // Retry: recv@200 (kept? original t_recv@0), dispatch@205, first@305, end@805.
        s.on_recv("sid", at(t0, 200)); // retry → t_recv kept at 0
        s.on_dispatch("sid", at(t0, 205));
        s.on_first_token("sid", at(t0, 305));
        s.on_turn_end("sid", at(t0, 805));

        let q = quads(&s, "sid");
        assert_eq!(q.len(), 1, "only the successful retry recorded");
        let [rc, ds, fs, en] = q[0];
        // router_ms = dispatch - recv = 205 - 0 = 205 (includes the retry gap).
        assert!(
            (ds - rc - 0.205).abs() < 0.03,
            "router≈205ms (incl retry): {}",
            ds - rc
        );
        // prefill clean from the successful dispatch = 305 - 205 = 100.
        assert!((fs - ds - 0.100).abs() < 0.02, "prefill≈100ms: {}", fs - ds);
        assert!((en - fs - 0.500).abs() < 0.02, "decode≈500ms: {}", en - fs);
    }

    #[test]
    fn errored_turn_without_first_token_is_not_recorded() {
        let s = test_stats();
        let t0 = Instant::now();
        s.on_recv("sid", at(t0, 0));
        s.on_dispatch("sid", at(t0, 5));
        s.on_turn_end("sid", at(t0, 50));
        assert!(quads(&s, "sid").is_empty());
    }

    #[test]
    fn json_shape_is_session_to_quads() {
        let s = test_stats();
        let t0 = Instant::now();
        for sid in ["a", "b"] {
            s.on_recv(sid, at(t0, 0));
            s.on_dispatch(sid, at(t0, 5));
            s.on_first_token(sid, at(t0, 15));
            s.on_turn_end(sid, at(t0, 55));
        }
        let snapshot: BTreeMap<String, Vec<Quad>> = s
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
    fn disabled_is_noop() {
        let s = SessionStats::disabled();
        let t0 = Instant::now();
        s.on_recv("sid", t0);
        s.on_dispatch("sid", at(t0, 5));
        s.on_first_token("sid", at(t0, 15));
        s.on_turn_end("sid", at(t0, 55));
        assert!(s.is_empty());
    }
}

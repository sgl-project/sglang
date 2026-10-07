//! Per-rank bootstrap state, the readiness deadline and the outcome tallies.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tracing::{debug, info, warn};

use super::DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP;
use super::{BootstrapState, RankOutcome, SnapshotOutcome, SweepOutcome};
use crate::state::kv_events::tree::KvWorkerId;

/// Rate-limit key for the snapshot-attempt log: one entry per peer per
/// verdict.
type PeerOutcomeKey = (String, &'static str);

/// Rate-limit state for one [`PeerOutcomeKey`]. The last-seen detail is kept
/// so a cause CHANGE under the same verdict is surfaced rather than throttled
/// away.
#[derive(Debug, Default)]
struct AttemptLog {
    attempts: u64,
    last_detail: String,
}

/// A repeated verdict against one peer is logged at info on every this-many
/// attempts, and at debug otherwise.
const ATTEMPT_LOG_EVERY: u64 = 100;

/// Counts keyed by a fixed `&'static str` label set, so cardinality is
/// bounded. Sampled by the metrics surface at scrape time.
#[derive(Debug, Default)]
struct LabelTally(Mutex<HashMap<&'static str, u64>>);

impl LabelTally {
    fn bump(&self, label: &'static str) {
        *self.0.lock().entry(label).or_insert(0) += 1;
    }

    fn snapshot(&self) -> Vec<(&'static str, u64)> {
        self.0.lock().iter().map(|(k, v)| (*k, *v)).collect()
    }
}

/// Per-rank bootstrap progress and whether initial bootstrap has settled.
/// Once settled, `settled()` stays true, so a rank registered later
/// (scale-up) cannot un-settle it.
#[derive(Debug)]
pub struct BootstrapTracker {
    states: Mutex<HashMap<KvWorkerId, BootstrapState>>,
    deadline: Mutex<Option<Instant>>,
    timeout: Duration,
    /// Upper bound on one peer-snapshot fetch; defaults to
    /// [`DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP`].
    fetch_cap: Duration,
    /// Whether peer bootstrap is configured. Not derivable from `settled()`:
    /// a disabled tracker and a finished one both report settled.
    enabled: bool,
    /// Set once `settled()` first answers true AFTER the first registration.
    /// An expiry observed before any rank is registered answers true without
    /// latching, so the re-arm at first registration still gates readiness.
    latched: AtomicBool,
    /// Whether the one-shot re-arm at first worker discovery has happened.
    ///
    /// Keyed on this rather than on `states.is_empty()`: the map empties again
    /// whenever `forget` removes the last rank, so an engine flapping
    /// remove/re-add would re-arm on every cycle and hold the readiness gate for
    /// an unbounded multiple of the configured timeout.
    rearmed: AtomicBool,
    /// Per-rank incarnation, minted afresh when a rank is registered after
    /// `forget`, so a result started for an earlier incarnation is
    /// recognisably stale.
    epochs: Mutex<HashMap<KvWorkerId, u64>>,
    epoch_seq: AtomicU64,
    /// Ranks already given their one post-gap retry this incarnation; `forget`
    /// clears the mark.
    gap_retried: Mutex<HashSet<KvWorkerId>>,
    /// Keyed by [`SnapshotOutcome::as_label`].
    peer_outcomes: LabelTally,
    /// Keyed by [`RankOutcome::as_label`].
    rank_outcomes: LabelTally,
    /// Rate-limit state for the attempt log in [`Self::record_peer_outcome`].
    peer_attempts: Mutex<HashMap<PeerOutcomeKey, AttemptLog>>,
    /// Keyed by [`SweepOutcome::as_label`].
    sweep_results: LabelTally,
}

/// `now + timeout`, clamped instead of panicking: `Instant + Duration` panics
/// on overflow, and an absurd timeout means "effectively never", not "abort".
fn deadline_after(timeout: Duration) -> Instant {
    const FAR_FUTURE: Duration = Duration::from_secs(100 * 365 * 24 * 60 * 60);
    let now = Instant::now();
    now.checked_add(timeout)
        .or_else(|| now.checked_add(FAR_FUTURE))
        .unwrap_or(now)
}

impl BootstrapTracker {
    pub fn new(timeout: Duration) -> Self {
        Self::new_with_fetch_cap(timeout, DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP)
    }

    pub fn new_with_fetch_cap(timeout: Duration, fetch_cap: Duration) -> Self {
        Self {
            fetch_cap,
            states: Mutex::new(HashMap::new()),
            // Armed at construction so `settled()` turns true by the deadline
            // even if no rank is ever registered; `register` re-arms once so the
            // budget normally runs from first worker discovery.
            deadline: Mutex::new(Some(deadline_after(timeout))),
            timeout,
            enabled: true,
            latched: AtomicBool::new(false),
            rearmed: AtomicBool::new(false),
            epochs: Mutex::new(HashMap::new()),
            epoch_seq: AtomicU64::new(0),
            gap_retried: Mutex::new(HashSet::new()),
            peer_outcomes: LabelTally::default(),
            rank_outcomes: LabelTally::default(),
            peer_attempts: Mutex::new(HashMap::new()),
            sweep_results: LabelTally::default(),
        }
    }

    /// Tally one snapshot FETCH against one peer, and log it.
    pub fn record_peer_outcome(&self, outcome: SnapshotOutcome, peer: &str, detail: Option<&str>) {
        self.peer_outcomes.bump(outcome.as_label());
        if outcome == SnapshotOutcome::Accepted {
            return;
        }
        let detail = detail.unwrap_or("");
        let (attempts, detail_changed) = {
            let mut logs = self.peer_attempts.lock();
            let log = logs
                .entry((peer.to_string(), outcome.as_label()))
                .or_default();
            log.attempts += 1;
            let changed = log.last_detail != detail;
            if changed {
                log.last_detail = detail.to_string();
            }
            (log.attempts, changed && log.attempts > 1)
        };
        // The first failure against a peer is signal; a repeat of the same
        // verdict is not — but repeated attempts against unreachable or
        // non-covering peers must still show progress at the default log level.
        // A detail CHANGE under the same verdict (connection-refused becoming
        // DNS, 404 becoming 503) is a new cause, not a repeat: surface it too.
        let surface = attempts == 1 || attempts % ATTEMPT_LOG_EVERY == 0 || detail_changed;
        macro_rules! log_attempt {
            ($level:ident) => {
                $level!(
                    peer = %peer,
                    outcome = outcome.as_label(),
                    attempts,
                    detail,
                    "kv-bootstrap: snapshot attempt did not yield state",
                )
            };
        }
        if surface {
            log_attempt!(info);
        } else {
            log_attempt!(debug);
        }
    }

    /// Tally one rank's final verdict; record each rank once.
    pub fn record_rank_outcome(&self, outcome: RankOutcome) {
        self.rank_outcomes.bump(outcome.as_label());
    }

    /// Per-peer-attempt tallies for the metrics surface.
    pub fn peer_outcome_counts(&self) -> Vec<(&'static str, u64)> {
        self.peer_outcomes.snapshot()
    }

    /// Per-rank tallies for the metrics surface.
    pub fn rank_outcome_counts(&self) -> Vec<(&'static str, u64)> {
        self.rank_outcomes.snapshot()
    }

    /// Tally one sweep's terminal verdict. `peers_tried` is accepted for
    /// callers that gate on it and is not stored.
    pub fn record_sweep_result(&self, result: SweepOutcome, peers_tried: usize) {
        let _ = peers_tried;
        self.sweep_results.bump(result.as_label());
    }

    /// Per-sweep-verdict tallies for the metrics surface.
    pub fn sweep_result_counts(&self) -> Vec<(&'static str, u64)> {
        self.sweep_results.snapshot()
    }

    /// A tracker that is settled from the start, for the paths where peer
    /// bootstrap is disabled entirely.
    pub fn disabled() -> Self {
        let mut t = Self::new(Duration::ZERO);
        t.enabled = false;
        t.latched.store(true, Ordering::Relaxed);
        t
    }

    /// Whether peer bootstrap is configured. Unlike [`BootstrapTracker::settled`]
    /// this never flips: it answers "may ranks bootstrap at all?", not "is
    /// readiness still waiting?".
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Upper bound on one peer-snapshot fetch.
    pub fn fetch_cap(&self) -> Duration {
        self.fetch_cap
    }

    /// Register ranks as `Pending` and return obligations for the ones newly
    /// inserted: each paired with a freshly minted incarnation number a later
    /// control message must still match to be allowed to act on it.
    ///
    /// A rank already registered keeps its state and incarnation and yields no
    /// obligation: whoever holds its existing obligation owns it. Decided under
    /// the `states` lock, so concurrent callers registering the same rank get
    /// exactly one obligation between them. A rank [`Self::forget`] removed is
    /// new again and gets a new incarnation.
    ///
    /// Ranks added after the tracker has latched are recorded (so metrics stay
    /// accurate) but cannot un-settle readiness. The first registration is not
    /// such a case: nothing latches before it (see [`Self::settled`]), so the
    /// re-arm below is what the readiness gate measures from.
    pub fn register(&self, ids: &[KvWorkerId]) -> Vec<(KvWorkerId, u64)> {
        if ids.is_empty() {
            return Vec::new();
        }
        let mut states = self.states.lock();
        // Re-arm exactly once, at first worker discovery, so the budget is not
        // spent by slow discovery. Strictly one-shot — see `rearmed`. Done
        // under the `states` lock, which `settled()` also holds while it
        // decides whether to latch, so the two cannot interleave.
        if !self.rearmed.swap(true, Ordering::Relaxed) {
            *self.deadline.lock() = Some(deadline_after(self.timeout));
        }
        let mut epochs = self.epochs.lock();
        let mut obligations = Vec::with_capacity(ids.len());
        for id in ids {
            if states.contains_key(id) {
                continue;
            }
            states.insert(id.clone(), BootstrapState::Pending);
            let epoch = self.epoch_seq.fetch_add(1, Ordering::Relaxed) + 1;
            epochs.insert(id.clone(), epoch);
            obligations.push((id.clone(), epoch));
        }
        obligations
    }

    /// Current incarnation of `id`, or `None` if it is not registered.
    pub fn epoch_of(&self, id: &KvWorkerId) -> Option<u64> {
        self.epochs.lock().get(id).copied()
    }

    /// Update a REGISTERED rank's state; an unregistered rank is ignored.
    ///
    /// Non-creating because `forget` is the authority on membership: an
    /// inserting `set` could resurrect a forgotten rank's state without an
    /// epoch, and `register` would then keep that stale state for the re-added
    /// rank instead of starting it `Pending`.
    pub fn set(&self, id: &KvWorkerId, state: BootstrapState) {
        if let Some(slot) = self.states.lock().get_mut(id) {
            *slot = state;
        }
    }

    pub fn forget(&self, ids: &[KvWorkerId]) {
        // Lock order: states, epochs, gap_retried — `retry_after_gap` takes
        // the same order.
        let mut states = self.states.lock();
        let mut epochs = self.epochs.lock();
        let mut gap_retried = self.gap_retried.lock();
        for id in ids {
            states.remove(id);
            epochs.remove(id);
            gap_retried.remove(id);
        }
    }

    /// Move a gap-discarded rank back to `Pending` so it can be swept again,
    /// returning the obligation to queue.
    ///
    /// One retry per incarnation; `forget` clears the mark. `None` when this
    /// incarnation already retried, the rank was forgotten, or it is not
    /// `Failed`. The caller must queue a returned obligation, since nothing
    /// else resolves the `Pending` state it leaves. A latched `settled()` is
    /// unaffected.
    pub fn retry_after_gap(&self, id: &KvWorkerId) -> Option<(KvWorkerId, u64)> {
        let mut states = self.states.lock();
        if states.get(id) != Some(&BootstrapState::Failed) {
            return None;
        }
        // Same lock order as `register` (states, then epochs).
        let epoch = *self.epochs.lock().get(id)?;
        if !self.gap_retried.lock().insert(id.clone()) {
            return None;
        }
        states.insert(id.clone(), BootstrapState::Pending);
        Some((id.clone(), epoch))
    }

    pub fn state_of(&self, id: &KvWorkerId) -> Option<BootstrapState> {
        self.states.lock().get(id).copied()
    }

    /// Snapshot of every tracked rank, for the metrics surface.
    pub fn states(&self) -> Vec<(KvWorkerId, BootstrapState)> {
        self.states
            .lock()
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    /// Whether every rank a `Pending` entry exists for has reached a terminal
    /// state, or the deadline has passed.
    ///
    /// Latches on the first `true` observed after the first registration. An
    /// expiry seen before any rank is registered answers true WITHOUT
    /// latching, so the first registration's re-arm still governs.
    ///
    /// The whole decision runs under the `states` lock, which `register` holds
    /// while it re-arms, so a latch can never be computed from the
    /// construction-time deadline and stored after the re-arm.
    ///
    /// The deadline is always armed (see [`BootstrapTracker::new`]), so this
    /// cannot stay `false` forever regardless of what does or does not get
    /// registered.
    pub fn settled(&self) -> bool {
        if self.latched.load(Ordering::Relaxed) {
            return true;
        }
        let pending = {
            let states = self.states.lock();
            let pending = states.values().filter(|s| !s.is_terminal()).count();
            let all_terminal = !states.is_empty() && pending == 0;
            let expired = self.deadline.lock().is_some_and(|d| Instant::now() >= d);
            if !(all_terminal || expired) {
                return false;
            }
            if !self.rearmed.load(Ordering::Relaxed) {
                return true;
            }
            self.latched.store(true, Ordering::Relaxed);
            pending
        };
        if pending > 0 {
            warn!(
                timeout_ms = self.timeout.as_millis(),
                pending,
                "kv-bootstrap: deadline elapsed with ranks still pending; \
                 serving with a partially warmed cache-aware tree",
            );
        }
        true
    }

    /// Time left before the deadline forces settlement, or `None` when no
    /// deadline is armed yet.
    pub fn time_remaining(&self) -> Option<Duration> {
        self.deadline
            .lock()
            .map(|d| d.saturating_duration_since(Instant::now()))
    }

    /// Move the deadline to now, so a test can observe expiry without
    /// sleeping against a wall-clock budget.
    #[cfg(test)]
    fn expire_deadline_for_test(&self) {
        *self.deadline.lock() = Some(Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Re-registering after a `forget` must mint a NEW incarnation, so a
    /// bootstrap task still in flight for the old one is recognisably stale.
    #[test]
    fn reregistration_after_forget_mints_a_new_epoch() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        let a = KvWorkerId::new("http://a".into(), 0);
        let first = t.register(std::slice::from_ref(&a));
        let e1 = first[0].1;

        // Re-registering a still-present rank yields no obligation and keeps
        // its incarnation.
        assert!(
            t.register(std::slice::from_ref(&a)).is_empty(),
            "a registered rank's obligation is already held",
        );
        assert_eq!(
            t.epoch_of(&a),
            Some(e1),
            "a live rank keeps its incarnation"
        );

        t.forget(std::slice::from_ref(&a));
        assert_eq!(t.epoch_of(&a), None);
        let second = t.register(std::slice::from_ref(&a));
        assert_ne!(
            second[0].1, e1,
            "remove + re-add must invalidate the previous incarnation",
        );
    }

    #[test]
    fn tracker_settles_when_all_ranks_terminal() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        let a = KvWorkerId::new("http://a".into(), 0);
        let b = KvWorkerId::new("http://a".into(), 1);
        t.register(&[a.clone(), b.clone()]);
        assert!(!t.settled());

        t.set(&a, BootstrapState::Recovered);
        assert!(!t.settled(), "one rank still pending");

        t.set(&b, BootstrapState::Failed);
        assert!(t.settled(), "Failed is terminal — cold is a valid outcome");
    }

    #[test]
    fn fetch_cap_round_trips_through_the_tracker() {
        let t =
            BootstrapTracker::new_with_fetch_cap(Duration::from_secs(120), Duration::from_secs(90));
        assert_eq!(t.timeout(), Duration::from_secs(120));
        assert_eq!(t.fetch_cap(), Duration::from_secs(90));
        assert_eq!(
            BootstrapTracker::new(Duration::from_secs(120)).fetch_cap(),
            DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP,
            "a tracker built without a cap takes the default",
        );
    }

    #[test]
    fn tracker_settles_on_deadline_with_pending_ranks() {
        let t = BootstrapTracker::new(Duration::ZERO);
        t.register(&[KvWorkerId::new("http://a".into(), 0)]);
        assert!(t.settled(), "zero timeout settles immediately");
    }

    /// An empty tracker is not settled *yet* — it may still be waiting on
    /// workers to be discovered — but its deadline is already armed, so it
    /// cannot wait forever.
    #[test]
    fn tracker_with_no_ranks_is_not_settled_before_deadline() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        assert!(!t.settled());
        assert!(
            t.time_remaining().is_some(),
            "the deadline must be armed at construction, not at first register",
        );
    }

    /// A tracker that is never registered still settles by its deadline.
    #[test]
    fn tracker_never_registered_still_settles() {
        let t = BootstrapTracker::new(Duration::ZERO);
        assert!(t.settled(), "an unregistered tracker must still settle",);
    }

    /// Registration keeps the deadline armed (the re-arm itself is not timed,
    /// to stay independent of the wall clock).
    #[test]
    fn registration_keeps_the_deadline_armed() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        t.register(&[KvWorkerId::new("http://a".into(), 0)]);
        assert!(t.time_remaining().is_some());
        assert!(!t.settled(), "a pending rank must still gate readiness");
    }

    #[test]
    fn disabled_tracker_is_settled_immediately() {
        assert!(BootstrapTracker::disabled().settled());
    }

    #[test]
    fn tracker_forget_removes_state() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        let a = KvWorkerId::new("http://a".into(), 0);
        t.register(std::slice::from_ref(&a));
        assert_eq!(t.state_of(&a), Some(BootstrapState::Pending));
        t.forget(std::slice::from_ref(&a));
        assert_eq!(t.state_of(&a), None);
    }

    /// An expiry observed before the first registration does not latch, so the
    /// registration's re-arm still governs `settled()`.
    #[test]
    fn a_premature_settle_does_not_disarm_the_gate() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        t.expire_deadline_for_test();
        assert!(t.settled(), "the construction deadline has expired");

        let rank = KvWorkerId::new("http://w1".into(), 0);
        t.register(std::slice::from_ref(&rank));
        assert!(
            !t.settled(),
            "the first registration re-arms the budget, so a Pending rank must \
             hold readiness again",
        );

        t.set(&rank, BootstrapState::Recovered);
        assert!(
            t.settled(),
            "and it settles normally once the rank resolves"
        );
    }

    /// Once latched after the first registration, a rank discovered later
    /// cannot pull readiness back to 503 — the scale-up case the latch exists
    /// for.
    #[test]
    fn a_latched_tracker_stays_settled_across_a_later_registration() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        let a = KvWorkerId::new("http://a".into(), 0);
        t.register(std::slice::from_ref(&a));
        t.set(&a, BootstrapState::Recovered);
        assert!(t.settled());

        t.register(&[KvWorkerId::new("http://b".into(), 0)]);
        assert!(
            t.settled(),
            "a scale-up must not un-ready a serving replica"
        );
    }

    /// A disabled tracker is settled from the start, and a registration that
    /// reaches it anyway must not open a gate peer bootstrap never configured.
    #[test]
    fn a_disabled_tracker_stays_settled_after_register() {
        let t = BootstrapTracker::disabled();
        t.register(&[KvWorkerId::new("http://a".into(), 0)]);
        assert!(t.settled());
    }

    /// `Instant + Duration` panics on overflow; an absurd timeout must mean
    /// "effectively never", not a crash at construction or first registration.
    #[test]
    fn an_absurd_timeout_does_not_panic() {
        let t = BootstrapTracker::new(Duration::MAX);
        t.register(&[KvWorkerId::new("http://a".into(), 0)]);
        assert!(!t.settled());
        assert!(t
            .time_remaining()
            .is_some_and(|d| d > Duration::from_secs(3600)));
    }

    /// One post-gap retry per incarnation: a second gap on the same one is
    /// final, but a remove + re-add is a fresh publisher and earns its own.
    #[test]
    fn retry_after_gap_is_one_shot_per_incarnation() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        let a = KvWorkerId::new("http://a".into(), 0);
        let first = t.register(std::slice::from_ref(&a));

        assert_eq!(t.retry_after_gap(&a), None, "Pending is not a gap state");
        t.set(&a, BootstrapState::Failed);
        assert_eq!(t.retry_after_gap(&a), Some(first[0].clone()));
        assert_eq!(t.state_of(&a), Some(BootstrapState::Pending));

        t.set(&a, BootstrapState::Failed);
        assert_eq!(t.retry_after_gap(&a), None, "already retried once");

        t.forget(std::slice::from_ref(&a));
        assert_eq!(
            t.retry_after_gap(&a),
            None,
            "a forgotten rank never retries"
        );
        let second = t.register(std::slice::from_ref(&a));
        t.set(&a, BootstrapState::Failed);
        assert_eq!(
            t.retry_after_gap(&a),
            Some(second[0].clone()),
            "a new incarnation gets its own retry",
        );
    }

    /// A worker discovered after readiness opened must still be allowed to warm,
    /// so `enabled()` does not follow `settled()`.
    #[test]
    fn enabled_survives_settling_so_late_workers_still_bootstrap() {
        let tracker = BootstrapTracker::new(Duration::from_millis(1));
        let id = KvWorkerId::new("http://w1:30000".into(), 0);
        tracker.register(std::slice::from_ref(&id));
        std::thread::sleep(Duration::from_millis(5));
        assert!(tracker.settled(), "deadline expiry settles readiness");
        assert!(
            tracker.enabled(),
            "settling must NOT disable bootstrap for workers found later",
        );
    }

    /// The disabled case must stay distinguishable from the finished case, or a
    /// router with no `--kv-peer-selector` would start holding batches.
    #[test]
    fn disabled_tracker_is_not_enabled() {
        let t = BootstrapTracker::disabled();
        assert!(t.settled());
        assert!(
            !t.enabled(),
            "no selector configured ⇒ never register ranks"
        );
    }

    /// The premise behind `BootstrapDeps::deadline`'s settled branch: the
    /// remaining window saturates at zero, so a late rank needs the configured
    /// timeout instead.
    #[test]
    fn time_remaining_saturates_to_zero_once_expired() {
        let tracker = BootstrapTracker::new(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(
            tracker.time_remaining(),
            Some(Duration::ZERO),
            "expired window must read as zero, not as None",
        );
        assert!(
            tracker.timeout() > Duration::ZERO,
            "the configured timeout must stay positive",
        );
    }
}

//! Per-rank bootstrap state, the readiness deadline, the seed gate and the outcome tallies.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tracing::{debug, info, warn};

use super::DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP;
use super::{BootstrapState, RankOutcome, SnapshotOutcome, SweepOutcome};
use crate::state::kv_events::tree::KvWorkerId;

/// Hard bound on the seed-required readiness gate: three times the bootstrap
/// budget, so a merely slow sweep still finishes inside it, floored at 60s so
/// a short budget leaves a usable window. Bounded at all because a
/// simultaneous fleet-wide restart, where no sweep can succeed, must degrade
/// to a delay rather than an outage with no exit.
fn seed_gate_timeout(bootstrap_timeout: Duration) -> Duration {
    bootstrap_timeout
        .saturating_mul(3)
        .max(Duration::from_secs(60))
}

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
    fn is_empty(&self) -> bool {
        self.0.lock().is_empty()
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
    /// Set once `settled()` first answers true AFTER the first registration,
    /// or by [`Self::admit_ready`]. An expiry observed before any rank is
    /// registered answers true without latching, so the re-arm at first
    /// registration still gates readiness.
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
    /// `--kv-bootstrap-seed-required`: whether a failed seed holds `/readyz`.
    seed_required: bool,
    /// Set by a [`SweepOutcome::TimedOut`] over a NON-EMPTY candidate set,
    /// cleared by a later `Found`.
    ///
    /// Only that verdict means "siblings were there and we failed to get their
    /// state". `NoPeers` (first deploy) and `FleetCold` (every sibling proved
    /// empty) must never gate: an unready replica leaves its own
    /// EndpointSlice, so gating on them deadlocks the fleet.
    seed_failed: AtomicBool,
    /// Hard bound on how long [`Self::seed_gate_open`] may hold readiness
    /// down; re-armed at first registration, like `deadline`.
    seed_gate_deadline: Mutex<Instant>,
    /// Latched by [`Self::admit_ready`] or by the hard bound expiring; the
    /// seed gate is then open for good.
    seed_gate_passed: AtomicBool,
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
        Self::new_with_opts(timeout, fetch_cap, false)
    }

    /// `seed_required`: hold `/readyz` at 503 when a sweep proves siblings were
    /// present and their state could not be pulled. See
    /// [`Self::seed_gate_open`].
    pub fn new_with_opts(timeout: Duration, fetch_cap: Duration, seed_required: bool) -> Self {
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
            seed_required,
            seed_failed: AtomicBool::new(false),
            seed_gate_deadline: Mutex::new(deadline_after(seed_gate_timeout(timeout))),
            seed_gate_passed: AtomicBool::new(false),
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

    /// Tally one sweep's terminal verdict. `peers_tried` is the size of the
    /// candidate set it was proven over: a `TimedOut` over a non-empty set is
    /// a failed seed, over an empty one only a discovery race. See
    /// [`Self::seed_gate_open`].
    pub fn record_sweep_result(&self, result: SweepOutcome, peers_tried: usize) {
        self.sweep_results.bump(result.as_label());
        match result {
            SweepOutcome::TimedOut if peers_tried > 0 => {
                self.seed_failed.store(true, Ordering::Relaxed);
            }
            // A later sweep that lands un-fails the replica.
            SweepOutcome::Found => self.seed_failed.store(false, Ordering::Relaxed),
            _ => {}
        }
    }

    /// Whether readiness may proceed despite the seed outcome.
    ///
    /// Closed only while ALL hold: seed-required is on, [`Self::admit_ready`]
    /// has never answered true, the hard bound has not expired, and either a sweep failed
    /// the seed or ranks are registered with no sweep verdict yet. The last
    /// arm exists because every rank can go terminal (overflow, publisher
    /// reset, the tracker deadline) before the boot sweep reports, and a 200
    /// in that window would latch the gate open unchecked.
    ///
    /// A NotReady pod is out of the Service, so a closed gate holds this
    /// replica back while the previous generation keeps serving; once the
    /// bound expires it serves cache-blind.
    pub fn seed_gate_open(&self) -> bool {
        if !self.seed_required || self.seed_gate_passed.load(Ordering::Relaxed) {
            return true;
        }
        let verdict_pending = self.rearmed.load(Ordering::Relaxed) && self.sweep_results.is_empty();
        if !verdict_pending && !self.seed_failed.load(Ordering::Relaxed) {
            return true;
        }
        if Instant::now() >= *self.seed_gate_deadline.lock() {
            // Latch, so the warning fires once rather than per probe.
            self.seed_gate_passed.store(true, Ordering::Relaxed);
            warn!(
                "kv-bootstrap: seed-required gate expired with no usable peer snapshot; \
                 serving cache-blind rather than holding this replica out forever",
            );
            return true;
        }
        false
    }

    /// Readiness conditions 3 and 4 together: `settled()` and
    /// `seed_gate_open()`. A `true` latches both, so neither a later sweep nor
    /// the first registration's re-arm can un-ready a replica that has served.
    pub fn admit_ready(&self) -> bool {
        if !(self.settled() && self.seed_gate_open()) {
            return false;
        }
        self.latched.store(true, Ordering::Relaxed);
        self.seed_gate_passed.store(true, Ordering::Relaxed);
        true
    }

    /// Whether a sweep has recorded a failed seed.
    pub fn seed_failed(&self) -> bool {
        self.seed_failed.load(Ordering::Relaxed)
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
    /// such a case: nothing latches before it (see [`Self::settled`]) unless
    /// [`Self::admit_ready`] already has, so the re-arm below is what the
    /// readiness gate measures from.
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
            *self.seed_gate_deadline.lock() = deadline_after(seed_gate_timeout(self.timeout));
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
    /// Whether any registered rank is still [`BootstrapState::Pending`]. Every
    /// such rank has an obligation some sweep owns, so its verdict is still to
    /// come.
    pub fn any_pending(&self) -> bool {
        self.states.lock().values().any(|s| !s.is_terminal())
    }

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

    #[cfg(test)]
    fn expire_seed_gate_for_test(&self) {
        *self.seed_gate_deadline.lock() = Instant::now();
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

    // ===== seed-required readiness gate =====
    //
    // Which sweep verdicts may hold readiness down. Both failure directions are
    // silent: gate too little and a cache-blind rollout ships, gate too much
    // and the fleet deadlocks.

    fn seed_tracker(seed_required: bool) -> BootstrapTracker {
        BootstrapTracker::new_with_opts(
            Duration::from_secs(300),
            DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP,
            seed_required,
        )
    }

    /// `NoPeers` (first deploy) and `FleetCold` (every sibling proved empty)
    /// have nothing to inherit; gating on either deadlocks the fleet, since an
    /// unready replica leaves its own EndpointSlice.
    #[test]
    fn seed_gate_never_closes_on_a_nothing_to_inherit_verdict() {
        for verdict in [SweepOutcome::NoPeers, SweepOutcome::FleetCold] {
            let t = seed_tracker(true);
            t.record_sweep_result(verdict, 9);
            assert!(
                t.seed_gate_open(),
                "{:?} means there was no state to inherit, not a failed seed; \
                 gating on it deadlocks a cold fleet",
                verdict.as_label(),
            );
            assert!(!t.seed_failed());
        }
    }

    /// `TimedOut` gates only when the candidate set was non-empty. Over zero
    /// peers it means discovery had not caught up, which is not evidence that
    /// anyone had state to give.
    #[test]
    fn seed_gate_closes_only_on_timed_out_over_a_non_empty_candidate_set() {
        let empty = seed_tracker(true);
        empty.record_sweep_result(SweepOutcome::TimedOut, 0);
        assert!(
            empty.seed_gate_open(),
            "a timeout over zero candidates is a discovery race, not a failed seed",
        );

        let peers = seed_tracker(true);
        peers.record_sweep_result(SweepOutcome::TimedOut, 9);
        assert!(
            !peers.seed_gate_open(),
            "siblings were present and their tree could not be pulled; this is \
             the one verdict that must hold readiness down",
        );
    }

    /// A later sweep that lands must un-fail the replica, or one unlucky sweep
    /// gates a pod that went on to seed perfectly.
    #[test]
    fn seed_gate_reopens_when_a_later_sweep_finds_a_snapshot() {
        let t = seed_tracker(true);
        t.record_sweep_result(SweepOutcome::TimedOut, 9);
        assert!(!t.seed_gate_open());
        t.record_sweep_result(SweepOutcome::Found, 1);
        assert!(t.seed_gate_open(), "a successful sweep clears the failure");
        assert!(!t.seed_failed());
    }

    /// Off by default: the same failing sweep must not gate a tracker that did
    /// not opt in.
    #[test]
    fn seed_gate_is_inert_unless_required() {
        let t = seed_tracker(false);
        t.record_sweep_result(SweepOutcome::TimedOut, 9);
        assert!(t.seed_failed(), "the failure is still recorded");
        assert!(
            t.seed_gate_open(),
            "but it must not hold readiness unless --kv-bootstrap-seed-required",
        );
    }

    /// Once the replica has served a ready 200, a sweep started by a
    /// late-discovered worker must not drag it back out of the Service.
    #[test]
    fn seed_gate_latches_open_once_readiness_has_passed() {
        let t = seed_tracker(true);
        let rank = KvWorkerId::new("http://w1".into(), 0);
        t.register(std::slice::from_ref(&rank));
        t.set(&rank, BootstrapState::Recovered);
        t.record_sweep_result(SweepOutcome::Found, 1);
        assert!(t.admit_ready());

        t.record_sweep_result(SweepOutcome::TimedOut, 9);
        assert!(
            t.seed_gate_open(),
            "an already-serving replica may not be un-readied by a later sweep",
        );
    }

    /// In a simultaneous fleet-wide restart every sweep ends `TimedOut` over a
    /// non-empty set; without the bound that is a permanent outage.
    #[test]
    fn seed_gate_opens_when_the_hard_bound_expires() {
        let t = seed_tracker(true);
        t.record_sweep_result(SweepOutcome::TimedOut, 9);
        assert!(!t.seed_gate_open(), "closed while the bound holds");
        t.expire_seed_gate_for_test();
        assert!(
            t.seed_gate_open(),
            "the bound must expire the gate; a gate with no exit is worse than \
             the degradation it prevents",
        );
    }

    /// A tracker that never bootstraps cannot have failed to.
    #[test]
    fn disabled_tracker_never_gates() {
        let t = BootstrapTracker::disabled();
        t.record_sweep_result(SweepOutcome::TimedOut, 9);
        assert!(t.seed_gate_open());
    }

    #[test]
    fn seed_gate_timeout_scales_and_floors() {
        assert_eq!(
            seed_gate_timeout(Duration::from_secs(300)),
            Duration::from_secs(900),
        );
        assert_eq!(
            seed_gate_timeout(Duration::from_secs(1)),
            Duration::from_secs(60),
            "floored, or a short budget makes the gate unobservably brief",
        );
        assert_eq!(
            seed_gate_timeout(Duration::from_millis(
                crate::config::DEFAULT_KV_BOOTSTRAP_TIMEOUT_MS
            )),
            Duration::from_secs(1800),
            "at the default budget the hold is three times it",
        );
    }

    /// The hard bound runs from first registration, so slow discovery cannot
    /// spend it before the first sweep has finished.
    #[test]
    fn the_seed_gate_bound_is_re_armed_at_first_registration() {
        let t = seed_tracker(true);
        t.expire_seed_gate_for_test();
        t.register(&[KvWorkerId::new("http://w1".into(), 0)]);

        t.record_sweep_result(SweepOutcome::TimedOut, 9);
        assert!(
            !t.seed_gate_open(),
            "the bound must be re-armed with the budget, not left expired",
        );
    }

    /// Every rank can go terminal before the boot sweep reports; a 200 in that
    /// window would latch the gate open before the verdict arrives.
    #[test]
    fn seed_gate_holds_until_the_boot_sweep_reports() {
        let t = seed_tracker(true);
        let rank = KvWorkerId::new("http://w1".into(), 0);
        t.register(std::slice::from_ref(&rank));
        t.set(&rank, BootstrapState::Failed);
        assert!(
            t.settled(),
            "condition 3 alone would let this replica serve"
        );
        assert!(
            !t.seed_gate_open(),
            "no sweep has reported yet, so the seed outcome is still unknown",
        );

        t.record_sweep_result(SweepOutcome::FleetCold, 9);
        assert!(t.seed_gate_open(), "a nothing-to-inherit verdict opens it");
    }

    /// A replica that registers no rank never runs a sweep, so there is no
    /// verdict to wait for.
    #[test]
    fn seed_gate_does_not_wait_for_a_sweep_that_cannot_run() {
        assert!(seed_tracker(true).seed_gate_open());
    }

    /// A pool can go ready on a worker that publishes no KV events before the
    /// first KV rank registers; that registration must not un-ready the
    /// serving replica.
    #[test]
    fn first_registration_after_a_served_200_keeps_the_replica_ready() {
        let t = seed_tracker(true);
        t.expire_deadline_for_test();
        assert!(t.admit_ready(), "the construction deadline has expired");

        t.register(&[KvWorkerId::new("http://w1".into(), 0)]);
        assert!(t.settled(), "condition 3 must stay latched");
        assert!(t.seed_gate_open(), "condition 4 must stay latched");
    }
}

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use thiserror::Error;
use uuid::{Variant, Version};

use crate::pd::config::PdProfileV1;
use crate::pd::protocol::FixedBytes;

macro_rules! uuid_identity {
    ($name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name([u8; 16]);

        impl $name {
            pub fn parse(value: &str) -> Result<Self, RoomError> {
                let uuid = uuid::Uuid::parse_str(value).map_err(|_| RoomError::InvalidIdentity)?;
                if uuid.to_string() != value {
                    return Err(RoomError::InvalidIdentity);
                }
                Self::from_bytes(*uuid.as_bytes())
            }

            pub fn from_bytes(bytes: [u8; 16]) -> Result<Self, RoomError> {
                let uuid = uuid::Uuid::from_bytes(bytes);
                if uuid.get_version() != Some(Version::Random)
                    || uuid.get_variant() != Variant::RFC4122
                {
                    return Err(RoomError::InvalidIdentity);
                }
                Ok(Self(bytes))
            }

            pub fn random() -> Self {
                Self(*uuid::Uuid::new_v4().as_bytes())
            }

            pub const fn as_bytes(self) -> [u8; 16] {
                self.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_struct(stringify!($name))
                    .finish_non_exhaustive()
            }
        }
    };
}

uuid_identity!(ProcessEpoch);
uuid_identity!(RegistrationEpoch);
uuid_identity!(AttemptId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RoomKey {
    pub decode_process_epoch: ProcessEpoch,
    pub bootstrap_room: u64,
    pub attempt_id: AttemptId,
}

impl RoomKey {
    pub fn new(
        decode_process_epoch: ProcessEpoch,
        bootstrap_room: u64,
        attempt_id: AttemptId,
    ) -> Result<Self, RoomError> {
        if bootstrap_room > i64::MAX as u64 {
            return Err(RoomError::BootstrapRoom);
        }
        Ok(Self {
            decode_process_epoch,
            bootstrap_room,
            attempt_id,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RoomId {
    pub key: RoomKey,
    pub generation: u64,
}

impl RoomId {
    pub fn new(key: RoomKey, generation: u64) -> Result<Self, RoomError> {
        if generation == 0 {
            return Err(RoomError::Generation);
        }
        Ok(Self { key, generation })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoomSpec {
    pub id: RoomId,
    pub request_digest: FixedBytes<32>,
    pub registration_epoch: RegistrationEpoch,
}

impl RoomSpec {
    pub fn new(
        id: RoomId,
        request_digest: FixedBytes<32>,
        registration_epoch: RegistrationEpoch,
    ) -> Result<Self, RoomError> {
        if request_digest.as_bytes().iter().all(|byte| *byte == 0) {
            return Err(RoomError::Digest);
        }
        Ok(Self {
            id,
            request_digest,
            registration_epoch,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoomRole {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PdReason {
    Success,
    RequestInvalid,
    Unsupported,
    CapacityExhausted,
    ProtocolMismatch,
    PeerUnavailable,
    RendezvousTimeout,
    TransferTimeout,
    TransferFailed,
    AckTimeout,
    Aborted,
    StaleEpoch,
    LocalFatal,
}

impl PdReason {
    pub const fn from_code(value: &str) -> Option<Self> {
        match value.as_bytes() {
            b"PD_SUCCESS" => Some(Self::Success),
            b"PD_REQUEST_INVALID" => Some(Self::RequestInvalid),
            b"PD_UNSUPPORTED" => Some(Self::Unsupported),
            b"PD_CAPACITY_EXHAUSTED" => Some(Self::CapacityExhausted),
            b"PD_PROTOCOL_MISMATCH" => Some(Self::ProtocolMismatch),
            b"PD_PEER_UNAVAILABLE" => Some(Self::PeerUnavailable),
            b"PD_RENDEZVOUS_TIMEOUT" => Some(Self::RendezvousTimeout),
            b"PD_TRANSFER_TIMEOUT" => Some(Self::TransferTimeout),
            b"PD_TRANSFER_FAILED" => Some(Self::TransferFailed),
            b"PD_ACK_TIMEOUT" => Some(Self::AckTimeout),
            b"PD_ABORTED" => Some(Self::Aborted),
            b"PD_STALE_EPOCH" => Some(Self::StaleEpoch),
            b"PD_LOCAL_FATAL" => Some(Self::LocalFatal),
            _ => None,
        }
    }

    pub const fn code(self) -> &'static str {
        match self {
            Self::Success => "PD_SUCCESS",
            Self::RequestInvalid => "PD_REQUEST_INVALID",
            Self::Unsupported => "PD_UNSUPPORTED",
            Self::CapacityExhausted => "PD_CAPACITY_EXHAUSTED",
            Self::ProtocolMismatch => "PD_PROTOCOL_MISMATCH",
            Self::PeerUnavailable => "PD_PEER_UNAVAILABLE",
            Self::RendezvousTimeout => "PD_RENDEZVOUS_TIMEOUT",
            Self::TransferTimeout => "PD_TRANSFER_TIMEOUT",
            Self::TransferFailed => "PD_TRANSFER_FAILED",
            Self::AckTimeout => "PD_ACK_TIMEOUT",
            Self::Aborted => "PD_ABORTED",
            Self::StaleEpoch => "PD_STALE_EPOCH",
            Self::LocalFatal => "PD_LOCAL_FATAL",
        }
    }

    pub const fn retryable(self) -> bool {
        matches!(
            self,
            Self::CapacityExhausted
                | Self::PeerUnavailable
                | Self::RendezvousTimeout
                | Self::TransferTimeout
                | Self::TransferFailed
                | Self::AckTimeout
                | Self::StaleEpoch
                | Self::LocalFatal
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RoomEvent {
    SourceReady,
    TransferSubmitted { plan_digest: FixedBytes<32> },
    TransferTerminal,
    PrepareAccepted { plan_digest: FixedBytes<32> },
    PrepareRejected(PdReason),
    DataReady { plan_digest: FixedBytes<32> },
    TransferComplete { plan_digest: FixedBytes<32> },
    TransferCompleteAck { plan_digest: FixedBytes<32> },
    TransferFailed(PdReason),
    Abort(PdReason),
    AbortReceived(PdReason),
    PeerLost,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoomEffect {
    SendPrepare,
    SendPrepareAccepted,
    SendPrepareRejected(PdReason),
    SubmitTransfer,
    SendDataReady,
    SendTransferComplete,
    SendTransferCompleteAck,
    SendAbort(PdReason),
    SendAbortAck(PdReason),
    NotifyTerminal(PdReason),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoomOutcome {
    Applied(Vec<RoomEffect>),
    Terminal {
        reason: PdReason,
        duplicate: bool,
        effects: Vec<RoomEffect>,
    },
    Rejected(PdReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RoomState {
    Waiting,
    Rendezvoused,
    SourceReady,
    Transferring,
    AwaitingComplete,
    AwaitingAck,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoomSnapshot {
    pub role: RoomRole,
    pub active_rooms: usize,
    pub tombstones: usize,
    pub timers: usize,
    pub terminal_notifications: u64,
    pub states: BTreeMap<RoomState, usize>,
    pub terminal_reasons: BTreeMap<PdReason, u64>,
}

pub trait Clock: Send + Sync {
    fn now_unix_ms(&self) -> u64;
    fn now_monotonic_ms(&self) -> u64;
}

#[derive(Debug)]
pub struct SystemClock {
    monotonic_origin: Instant,
}

impl Default for SystemClock {
    fn default() -> Self {
        Self {
            monotonic_origin: Instant::now(),
        }
    }
}

impl Clock for SystemClock {
    fn now_unix_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or(0)
    }

    fn now_monotonic_ms(&self) -> u64 {
        self.monotonic_origin.elapsed().as_millis() as u64
    }
}

#[derive(Debug)]
pub struct ManualClock {
    now_unix_ms: AtomicU64,
    now_monotonic_ms: AtomicU64,
}

impl ManualClock {
    pub const fn new(now_unix_ms: u64) -> Self {
        Self {
            now_unix_ms: AtomicU64::new(now_unix_ms),
            now_monotonic_ms: AtomicU64::new(now_unix_ms),
        }
    }

    pub fn advance(&self, milliseconds: u64) {
        self.now_unix_ms.fetch_add(milliseconds, Ordering::SeqCst);
        self.now_monotonic_ms
            .fetch_add(milliseconds, Ordering::SeqCst);
    }

    pub fn advance_unix(&self, milliseconds: u64) {
        self.now_unix_ms.fetch_add(milliseconds, Ordering::SeqCst);
    }

    pub fn advance_monotonic(&self, milliseconds: u64) {
        self.now_monotonic_ms
            .fetch_add(milliseconds, Ordering::SeqCst);
    }
}

impl Clock for ManualClock {
    fn now_unix_ms(&self) -> u64 {
        self.now_unix_ms.load(Ordering::SeqCst)
    }

    fn now_monotonic_ms(&self) -> u64 {
        self.now_monotonic_ms.load(Ordering::SeqCst)
    }
}

struct RoomEntry {
    spec: RoomSpec,
    local_arrived: bool,
    peer_arrived: bool,
    state: RoomState,
    plan_digest: Option<FixedBytes<32>>,
    deadline_monotonic_ms: u64,
    seen_events: HashSet<RoomEvent>,
}

#[derive(Clone)]
struct Tombstone {
    reason: PdReason,
    completed_unix_ms: u64,
}

#[derive(Debug, Clone, Copy)]
struct Limits {
    active_rooms: usize,
    tombstones: usize,
    rendezvous_ms: u64,
    transfer_ms: u64,
    ack_ms: u64,
    retention_ms: u64,
}

pub struct RoomTable {
    role: RoomRole,
    process_epoch: ProcessEpoch,
    registration_epoch: RegistrationEpoch,
    limits: Limits,
    clock: Arc<dyn Clock>,
    active: HashMap<RoomId, RoomEntry>,
    tombstones: HashMap<RoomId, Tombstone>,
    tombstone_order: VecDeque<RoomId>,
    terminal_notifications: u64,
    terminal_reasons: BTreeMap<PdReason, u64>,
}

impl RoomTable {
    pub fn new<C>(
        role: RoomRole,
        process_epoch: ProcessEpoch,
        registration_epoch: RegistrationEpoch,
        profile: &PdProfileV1,
        clock: Arc<C>,
    ) -> Result<Self, RoomError>
    where
        C: Clock + 'static,
    {
        let limits = Limits {
            active_rooms: usize::try_from(profile.capacity.active_rooms_per_pair)
                .map_err(|_| RoomError::Profile)?,
            tombstones: usize::try_from(profile.capacity.tombstones_per_pair)
                .map_err(|_| RoomError::Profile)?,
            rendezvous_ms: profile.deadline_ms.room_rendezvous,
            transfer_ms: profile.deadline_ms.native_transfer,
            ack_ms: profile
                .deadline_ms
                .completion_ack
                .min(profile.deadline_ms.abort_ack),
            retention_ms: profile.deadline_ms.tombstone_retention,
        };
        if limits.active_rooms != 32
            || limits.tombstones != 4096
            || limits.rendezvous_ms != 300_000
            || limits.transfer_ms != 60_000
            || limits.ack_ms != 10_000
            || limits.retention_ms != 300_000
        {
            return Err(RoomError::Profile);
        }
        Ok(Self {
            role,
            process_epoch,
            registration_epoch,
            limits,
            clock,
            active: HashMap::new(),
            tombstones: HashMap::new(),
            tombstone_order: VecDeque::new(),
            terminal_notifications: 0,
            terminal_reasons: BTreeMap::new(),
        })
    }

    pub fn observe_local(&mut self, spec: RoomSpec) -> RoomOutcome {
        self.observe(spec, Arrival::Local)
    }

    pub fn observe_peer(&mut self, spec: RoomSpec) -> RoomOutcome {
        self.observe(spec, Arrival::Peer)
    }

    pub fn apply(&mut self, id: RoomId, event: RoomEvent) -> RoomOutcome {
        let now = self.clock.now_monotonic_ms();
        self.prune_expired_tombstones(now);
        if let Some(tombstone) = self.tombstones.get(&id) {
            return terminal_outcome(tombstone.reason, true, Vec::new());
        }
        if !self.active.contains_key(&id) {
            return RoomOutcome::Rejected(PdReason::StaleEpoch);
        }

        let duplicate = self
            .active
            .get_mut(&id)
            .expect("active room checked")
            .seen_events
            .replace(event.clone())
            .is_some();
        if duplicate {
            return RoomOutcome::Applied(Vec::new());
        }

        match event {
            RoomEvent::Abort(reason) => {
                return self.terminalize(
                    id,
                    normalize_failure(reason),
                    vec![
                        RoomEffect::SendAbort(normalize_failure(reason)),
                        RoomEffect::NotifyTerminal(normalize_failure(reason)),
                    ],
                    now,
                );
            }
            RoomEvent::AbortReceived(reason) => {
                return self.terminalize(
                    id,
                    normalize_failure(reason),
                    vec![
                        RoomEffect::SendAbortAck(normalize_failure(reason)),
                        RoomEffect::NotifyTerminal(normalize_failure(reason)),
                    ],
                    now,
                );
            }
            RoomEvent::PeerLost => {
                return self.terminalize(
                    id,
                    PdReason::PeerUnavailable,
                    vec![RoomEffect::NotifyTerminal(PdReason::PeerUnavailable)],
                    now,
                );
            }
            RoomEvent::TransferFailed(reason) | RoomEvent::PrepareRejected(reason) => {
                let reason = normalize_failure(reason);
                return self.terminalize(id, reason, vec![RoomEffect::NotifyTerminal(reason)], now);
            }
            _ => {}
        }

        let result = match self.role {
            RoomRole::Prefill => self.apply_prefill(id, event, now),
            RoomRole::Decode => self.apply_decode(id, event, now),
        };
        result.unwrap_or_else(|| {
            self.terminalize(
                id,
                PdReason::ProtocolMismatch,
                vec![
                    RoomEffect::SendAbort(PdReason::ProtocolMismatch),
                    RoomEffect::NotifyTerminal(PdReason::ProtocolMismatch),
                ],
                now,
            )
        })
    }

    pub fn expire_due(&mut self) -> Vec<(RoomId, PdReason)> {
        let now = self.clock.now_monotonic_ms();
        self.prune_expired_tombstones(now);
        let expired: Vec<(RoomId, PdReason)> = self
            .active
            .iter()
            .filter(|(_, entry)| entry.deadline_monotonic_ms <= now)
            .map(|(id, entry)| {
                let reason = match entry.state {
                    RoomState::Waiting | RoomState::Rendezvoused => PdReason::RendezvousTimeout,
                    RoomState::SourceReady | RoomState::Transferring => PdReason::TransferTimeout,
                    RoomState::AwaitingComplete | RoomState::AwaitingAck => PdReason::AckTimeout,
                };
                (*id, reason)
            })
            .collect();
        for (id, reason) in &expired {
            self.terminalize(
                *id,
                *reason,
                vec![
                    RoomEffect::SendAbort(*reason),
                    RoomEffect::NotifyTerminal(*reason),
                ],
                now,
            );
        }
        expired
    }

    pub fn fail_all(&mut self, reason: PdReason) -> Vec<RoomId> {
        let now = self.clock.now_monotonic_ms();
        let ids: Vec<RoomId> = self.active.keys().copied().collect();
        for id in &ids {
            self.terminalize(
                *id,
                normalize_failure(reason),
                vec![RoomEffect::NotifyTerminal(normalize_failure(reason))],
                now,
            );
        }
        ids
    }

    pub fn snapshot(&self) -> RoomSnapshot {
        let mut states = BTreeMap::new();
        for entry in self.active.values() {
            *states.entry(entry.state).or_insert(0) += 1;
        }
        RoomSnapshot {
            role: self.role,
            active_rooms: self.active.len(),
            tombstones: self.tombstones.len(),
            timers: self.active.len(),
            terminal_notifications: self.terminal_notifications,
            states,
            terminal_reasons: self.terminal_reasons.clone(),
        }
    }

    fn observe(&mut self, spec: RoomSpec, arrival: Arrival) -> RoomOutcome {
        let now = self.clock.now_monotonic_ms();
        self.prune_expired_tombstones(now);
        if spec.id.key.decode_process_epoch != self.process_epoch
            || spec.registration_epoch != self.registration_epoch
        {
            return RoomOutcome::Rejected(PdReason::StaleEpoch);
        }
        if let Some(tombstone) = self.tombstones.get(&spec.id) {
            return terminal_outcome(tombstone.reason, true, Vec::new());
        }
        if self.active.contains_key(&spec.id) {
            if self
                .active
                .get(&spec.id)
                .is_some_and(|entry| entry.spec.request_digest != spec.request_digest)
            {
                return self.terminalize(
                    spec.id,
                    PdReason::ProtocolMismatch,
                    vec![
                        RoomEffect::SendAbort(PdReason::ProtocolMismatch),
                        RoomEffect::NotifyTerminal(PdReason::ProtocolMismatch),
                    ],
                    now,
                );
            }
            return self.mark_arrival(spec.id, arrival);
        }
        if self.has_room_key(spec.id.key) {
            return RoomOutcome::Rejected(PdReason::StaleEpoch);
        }
        if self.active.len() >= self.limits.active_rooms
            || self.tombstones.len() + self.active.len() >= self.limits.tombstones
        {
            return RoomOutcome::Rejected(PdReason::CapacityExhausted);
        }

        let (local_arrived, peer_arrived) = match arrival {
            Arrival::Local => (true, false),
            Arrival::Peer => (false, true),
        };
        self.active.insert(
            spec.id,
            RoomEntry {
                spec,
                local_arrived,
                peer_arrived,
                state: RoomState::Waiting,
                plan_digest: None,
                deadline_monotonic_ms: now.saturating_add(self.limits.rendezvous_ms),
                seen_events: HashSet::new(),
            },
        );
        RoomOutcome::Applied(Vec::new())
    }

    fn mark_arrival(&mut self, id: RoomId, arrival: Arrival) -> RoomOutcome {
        let entry = self.active.get_mut(&id).expect("active room checked");
        match arrival {
            Arrival::Local if entry.local_arrived => return RoomOutcome::Applied(Vec::new()),
            Arrival::Peer if entry.peer_arrived => return RoomOutcome::Applied(Vec::new()),
            Arrival::Local => entry.local_arrived = true,
            Arrival::Peer => entry.peer_arrived = true,
        }
        if entry.local_arrived && entry.peer_arrived {
            entry.state = RoomState::Rendezvoused;
            RoomOutcome::Applied(vec![match self.role {
                RoomRole::Prefill => RoomEffect::SendPrepareAccepted,
                RoomRole::Decode => RoomEffect::SendPrepare,
            }])
        } else {
            RoomOutcome::Applied(Vec::new())
        }
    }

    fn apply_prefill(&mut self, id: RoomId, event: RoomEvent, now: u64) -> Option<RoomOutcome> {
        let entry = self.active.get_mut(&id)?;
        match (entry.state, event) {
            (RoomState::Rendezvoused, RoomEvent::SourceReady) => {
                entry.state = RoomState::SourceReady;
                entry.deadline_monotonic_ms = now.saturating_add(self.limits.transfer_ms);
                Some(RoomOutcome::Applied(vec![RoomEffect::SubmitTransfer]))
            }
            (RoomState::SourceReady, RoomEvent::TransferSubmitted { plan_digest })
                if valid_digest(&plan_digest) =>
            {
                entry.plan_digest = Some(plan_digest);
                entry.state = RoomState::Transferring;
                entry.deadline_monotonic_ms = now.saturating_add(self.limits.transfer_ms);
                Some(RoomOutcome::Applied(Vec::new()))
            }
            (RoomState::Transferring, RoomEvent::TransferTerminal) => {
                entry.state = RoomState::AwaitingComplete;
                entry.deadline_monotonic_ms = now.saturating_add(self.limits.ack_ms);
                Some(RoomOutcome::Applied(vec![RoomEffect::SendDataReady]))
            }
            (RoomState::AwaitingComplete, RoomEvent::TransferComplete { plan_digest })
                if entry.plan_digest == Some(plan_digest) =>
            {
                Some(self.terminalize(
                    id,
                    PdReason::Success,
                    vec![
                        RoomEffect::SendTransferCompleteAck,
                        RoomEffect::NotifyTerminal(PdReason::Success),
                    ],
                    now,
                ))
            }
            _ => None,
        }
    }

    fn apply_decode(&mut self, id: RoomId, event: RoomEvent, now: u64) -> Option<RoomOutcome> {
        let entry = self.active.get_mut(&id)?;
        match (entry.state, event) {
            (RoomState::Rendezvoused, RoomEvent::PrepareAccepted { plan_digest })
                if valid_digest(&plan_digest) =>
            {
                entry.plan_digest = Some(plan_digest);
                entry.state = RoomState::Transferring;
                entry.deadline_monotonic_ms = now.saturating_add(self.limits.transfer_ms);
                Some(RoomOutcome::Applied(Vec::new()))
            }
            (RoomState::Transferring, RoomEvent::DataReady { plan_digest })
                if entry.plan_digest == Some(plan_digest) =>
            {
                entry.state = RoomState::AwaitingAck;
                entry.deadline_monotonic_ms = now.saturating_add(self.limits.ack_ms);
                Some(RoomOutcome::Applied(vec![RoomEffect::SendTransferComplete]))
            }
            (RoomState::AwaitingAck, RoomEvent::TransferCompleteAck { plan_digest })
                if entry.plan_digest == Some(plan_digest) =>
            {
                Some(self.terminalize(
                    id,
                    PdReason::Success,
                    vec![RoomEffect::NotifyTerminal(PdReason::Success)],
                    now,
                ))
            }
            _ => None,
        }
    }

    fn terminalize(
        &mut self,
        id: RoomId,
        reason: PdReason,
        effects: Vec<RoomEffect>,
        now: u64,
    ) -> RoomOutcome {
        if let Some(tombstone) = self.tombstones.get(&id) {
            return terminal_outcome(tombstone.reason, true, Vec::new());
        }
        if self.active.remove(&id).is_none() {
            return RoomOutcome::Rejected(PdReason::StaleEpoch);
        }
        debug_assert!(self.tombstones.len() < self.limits.tombstones);
        self.tombstones.insert(
            id,
            Tombstone {
                reason,
                completed_unix_ms: now,
            },
        );
        self.tombstone_order.push_back(id);
        self.terminal_notifications = self.terminal_notifications.saturating_add(1);
        *self.terminal_reasons.entry(reason).or_insert(0) += 1;
        tracing::debug!(
            role = ?self.role,
            state = "terminal",
            reason = reason.code(),
            generation = id.generation,
            "PD Room reached its first terminal transition"
        );
        terminal_outcome(reason, false, effects)
    }

    fn prune_expired_tombstones(&mut self, now: u64) {
        while let Some(id) = self.tombstone_order.front().copied() {
            let Some(tombstone) = self.tombstones.get(&id) else {
                self.tombstone_order.pop_front();
                continue;
            };
            if now.saturating_sub(tombstone.completed_unix_ms) < self.limits.retention_ms {
                break;
            }
            self.tombstone_order.pop_front();
            self.tombstones.remove(&id);
        }
    }

    fn has_room_key(&self, key: RoomKey) -> bool {
        self.active.keys().any(|id| id.key == key) || self.tombstones.keys().any(|id| id.key == key)
    }
}

#[derive(Debug, Clone, Copy)]
enum Arrival {
    Local,
    Peer,
}

fn valid_digest(digest: &FixedBytes<32>) -> bool {
    digest.as_bytes().iter().any(|byte| *byte != 0)
}

fn normalize_failure(reason: PdReason) -> PdReason {
    if reason == PdReason::Success {
        PdReason::ProtocolMismatch
    } else {
        reason
    }
}

fn terminal_outcome(reason: PdReason, duplicate: bool, effects: Vec<RoomEffect>) -> RoomOutcome {
    RoomOutcome::Terminal {
        reason,
        duplicate,
        effects,
    }
}

#[derive(Debug, Error)]
pub enum RoomError {
    #[error("PD identity must be a lowercase canonical RFC4122 UUIDv4")]
    InvalidIdentity,
    #[error("bootstrap_room must fit the frozen u63 range")]
    BootstrapRoom,
    #[error("room generation must be non-zero")]
    Generation,
    #[error("request digest must be a non-zero 32-byte digest")]
    Digest,
    #[error("room limits do not match the frozen PD profile")]
    Profile,
}

use std::collections::BTreeSet;
use std::sync::{Arc, RwLock};

use thiserror::Error;

use crate::pd::protocol::FixedBytes;
use crate::pd::request::KVPoll;
use crate::pd::room::{Clock, PdReason, ProcessEpoch, RegistrationEpoch, RoomId};
use crate::pd::runtime::RuntimeSnapshot;

pub const PD_REGION_COUNT: usize = 58;
pub const MAX_TRANSPORT_HANDLES: usize = 32;
pub const MAX_TRANSPORT_BATCH: usize = 8;

pub(super) const HANDLE_SLOT_BITS: u32 = 6;
const HANDLE_ROLE_BITS: u32 = 1;
const HANDLE_GENERATION_BITS: u32 = 25;
pub(super) const HANDLE_GENERATION_SHIFT: u32 = HANDLE_SLOT_BITS + HANDLE_ROLE_BITS;
pub(super) const HANDLE_OWNER_SHIFT: u32 = HANDLE_GENERATION_SHIFT + HANDLE_GENERATION_BITS;
const HANDLE_SLOT_MASK: u64 = (1_u64 << HANDLE_SLOT_BITS) - 1;
pub(super) const HANDLE_GENERATION_MASK: u64 = (1_u64 << HANDLE_GENERATION_BITS) - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportHandleRole {
    Sender,
    Receiver,
}

impl TransportHandleRole {
    pub(super) const fn bit(self) -> u64 {
        match self {
            Self::Sender => 0,
            Self::Receiver => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpaqueHandle(pub(super) u64);

impl OpaqueHandle {
    pub const fn raw(self) -> u64 {
        self.0
    }

    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    pub const fn slot(self) -> usize {
        (self.0 & HANDLE_SLOT_MASK) as usize
    }

    pub(super) const fn role(self) -> TransportHandleRole {
        if (self.0 >> HANDLE_SLOT_BITS) & 1 == 0 {
            TransportHandleRole::Sender
        } else {
            TransportHandleRole::Receiver
        }
    }

    pub const fn generation(self) -> u64 {
        (self.0 >> HANDLE_GENERATION_SHIFT) & HANDLE_GENERATION_MASK
    }

    pub(super) const fn owner_tag(self) -> u32 {
        (self.0 >> HANDLE_OWNER_SHIFT) as u32
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SenderCreateInput {
    pub decode_process_epoch: ProcessEpoch,
    pub bootstrap_room: u64,
    pub attempt_id: crate::pd::room::AttemptId,
    pub request_digest: FixedBytes<32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiverCreateInput {
    pub bootstrap_room: u64,
    pub attempt_id: crate::pd::room::AttemptId,
    pub request_digest: FixedBytes<32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SenderChunk {
    pub handle: OpaqueHandle,
    pub transfer_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalEvent {
    pub handle: OpaqueHandle,
    pub reason: PdReason,
    pub first_token_id: Option<i32>,
    pub transfer_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportPollResult {
    pub handle: OpaqueHandle,
    pub status: KVPoll,
    pub reason: PdReason,
    pub retryable: bool,
    pub transfer_bytes: u64,
    pub transfer_latency_ms: u64,
    pub terminal_generation: u64,
    pub first_token_id: Option<i32>,
    pub first_token_consumed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransportRoomContext {
    pub room: RoomId,
    pub request_digest: FixedBytes<32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportSnapshot {
    pub runtime: RuntimeSnapshot,
    pub model_manifest_digest: FixedBytes<32>,
    pub tokenizer_manifest_digest: FixedBytes<32>,
    pub layout_fingerprint: FixedBytes<32>,
    pub expected_bootstrap_host: String,
    pub allowed_bootstrap_ports: BTreeSet<u16>,
    pub accepting_rooms: bool,
    pub active_handles: usize,
    pub result_slots: usize,
    pub abort_generation: u64,
    pub last_abort_reason: Option<PdReason>,
}

#[derive(Clone, Debug)]
pub struct PdReadinessHandle {
    pub(super) shared: Arc<RwLock<TransportSnapshot>>,
}

impl PdReadinessHandle {
    pub fn snapshot(&self) -> TransportSnapshot {
        self.shared
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TransportError {
    #[error("PD transport is not PairReady")]
    NotReady,
    #[error("PD transport batch cardinality is invalid")]
    InvalidBatch,
    #[error("PD transport handle capacity is exhausted")]
    CapacityExhausted,
    #[error("PD transport handle is stale or belongs to another epoch")]
    StaleHandle,
    #[error("PD transport handle has the wrong role")]
    WrongRole,
    #[error("PD transport transition is invalid")]
    InvalidTransition,
    #[error("PD Room reached a typed terminal: {0:?}")]
    Room(PdReason),
    #[error("PD peer session failed: {0:?}")]
    Peer(PdReason),
    #[error("PD transport entered a local fatal state: {0:?}")]
    LocalFatal(PdReason),
}

impl TransportError {
    pub const fn reason(&self) -> PdReason {
        match self {
            Self::NotReady => PdReason::PeerUnavailable,
            Self::InvalidBatch => PdReason::RequestInvalid,
            Self::CapacityExhausted => PdReason::CapacityExhausted,
            Self::StaleHandle => PdReason::StaleEpoch,
            Self::WrongRole => PdReason::Unsupported,
            Self::InvalidTransition => PdReason::ProtocolMismatch,
            Self::Room(reason) | Self::Peer(reason) | Self::LocalFatal(reason) => *reason,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HandleState {
    Created,
    WaitingForInput,
    Transferring,
    Terminal,
}

pub(super) struct TerminalResult {
    pub(super) reason: PdReason,
    pub(super) transfer_bytes: u64,
    pub(super) transfer_latency_ms: u64,
    pub(super) first_token_id: Option<i32>,
    pub(super) first_token_consumed: bool,
}

pub(super) struct HandleEntry {
    pub(super) role: TransportHandleRole,
    pub(super) room: RoomId,
    pub(super) request_digest: FixedBytes<32>,
    pub(super) process_epoch: ProcessEpoch,
    pub(super) registration_epoch: RegistrationEpoch,
    pub(super) state: HandleState,
    pub(super) created_monotonic_ms: u64,
    pub(super) transfer_bytes: u64,
    pub(super) terminal: Option<TerminalResult>,
}

#[derive(Default)]
pub(super) struct HandleSlot {
    pub(super) generation: u64,
    pub(super) entry: Option<HandleEntry>,
}

pub(super) struct SharedClock(pub(super) Arc<dyn Clock>);

impl Clock for SharedClock {
    fn now_unix_ms(&self) -> u64 {
        self.0.now_unix_ms()
    }

    fn now_monotonic_ms(&self) -> u64 {
        self.0.now_monotonic_ms()
    }
}

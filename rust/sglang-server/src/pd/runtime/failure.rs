use crate::mooncake::{EngineError, NativeOperation};
use crate::pd::buffer::BufferError;
use crate::pd::room::PdReason;
use crate::pd::transport::TransportError;

use super::RuntimeError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureScope {
    Request,
    Room,
    PeerSession,
    LocalFatal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FailureClass {
    pub scope: FailureScope,
    pub reason: PdReason,
}

impl FailureClass {
    pub const fn new(scope: FailureScope, reason: PdReason) -> Self {
        Self { scope, reason }
    }

    pub const fn for_runtime(error: &RuntimeError) -> Self {
        match error {
            RuntimeError::Worker | RuntimeError::Configuration | RuntimeError::Profile(_) => {
                Self::new(FailureScope::LocalFatal, PdReason::LocalFatal)
            }
            RuntimeError::Compatibility
            | RuntimeError::PeerRejected
            | RuntimeError::UnexpectedMessage
            | RuntimeError::Frame(_)
            | RuntimeError::Crypto(_) => {
                Self::new(FailureScope::PeerSession, PdReason::ProtocolMismatch)
            }
            RuntimeError::Timeout | RuntimeError::Session(_) | RuntimeError::PeerDraining => {
                Self::new(FailureScope::PeerSession, PdReason::PeerUnavailable)
            }
            RuntimeError::Bootstrap(reason) => match reason {
                PdReason::LocalFatal => Self::new(FailureScope::LocalFatal, PdReason::LocalFatal),
                PdReason::PeerUnavailable => {
                    Self::new(FailureScope::PeerSession, PdReason::PeerUnavailable)
                }
                _ => Self::new(FailureScope::PeerSession, *reason),
            },
        }
    }

    pub const fn for_transport(error: &TransportError) -> Self {
        match error {
            TransportError::InvalidBatch => {
                Self::new(FailureScope::Request, PdReason::RequestInvalid)
            }
            TransportError::CapacityExhausted => {
                Self::new(FailureScope::Request, PdReason::CapacityExhausted)
            }
            TransportError::StaleHandle => Self::new(FailureScope::Request, PdReason::StaleEpoch),
            TransportError::WrongRole => Self::new(FailureScope::Request, PdReason::Unsupported),
            TransportError::InvalidTransition => {
                Self::new(FailureScope::Room, PdReason::ProtocolMismatch)
            }
            TransportError::NotReady => {
                Self::new(FailureScope::PeerSession, PdReason::PeerUnavailable)
            }
            TransportError::Room(reason) => Self::new(FailureScope::Room, *reason),
            TransportError::Peer(reason) => Self::new(FailureScope::PeerSession, *reason),
            TransportError::LocalFatal(reason) => Self::new(FailureScope::LocalFatal, *reason),
        }
    }

    pub const fn for_buffer(error: &BufferError) -> Self {
        match error {
            BufferError::InvalidDescriptor { .. }
            | BufferError::PlanLimit { .. }
            | BufferError::PlanMismatch { .. } => {
                Self::new(FailureScope::Request, PdReason::RequestInvalid)
            }
            BufferError::CapacityExhausted { .. } | BufferError::WorkerFull => {
                Self::new(FailureScope::Request, PdReason::CapacityExhausted)
            }
            BufferError::StaleRegistration | BufferError::StaleHandle => {
                Self::new(FailureScope::Request, PdReason::StaleEpoch)
            }
            BufferError::DataRecord { .. }
            | BufferError::SourceFence
            | BufferError::VisibilityFence
            | BufferError::NativeTransfer => {
                Self::new(FailureScope::Room, PdReason::TransferFailed)
            }
            BufferError::Deadline => Self::new(FailureScope::Room, PdReason::TransferTimeout),
            BufferError::Registration { .. }
            | BufferError::Unregistration { .. }
            | BufferError::TableInUse { .. }
            | BufferError::ResourceOwned
            | BufferError::InvalidTransition => {
                Self::new(FailureScope::LocalFatal, PdReason::LocalFatal)
            }
        }
    }

    pub const fn for_engine(error: &EngineError) -> Self {
        match error {
            EngineError::QueueFull | EngineError::InFlightLimit { .. } => {
                Self::new(FailureScope::Request, PdReason::CapacityExhausted)
            }
            EngineError::BatchNotTerminal { .. } => {
                Self::new(FailureScope::Room, PdReason::TransferTimeout)
            }
            EngineError::Native { operation, .. } => match operation {
                NativeOperation::OpenPeer | NativeOperation::ClosePeer => {
                    Self::new(FailureScope::PeerSession, PdReason::PeerUnavailable)
                }
                NativeOperation::AllocateBatch
                | NativeOperation::SubmitBatch
                | NativeOperation::Poll
                | NativeOperation::FreeBatch => {
                    Self::new(FailureScope::Room, PdReason::TransferFailed)
                }
                _ => Self::new(FailureScope::LocalFatal, PdReason::LocalFatal),
            },
            EngineError::UnsupportedGpu { .. }
            | EngineError::InvalidDescriptor { .. }
            | EngineError::RangeOverflow { .. }
            | EngineError::RangeOutOfBounds { .. }
            | EngineError::WorkerClosed
            | EngineError::ResponseTimeout { .. }
            | EngineError::ResourceClosed { .. }
            | EngineError::NativeHandle { .. }
            | EngineError::LibraryMissing { .. }
            | EngineError::ManifestMissing { .. }
            | EngineError::ArtifactMismatch { .. }
            | EngineError::AbiMismatch { .. }
            | EngineError::SymbolMissing { .. }
            | EngineError::LoaderFailure { .. }
            | EngineError::ForkDetected { .. }
            | EngineError::WorkerStart { .. }
            | EngineError::Rollback { .. }
            | EngineError::LockPoisoned => {
                Self::new(FailureScope::LocalFatal, PdReason::LocalFatal)
            }
        }
    }

    pub const fn for_quarantine(hard_deadline_expired: bool) -> Self {
        if hard_deadline_expired {
            Self::new(FailureScope::LocalFatal, PdReason::LocalFatal)
        } else {
            Self::new(FailureScope::Room, PdReason::TransferTimeout)
        }
    }
}

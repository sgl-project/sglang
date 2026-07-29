use sha2::{Digest, Sha256};

use super::*;

pub(super) fn validate_batch(length: usize) -> Result<(), TransportError> {
    if (1..=MAX_TRANSPORT_BATCH).contains(&length) {
        Ok(())
    } else {
        Err(TransportError::InvalidBatch)
    }
}

pub(super) fn expect_applied(outcome: RoomOutcome) -> Result<(), TransportError> {
    match outcome {
        RoomOutcome::Applied(_) => Ok(()),
        RoomOutcome::Terminal { reason, .. } | RoomOutcome::Rejected(reason) => {
            Err(error_for_reason(reason))
        }
    }
}

pub(super) fn error_for_reason(reason: PdReason) -> TransportError {
    match reason {
        PdReason::CapacityExhausted => TransportError::CapacityExhausted,
        PdReason::StaleEpoch => TransportError::StaleHandle,
        PdReason::PeerUnavailable => TransportError::NotReady,
        PdReason::LocalFatal => TransportError::LocalFatal(reason),
        PdReason::RendezvousTimeout
        | PdReason::TransferTimeout
        | PdReason::TransferFailed
        | PdReason::AckTimeout
        | PdReason::Aborted => TransportError::Room(reason),
        PdReason::Success
        | PdReason::RequestInvalid
        | PdReason::Unsupported
        | PdReason::ProtocolMismatch => TransportError::InvalidTransition,
    }
}

pub(super) fn owner_tag(identity: &RuntimeIdentity) -> u32 {
    let mut digest = Sha256::new();
    digest.update(identity.process_epoch.as_bytes());
    digest.update(identity.registration_epoch.as_bytes());
    digest.update([match identity.role {
        Role::Prefill => 0,
        Role::Decode => 1,
    }]);
    u32::from_be_bytes(digest.finalize()[..4].try_into().expect("SHA-256 prefix"))
}

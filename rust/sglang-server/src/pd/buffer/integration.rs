use crate::pd::buffer::{BufferError, DataPlaneEffect, DataPlaneIdentity};
use crate::pd::protocol::FixedBytes;
use crate::pd::room::{RoomEvent, RoomOutcome, RoomTable};

pub fn apply_prefill_data_effect(
    rooms: &mut RoomTable,
    effect: DataPlaneEffect,
) -> Result<Vec<RoomOutcome>, BufferError> {
    match effect {
        DataPlaneEffect::DataReady { identity } => {
            let digest = digest(identity);
            Ok(vec![
                rooms.apply(
                    identity.room,
                    RoomEvent::TransferSubmitted {
                        plan_digest: digest,
                    },
                ),
                rooms.apply(identity.room, RoomEvent::TransferTerminal),
            ])
        }
        DataPlaneEffect::TransferFailed {
            identity, reason, ..
        }
        | DataPlaneEffect::Quarantined {
            identity, reason, ..
        } => Ok(vec![
            rooms.apply(identity.room, RoomEvent::TransferFailed(reason)),
        ]),
        DataPlaneEffect::TransferComplete { .. } => Err(BufferError::InvalidTransition),
    }
}

pub fn apply_prepare_accepted(rooms: &mut RoomTable, identity: DataPlaneIdentity) -> RoomOutcome {
    rooms.apply(
        identity.room,
        RoomEvent::PrepareAccepted {
            plan_digest: digest(identity),
        },
    )
}

pub fn apply_decode_data_effect(
    rooms: &mut RoomTable,
    effect: DataPlaneEffect,
) -> Result<RoomOutcome, BufferError> {
    match effect {
        DataPlaneEffect::TransferComplete { identity } => Ok(rooms.apply(
            identity.room,
            RoomEvent::DataReady {
                plan_digest: digest(identity),
            },
        )),
        DataPlaneEffect::TransferFailed {
            identity, reason, ..
        }
        | DataPlaneEffect::Quarantined {
            identity, reason, ..
        } => Ok(rooms.apply(identity.room, RoomEvent::TransferFailed(reason))),
        DataPlaneEffect::DataReady { .. } => Err(BufferError::InvalidTransition),
    }
}

pub fn apply_decode_ack(rooms: &mut RoomTable, identity: DataPlaneIdentity) -> RoomOutcome {
    rooms.apply(
        identity.room,
        RoomEvent::TransferCompleteAck {
            plan_digest: digest(identity),
        },
    )
}

fn digest(identity: DataPlaneIdentity) -> FixedBytes<32> {
    FixedBytes::new(*identity.transfer_plan_digest.as_bytes())
}

use std::collections::BTreeSet;
use std::sync::Arc;

use sglang_server::pd::config::PdProfileV1;
use sglang_server::pd::protocol::{
    DestinationBlock, FixedBytes, KvBlock, PrepareAccepted, PrepareRoom, Role, RoomFields,
};
use sglang_server::pd::room::{
    AttemptId, PdReason, ProcessEpoch, RegistrationEpoch, RoomId, RoomKey, RoomSpec,
};
use sglang_server::pd::runtime::RuntimeIdentity;

#[derive(Clone)]
pub(super) struct RoomContext {
    pub(super) spec: RoomSpec,
    pub(super) wire: RoomFields,
    pub(super) plan_digest: FixedBytes<32>,
}

pub(super) fn context(
    process: ProcessEpoch,
    registration: RegistrationEpoch,
    room_number: u64,
    attempt_index: u64,
) -> Result<RoomContext, String> {
    let attempt = attempt(attempt_index)?;
    let key = RoomKey::new(process, room_number, attempt).map_err(room_error)?;
    let id = RoomId::new(key, 1).map_err(room_error)?;
    let request_digest = digest(0xa1_u8.wrapping_add(room_number as u8));
    let spec = RoomSpec::new(id, request_digest, registration).map_err(room_error)?;
    Ok(RoomContext {
        spec,
        wire: RoomFields {
            decode_process_epoch: FixedBytes::new(process.as_bytes()),
            bootstrap_room: room_number,
            attempt_id: FixedBytes::new(attempt.as_bytes()),
            generation: 1,
            request_contract_digest: request_digest,
        },
        plan_digest: digest(0xb1_u8.wrapping_add(room_number as u8)),
    })
}

pub(super) fn context_from_prepare(
    prepare: &PrepareRoom,
    registration: RegistrationEpoch,
) -> Result<RoomContext, String> {
    let process = ProcessEpoch::from_bytes(prepare.room.decode_process_epoch.into_array())
        .map_err(room_error)?;
    let attempt =
        AttemptId::from_bytes(prepare.room.attempt_id.into_array()).map_err(room_error)?;
    let key = RoomKey::new(process, prepare.room.bootstrap_room, attempt).map_err(room_error)?;
    let id = RoomId::new(key, prepare.room.generation).map_err(room_error)?;
    let spec = RoomSpec::new(id, prepare.room.request_contract_digest, registration)
        .map_err(room_error)?;
    Ok(RoomContext {
        spec,
        wire: prepare.room.clone(),
        plan_digest: digest(0xb1_u8.wrapping_add(prepare.room.bootstrap_room as u8)),
    })
}

pub(super) fn prepare_payload(
    context: &RoomContext,
    destination_registration_epoch: FixedBytes<16>,
) -> PrepareRoom {
    PrepareRoom {
        room: context.wire.clone(),
        destination_registration_epoch,
        destination_blocks: vec![DestinationBlock {
            region_id: 1,
            destination_page: context.spec.id.key.bootstrap_room as u32,
            byte_offset: 0,
            byte_length: 131_072,
        }],
        destination_aux_slot: (context.spec.id.key.bootstrap_room % 31 + 1) as u16,
        destination_completion_slot: (context.spec.id.key.bootstrap_room % 31 + 1) as u16,
        valid_token_count: 64,
        chunk_sequence: 0,
        chunk_count: 1,
        is_last_chunk: true,
    }
}

pub(super) fn accepted_payload(
    context: &RoomContext,
    identity: &RuntimeIdentity,
    destination_registration_epoch: FixedBytes<16>,
) -> PrepareAccepted {
    PrepareAccepted {
        room: context.wire.clone(),
        source_registration_epoch: FixedBytes::new(identity.registration_epoch.as_bytes()),
        destination_registration_epoch,
        kv_blocks: vec![KvBlock {
            region_id: 1,
            source_page: context.spec.id.key.bootstrap_room as u32,
            destination_page: context.spec.id.key.bootstrap_room as u32,
            byte_offset: 0,
            byte_length: 131_072,
        }],
        source_aux_slot: (context.spec.id.key.bootstrap_room % 31 + 1) as u16,
        destination_aux_slot: (context.spec.id.key.bootstrap_room % 31 + 1) as u16,
        source_completion_slot: (context.spec.id.key.bootstrap_room % 31 + 1) as u16,
        destination_completion_slot: (context.spec.id.key.bootstrap_room % 31 + 1) as u16,
        valid_token_count: 64,
        chunk_sequence: 0,
        chunk_count: 1,
        is_last_chunk: true,
        transfer_plan_digest: context.plan_digest,
    }
}

pub(super) fn identity(role: Role) -> Result<RuntimeIdentity, String> {
    RuntimeIdentity::new(
        role,
        ProcessEpoch::random(),
        RegistrationEpoch::random(),
        digest(0x11),
        digest(0x22),
        digest(0x33),
        digest(0x44),
        "127.0.0.1".into(),
        BTreeSet::from([19000]),
        Arc::new(PdProfileV1::load_embedded().map_err(|error| error.to_string())?),
    )
    .map_err(|error| error.to_string())
}

pub(super) fn parse_reason(value: &str) -> Result<PdReason, String> {
    Ok(match value {
        "PD_PROTOCOL_MISMATCH" => PdReason::ProtocolMismatch,
        "PD_RENDEZVOUS_TIMEOUT" => PdReason::RendezvousTimeout,
        "PD_STALE_EPOCH" => PdReason::StaleEpoch,
        "PD_TRANSFER_FAILED" => PdReason::TransferFailed,
        "PD_PEER_UNAVAILABLE" => PdReason::PeerUnavailable,
        "PD_ABORTED" => PdReason::Aborted,
        _ => return Err("unexpected typed PD reason".into()),
    })
}

pub(super) fn reason(reason: PdReason) -> String {
    reason.code().into()
}

pub(super) fn room_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn attempt(index: u64) -> Result<AttemptId, String> {
    let mut bytes = (index as u128).to_be_bytes();
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    AttemptId::from_bytes(bytes).map_err(room_error)
}

fn digest(byte: u8) -> FixedBytes<32> {
    FixedBytes::new([byte.max(1); 32])
}

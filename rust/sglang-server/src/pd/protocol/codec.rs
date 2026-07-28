use std::collections::BTreeSet;
use std::io::Cursor;

use hmac::{Hmac, Mac};
use rmpv::Value;
use sha2::Sha256;
use thiserror::Error;

use crate::pd::config::MAX_CONTROL_PAYLOAD_BYTES;
use crate::pd::protocol::types::{ControlPayload, Direction, MessageKind, PayloadError};

pub const HEADER_BYTES: usize = 32;
pub const TAG_BYTES: usize = 32;

const MAGIC: &[u8; 4] = b"SGPD";
const SCHEMA_MAJOR: u16 = 1;
const SCHEMA_MINOR: u16 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    pub kind: MessageKind,
    pub payload_len: u32,
    pub sequence: u64,
    pub deadline_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedFrame {
    pub header: FrameHeader,
    pub direction: Direction,
    pub payload: ControlPayload,
}

pub struct FrameCodec;

impl FrameCodec {
    pub fn encode(
        kind: MessageKind,
        direction: Direction,
        sequence: u64,
        deadline_unix_ms: u64,
        payload: &ControlPayload,
        key: &[u8; 32],
    ) -> Result<Vec<u8>, FrameError> {
        if payload.kind() != kind {
            return Err(FrameError::PayloadKind);
        }
        if !kind.allows(direction) {
            return Err(FrameError::Direction);
        }
        if sequence == 0 {
            return Err(FrameError::Sequence);
        }
        if deadline_unix_ms == 0 {
            return Err(FrameError::Deadline);
        }

        let payload_bytes = encode_payload(payload)?;
        if payload_bytes.len() > MAX_CONTROL_PAYLOAD_BYTES {
            return Err(FrameError::PayloadTooLarge);
        }
        let payload_len =
            u32::try_from(payload_bytes.len()).map_err(|_| FrameError::PayloadTooLarge)?;
        let mut frame = Vec::with_capacity(HEADER_BYTES + payload_bytes.len() + TAG_BYTES);
        frame.extend_from_slice(MAGIC);
        frame.extend_from_slice(&SCHEMA_MAJOR.to_be_bytes());
        frame.extend_from_slice(&SCHEMA_MINOR.to_be_bytes());
        frame.extend_from_slice(&(kind as u16).to_be_bytes());
        frame.extend_from_slice(&0_u16.to_be_bytes());
        frame.extend_from_slice(&payload_len.to_be_bytes());
        frame.extend_from_slice(&sequence.to_be_bytes());
        frame.extend_from_slice(&deadline_unix_ms.to_be_bytes());
        frame.extend_from_slice(&payload_bytes);
        let mut mac = Hmac::<Sha256>::new_from_slice(key).map_err(|_| FrameError::Auth)?;
        mac.update(&frame);
        frame.extend_from_slice(&mac.finalize().into_bytes());
        Ok(frame)
    }

    pub fn decode(
        frame: &[u8],
        direction: Direction,
        expected_sequence: u64,
        now_unix_ms: u64,
        key: &[u8; 32],
    ) -> Result<DecodedFrame, FrameError> {
        if frame.len() < HEADER_BYTES + TAG_BYTES {
            return Err(FrameError::Truncated);
        }
        if &frame[..4] != MAGIC {
            return Err(FrameError::Magic);
        }
        let major = read_u16(frame, 4)?;
        let minor = read_u16(frame, 6)?;
        if major != SCHEMA_MAJOR || minor != SCHEMA_MINOR {
            return Err(FrameError::Version);
        }
        let kind = MessageKind::try_from(read_u16(frame, 8)?)?;
        if read_u16(frame, 10)? != 0 {
            return Err(FrameError::Flags);
        }
        let payload_len = read_u32(frame, 12)?;
        if payload_len as usize > MAX_CONTROL_PAYLOAD_BYTES {
            return Err(FrameError::PayloadTooLarge);
        }
        let sequence = read_u64(frame, 16)?;
        let deadline_unix_ms = read_u64(frame, 24)?;
        let authenticated_len = HEADER_BYTES
            .checked_add(payload_len as usize)
            .ok_or(FrameError::PayloadTooLarge)?;
        let expected_len = authenticated_len
            .checked_add(TAG_BYTES)
            .ok_or(FrameError::PayloadTooLarge)?;
        if frame.len() != expected_len {
            return Err(FrameError::Length);
        }

        let mut mac = Hmac::<Sha256>::new_from_slice(key).map_err(|_| FrameError::Auth)?;
        mac.update(&frame[..authenticated_len]);
        mac.verify_slice(&frame[authenticated_len..])
            .map_err(|_| FrameError::Auth)?;

        if !kind.allows(direction) {
            return Err(FrameError::Direction);
        }
        if expected_sequence == 0 || sequence != expected_sequence {
            return Err(FrameError::Sequence);
        }
        if deadline_unix_ms < now_unix_ms {
            return Err(FrameError::Deadline);
        }

        let payload_bytes = &frame[HEADER_BYTES..authenticated_len];
        let payload = decode_payload(kind, payload_bytes)?;
        Ok(DecodedFrame {
            header: FrameHeader {
                kind,
                payload_len,
                sequence,
                deadline_unix_ms,
            },
            direction,
            payload,
        })
    }
}

fn encode_payload(payload: &ControlPayload) -> Result<Vec<u8>, FrameError> {
    payload.validate()?;
    let mut value = payload.to_value()?;
    canonicalize_value(&mut value)?;
    let mut bytes = Vec::new();
    rmpv::encode::write_value(&mut bytes, &value).map_err(FrameError::Encode)?;
    Ok(bytes)
}

fn decode_payload(kind: MessageKind, bytes: &[u8]) -> Result<ControlPayload, FrameError> {
    let mut cursor = Cursor::new(bytes);
    let value = rmpv::decode::read_value(&mut cursor).map_err(FrameError::Decode)?;
    if cursor.position() as usize != bytes.len() {
        return Err(FrameError::TrailingPayload);
    }
    validate_wire_value(&value)?;
    validate_schema(kind, &value)?;
    let payload = ControlPayload::from_value(kind, value)?;
    let canonical = encode_payload(&payload)?;
    if canonical != bytes {
        return Err(FrameError::NonCanonical);
    }
    Ok(payload)
}

fn canonicalize_value(value: &mut Value) -> Result<(), FrameError> {
    match value {
        Value::Boolean(_) | Value::Integer(_) | Value::Binary(_) => Ok(()),
        Value::String(string) => {
            if string.as_str().is_none() {
                return Err(FrameError::InvalidUtf8);
            }
            Ok(())
        }
        Value::Array(items) => {
            for item in items {
                canonicalize_value(item)?;
            }
            Ok(())
        }
        Value::Map(entries) => {
            for (key, value) in entries.iter_mut() {
                if key.as_str().is_none() {
                    return Err(FrameError::NonStringKey);
                }
                canonicalize_value(value)?;
            }
            entries.sort_unstable_by(|left, right| {
                left.0
                    .as_str()
                    .expect("validated string key")
                    .as_bytes()
                    .cmp(right.0.as_str().expect("validated string key").as_bytes())
            });
            if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                return Err(FrameError::DuplicateKey);
            }
            Ok(())
        }
        Value::Nil | Value::F32(_) | Value::F64(_) | Value::Ext(_, _) => {
            Err(FrameError::UnsupportedType)
        }
    }
}

fn validate_wire_value(value: &Value) -> Result<(), FrameError> {
    match value {
        Value::Boolean(_) | Value::Integer(_) | Value::Binary(_) => Ok(()),
        Value::String(string) => {
            if string.as_str().is_none() {
                return Err(FrameError::InvalidUtf8);
            }
            Ok(())
        }
        Value::Array(items) => {
            for item in items {
                validate_wire_value(item)?;
            }
            Ok(())
        }
        Value::Map(entries) => {
            let mut previous: Option<&[u8]> = None;
            for (key, value) in entries {
                let key = key.as_str().ok_or(FrameError::NonStringKey)?;
                let key_bytes = key.as_bytes();
                if let Some(previous) = previous {
                    if previous == key_bytes {
                        return Err(FrameError::DuplicateKey);
                    }
                    if previous > key_bytes {
                        return Err(FrameError::KeyOrder);
                    }
                }
                previous = Some(key_bytes);
                validate_wire_value(value)?;
            }
            Ok(())
        }
        Value::Nil | Value::F32(_) | Value::F64(_) | Value::Ext(_, _) => {
            Err(FrameError::UnsupportedType)
        }
    }
}

fn validate_schema(kind: MessageKind, value: &Value) -> Result<(), FrameError> {
    let Value::Map(entries) = value else {
        return Err(FrameError::PayloadMap);
    };
    let actual: BTreeSet<&str> = entries
        .iter()
        .map(|(key, _)| key.as_str().ok_or(FrameError::NonStringKey))
        .collect::<Result<_, _>>()?;
    let expected: BTreeSet<&str> = schema_keys(kind).iter().copied().collect();
    if actual != expected || entries.len() != expected.len() {
        return Err(FrameError::FieldSet);
    }
    Ok(())
}

fn schema_keys(kind: MessageKind) -> &'static [&'static str] {
    const HELLO: &[&str] = &[
        "role",
        "rank",
        "process_epoch",
        "gpu",
        "tp",
        "pp",
        "dp",
        "capabilities",
        "profile_digest",
        "model_manifest_digest",
        "tokenizer_manifest_digest",
        "layout_fingerprint",
        "native_abi_digest",
        "psk_id",
        "nonce",
    ];
    const SERVER_HELLO: &[&str] = &[
        "role",
        "rank",
        "process_epoch",
        "gpu",
        "tp",
        "pp",
        "dp",
        "capabilities",
        "profile_digest",
        "model_manifest_digest",
        "tokenizer_manifest_digest",
        "layout_fingerprint",
        "native_abi_digest",
        "psk_id",
        "nonce",
        "client_hello_hash",
        "accepted",
        "reason",
    ];
    const TRANSCRIPT: &[&str] = &["transcript_hash"];
    const REGISTER: &[&str] = &[
        "registration_epoch",
        "layout_fingerprint",
        "mooncake_host",
        "mooncake_port",
        "regions",
    ];
    const REGISTER_ACK: &[&str] = &["registration_epoch", "accepted", "reason"];
    const PROBE_READY: &[&str] = &[
        "registration_epoch",
        "probe_generation",
        "aux_slot",
        "probe_data",
    ];
    const PROBE_ACK: &[&str] = &[
        "registration_epoch",
        "probe_generation",
        "aux_slot",
        "accepted",
        "reason",
    ];
    const PREPARE_ROOM: &[&str] = &[
        "decode_process_epoch",
        "bootstrap_room",
        "attempt_id",
        "generation",
        "request_contract_digest",
        "destination_registration_epoch",
        "destination_blocks",
        "destination_aux_slot",
        "destination_completion_slot",
        "valid_token_count",
        "chunk_sequence",
        "chunk_count",
        "is_last_chunk",
    ];
    const PREPARE_ACCEPTED: &[&str] = &[
        "decode_process_epoch",
        "bootstrap_room",
        "attempt_id",
        "generation",
        "request_contract_digest",
        "source_registration_epoch",
        "destination_registration_epoch",
        "kv_blocks",
        "source_aux_slot",
        "destination_aux_slot",
        "source_completion_slot",
        "destination_completion_slot",
        "valid_token_count",
        "chunk_sequence",
        "chunk_count",
        "is_last_chunk",
        "transfer_plan_digest",
    ];
    const REJECTED: &[&str] = &[
        "decode_process_epoch",
        "bootstrap_room",
        "attempt_id",
        "generation",
        "request_contract_digest",
        "reason",
    ];
    const PLANNED: &[&str] = &[
        "decode_process_epoch",
        "bootstrap_room",
        "attempt_id",
        "generation",
        "request_contract_digest",
        "transfer_plan_digest",
    ];
    const TERMINAL: &[&str] = &[
        "decode_process_epoch",
        "bootstrap_room",
        "attempt_id",
        "generation",
        "request_contract_digest",
        "transfer_plan_digest",
        "reason",
    ];
    const PING: &[&str] = &["ping_id"];
    const DRAIN: &[&str] = &["drain_generation"];

    match kind {
        MessageKind::ClientHello => HELLO,
        MessageKind::ServerHello => SERVER_HELLO,
        MessageKind::SessionReady | MessageKind::SessionReadyAck => TRANSCRIPT,
        MessageKind::RegisterRegions => REGISTER,
        MessageKind::RegisterRegionsAck => REGISTER_ACK,
        MessageKind::ProbeReady => PROBE_READY,
        MessageKind::ProbeAck => PROBE_ACK,
        MessageKind::PrepareRoom => PREPARE_ROOM,
        MessageKind::PrepareAccepted => PREPARE_ACCEPTED,
        MessageKind::PrepareRejected => REJECTED,
        MessageKind::DataReady
        | MessageKind::TransferComplete
        | MessageKind::TransferCompleteAck => PLANNED,
        MessageKind::TransferFailed | MessageKind::Abort | MessageKind::AbortAck => TERMINAL,
        MessageKind::Ping | MessageKind::Pong => PING,
        MessageKind::GoAway | MessageKind::GoAwayAck => DRAIN,
    }
}

fn read_u16(frame: &[u8], offset: usize) -> Result<u16, FrameError> {
    let bytes = frame
        .get(offset..offset + 2)
        .ok_or(FrameError::Truncated)?
        .try_into()
        .map_err(|_| FrameError::Truncated)?;
    Ok(u16::from_be_bytes(bytes))
}

fn read_u32(frame: &[u8], offset: usize) -> Result<u32, FrameError> {
    let bytes = frame
        .get(offset..offset + 4)
        .ok_or(FrameError::Truncated)?
        .try_into()
        .map_err(|_| FrameError::Truncated)?;
    Ok(u32::from_be_bytes(bytes))
}

fn read_u64(frame: &[u8], offset: usize) -> Result<u64, FrameError> {
    let bytes = frame
        .get(offset..offset + 8)
        .ok_or(FrameError::Truncated)?
        .try_into()
        .map_err(|_| FrameError::Truncated)?;
    Ok(u64::from_be_bytes(bytes))
}

#[derive(Debug, Error)]
pub enum FrameError {
    #[error("PD control frame is truncated")]
    Truncated,
    #[error("PD control frame magic does not match")]
    Magic,
    #[error("PD control frame requires exact schema 1.0")]
    Version,
    #[error("PD control frame flags must be zero")]
    Flags,
    #[error("PD control payload exceeds 524288 bytes")]
    PayloadTooLarge,
    #[error("PD control frame length does not match its header")]
    Length,
    #[error("PD control authentication failed")]
    Auth,
    #[error("PD control kind is not valid in this direction")]
    Direction,
    #[error("PD control sequence is duplicate, stale, or non-contiguous")]
    Sequence,
    #[error("PD control deadline is missing or expired")]
    Deadline,
    #[error("PD control payload kind does not match its frame kind")]
    PayloadKind,
    #[error("PD control payload must be a map")]
    PayloadMap,
    #[error("PD control payload contains a non-string map key")]
    NonStringKey,
    #[error("PD control payload contains duplicate map keys")]
    DuplicateKey,
    #[error("PD control payload map keys are not in canonical order")]
    KeyOrder,
    #[error("PD control payload contains invalid UTF-8")]
    InvalidUtf8,
    #[error("PD control payload contains a forbidden MessagePack type")]
    UnsupportedType,
    #[error("PD control payload field set does not match its kind")]
    FieldSet,
    #[error("PD control payload is not canonically encoded")]
    NonCanonical,
    #[error("PD control payload contains trailing bytes")]
    TrailingPayload,
    #[error("PD control kind is unknown: {0}")]
    Payload(#[from] PayloadError),
    #[error("could not decode PD MessagePack payload")]
    Decode(#[source] rmpv::decode::Error),
    #[error("could not encode PD MessagePack payload")]
    Encode(#[source] rmpv::encode::Error),
}

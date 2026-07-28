use uuid::{Variant, Version};

use crate::pd::config::EXPECTED_PROFILE_DIGEST_HEX;
use crate::pd::protocol::types::{
    ClientHello, ControlPayload, DestinationBlock, FixedBytes, KvBlock, PayloadError, PlanDigest,
    ProbeAck, RegionRecord, Role, RoomFields, ServerHello,
};

const MAX_BOOTSTRAP_ROOM: u64 = i64::MAX as u64;
const MAX_REGION_ID: u16 = 57;
const SLOT_CAPACITY: u16 = 32;
const MAX_TOKEN_COUNT: u32 = 4096;

impl ControlPayload {
    pub(crate) fn validate(&self) -> Result<(), PayloadError> {
        match self {
            Self::ClientHello(hello) => validate_client_hello(hello),
            Self::ServerHello(hello) => validate_server_hello(hello),
            Self::SessionReady(confirmation) | Self::SessionReadyAck(confirmation) => {
                validate_digest("transcript_hash", confirmation.transcript_hash.as_bytes())
            }
            Self::RegisterRegions(registration) => {
                validate_uuid("registration_epoch", &registration.registration_epoch)?;
                if registration.mooncake_host.is_empty() {
                    return invalid("mooncake_host", "must not be empty");
                }
                if registration.mooncake_port == 0 {
                    return invalid("mooncake_port", "must be non-zero");
                }
                validate_regions(&registration.regions)
            }
            Self::RegisterRegionsAck(decision) => {
                validate_uuid("registration_epoch", &decision.registration_epoch)?;
                validate_decision(decision.accepted, &decision.reason)
            }
            Self::ProbeReady(probe) => {
                validate_uuid("registration_epoch", &probe.registration_epoch)?;
                if probe.probe_generation == 0 {
                    return invalid("probe_generation", "must be non-zero");
                }
                if probe.aux_slot != 0 {
                    return invalid("aux_slot", "the bootstrap canary must use reserved slot 0");
                }
                if is_all_zero(probe.probe_data.as_bytes()) {
                    return invalid("probe_data", "must not be all zero");
                }
                Ok(())
            }
            Self::ProbeAck(probe) => validate_probe_ack(probe),
            Self::PrepareRoom(prepare) => {
                validate_room(&prepare.room)?;
                validate_uuid(
                    "destination_registration_epoch",
                    &prepare.destination_registration_epoch,
                )?;
                validate_destination_blocks(&prepare.destination_blocks)?;
                validate_slot("destination_aux_slot", prepare.destination_aux_slot)?;
                validate_slot(
                    "destination_completion_slot",
                    prepare.destination_completion_slot,
                )?;
                validate_chunk(
                    prepare.valid_token_count,
                    prepare.chunk_sequence,
                    prepare.chunk_count,
                    prepare.is_last_chunk,
                )
            }
            Self::PrepareAccepted(accepted) => {
                validate_room(&accepted.room)?;
                validate_uuid(
                    "source_registration_epoch",
                    &accepted.source_registration_epoch,
                )?;
                validate_uuid(
                    "destination_registration_epoch",
                    &accepted.destination_registration_epoch,
                )?;
                validate_kv_blocks(&accepted.kv_blocks)?;
                for (field, slot) in [
                    ("source_aux_slot", accepted.source_aux_slot),
                    ("destination_aux_slot", accepted.destination_aux_slot),
                    ("source_completion_slot", accepted.source_completion_slot),
                    (
                        "destination_completion_slot",
                        accepted.destination_completion_slot,
                    ),
                ] {
                    validate_slot(field, slot)?;
                }
                validate_chunk(
                    accepted.valid_token_count,
                    accepted.chunk_sequence,
                    accepted.chunk_count,
                    accepted.is_last_chunk,
                )?;
                validate_digest(
                    "transfer_plan_digest",
                    accepted.transfer_plan_digest.as_bytes(),
                )
            }
            Self::PrepareRejected(rejected) => {
                validate_room(&rejected.room)?;
                validate_reason(&rejected.reason)
            }
            Self::DataReady(planned)
            | Self::TransferComplete(planned)
            | Self::TransferCompleteAck(planned) => {
                validate_room(&planned.room)?;
                validate_digest(
                    "transfer_plan_digest",
                    planned.transfer_plan_digest.as_bytes(),
                )
            }
            Self::TransferFailed(terminal) => {
                validate_terminal(&terminal.room, &terminal.transfer_plan_digest)?;
                validate_reason(&terminal.reason)
            }
            Self::Abort(terminal) | Self::AbortAck(terminal) => {
                validate_terminal(&terminal.room, &terminal.transfer_plan_digest)?;
                validate_reason(&terminal.reason)
            }
            Self::Ping(ping) | Self::Pong(ping) => {
                if ping.ping_id == 0 {
                    return invalid("ping_id", "must be non-zero");
                }
                Ok(())
            }
            Self::GoAway(drain) | Self::GoAwayAck(drain) => {
                if drain.drain_generation == 0 {
                    return invalid("drain_generation", "must be non-zero");
                }
                Ok(())
            }
        }
    }
}

fn validate_client_hello(hello: &ClientHello) -> Result<(), PayloadError> {
    if hello.role != Role::Decode || hello.gpu != 5 {
        return invalid("role", "ClientHello must identify decode on GPU 5");
    }
    validate_hello_topology(hello.rank, hello.tp, hello.pp, hello.dp, hello.capabilities)?;
    validate_hello_identity(
        &hello.process_epoch,
        &hello.profile_digest,
        &hello.model_manifest_digest,
        &hello.tokenizer_manifest_digest,
        &hello.layout_fingerprint,
        &hello.native_abi_digest,
        &hello.nonce,
    )
}

fn validate_server_hello(hello: &ServerHello) -> Result<(), PayloadError> {
    if hello.role != Role::Prefill || hello.gpu != 4 {
        return invalid("role", "ServerHello must identify prefill on GPU 4");
    }
    validate_hello_topology(hello.rank, hello.tp, hello.pp, hello.dp, hello.capabilities)?;
    validate_hello_identity(
        &hello.process_epoch,
        &hello.profile_digest,
        &hello.model_manifest_digest,
        &hello.tokenizer_manifest_digest,
        &hello.layout_fingerprint,
        &hello.native_abi_digest,
        &hello.nonce,
    )?;
    validate_digest("client_hello_hash", hello.client_hello_hash.as_bytes())?;
    validate_decision(hello.accepted, &hello.reason)
}

fn validate_hello_topology(
    rank: u16,
    tp: u16,
    pp: u16,
    dp: u16,
    capabilities: u64,
) -> Result<(), PayloadError> {
    if rank != 0 {
        return invalid("rank", "must be zero");
    }
    if (tp, pp, dp) != (1, 1, 1) {
        return invalid("topology", "TP, PP and DP must all equal one");
    }
    if capabilities != 0 {
        return invalid("capabilities", "v1 does not negotiate capabilities");
    }
    Ok(())
}

fn validate_hello_identity(
    process_epoch: &FixedBytes<16>,
    profile_digest: &FixedBytes<32>,
    model_manifest_digest: &FixedBytes<32>,
    tokenizer_manifest_digest: &FixedBytes<32>,
    layout_fingerprint: &FixedBytes<32>,
    native_abi_digest: &FixedBytes<32>,
    nonce: &FixedBytes<32>,
) -> Result<(), PayloadError> {
    validate_uuid("process_epoch", process_epoch)?;
    let expected = FixedBytes::<32>::from_hex(EXPECTED_PROFILE_DIGEST_HEX).map_err(|error| {
        PayloadError::InvalidField {
            field: "profile_digest",
            detail: error.to_string(),
        }
    })?;
    if profile_digest != &expected {
        return invalid("profile_digest", "does not match frozen profile v1");
    }
    for (field, digest) in [
        ("model_manifest_digest", model_manifest_digest.as_bytes()),
        (
            "tokenizer_manifest_digest",
            tokenizer_manifest_digest.as_bytes(),
        ),
        ("layout_fingerprint", layout_fingerprint.as_bytes()),
        ("native_abi_digest", native_abi_digest.as_bytes()),
        ("nonce", nonce.as_bytes()),
    ] {
        validate_digest(field, digest)?;
    }
    Ok(())
}

fn validate_decision(accepted: bool, reason: &str) -> Result<(), PayloadError> {
    if accepted {
        if !reason.is_empty() {
            return invalid("reason", "must be empty when accepted is true");
        }
    } else {
        validate_reason(reason)?;
    }
    Ok(())
}

fn validate_probe_ack(probe: &ProbeAck) -> Result<(), PayloadError> {
    validate_uuid("registration_epoch", &probe.registration_epoch)?;
    if probe.probe_generation == 0 {
        return invalid("probe_generation", "must be non-zero");
    }
    if probe.aux_slot != 0 {
        return invalid("aux_slot", "the bootstrap canary must use reserved slot 0");
    }
    validate_decision(probe.accepted, &probe.reason)
}

fn validate_regions(regions: &[RegionRecord]) -> Result<(), PayloadError> {
    if regions.is_empty() {
        return invalid("regions", "must not be empty");
    }
    if regions
        .windows(2)
        .any(|pair| pair[0].region_id >= pair[1].region_id)
    {
        return invalid("regions", "must be strictly sorted by region_id");
    }
    for region in regions {
        if region.region_id > MAX_REGION_ID {
            return invalid("region_id", "must be in the frozen 0..=57 mapping");
        }
        if region.length_bytes == 0 {
            return invalid("length_bytes", "must be non-zero");
        }
        region
            .remote_base_addr
            .checked_add(region.length_bytes)
            .ok_or_else(|| PayloadError::InvalidField {
                field: "remote_base_addr",
                detail: "address range overflows u64".into(),
            })?;
        if !matches!(
            region.location.as_str(),
            "cpu:0" | "cpu:1" | "cuda:4" | "cuda:5"
        ) {
            return invalid("location", "is not in the frozen location allowlist");
        }
    }
    Ok(())
}

fn validate_room(room: &RoomFields) -> Result<(), PayloadError> {
    validate_uuid("decode_process_epoch", &room.decode_process_epoch)?;
    validate_uuid("attempt_id", &room.attempt_id)?;
    if room.bootstrap_room > MAX_BOOTSTRAP_ROOM {
        return invalid("bootstrap_room", "must fit the frozen u63 range");
    }
    if room.generation == 0 {
        return invalid("generation", "must be non-zero");
    }
    validate_digest(
        "request_contract_digest",
        room.request_contract_digest.as_bytes(),
    )
}

fn validate_terminal(room: &RoomFields, plan: &PlanDigest) -> Result<(), PayloadError> {
    validate_room(room)?;
    if !plan.is_empty() {
        validate_digest("transfer_plan_digest", plan.as_bytes())?;
    }
    Ok(())
}

fn validate_destination_blocks(blocks: &[DestinationBlock]) -> Result<(), PayloadError> {
    if blocks.is_empty() {
        return invalid("destination_blocks", "must not be empty");
    }
    if blocks.windows(2).any(|pair| pair[0] >= pair[1]) {
        return invalid(
            "destination_blocks",
            "must be strictly sorted without duplicates",
        );
    }
    for block in blocks {
        validate_range(
            block.region_id,
            block.byte_offset,
            block.byte_length,
            "destination_blocks",
        )?;
    }
    for pair in blocks.windows(2) {
        if pair[0].region_id == pair[1].region_id
            && pair[0].destination_page == pair[1].destination_page
            && pair[0].byte_offset + pair[0].byte_length > pair[1].byte_offset
        {
            return invalid("destination_blocks", "ranges must not overlap");
        }
    }
    Ok(())
}

fn validate_kv_blocks(blocks: &[KvBlock]) -> Result<(), PayloadError> {
    if blocks.is_empty() {
        return invalid("kv_blocks", "must not be empty");
    }
    if blocks.windows(2).any(|pair| pair[0] >= pair[1]) {
        return invalid("kv_blocks", "must be strictly sorted without duplicates");
    }
    for block in blocks {
        validate_range(
            block.region_id,
            block.byte_offset,
            block.byte_length,
            "kv_blocks",
        )?;
    }
    for pair in blocks.windows(2) {
        if pair[0].region_id == pair[1].region_id
            && pair[0].source_page == pair[1].source_page
            && pair[0].destination_page == pair[1].destination_page
            && pair[0].byte_offset + pair[0].byte_length > pair[1].byte_offset
        {
            return invalid("kv_blocks", "ranges must not overlap");
        }
    }
    Ok(())
}

fn validate_range(
    region_id: u16,
    offset: u64,
    length: u64,
    field: &'static str,
) -> Result<(), PayloadError> {
    if region_id > MAX_REGION_ID {
        return invalid("region_id", "must be in the frozen 0..=57 mapping");
    }
    if length == 0 {
        return invalid(field, "range length must be non-zero");
    }
    offset
        .checked_add(length)
        .ok_or_else(|| PayloadError::InvalidField {
            field,
            detail: "range overflows u64".into(),
        })?;
    Ok(())
}

fn validate_chunk(
    valid_token_count: u32,
    chunk_sequence: u32,
    chunk_count: u32,
    is_last_chunk: bool,
) -> Result<(), PayloadError> {
    if !(1..=MAX_TOKEN_COUNT).contains(&valid_token_count) {
        return invalid("valid_token_count", "must be in 1..=4096");
    }
    if chunk_count == 0 || chunk_sequence >= chunk_count {
        return invalid("chunk_sequence", "must identify an existing chunk");
    }
    if is_last_chunk != (chunk_sequence + 1 == chunk_count) {
        return invalid("is_last_chunk", "must match chunk_sequence/chunk_count");
    }
    Ok(())
}

fn validate_slot(field: &'static str, slot: u16) -> Result<(), PayloadError> {
    if slot >= SLOT_CAPACITY {
        return invalid(field, "must fit the frozen 32-slot capacity");
    }
    Ok(())
}

fn validate_uuid(field: &'static str, value: &FixedBytes<16>) -> Result<(), PayloadError> {
    let uuid = uuid::Uuid::from_bytes(value.into_array());
    if uuid.get_version() != Some(Version::Random) || uuid.get_variant() != Variant::RFC4122 {
        return invalid(field, "must be an RFC4122 version-4 UUID");
    }
    Ok(())
}

fn validate_digest(field: &'static str, value: &[u8]) -> Result<(), PayloadError> {
    if value.len() != 32 || is_all_zero(value) {
        return invalid(field, "must be a non-zero 32-byte digest");
    }
    Ok(())
}

fn validate_reason(reason: &str) -> Result<(), PayloadError> {
    if !matches!(
        reason,
        "PD_REQUEST_INVALID"
            | "PD_UNSUPPORTED"
            | "PD_CAPACITY_EXHAUSTED"
            | "PD_PROTOCOL_MISMATCH"
            | "PD_PEER_UNAVAILABLE"
            | "PD_RENDEZVOUS_TIMEOUT"
            | "PD_TRANSFER_TIMEOUT"
            | "PD_TRANSFER_FAILED"
            | "PD_ACK_TIMEOUT"
            | "PD_ABORTED"
            | "PD_STALE_EPOCH"
            | "PD_LOCAL_FATAL"
    ) {
        return invalid("reason", "must be a frozen PD_* reason");
    }
    Ok(())
}

fn is_all_zero(value: &[u8]) -> bool {
    value.iter().all(|byte| *byte == 0)
}

fn invalid<T>(field: &'static str, detail: impl Into<String>) -> Result<T, PayloadError> {
    Err(PayloadError::InvalidField {
        field,
        detail: detail.into(),
    })
}

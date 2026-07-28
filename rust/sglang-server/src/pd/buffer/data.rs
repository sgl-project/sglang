use crate::pd::buffer::BufferError;
use crate::pd::buffer::plan::TransferPlanDigest;
use crate::pd::protocol::FixedBytes;
use crate::pd::room::{AttemptId, ProcessEpoch, RegistrationEpoch};

pub const AUX_BYTES: usize = 64;
pub const COMPLETION_BYTES: usize = 192;
const COMPLETION_BODY_AND_CRC_BYTES: usize = 188;
const KV_ROW_BYTES: usize = 2_048;
const KV_PAGE_TOKENS: u32 = 64;
const KV_PAGE_BYTES: usize = KV_ROW_BYTES * KV_PAGE_TOKENS as usize;
const MAX_TOKEN_COUNT: u32 = 4_096;
const DONE_MARKER: [u8; 4] = *b"DONE";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuxRecordInput {
    pub first_token_valid: bool,
    pub first_token_id: i32,
    pub prompt_token_count: u32,
    pub prefill_output_count: u32,
    pub request_digest: FixedBytes<32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuxRecord {
    pub first_token_valid: bool,
    pub first_token_id: i32,
    pub prompt_token_count: u32,
    pub prefill_output_count: u32,
    pub request_digest: FixedBytes<32>,
}

impl AuxRecord {
    pub fn encode(input: AuxRecordInput) -> Result<[u8; AUX_BYTES], BufferError> {
        validate_aux_fields(&input)?;
        let mut output = [0_u8; AUX_BYTES];
        output[0..4].copy_from_slice(b"SGAX");
        output[4..6].copy_from_slice(&1_u16.to_be_bytes());
        output[6..8].copy_from_slice(&(AUX_BYTES as u16).to_be_bytes());
        output[8..12].copy_from_slice(&u32::from(input.first_token_valid).to_be_bytes());
        output[12..16].copy_from_slice(&input.first_token_id.to_be_bytes());
        output[16..20].copy_from_slice(&input.prompt_token_count.to_be_bytes());
        output[20..24].copy_from_slice(&input.prefill_output_count.to_be_bytes());
        output[32..64].copy_from_slice(input.request_digest.as_bytes());
        Ok(output)
    }

    pub fn decode(bytes: &[u8; AUX_BYTES]) -> Result<Self, BufferError> {
        if &bytes[0..4] != b"SGAX" {
            return invalid("aux_magic");
        }
        if read_u16(bytes, 4)? != 1 || read_u16(bytes, 6)? != AUX_BYTES as u16 {
            return invalid("aux_schema");
        }
        let flags = read_u32(bytes, 8)?;
        if flags & !1 != 0 {
            return invalid("aux_flags");
        }
        if bytes[24..32].iter().any(|byte| *byte != 0) {
            return invalid("aux_reserved");
        }
        let input = AuxRecordInput {
            first_token_valid: flags & 1 != 0,
            first_token_id: read_i32(bytes, 12)?,
            prompt_token_count: read_u32(bytes, 16)?,
            prefill_output_count: read_u32(bytes, 20)?,
            request_digest: FixedBytes::new(bytes[32..64].try_into().map_err(|_| {
                BufferError::DataRecord {
                    check: "aux_request_digest",
                }
            })?),
        };
        validate_aux_fields(&input)?;
        Ok(Self {
            first_token_valid: input.first_token_valid,
            first_token_id: input.first_token_id,
            prompt_token_count: input.prompt_token_count,
            prefill_output_count: input.prefill_output_count,
            request_digest: input.request_digest,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionRecordInput {
    pub decode_process_epoch: ProcessEpoch,
    pub attempt_id: AttemptId,
    pub source_registration_epoch: RegistrationEpoch,
    pub destination_registration_epoch: RegistrationEpoch,
    pub bootstrap_room: u64,
    pub transfer_generation: u64,
    pub chunk_sequence: u32,
    pub chunk_count: u32,
    pub page_count: u32,
    pub valid_token_count: u32,
    pub request_digest: FixedBytes<32>,
    pub transfer_plan_digest: TransferPlanDigest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionWrites {
    body_and_crc: [u8; COMPLETION_BODY_AND_CRC_BYTES],
    commit_marker: [u8; 4],
}

impl CompletionWrites {
    pub fn encode(
        input: &CompletionRecordInput,
        aux: &[u8; AUX_BYTES],
    ) -> Result<Self, BufferError> {
        validate_completion_input(input)?;
        let decoded_aux = AuxRecord::decode(aux)?;
        if decoded_aux.request_digest != input.request_digest
            || decoded_aux.prompt_token_count != input.valid_token_count
        {
            return invalid("aux_identity");
        }
        let mut record = [0_u8; COMPLETION_BYTES];
        record[0..4].copy_from_slice(b"SGCP");
        record[4..6].copy_from_slice(&1_u16.to_be_bytes());
        record[6..8].copy_from_slice(&(COMPLETION_BYTES as u16).to_be_bytes());
        record[8..12].copy_from_slice(&1_u32.to_be_bytes());
        record[16..32].copy_from_slice(&input.decode_process_epoch.as_bytes());
        record[32..48].copy_from_slice(&input.attempt_id.as_bytes());
        record[48..64].copy_from_slice(&input.source_registration_epoch.as_bytes());
        record[64..80].copy_from_slice(&input.destination_registration_epoch.as_bytes());
        record[80..88].copy_from_slice(&input.bootstrap_room.to_be_bytes());
        record[88..96].copy_from_slice(&input.transfer_generation.to_be_bytes());
        record[96..100].copy_from_slice(&input.chunk_sequence.to_be_bytes());
        record[100..104].copy_from_slice(&input.chunk_count.to_be_bytes());
        record[104..108].copy_from_slice(&input.page_count.to_be_bytes());
        record[108..112].copy_from_slice(&input.valid_token_count.to_be_bytes());
        record[112..116].copy_from_slice(&(AUX_BYTES as u32).to_be_bytes());
        record[116..120].copy_from_slice(&crc32c(aux).to_be_bytes());
        record[120..152].copy_from_slice(input.request_digest.as_bytes());
        record[152..184].copy_from_slice(input.transfer_plan_digest.as_bytes());
        let record_crc = crc32c(&record[..184]);
        record[184..188].copy_from_slice(&record_crc.to_be_bytes());
        Ok(Self {
            body_and_crc: record[..188]
                .try_into()
                .map_err(|_| BufferError::DataRecord {
                    check: "completion_length",
                })?,
            commit_marker: DONE_MARKER,
        })
    }

    pub const fn body_and_crc(&self) -> &[u8; COMPLETION_BODY_AND_CRC_BYTES] {
        &self.body_and_crc
    }

    pub const fn commit_marker(&self) -> &[u8; 4] {
        &self.commit_marker
    }

    pub fn committed_bytes(&self) -> [u8; COMPLETION_BYTES] {
        let mut output = [0_u8; COMPLETION_BYTES];
        output[..188].copy_from_slice(&self.body_and_crc);
        output[188..192].copy_from_slice(&self.commit_marker);
        output
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedCompletion {
    pub aux: AuxRecord,
    pub page_count: u32,
    pub valid_token_count: u32,
}

pub fn validate_completion(
    completion: &[u8; COMPLETION_BYTES],
    aux: &[u8; AUX_BYTES],
    expected: &CompletionRecordInput,
) -> Result<ValidatedCompletion, BufferError> {
    if completion[188..192] != DONE_MARKER {
        return invalid("completion_marker");
    }
    if read_u32(completion, 184)? != crc32c(&completion[..184]) {
        return invalid("completion_crc");
    }
    if read_u32(completion, 116)? != crc32c(aux) {
        return invalid("aux_crc");
    }
    if &completion[0..4] != b"SGCP"
        || read_u16(completion, 4)? != 1
        || read_u16(completion, 6)? != COMPLETION_BYTES as u16
    {
        return invalid("completion_schema");
    }
    if read_u32(completion, 8)? != 1 || completion[12..16].iter().any(|byte| *byte != 0) {
        return invalid("completion_flags");
    }
    if completion[16..32] != expected.decode_process_epoch.as_bytes()
        || completion[32..48] != expected.attempt_id.as_bytes()
        || completion[48..64] != expected.source_registration_epoch.as_bytes()
        || completion[64..80] != expected.destination_registration_epoch.as_bytes()
        || read_u64(completion, 80)? != expected.bootstrap_room
        || read_u64(completion, 88)? != expected.transfer_generation
        || read_u32(completion, 96)? != expected.chunk_sequence
        || read_u32(completion, 100)? != expected.chunk_count
        || read_u32(completion, 104)? != expected.page_count
        || read_u32(completion, 108)? != expected.valid_token_count
        || read_u32(completion, 112)? != AUX_BYTES as u32
        || completion[120..152] != *expected.request_digest.as_array()
        || completion[152..184] != *expected.transfer_plan_digest.as_bytes()
    {
        return invalid("completion_identity");
    }
    validate_completion_input(expected)?;
    let aux = AuxRecord::decode(aux)?;
    if aux.request_digest != expected.request_digest
        || aux.prompt_token_count != expected.valid_token_count
    {
        return invalid("aux_identity");
    }
    Ok(ValidatedCompletion {
        aux,
        page_count: expected.page_count,
        valid_token_count: expected.valid_token_count,
    })
}

pub fn clear_partial_page_tail(
    final_page: &mut [u8],
    valid_token_count: u32,
) -> Result<(), BufferError> {
    if final_page.len() != KV_PAGE_BYTES || !(1..=MAX_TOKEN_COUNT).contains(&valid_token_count) {
        return invalid("partial_page");
    }
    let valid_rows = match valid_token_count % KV_PAGE_TOKENS {
        0 => KV_PAGE_TOKENS,
        remainder => remainder,
    };
    let valid_bytes = usize::try_from(valid_rows)
        .map_err(|_| BufferError::DataRecord {
            check: "partial_page",
        })?
        .checked_mul(KV_ROW_BYTES)
        .ok_or(BufferError::DataRecord {
            check: "partial_page",
        })?;
    final_page[valid_bytes..].fill(0);
    Ok(())
}

pub fn crc32c(bytes: &[u8]) -> u32 {
    crc_fast::crc32_iscsi(bytes)
}

fn validate_aux_fields(input: &AuxRecordInput) -> Result<(), BufferError> {
    if !(1..=MAX_TOKEN_COUNT).contains(&input.prompt_token_count) {
        return invalid("aux_prompt_token_count");
    }
    if input.prefill_output_count > 1
        || input.first_token_valid != (input.prefill_output_count == 1)
        || (!input.first_token_valid && input.first_token_id != 0)
        || input
            .request_digest
            .as_bytes()
            .iter()
            .all(|byte| *byte == 0)
    {
        return invalid("aux_fields");
    }
    Ok(())
}

fn validate_completion_input(input: &CompletionRecordInput) -> Result<(), BufferError> {
    if input.bootstrap_room > i64::MAX as u64
        || input.transfer_generation == 0
        || input.chunk_sequence != 0
        || input.chunk_count != 1
        || !(1..=MAX_TOKEN_COUNT).contains(&input.valid_token_count)
        || input.page_count != input.valid_token_count.div_ceil(KV_PAGE_TOKENS)
        || input
            .request_digest
            .as_bytes()
            .iter()
            .all(|byte| *byte == 0)
        || input
            .transfer_plan_digest
            .as_bytes()
            .iter()
            .all(|byte| *byte == 0)
    {
        return invalid("completion_input");
    }
    Ok(())
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, BufferError> {
    bytes
        .get(offset..offset + 2)
        .and_then(|value| value.try_into().ok())
        .map(u16::from_be_bytes)
        .ok_or(BufferError::DataRecord {
            check: "record_length",
        })
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, BufferError> {
    bytes
        .get(offset..offset + 4)
        .and_then(|value| value.try_into().ok())
        .map(u32::from_be_bytes)
        .ok_or(BufferError::DataRecord {
            check: "record_length",
        })
}

fn read_i32(bytes: &[u8], offset: usize) -> Result<i32, BufferError> {
    bytes
        .get(offset..offset + 4)
        .and_then(|value| value.try_into().ok())
        .map(i32::from_be_bytes)
        .ok_or(BufferError::DataRecord {
            check: "record_length",
        })
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, BufferError> {
    bytes
        .get(offset..offset + 8)
        .and_then(|value| value.try_into().ok())
        .map(u64::from_be_bytes)
        .ok_or(BufferError::DataRecord {
            check: "record_length",
        })
}

fn invalid<T>(check: &'static str) -> Result<T, BufferError> {
    Err(BufferError::DataRecord { check })
}

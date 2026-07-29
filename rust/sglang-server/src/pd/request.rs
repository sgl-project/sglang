use std::collections::BTreeSet;

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::message::SamplingParams;
use crate::pd::protocol::FixedBytes;
use crate::pd::room::PdReason;

pub const REQUEST_DOMAIN: &[u8] = b"SGLANG-PD-REQUEST-V1\0";
const MAX_BATCH_ITEMS: u32 = 8;
const MAX_INPUT_TOKENS: usize = 4_096;
const MAX_NEW_TOKENS: u32 = 256;

#[derive(Debug, Clone, PartialEq)]
pub struct RequestSampling {
    pub temperature: f64,
    pub top_k: u32,
    pub top_p: f64,
    pub min_p: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
    pub repetition_penalty: f64,
    pub min_new_tokens: u32,
    pub max_new_tokens: u32,
    pub stop: Vec<String>,
    pub stop_regex: Vec<String>,
    pub stop_token_ids: Vec<u32>,
    pub ignore_eos: bool,
    pub n: u32,
}

impl RequestSampling {
    /// Extract the frozen digest fields from the upstream typed, already-normalized
    /// sampling object. Unsupported capabilities are rejected by the HTTP support
    /// gate before this conversion.
    pub fn from_normalized(value: &SamplingParams) -> Result<Self, RequestContractError> {
        Ok(Self {
            temperature: value.temperature,
            top_k: u32::try_from(value.top_k)
                .map_err(|_| RequestContractError::InvalidField("top_k"))?,
            top_p: value.top_p,
            min_p: value.min_p,
            frequency_penalty: value.frequency_penalty,
            presence_penalty: value.presence_penalty,
            repetition_penalty: value.repetition_penalty,
            min_new_tokens: u32::try_from(value.min_new_tokens)
                .map_err(|_| RequestContractError::InvalidField("min_new_tokens"))?,
            max_new_tokens: u32::try_from(value.max_new_tokens.unwrap_or(128))
                .map_err(|_| RequestContractError::InvalidField("max_new_tokens"))?,
            stop: value.stop_strs.clone(),
            stop_regex: value.stop_regex_strs.clone(),
            stop_token_ids: value
                .stop_token_ids
                .as_deref()
                .unwrap_or_default()
                .iter()
                .map(|value| {
                    u32::try_from(*value)
                        .map_err(|_| RequestContractError::InvalidField("stop_token_ids"))
                })
                .collect::<Result<_, _>>()?,
            ignore_eos: value.ignore_eos,
            n: u32::try_from(value.n).map_err(|_| RequestContractError::InvalidField("n"))?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuxSchema {
    pub version: u16,
    pub bytes: u16,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RequestContractInput {
    pub model_manifest_digest: FixedBytes<32>,
    pub tokenizer_manifest_digest: FixedBytes<32>,
    pub layout_fingerprint: FixedBytes<32>,
    pub profile_digest: FixedBytes<32>,
    pub batch_index: u32,
    pub normalized_input_ids: Vec<u32>,
    pub sampling: RequestSampling,
    pub aux_schema: AuxSchema,
}

#[derive(Clone, PartialEq, Eq)]
pub struct RequestContractDigest {
    digest: FixedBytes<32>,
    canonical: Vec<u8>,
}

impl RequestContractDigest {
    pub fn new(input: RequestContractInput) -> Result<Self, RequestContractError> {
        validate(&input)?;
        let canonical = canonical_bytes(&input)?;
        let digest = FixedBytes::new(Sha256::digest(&canonical).into());
        Ok(Self { digest, canonical })
    }

    pub const fn as_fixed_bytes(&self) -> FixedBytes<32> {
        self.digest
    }

    pub const fn as_bytes(&self) -> &[u8] {
        self.digest.as_bytes()
    }

    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical
    }

    pub fn to_hex(&self) -> String {
        hex::encode(self.digest.as_bytes())
    }
}

impl std::fmt::Debug for RequestContractDigest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("RequestContractDigest")
            .field(&self.to_hex())
            .finish()
    }
}

fn validate(input: &RequestContractInput) -> Result<(), RequestContractError> {
    for (field, digest) in [
        ("model_manifest_digest", input.model_manifest_digest),
        ("tokenizer_manifest_digest", input.tokenizer_manifest_digest),
        ("layout_fingerprint", input.layout_fingerprint),
        ("profile_digest", input.profile_digest),
    ] {
        if digest.as_bytes().iter().all(|byte| *byte == 0) {
            return invalid(field);
        }
    }
    if input.batch_index >= MAX_BATCH_ITEMS {
        return invalid("batch_index");
    }
    if !(1..=MAX_INPUT_TOKENS).contains(&input.normalized_input_ids.len())
        || input
            .normalized_input_ids
            .iter()
            .any(|token| *token > i32::MAX as u32)
    {
        return invalid("normalized_input_ids");
    }

    let sampling = &input.sampling;
    for (field, value) in [
        ("temperature", sampling.temperature),
        ("top_p", sampling.top_p),
        ("min_p", sampling.min_p),
        ("frequency_penalty", sampling.frequency_penalty),
        ("presence_penalty", sampling.presence_penalty),
        ("repetition_penalty", sampling.repetition_penalty),
    ] {
        if !value.is_finite() {
            return invalid(field);
        }
    }
    if sampling.temperature != 1.0
        || sampling.top_k != 1
        || sampling.top_p != 1.0
        || sampling.min_p != 0.0
    {
        return invalid("greedy_sampling");
    }
    if !((-2.0..=2.0).contains(&sampling.frequency_penalty)
        && (-2.0..=2.0).contains(&sampling.presence_penalty)
        && sampling.repetition_penalty > 0.0
        && sampling.repetition_penalty <= 2.0)
    {
        return invalid("sampling_penalty");
    }
    if sampling.min_new_tokens > sampling.max_new_tokens
        || sampling.max_new_tokens > MAX_NEW_TOKENS
        || sampling.n != 1
    {
        return invalid("sampling_bounds");
    }
    if input.aux_schema
        != (AuxSchema {
            version: 1,
            bytes: 64,
        })
    {
        return invalid("aux_schema");
    }
    Ok(())
}

fn canonical_bytes(input: &RequestContractInput) -> Result<Vec<u8>, RequestContractError> {
    let sampling = &input.sampling;
    let mut output = Vec::with_capacity(REQUEST_DOMAIN.len() + 512);
    output.extend_from_slice(REQUEST_DOMAIN);
    for digest in [
        input.model_manifest_digest,
        input.tokenizer_manifest_digest,
        input.layout_fingerprint,
        input.profile_digest,
    ] {
        output.extend_from_slice(digest.as_bytes());
    }
    push_u32(&mut output, input.batch_index);
    push_len(&mut output, input.normalized_input_ids.len())?;
    for token_id in &input.normalized_input_ids {
        push_u32(&mut output, *token_id);
    }
    push_f64(&mut output, sampling.temperature);
    push_u32(&mut output, sampling.top_k);
    push_f64(&mut output, sampling.top_p);
    push_f64(&mut output, sampling.min_p);
    push_f64(&mut output, sampling.frequency_penalty);
    push_f64(&mut output, sampling.presence_penalty);
    push_f64(&mut output, sampling.repetition_penalty);
    push_u32(&mut output, sampling.min_new_tokens);
    push_u32(&mut output, sampling.max_new_tokens);
    push_string_set(&mut output, &sampling.stop)?;
    push_string_set(&mut output, &sampling.stop_regex)?;
    push_u32_set(&mut output, &sampling.stop_token_ids)?;
    output.push(u8::from(sampling.ignore_eos));
    push_u32(&mut output, sampling.n);
    output.extend_from_slice(&input.aux_schema.version.to_be_bytes());
    output.extend_from_slice(&input.aux_schema.bytes.to_be_bytes());
    Ok(output)
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_be_bytes());
}

fn push_f64(output: &mut Vec<u8>, value: f64) {
    let normalized = if value == 0.0 { 0.0 } else { value };
    output.extend_from_slice(&normalized.to_be_bytes());
}

fn push_len(output: &mut Vec<u8>, length: usize) -> Result<(), RequestContractError> {
    let length = u32::try_from(length).map_err(|_| RequestContractError::Length)?;
    push_u32(output, length);
    Ok(())
}

fn encoded_string(value: &str) -> Result<Vec<u8>, RequestContractError> {
    let bytes = value.as_bytes();
    let mut encoded = Vec::with_capacity(4 + bytes.len());
    push_len(&mut encoded, bytes.len())?;
    encoded.extend_from_slice(bytes);
    Ok(encoded)
}

fn push_string_set(output: &mut Vec<u8>, values: &[String]) -> Result<(), RequestContractError> {
    let encoded = values
        .iter()
        .map(|value| encoded_string(value))
        .collect::<Result<BTreeSet<_>, _>>()?;
    push_len(output, encoded.len())?;
    for item in encoded {
        output.extend_from_slice(&item);
    }
    Ok(())
}

fn push_u32_set(output: &mut Vec<u8>, values: &[u32]) -> Result<(), RequestContractError> {
    let encoded: BTreeSet<[u8; 4]> = values.iter().map(|value| value.to_be_bytes()).collect();
    push_len(output, encoded.len())?;
    for item in encoded {
        output.extend_from_slice(&item);
    }
    Ok(())
}

fn invalid<T>(field: &'static str) -> Result<T, RequestContractError> {
    Err(RequestContractError::InvalidField(field))
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RequestContractError {
    #[error("invalid frozen PD request contract field {0}")]
    InvalidField(&'static str),
    #[error("PD request contract collection is too large")]
    Length,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KVPoll {
    Failed = 0,
    Bootstrapping = 1,
    WaitingForInput = 2,
    Transferring = 3,
    Success = 4,
}

impl KVPoll {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Failed => "Failed",
            Self::Bootstrapping => "Bootstrapping",
            Self::WaitingForInput => "WaitingForInput",
            Self::Transferring => "Transferring",
            Self::Success => "Success",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdPublicError {
    reason: PdReason,
    status: u16,
    message: &'static str,
}

impl PdPublicError {
    pub fn new(reason: PdReason) -> Result<Self, RequestContractError> {
        let (status, message) = match reason {
            PdReason::RequestInvalid => (400, "PD request is invalid"),
            PdReason::Unsupported => (422, "PD request uses an unsupported capability"),
            PdReason::CapacityExhausted => (429, "PD capacity is exhausted"),
            PdReason::ProtocolMismatch => (503, "PD protocol mismatch"),
            PdReason::PeerUnavailable => (503, "PD peer is unavailable"),
            PdReason::RendezvousTimeout => (504, "PD room rendezvous timed out"),
            PdReason::TransferTimeout => (504, "PD transfer timed out"),
            PdReason::TransferFailed => (502, "PD transfer failed"),
            PdReason::AckTimeout => (504, "PD completion acknowledgement timed out"),
            PdReason::Aborted => (499, "PD request was aborted"),
            PdReason::StaleEpoch => (409, "PD handle or epoch is stale"),
            PdReason::LocalFatal => (503, "PD local runtime failed"),
            PdReason::Success => return invalid("public_error_reason"),
        };
        Ok(Self {
            reason,
            status,
            message,
        })
    }

    pub const fn reason(&self) -> &'static str {
        self.reason.code()
    }

    pub const fn status(&self) -> u16 {
        self.status
    }

    pub const fn retryable(&self) -> bool {
        self.reason.retryable()
    }

    pub fn body(&self) -> serde_json::Value {
        serde_json::json!({
            "error": {
                "message": self.message,
                "type": "pd_error",
                "code": self.status,
                "pd_reason": self.reason.code(),
                "retryable": self.reason.retryable(),
            }
        })
    }
}

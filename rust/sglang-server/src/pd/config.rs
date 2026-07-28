use sha2::{Digest, Sha256};
use thiserror::Error;

pub const EXPECTED_PROFILE_CANONICAL_BYTES: usize = 769;
pub const EXPECTED_PROFILE_DIGEST_HEX: &str =
    "9a9b82e5536d19d1cc17a395c25a906b4e7185fb42612a0cfb30ed8b9ad9e881";
pub const PROFILE_DOMAIN: &[u8] = b"SGLANG-PD-PROFILE-V1\0";

const PROFILE_SOURCE: &[u8] = include_bytes!("../../contracts/profile-v1.json");

#[derive(Debug, Error)]
pub enum ProfileError {
    #[error("invalid PD profile JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid PD profile field {field}: {detail}")]
    InvalidField { field: &'static str, detail: String },
    #[error("PD profile canonical body has {actual} bytes, expected {expected}")]
    CanonicalLength { actual: usize, expected: usize },
    #[error("PD profile digest {actual} does not match the frozen profile")]
    DigestMismatch { actual: String },
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PdProfileV1 {
    pub profile_version: u16,
    pub control: ControlProfile,
    pub security: SecurityProfile,
    pub topology: TopologyProfile,
    pub platform: PlatformProfile,
    pub model: ModelProfile,
    pub request: RequestProfile,
    pub transport: TransportProfile,
    pub capacity: CapacityProfile,
    pub deadline_ms: DeadlineProfile,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlProfile {
    pub magic: String,
    pub schema_major: u16,
    pub schema_minor: u16,
    pub max_payload_bytes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SecurityProfile {
    pub mac: String,
    pub kdf: String,
    pub psk_bytes: u32,
    pub nonce_bytes: u32,
    pub session_key_bytes: u32,
    pub tag_bytes: u32,
    pub psk_id_bytes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyProfile {
    pub host_count: u32,
    pub prefill_ranks: u32,
    pub decode_ranks: u32,
    pub tp: u32,
    pub pp: u32,
    pub dp: u32,
    pub prefill_gpu: u32,
    pub decode_gpu: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlatformProfile {
    pub os: String,
    pub arch: String,
    pub distribution: String,
    pub distribution_version: String,
    pub gpu: String,
    pub sm: u32,
    pub cuda: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelProfile {
    pub architecture: String,
    pub dtype: String,
    pub quantization: String,
    pub attention: String,
    pub state: String,
    pub layers: u32,
    pub attention_heads: u32,
    pub kv_heads: u32,
    pub head_dimension: u32,
    pub kv_layout: String,
    pub region_layout: String,
    pub page_size_tokens: u32,
    pub kernel: String,
    pub enabled_runtime_features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestProfile {
    pub endpoint: String,
    pub input_kinds: Vec<String>,
    pub batch_min: u32,
    pub batch_max: u32,
    pub input_tokens_min: u32,
    pub input_tokens_max: u32,
    pub max_new_tokens_min: u32,
    pub max_new_tokens_max: u32,
    pub n: u32,
    pub temperature: f64,
    pub top_k: u32,
    pub top_p: f64,
    pub min_p: f64,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportProfile {
    pub backend: String,
    pub version: String,
    pub commit: String,
    pub header_sha256: String,
    pub components: Vec<String>,
    pub auto_discover: bool,
    pub fallback_allowed: bool,
    pub rdma_devices: Vec<String>,
    pub gpudirect_host_flush_required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityProfile {
    pub active_rooms_per_pair: u64,
    pub native_transfers_per_pair: u64,
    pub kv_pages_per_room: u64,
    pub leased_kv_pages_per_endpoint: u64,
    pub aux_slots_per_endpoint: u64,
    pub completion_slots_per_endpoint: u64,
    pub request_slots_per_endpoint: u64,
    pub pending_transfer_bytes_per_pair: u64,
    pub tombstones_per_pair: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeadlineProfile {
    pub local_initialization: u64,
    pub connect_and_hello: u64,
    pub room_rendezvous: u64,
    pub destination_allocation: u64,
    pub native_transfer: u64,
    pub completion_ack: u64,
    pub abort_ack: u64,
    pub heartbeat_interval: u64,
    pub heartbeat_misses: u8,
    pub tombstone_retention: u64,
}

impl PdProfileV1 {
    pub fn load_embedded() -> Result<Self, ProfileError> {
        Self::from_slice(PROFILE_SOURCE)
    }

    pub fn from_slice(bytes: &[u8]) -> Result<Self, ProfileError> {
        let profile: Self = serde_json::from_slice(bytes)?;
        profile.validate()?;
        let body = profile.canonical_body()?;
        if body.len() != EXPECTED_PROFILE_CANONICAL_BYTES {
            return Err(ProfileError::CanonicalLength {
                actual: body.len(),
                expected: EXPECTED_PROFILE_CANONICAL_BYTES,
            });
        }
        let actual_digest = profile.digest_hex()?;
        if actual_digest != EXPECTED_PROFILE_DIGEST_HEX {
            return Err(ProfileError::DigestMismatch {
                actual: actual_digest,
            });
        }
        Ok(profile)
    }

    pub fn canonical_body(&self) -> Result<Vec<u8>, ProfileError> {
        self.validate()?;
        let mut output = Vec::with_capacity(EXPECTED_PROFILE_CANONICAL_BYTES);

        push_u16(&mut output, self.profile_version);
        push_string(&mut output, &self.control.magic)?;
        push_u16(&mut output, self.control.schema_major);
        push_u16(&mut output, self.control.schema_minor);
        push_u32(&mut output, self.control.max_payload_bytes);

        push_string(&mut output, &self.security.mac)?;
        push_string(&mut output, &self.security.kdf)?;
        push_u32(&mut output, self.security.psk_bytes);
        push_u32(&mut output, self.security.nonce_bytes);
        push_u32(&mut output, self.security.session_key_bytes);
        push_u32(&mut output, self.security.tag_bytes);
        push_u32(&mut output, self.security.psk_id_bytes);

        for value in [
            self.topology.host_count,
            self.topology.prefill_ranks,
            self.topology.decode_ranks,
            self.topology.tp,
            self.topology.pp,
            self.topology.dp,
            self.topology.prefill_gpu,
            self.topology.decode_gpu,
        ] {
            push_u32(&mut output, value);
        }

        push_string(&mut output, &self.platform.os)?;
        push_string(&mut output, &self.platform.arch)?;
        push_string(&mut output, &self.platform.distribution)?;
        push_string(&mut output, &self.platform.distribution_version)?;
        push_string(&mut output, &self.platform.gpu)?;
        push_u32(&mut output, self.platform.sm);
        push_string(&mut output, &self.platform.cuda)?;

        push_string(&mut output, &self.model.architecture)?;
        push_string(&mut output, &self.model.dtype)?;
        push_string(&mut output, &self.model.quantization)?;
        push_string(&mut output, &self.model.attention)?;
        push_string(&mut output, &self.model.state)?;
        for value in [
            self.model.layers,
            self.model.attention_heads,
            self.model.kv_heads,
            self.model.head_dimension,
        ] {
            push_u32(&mut output, value);
        }
        push_string(&mut output, &self.model.kv_layout)?;
        push_string(&mut output, &self.model.region_layout)?;
        push_u32(&mut output, self.model.page_size_tokens);
        push_string(&mut output, &self.model.kernel)?;
        push_string_list(
            &mut output,
            "model.enabled_runtime_features",
            &self.model.enabled_runtime_features,
        )?;

        push_string(&mut output, &self.request.endpoint)?;
        push_string_list(
            &mut output,
            "request.input_kinds",
            &self.request.input_kinds,
        )?;
        for value in [
            self.request.batch_min,
            self.request.batch_max,
            self.request.input_tokens_min,
            self.request.input_tokens_max,
            self.request.max_new_tokens_min,
            self.request.max_new_tokens_max,
            self.request.n,
        ] {
            push_u32(&mut output, value);
        }
        push_f64(&mut output, self.request.temperature)?;
        push_u32(&mut output, self.request.top_k);
        push_f64(&mut output, self.request.top_p)?;
        push_f64(&mut output, self.request.min_p)?;
        push_string_list(&mut output, "request.features", &self.request.features)?;

        push_string(&mut output, &self.transport.backend)?;
        push_string(&mut output, &self.transport.version)?;
        push_fixed_hex(&mut output, "transport.commit", &self.transport.commit, 20)?;
        push_fixed_hex(
            &mut output,
            "transport.header_sha256",
            &self.transport.header_sha256,
            32,
        )?;
        push_string_list(
            &mut output,
            "transport.components",
            &self.transport.components,
        )?;
        push_bool(&mut output, self.transport.auto_discover);
        push_bool(&mut output, self.transport.fallback_allowed);
        push_string_list(
            &mut output,
            "transport.rdma_devices",
            &self.transport.rdma_devices,
        )?;
        push_bool(&mut output, self.transport.gpudirect_host_flush_required);

        for value in [
            self.capacity.active_rooms_per_pair,
            self.capacity.native_transfers_per_pair,
            self.capacity.kv_pages_per_room,
            self.capacity.leased_kv_pages_per_endpoint,
            self.capacity.aux_slots_per_endpoint,
            self.capacity.completion_slots_per_endpoint,
            self.capacity.request_slots_per_endpoint,
            self.capacity.pending_transfer_bytes_per_pair,
            self.capacity.tombstones_per_pair,
        ] {
            push_u64(&mut output, value);
        }

        for value in [
            self.deadline_ms.local_initialization,
            self.deadline_ms.connect_and_hello,
            self.deadline_ms.room_rendezvous,
            self.deadline_ms.destination_allocation,
            self.deadline_ms.native_transfer,
            self.deadline_ms.completion_ack,
            self.deadline_ms.abort_ack,
            self.deadline_ms.heartbeat_interval,
        ] {
            push_u64(&mut output, value);
        }
        output.push(self.deadline_ms.heartbeat_misses);
        push_u64(&mut output, self.deadline_ms.tombstone_retention);

        Ok(output)
    }

    pub fn digest(&self) -> Result<[u8; 32], ProfileError> {
        let body = self.canonical_body()?;
        let mut hasher = Sha256::new();
        hasher.update(PROFILE_DOMAIN);
        hasher.update(body);
        Ok(hasher.finalize().into())
    }

    pub fn digest_hex(&self) -> Result<String, ProfileError> {
        Ok(hex::encode(self.digest()?))
    }

    fn validate(&self) -> Result<(), ProfileError> {
        validate_sorted_unique(
            "model.enabled_runtime_features",
            &self.model.enabled_runtime_features,
        )?;
        validate_sorted_unique("request.input_kinds", &self.request.input_kinds)?;
        validate_sorted_unique("request.features", &self.request.features)?;
        validate_sorted_unique("transport.components", &self.transport.components)?;
        validate_sorted_unique("transport.rdma_devices", &self.transport.rdma_devices)?;
        validate_finite("request.temperature", self.request.temperature)?;
        validate_finite("request.top_p", self.request.top_p)?;
        validate_finite("request.min_p", self.request.min_p)?;
        validate_lower_hex("transport.commit", &self.transport.commit, 20)?;
        validate_lower_hex("transport.header_sha256", &self.transport.header_sha256, 32)?;
        Ok(())
    }
}

fn validate_sorted_unique(field: &'static str, values: &[String]) -> Result<(), ProfileError> {
    if values.windows(2).any(|items| items[0] >= items[1]) {
        return Err(ProfileError::InvalidField {
            field,
            detail: "must be strictly sorted by encoded bytes without duplicates".into(),
        });
    }
    Ok(())
}

fn validate_finite(field: &'static str, value: f64) -> Result<(), ProfileError> {
    if !value.is_finite() {
        return Err(ProfileError::InvalidField {
            field,
            detail: "must be a finite binary64 value".into(),
        });
    }
    Ok(())
}

fn validate_lower_hex(
    field: &'static str,
    value: &str,
    expected_bytes: usize,
) -> Result<(), ProfileError> {
    if value.len() != expected_bytes * 2
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ProfileError::InvalidField {
            field,
            detail: format!("must be {} bytes of lowercase hexadecimal", expected_bytes),
        });
    }
    Ok(())
}

fn push_string(output: &mut Vec<u8>, value: &str) -> Result<(), ProfileError> {
    let length = u32::try_from(value.len()).map_err(|_| ProfileError::InvalidField {
        field: "string",
        detail: "UTF-8 value exceeds u32 length".into(),
    })?;
    push_u32(output, length);
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn push_string_list(
    output: &mut Vec<u8>,
    field: &'static str,
    values: &[String],
) -> Result<(), ProfileError> {
    validate_sorted_unique(field, values)?;
    let count = u32::try_from(values.len()).map_err(|_| ProfileError::InvalidField {
        field,
        detail: "list exceeds u32 item count".into(),
    })?;
    push_u32(output, count);
    let mut encoded_items = Vec::with_capacity(values.len());
    for value in values {
        let mut encoded = Vec::with_capacity(4 + value.len());
        push_string(&mut encoded, value)?;
        encoded_items.push(encoded);
    }
    encoded_items.sort_unstable();
    for encoded in encoded_items {
        output.extend_from_slice(&encoded);
    }
    Ok(())
}

fn push_fixed_hex(
    output: &mut Vec<u8>,
    field: &'static str,
    value: &str,
    expected_bytes: usize,
) -> Result<(), ProfileError> {
    validate_lower_hex(field, value, expected_bytes)?;
    let decoded = hex::decode(value).map_err(|error| ProfileError::InvalidField {
        field,
        detail: error.to_string(),
    })?;
    output.extend_from_slice(&decoded);
    Ok(())
}

fn push_f64(output: &mut Vec<u8>, value: f64) -> Result<(), ProfileError> {
    validate_finite("float", value)?;
    let normalized = if value == 0.0 { 0.0 } else { value };
    output.extend_from_slice(&normalized.to_bits().to_be_bytes());
    Ok(())
}

fn push_bool(output: &mut Vec<u8>, value: bool) {
    output.push(u8::from(value));
}

fn push_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_be_bytes());
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_be_bytes());
}

fn push_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_be_bytes());
}

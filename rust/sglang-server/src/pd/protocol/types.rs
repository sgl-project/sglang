use std::fmt;
use std::io::Cursor;

use serde::de::Visitor;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;
use uuid::{Variant, Version};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FixedBytes<const N: usize>([u8; N]);

impl<const N: usize> FixedBytes<N> {
    pub const fn new(bytes: [u8; N]) -> Self {
        Self(bytes)
    }

    pub fn from_hex(value: &str) -> Result<Self, PayloadError> {
        if value.len() != N * 2
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(PayloadError::InvalidField {
                field: "fixed_bytes",
                detail: format!("expected {N} bytes of lowercase hexadecimal"),
            });
        }
        let decoded = hex::decode(value).map_err(|error| PayloadError::InvalidField {
            field: "fixed_bytes",
            detail: error.to_string(),
        })?;
        let bytes = decoded.try_into().map_err(|_| PayloadError::InvalidField {
            field: "fixed_bytes",
            detail: format!("expected exactly {N} bytes"),
        })?;
        Ok(Self(bytes))
    }

    pub const fn as_array(&self) -> &[u8; N] {
        &self.0
    }

    pub const fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub const fn into_array(self) -> [u8; N] {
        self.0
    }
}

impl<const N: usize> fmt::Debug for FixedBytes<N> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FixedBytes")
            .field("length", &N)
            .finish_non_exhaustive()
    }
}

impl<const N: usize> Serialize for FixedBytes<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(&self.0)
    }
}

struct FixedBytesVisitor<const N: usize>;

impl<const N: usize> FixedBytesVisitor<N> {
    fn from_bytes<E>(bytes: &[u8]) -> Result<FixedBytes<N>, E>
    where
        E: serde::de::Error,
    {
        let bytes: [u8; N] = bytes
            .try_into()
            .map_err(|_| E::custom(format!("expected exactly {N} binary bytes")))?;
        if N == 16 {
            let uuid_bytes: &[u8; 16] = bytes
                .as_slice()
                .try_into()
                .map_err(|_| E::custom("expected exactly 16 UUID bytes"))?;
            validate_uuid_v4(uuid_bytes).map_err(E::custom)?;
        }
        Ok(FixedBytes::new(bytes))
    }
}

impl<'de, const N: usize> Visitor<'de> for FixedBytesVisitor<N> {
    type Value = FixedBytes<N>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{N} binary bytes, lowercase hex, or a canonical UUIDv4"
        )
    }

    fn visit_bytes<E>(self, value: &[u8]) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Self::from_bytes(value)
    }

    fn visit_byte_buf<E>(self, value: Vec<u8>) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Self::from_bytes(&value)
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        if N == 16 && value.contains('-') {
            let uuid = uuid::Uuid::parse_str(value).map_err(E::custom)?;
            if uuid.to_string() != value {
                return Err(E::custom("UUID must be lowercase canonical text"));
            }
            validate_uuid_v4(uuid.as_bytes()).map_err(E::custom)?;
            return Self::from_bytes(uuid.as_bytes());
        }
        FixedBytes::from_hex(value).map_err(E::custom)
    }
}

impl<'de, const N: usize> Deserialize<'de> for FixedBytes<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(FixedBytesVisitor::<N>)
    }
}

fn validate_uuid_v4(bytes: &[u8; 16]) -> Result<(), &'static str> {
    let uuid = uuid::Uuid::from_bytes(*bytes);
    if uuid.get_version() != Some(Version::Random) || uuid.get_variant() != Variant::RFC4122 {
        return Err("UUID must be canonical RFC4122 version 4");
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanDigest(Vec<u8>);

impl PlanDigest {
    pub fn empty() -> Self {
        Self(Vec::new())
    }

    pub fn from_digest(digest: FixedBytes<32>) -> Self {
        Self(digest.as_bytes().to_vec())
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Serialize for PlanDigest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(&self.0)
    }
}

impl<'de> Deserialize<'de> for PlanDigest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct PlanDigestVisitor;

        impl<'de> Visitor<'de> for PlanDigestVisitor {
            type Value = PlanDigest;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an empty binary value or a 32-byte digest")
            }

            fn visit_bytes<E>(self, value: &[u8]) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if value.is_empty() || value.len() == 32 {
                    Ok(PlanDigest(value.to_vec()))
                } else {
                    Err(E::invalid_length(value.len(), &"0 or 32 binary bytes"))
                }
            }

            fn visit_byte_buf<E>(self, value: Vec<u8>) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                self.visit_bytes(&value)
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if value.is_empty() {
                    return Ok(PlanDigest::empty());
                }
                FixedBytes::<32>::from_hex(value)
                    .map(PlanDigest::from_digest)
                    .map_err(E::custom)
            }
        }

        deserializer.deserialize_any(PlanDigestVisitor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    DecodeToPrefill,
    PrefillToDecode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthKind {
    Psk,
    Session,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Prefill,
    Decode,
}

impl Serialize for Role {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        })
    }
}

impl<'de> Deserialize<'de> for Role {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct RoleVisitor;

        impl Visitor<'_> for RoleVisitor {
            type Value = Role;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("prefill or decode")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "prefill" => Ok(Role::Prefill),
                    "decode" => Ok(Role::Decode),
                    _ => Err(E::unknown_variant(value, &["prefill", "decode"])),
                }
            }
        }

        deserializer.deserialize_str(RoleVisitor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum MessageKind {
    ClientHello = 1,
    ServerHello = 2,
    SessionReady = 3,
    SessionReadyAck = 4,
    RegisterRegions = 5,
    RegisterRegionsAck = 6,
    ProbeReady = 7,
    ProbeAck = 8,
    PrepareRoom = 9,
    PrepareAccepted = 10,
    PrepareRejected = 11,
    DataReady = 12,
    TransferComplete = 13,
    TransferCompleteAck = 14,
    TransferFailed = 15,
    Abort = 16,
    AbortAck = 17,
    Ping = 18,
    Pong = 19,
    GoAway = 20,
    GoAwayAck = 21,
}

impl TryFrom<u16> for MessageKind {
    type Error = PayloadError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        Ok(match value {
            1 => Self::ClientHello,
            2 => Self::ServerHello,
            3 => Self::SessionReady,
            4 => Self::SessionReadyAck,
            5 => Self::RegisterRegions,
            6 => Self::RegisterRegionsAck,
            7 => Self::ProbeReady,
            8 => Self::ProbeAck,
            9 => Self::PrepareRoom,
            10 => Self::PrepareAccepted,
            11 => Self::PrepareRejected,
            12 => Self::DataReady,
            13 => Self::TransferComplete,
            14 => Self::TransferCompleteAck,
            15 => Self::TransferFailed,
            16 => Self::Abort,
            17 => Self::AbortAck,
            18 => Self::Ping,
            19 => Self::Pong,
            20 => Self::GoAway,
            21 => Self::GoAwayAck,
            _ => return Err(PayloadError::UnknownKind(value)),
        })
    }
}

impl MessageKind {
    pub const fn auth(self) -> AuthKind {
        match self {
            Self::ClientHello | Self::ServerHello => AuthKind::Psk,
            _ => AuthKind::Session,
        }
    }

    pub const fn allows(self, direction: Direction) -> bool {
        match self {
            Self::ClientHello
            | Self::SessionReady
            | Self::RegisterRegions
            | Self::ProbeAck
            | Self::PrepareRoom
            | Self::TransferComplete => matches!(direction, Direction::DecodeToPrefill),
            Self::ServerHello
            | Self::SessionReadyAck
            | Self::RegisterRegionsAck
            | Self::ProbeReady
            | Self::PrepareAccepted
            | Self::PrepareRejected
            | Self::DataReady
            | Self::TransferCompleteAck => matches!(direction, Direction::PrefillToDecode),
            Self::TransferFailed
            | Self::Abort
            | Self::AbortAck
            | Self::Ping
            | Self::Pong
            | Self::GoAway
            | Self::GoAwayAck => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientHello {
    pub role: Role,
    pub rank: u16,
    pub process_epoch: FixedBytes<16>,
    pub gpu: u16,
    pub tp: u16,
    pub pp: u16,
    pub dp: u16,
    pub capabilities: u64,
    pub profile_digest: FixedBytes<32>,
    pub model_manifest_digest: FixedBytes<32>,
    pub tokenizer_manifest_digest: FixedBytes<32>,
    pub layout_fingerprint: FixedBytes<32>,
    pub native_abi_digest: FixedBytes<32>,
    pub psk_id: FixedBytes<8>,
    pub nonce: FixedBytes<32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerHello {
    pub role: Role,
    pub rank: u16,
    pub process_epoch: FixedBytes<16>,
    pub gpu: u16,
    pub tp: u16,
    pub pp: u16,
    pub dp: u16,
    pub capabilities: u64,
    pub profile_digest: FixedBytes<32>,
    pub model_manifest_digest: FixedBytes<32>,
    pub tokenizer_manifest_digest: FixedBytes<32>,
    pub layout_fingerprint: FixedBytes<32>,
    pub native_abi_digest: FixedBytes<32>,
    pub psk_id: FixedBytes<8>,
    pub nonce: FixedBytes<32>,
    pub client_hello_hash: FixedBytes<32>,
    pub accepted: bool,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranscriptConfirmation {
    pub transcript_hash: FixedBytes<32>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionRecord {
    pub region_id: u16,
    pub remote_base_addr: u64,
    pub length_bytes: u64,
    pub location: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterRegions {
    pub registration_epoch: FixedBytes<16>,
    pub layout_fingerprint: FixedBytes<32>,
    pub mooncake_host: String,
    pub mooncake_port: u16,
    pub regions: Vec<RegionRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistrationDecision {
    pub registration_epoch: FixedBytes<16>,
    pub accepted: bool,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeReady {
    pub registration_epoch: FixedBytes<16>,
    pub probe_generation: u64,
    pub aux_slot: u16,
    pub probe_data: FixedBytes<64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeAck {
    pub registration_epoch: FixedBytes<16>,
    pub probe_generation: u64,
    pub aux_slot: u16,
    pub accepted: bool,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoomFields {
    pub decode_process_epoch: FixedBytes<16>,
    pub bootstrap_room: u64,
    pub attempt_id: FixedBytes<16>,
    pub generation: u64,
    pub request_contract_digest: FixedBytes<32>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DestinationBlock {
    pub region_id: u16,
    pub destination_page: u32,
    pub byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvBlock {
    pub region_id: u16,
    pub source_page: u32,
    pub destination_page: u32,
    pub byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrepareRoom {
    #[serde(flatten)]
    pub room: RoomFields,
    pub destination_registration_epoch: FixedBytes<16>,
    pub destination_blocks: Vec<DestinationBlock>,
    pub destination_aux_slot: u16,
    pub destination_completion_slot: u16,
    pub valid_token_count: u32,
    pub chunk_sequence: u32,
    pub chunk_count: u32,
    pub is_last_chunk: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrepareAccepted {
    #[serde(flatten)]
    pub room: RoomFields,
    pub source_registration_epoch: FixedBytes<16>,
    pub destination_registration_epoch: FixedBytes<16>,
    pub kv_blocks: Vec<KvBlock>,
    pub source_aux_slot: u16,
    pub destination_aux_slot: u16,
    pub source_completion_slot: u16,
    pub destination_completion_slot: u16,
    pub valid_token_count: u32,
    pub chunk_sequence: u32,
    pub chunk_count: u32,
    pub is_last_chunk: bool,
    pub transfer_plan_digest: FixedBytes<32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrepareRejected {
    #[serde(flatten)]
    pub room: RoomFields,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedRoom {
    #[serde(flatten)]
    pub room: RoomFields,
    pub transfer_plan_digest: FixedBytes<32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalRoom {
    #[serde(flatten)]
    pub room: RoomFields,
    pub transfer_plan_digest: PlanDigest,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PingPong {
    pub ping_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Drain {
    pub drain_generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlPayload {
    ClientHello(ClientHello),
    ServerHello(ServerHello),
    SessionReady(TranscriptConfirmation),
    SessionReadyAck(TranscriptConfirmation),
    RegisterRegions(RegisterRegions),
    RegisterRegionsAck(RegistrationDecision),
    ProbeReady(ProbeReady),
    ProbeAck(ProbeAck),
    PrepareRoom(PrepareRoom),
    PrepareAccepted(PrepareAccepted),
    PrepareRejected(PrepareRejected),
    DataReady(PlannedRoom),
    TransferComplete(PlannedRoom),
    TransferCompleteAck(PlannedRoom),
    TransferFailed(TerminalRoom),
    Abort(TerminalRoom),
    AbortAck(TerminalRoom),
    Ping(PingPong),
    Pong(PingPong),
    GoAway(Drain),
    GoAwayAck(Drain),
}

impl ControlPayload {
    pub fn kind(&self) -> MessageKind {
        match self {
            Self::ClientHello(_) => MessageKind::ClientHello,
            Self::ServerHello(_) => MessageKind::ServerHello,
            Self::SessionReady(_) => MessageKind::SessionReady,
            Self::SessionReadyAck(_) => MessageKind::SessionReadyAck,
            Self::RegisterRegions(_) => MessageKind::RegisterRegions,
            Self::RegisterRegionsAck(_) => MessageKind::RegisterRegionsAck,
            Self::ProbeReady(_) => MessageKind::ProbeReady,
            Self::ProbeAck(_) => MessageKind::ProbeAck,
            Self::PrepareRoom(_) => MessageKind::PrepareRoom,
            Self::PrepareAccepted(_) => MessageKind::PrepareAccepted,
            Self::PrepareRejected(_) => MessageKind::PrepareRejected,
            Self::DataReady(_) => MessageKind::DataReady,
            Self::TransferComplete(_) => MessageKind::TransferComplete,
            Self::TransferCompleteAck(_) => MessageKind::TransferCompleteAck,
            Self::TransferFailed(_) => MessageKind::TransferFailed,
            Self::Abort(_) => MessageKind::Abort,
            Self::AbortAck(_) => MessageKind::AbortAck,
            Self::Ping(_) => MessageKind::Ping,
            Self::Pong(_) => MessageKind::Pong,
            Self::GoAway(_) => MessageKind::GoAway,
            Self::GoAwayAck(_) => MessageKind::GoAwayAck,
        }
    }

    pub fn from_json(kind: MessageKind, value: serde_json::Value) -> Result<Self, PayloadError> {
        macro_rules! parse {
            ($type:ty, $variant:ident) => {
                serde_json::from_value::<$type>(value)
                    .map(Self::$variant)
                    .map_err(PayloadError::Json)
            };
        }
        let payload = match kind {
            MessageKind::ClientHello => parse!(ClientHello, ClientHello),
            MessageKind::ServerHello => parse!(ServerHello, ServerHello),
            MessageKind::SessionReady => parse!(TranscriptConfirmation, SessionReady),
            MessageKind::SessionReadyAck => parse!(TranscriptConfirmation, SessionReadyAck),
            MessageKind::RegisterRegions => parse!(RegisterRegions, RegisterRegions),
            MessageKind::RegisterRegionsAck => {
                parse!(RegistrationDecision, RegisterRegionsAck)
            }
            MessageKind::ProbeReady => parse!(ProbeReady, ProbeReady),
            MessageKind::ProbeAck => parse!(ProbeAck, ProbeAck),
            MessageKind::PrepareRoom => parse!(PrepareRoom, PrepareRoom),
            MessageKind::PrepareAccepted => parse!(PrepareAccepted, PrepareAccepted),
            MessageKind::PrepareRejected => parse!(PrepareRejected, PrepareRejected),
            MessageKind::DataReady => parse!(PlannedRoom, DataReady),
            MessageKind::TransferComplete => parse!(PlannedRoom, TransferComplete),
            MessageKind::TransferCompleteAck => parse!(PlannedRoom, TransferCompleteAck),
            MessageKind::TransferFailed => parse!(TerminalRoom, TransferFailed),
            MessageKind::Abort => parse!(TerminalRoom, Abort),
            MessageKind::AbortAck => parse!(TerminalRoom, AbortAck),
            MessageKind::Ping => parse!(PingPong, Ping),
            MessageKind::Pong => parse!(PingPong, Pong),
            MessageKind::GoAway => parse!(Drain, GoAway),
            MessageKind::GoAwayAck => parse!(Drain, GoAwayAck),
        }?;
        payload.validate()?;
        Ok(payload)
    }

    pub(crate) fn from_value(kind: MessageKind, value: rmpv::Value) -> Result<Self, PayloadError> {
        macro_rules! parse {
            ($type:ty, $variant:ident) => {
                rmpv::ext::from_value::<$type>(value)
                    .map(Self::$variant)
                    .map_err(PayloadError::MessagePackValue)
            };
        }
        let payload = match kind {
            MessageKind::ClientHello => parse!(ClientHello, ClientHello),
            MessageKind::ServerHello => parse!(ServerHello, ServerHello),
            MessageKind::SessionReady => parse!(TranscriptConfirmation, SessionReady),
            MessageKind::SessionReadyAck => parse!(TranscriptConfirmation, SessionReadyAck),
            MessageKind::RegisterRegions => parse!(RegisterRegions, RegisterRegions),
            MessageKind::RegisterRegionsAck => {
                parse!(RegistrationDecision, RegisterRegionsAck)
            }
            MessageKind::ProbeReady => parse!(ProbeReady, ProbeReady),
            MessageKind::ProbeAck => parse!(ProbeAck, ProbeAck),
            MessageKind::PrepareRoom => parse!(PrepareRoom, PrepareRoom),
            MessageKind::PrepareAccepted => parse!(PrepareAccepted, PrepareAccepted),
            MessageKind::PrepareRejected => parse!(PrepareRejected, PrepareRejected),
            MessageKind::DataReady => parse!(PlannedRoom, DataReady),
            MessageKind::TransferComplete => parse!(PlannedRoom, TransferComplete),
            MessageKind::TransferCompleteAck => parse!(PlannedRoom, TransferCompleteAck),
            MessageKind::TransferFailed => parse!(TerminalRoom, TransferFailed),
            MessageKind::Abort => parse!(TerminalRoom, Abort),
            MessageKind::AbortAck => parse!(TerminalRoom, AbortAck),
            MessageKind::Ping => parse!(PingPong, Ping),
            MessageKind::Pong => parse!(PingPong, Pong),
            MessageKind::GoAway => parse!(Drain, GoAway),
            MessageKind::GoAwayAck => parse!(Drain, GoAwayAck),
        }?;
        payload.validate()?;
        Ok(payload)
    }

    pub(crate) fn to_value(&self) -> Result<rmpv::Value, PayloadError> {
        macro_rules! value {
            ($inner:expr) => {
                serialize_struct_value($inner)
            };
        }
        match self {
            Self::ClientHello(inner) => value!(inner),
            Self::ServerHello(inner) => value!(inner),
            Self::SessionReady(inner) | Self::SessionReadyAck(inner) => value!(inner),
            Self::RegisterRegions(inner) => value!(inner),
            Self::RegisterRegionsAck(inner) => value!(inner),
            Self::ProbeReady(inner) => value!(inner),
            Self::ProbeAck(inner) => value!(inner),
            Self::PrepareRoom(inner) => value!(inner),
            Self::PrepareAccepted(inner) => value!(inner),
            Self::PrepareRejected(inner) => value!(inner),
            Self::DataReady(inner)
            | Self::TransferComplete(inner)
            | Self::TransferCompleteAck(inner) => value!(inner),
            Self::TransferFailed(inner) | Self::Abort(inner) | Self::AbortAck(inner) => {
                value!(inner)
            }
            Self::Ping(inner) | Self::Pong(inner) => value!(inner),
            Self::GoAway(inner) | Self::GoAwayAck(inner) => value!(inner),
        }
    }
}

fn serialize_struct_value<T>(value: &T) -> Result<rmpv::Value, PayloadError>
where
    T: Serialize,
{
    let mut bytes = Vec::new();
    value
        .serialize(&mut rmp_serde::Serializer::new(&mut bytes).with_struct_map())
        .map_err(PayloadError::MessagePackEncode)?;
    let mut cursor = Cursor::new(bytes);
    rmpv::decode::read_value(&mut cursor).map_err(PayloadError::MessagePackDecode)
}

#[derive(Debug, Error)]
pub enum PayloadError {
    #[error("unknown PD control kind {0}")]
    UnknownKind(u16),
    #[error("invalid PD payload JSON: {0}")]
    Json(serde_json::Error),
    #[error("invalid PD MessagePack payload: {0}")]
    MessagePackValue(rmpv::ext::Error),
    #[error("could not serialize typed PD MessagePack payload: {0}")]
    MessagePackEncode(rmp_serde::encode::Error),
    #[error("could not decode typed PD MessagePack payload: {0}")]
    MessagePackDecode(rmpv::decode::Error),
    #[error("invalid PD payload field {field}: {detail}")]
    InvalidField { field: &'static str, detail: String },
}

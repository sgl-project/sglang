//! Hand-written extensions on the generated types: the accessors and
//! constructors server logic needs beyond the wire shapes. Grows with each
//! migration stage.

use std::borrow::Cow;

use crate::api::v1::{FinishReason, FinishStop, Matched, MatchedTokens, finish_reason, matched};

impl FinishReason {
    /// A stop reason (`type: "stop"`), optionally carrying what matched.
    pub fn stop(matched: Option<matched::Value>) -> Self {
        FinishReason {
            kind: Some(finish_reason::Kind::Stop(FinishStop {
                matched: matched.map(|value| Matched { value: Some(value) }),
            })),
        }
    }

    /// Returns the wire `type` without reserializing the finish reason (an
    /// unknown reason's is read out of its raw JSON, hence the `Cow`).
    pub fn kind_name(&self) -> Option<Cow<'_, str>> {
        match self.kind.as_ref()? {
            finish_reason::Kind::Stop(_) => Some(Cow::Borrowed("stop")),
            finish_reason::Kind::Length(_) => Some(Cow::Borrowed("length")),
            finish_reason::Kind::Abort(_) => Some(Cow::Borrowed("abort")),
            finish_reason::Kind::Unknown(raw) => {
                let parsed: serde_json::Value = serde_json::from_str(raw).ok()?;
                parsed
                    .get("type")
                    .and_then(serde_json::Value::as_str)
                    .map(|s| Cow::Owned(s.to_owned()))
            }
        }
    }

    /// The stop this request matched, if it stopped on one. `None` for
    /// length/abort and for an unknown type.
    pub fn matched(&self) -> Option<&matched::Value> {
        match self.kind.as_ref()? {
            finish_reason::Kind::Stop(stop) => stop.matched.as_ref()?.value.as_ref(),
            _ => None,
        }
    }

    /// `Some((status, message))` when this is an abort carrying a `status_code` —
    /// a scheduler-side request error the API surfaces as that HTTP status instead
    /// of as a normal completion. A plain abort (no code) reads as `None`.
    pub fn abort_status(&self) -> Option<(u16, &str)> {
        match self.kind.as_ref()? {
            finish_reason::Kind::Abort(abort) => Some((
                u16::try_from(abort.status_code?).ok()?,
                abort.message.as_deref().unwrap_or("request aborted"),
            )),
            _ => None,
        }
    }
}

impl From<Vec<i64>> for MatchedTokens {
    fn from(ids: Vec<i64>) -> Self {
        MatchedTokens { ids }
    }
}

/// The shared JSON contract encodes a protobuf bytes field as canonical padded
/// base64, never a JSON integer array. gRPC keeps the binary protobuf field.
pub(crate) fn encode_bytes(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

pub(crate) struct JsonBytes(pub Vec<u8>);

impl<'de> serde::Deserialize<'de> for JsonBytes {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use base64::Engine;
        let text = <String as serde::Deserialize>::deserialize(deserializer)?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&text)
            .map_err(|_| serde::de::Error::custom("invalid base64 bytes"))?;
        if encode_bytes(&bytes) != text {
            return Err(serde::de::Error::custom("invalid base64 bytes"));
        }
        Ok(Self(bytes))
    }
}

//! Hand-written extensions on the generated types: the accessors and
//! constructors server logic needs beyond the wire shapes. Grows with each
//! migration stage.

use std::borrow::Cow;

use crate::api::v1::{
    FinishReason, FinishStop, GenerateResponse, GenerateStreamError, Matched, MatchedTokens,
    finish_reason, matched,
};

/// Decode one HTTP `/generate` object or a batch into successes or engine errors.
pub fn decode_generate_http_response(
    bytes: &[u8],
) -> serde_json::Result<Vec<Result<GenerateResponse, GenerateStreamError>>> {
    fn item(
        value: serde_json::Value,
    ) -> serde_json::Result<Result<GenerateResponse, GenerateStreamError>> {
        if value.get("error").is_some() {
            serde_json::from_value::<GenerateStreamError>(value).map(Err)
        } else {
            serde_json::from_value::<GenerateResponse>(value).map(Ok)
        }
    }

    match serde_json::from_slice(bytes)? {
        serde_json::Value::Array(items) => items.into_iter().map(item).collect(),
        value => item(value).map(|item| vec![item]),
    }
}

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

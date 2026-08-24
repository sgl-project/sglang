//! The terminal finish reason: Python's `FinishReasonDict` — what
//! `BaseFinishReason.to_json()` (schedule_batch.py) puts on the response, and
//! what the API echoes back as `meta_info.finish_reason`.

use serde::{Deserialize, Serialize};

/// The stop that ended a request — Python's `matched` key, typed
/// `Union[str, int, List[int]]`: a stop token id, a stop string
/// (`FINISH_MATCHED_STR` / `FINISHED_MATCHED_REGEX`), or a multi-token stop
/// sequence. Untagged because the three wire shapes are disjoint, so the shape
/// alone picks the arm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Matched {
    Token(i64),
    Str(String),
    Tokens(Vec<i64>),
}

/// The finish reasons this build knows, keyed by the `type` tag Python writes.
/// Every field is optional so a reason missing one still classifies (and keeps its
/// tag) instead of falling through to [`FinishReason::Unknown`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum FinishKind {
    /// `FINISH_MATCHED_TOKEN` / `FINISH_MATCHED_STR` / `FINISHED_MATCHED_REGEX` —
    /// all three report `type: "stop"`, to match the OpenAI API's value.
    Stop {
        #[serde(default)]
        matched: Option<Matched>,
    },
    /// `FINISH_LENGTH` — hit `max_new_tokens` (or the context limit).
    Length {
        #[serde(default)]
        length: Option<u64>,
    },
    /// `FINISH_ABORT` — a scheduler-side termination. Boxed because it is by far
    /// the widest variant and the rarest: unboxed it would set the size of every
    /// [`ChunkEvent`], including the plain stop/length ones (see
    /// `chunk_event_frame_stays_small`).
    Abort(Box<AbortReason>),
}

/// The `FINISH_ABORT` payload. `status_code`/`err_type` are `None` for a plain
/// abort and set for a request error (e.g. over-context → 400); Python emits all
/// three keys, nulls included, so none of them is skipped on the way out.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbortReason {
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub status_code: Option<u16>,
    #[serde(default)]
    pub err_type: Option<String>,
}

/// A terminal finish reason exactly as `BaseFinishReason.to_json()`
/// (schedule_batch.py) puts it on the wire — Python's `FinishReasonDict`. It
/// round-trips: the API echoes it verbatim as `meta_info.finish_reason`, so the
/// serialized form must stay key-for-key what Python would have sent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FinishReason {
    Known(FinishKind),
    /// A `type` this build doesn't know, kept as its raw map and echoed unchanged.
    /// This arm is why the outer enum is untagged: a finish reason added Python-side
    /// must not fail the header decode, which rejects the whole frame — every
    /// request in the batch, not just the one that carried it.
    // Keep the native frame compact even when HTTP/rendering dependencies turn
    // on serde_json's large `preserve_order` map representation.
    Unknown(Box<serde_json::Map<String, serde_json::Value>>),
}

impl From<FinishKind> for FinishReason {
    fn from(kind: FinishKind) -> Self {
        FinishReason::Known(kind)
    }
}

impl FinishReason {
    /// Returns the wire type without reserializing the finish reason.
    pub fn kind_name(&self) -> Option<&str> {
        match self {
            FinishReason::Known(FinishKind::Stop { .. }) => Some("stop"),
            FinishReason::Known(FinishKind::Length { .. }) => Some("length"),
            FinishReason::Known(FinishKind::Abort(_)) => Some("abort"),
            FinishReason::Unknown(fields) => fields.get("type").and_then(|value| value.as_str()),
        }
    }

    /// The stop this request matched, if it stopped on one. `None` for
    /// length/abort and for an unknown type.
    pub fn matched(&self) -> Option<&Matched> {
        match self {
            FinishReason::Known(FinishKind::Stop { matched }) => matched.as_ref(),
            _ => None,
        }
    }

    /// `Some((status, message))` when this is an abort carrying a `status_code` —
    /// a scheduler-side request error the API surfaces as that HTTP status instead
    /// of as a normal completion. A plain abort (no code) reads as `None`.
    pub fn abort_status(&self) -> Option<(u16, &str)> {
        match self {
            FinishReason::Known(FinishKind::Abort(a)) => Some((
                a.status_code?,
                a.message.as_deref().unwrap_or("request aborted"),
            )),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse a finish reason from the map Python puts on the wire, so these tests
    /// exercise the real deserialization and not just the classifier.
    fn fr(v: serde_json::Value) -> Option<FinishReason> {
        Some(serde_json::from_value(v).expect("finish reason must parse"))
    }

    /// Classify through an `Option`, the shape both API paths actually hold.
    /// Owned message so a caller can classify a temporary inline.
    fn abort_status(f: &Option<FinishReason>) -> Option<(u16, String)> {
        let (code, message) = f.as_ref()?.abort_status()?;
        Some((code, message.to_string()))
    }

    /// The classifier both API paths use: a validation abort yields its
    /// `(code, message)` (the streaming path turns this into an SSE error event
    /// instead of a normal `Done` frame); anything else yields `None`.
    #[test]
    fn abort_status_extracts_code_and_message() {
        let (code, msg) = abort_status(&fr(serde_json::json!({
            "type": "abort", "message": "over the limit", "status_code": 400
        })))
        .expect("validation abort → (code, message)");
        assert_eq!(code, 400);
        assert_eq!(msg, "over the limit");
        // Normal finish, bare abort (no status), and no finish → not an error.
        assert!(abort_status(&fr(serde_json::json!({"type": "stop"}))).is_none());
        assert!(abort_status(&fr(serde_json::json!({"type": "abort"}))).is_none());
        assert!(abort_status(&None).is_none());
    }

    /// A normal finish, a bare abort (no status), and no finish are not errors
    /// (the unary path returns them as a 200 result frame).
    #[test]
    fn non_error_finishes_stay_ok() {
        assert!(abort_status(&fr(serde_json::json!({"type": "stop", "matched": 5}))).is_none());
        assert!(abort_status(&fr(serde_json::json!({"type": "length", "length": 8}))).is_none());
        assert!(
            abort_status(&fr(
                serde_json::json!({"type": "abort", "message": "Aborted"})
            ))
            .is_none()
        );
        assert!(abort_status(&None).is_none());
    }

    /// The `matched` accessor reads only a stop, and reads every shape of one.
    #[test]
    fn matched_reads_stops_only() {
        let m = |v| fr(v).unwrap().matched().cloned();
        assert_eq!(
            m(serde_json::json!({"type": "stop", "matched": 9})),
            Some(Matched::Token(9))
        );
        assert_eq!(
            m(serde_json::json!({"type": "stop", "matched": "</s>"})),
            Some(Matched::Str("</s>".into()))
        );
        assert_eq!(
            m(serde_json::json!({"type": "stop", "matched": [9, 10]})),
            Some(Matched::Tokens(vec![9, 10]))
        );
        // A stop with no `matched`, and the non-stop reasons.
        assert_eq!(m(serde_json::json!({"type": "stop"})), None);
        assert_eq!(m(serde_json::json!({"type": "length", "length": 8})), None);
        assert_eq!(m(serde_json::json!({"type": "abort"})), None);
    }

    /// Python's abort dict always carries `status_code`/`err_type`, null included,
    /// and the reason is echoed verbatim into `meta_info.finish_reason` — so the
    /// typed form must serialize back key-for-key, nulls and all. A dropped null
    /// key is a silently changed response body.
    #[test]
    fn finish_reason_round_trips_python_shapes() {
        for wire in [
            serde_json::json!({"type": "stop", "matched": 9}),
            serde_json::json!({"type": "stop", "matched": "</s>"}),
            serde_json::json!({"type": "stop", "matched": [9, 10]}),
            serde_json::json!({"type": "length", "length": 8}),
            serde_json::json!({
                "type": "abort", "message": "Aborted",
                "status_code": null, "err_type": null
            }),
            serde_json::json!({
                "type": "abort", "message": "over the limit",
                "status_code": 400, "err_type": "BadRequestError"
            }),
        ] {
            let parsed: FinishReason = serde_json::from_value(wire.clone()).unwrap();
            assert!(
                matches!(parsed, FinishReason::Known(_)),
                "must classify, not fall back to Unknown: {wire}"
            );
            assert_eq!(serde_json::to_value(&parsed).unwrap(), wire);
        }
    }

    /// A `type` this build doesn't know must not fail the decode — the header holds
    /// the whole batch, so a rejected frame drops every request in it, not just the
    /// one carrying the new reason. It is kept verbatim and echoed unchanged.
    #[test]
    fn unknown_finish_type_is_preserved_not_rejected() {
        let wire = serde_json::json!({"type": "tool_calls", "name": "search"});
        let parsed: FinishReason = serde_json::from_value(wire.clone()).unwrap();
        assert!(matches!(parsed, FinishReason::Unknown(_)));
        assert_eq!(serde_json::to_value(&parsed).unwrap(), wire);
        // Unknown reasons classify as neither an abort nor a matched stop.
        assert!(parsed.abort_status().is_none());
        assert!(parsed.matched().is_none());
    }
}

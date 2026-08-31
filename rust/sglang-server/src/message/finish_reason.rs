//! The terminal finish reason — re-exported from the schema-generated types:
//! `proto/sglang/api/v1/finish.proto` is the ground truth for Python's
//! `FinishReasonDict` (`BaseFinishReason.to_json()` in schedule_batch.py),
//! which the API echoes back as `meta_info.finish_reason`. The accessors
//! (`kind_name`, `matched`, `abort_status`) live in sglang-api-types' ext.

pub use sglang_api_types::api::v1::FinishReason;
#[cfg(test)]
pub use sglang_api_types::api::v1::finish_reason;
pub use sglang_api_types::api::v1::matched::Value as Matched;

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
            Some(Matched::Tokens(vec![9, 10].into()))
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
                !matches!(parsed.kind, Some(finish_reason::Kind::Unknown(_))),
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
        assert!(matches!(parsed.kind, Some(finish_reason::Kind::Unknown(_))));
        assert_eq!(serde_json::to_value(&parsed).unwrap(), wire);
        // Unknown reasons classify as neither an abort nor a matched stop.
        assert!(parsed.abort_status().is_none());
        assert!(parsed.matched().is_none());
        // But their `type` still reads out for the OpenAI finish mapping.
        assert_eq!(parsed.kind_name().as_deref(), Some("tool_calls"));
    }
}

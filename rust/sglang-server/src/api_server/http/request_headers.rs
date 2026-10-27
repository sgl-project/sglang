//! Router-computed request fields sent as headers, so the caller can leave the
//! `/generate` body opaque: the port of Python `request_headers.py`. Each header
//! overrides the body field of the same name before the body is decoded.

use axum::http::HeaderMap;
use serde_json::Value;

#[derive(Clone, Copy)]
enum Kind {
    Str,
    Int,
}

/// Header → body field, Python's `_HEADER_OVERRIDES`. `conversation_id` has no
/// consumer past Python's TokenizerManager, so the schema drops it here.
const HEADER_OVERRIDES: &[(&str, &str, Kind)] = &[
    ("x-override-rid", "rid", Kind::Str),
    ("x-override-bootstrap-host", "bootstrap_host", Kind::Str),
    ("x-override-bootstrap-port", "bootstrap_port", Kind::Int),
    ("x-override-bootstrap-room", "bootstrap_room", Kind::Int),
    ("x-override-conversation-id", "conversation_id", Kind::Str),
    ("x-override-routed-dp-rank", "routed_dp_rank", Kind::Int),
    (
        "x-override-disagg-prefill-dp-rank",
        "disagg_prefill_dp_rank",
        Kind::Int,
    ),
    ("x-override-priority", "priority", Kind::Int),
];

/// Write each present override header into `body`. An unparsable value fails
/// the request (Python answers 400). A non-object body is left alone; decoding
/// rejects it.
pub(super) fn apply_header_overrides(body: &mut Value, headers: &HeaderMap) -> Result<(), String> {
    let Some(fields) = body.as_object_mut() else {
        return Ok(());
    };
    for &(header, field, kind) in HEADER_OVERRIDES {
        let Some(value) = headers.get(header) else {
            continue;
        };
        let text = value
            .to_str()
            .map_err(|_| format!("invalid {header} header: not visible ASCII"))?;
        let value = match kind {
            Kind::Str => Value::from(text),
            Kind::Int => text
                .parse::<i64>()
                .map(Value::from)
                .map_err(|e| format!("invalid {header} header {text:?}: {e}"))?,
        };
        fields.insert(field.to_owned(), value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::request::{GenerationPolicy, into_requests};

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        pairs
            .iter()
            .map(|&(name, value)| (name.parse().unwrap(), value.parse().unwrap()))
            .collect()
    }

    /// Headers override the body's fields, typed as Python casts them; absent
    /// headers leave the body alone. The result decodes with every field in its
    /// request slot, but `conversation_id`, which has none and is dropped.
    #[test]
    fn headers_override_body_fields() {
        let mut body = serde_json::json!({"text": "hi", "rid": "client", "routed_dp_rank": 0});
        apply_header_overrides(
            &mut body,
            &headers(&[
                ("x-override-rid", "router"),
                ("x-override-bootstrap-host", "10.0.0.1"),
                ("x-override-bootstrap-port", "8998"),
                ("x-override-bootstrap-room", "9223372036854775807"),
                ("x-override-conversation-id", "conv"),
                ("x-override-routed-dp-rank", "3"),
                ("x-override-disagg-prefill-dp-rank", "1"),
                ("x-override-priority", "-2"),
            ]),
        )
        .unwrap();
        assert_eq!(
            body,
            serde_json::json!({
                "text": "hi", "rid": "router", "routed_dp_rank": 3,
                "bootstrap_host": "10.0.0.1", "bootstrap_port": 8998,
                "bootstrap_room": i64::MAX, "conversation_id": "conv",
                "disagg_prefill_dp_rank": 1, "priority": -2,
            })
        );
        let (requests, _) = into_requests(
            serde_json::from_value(body.clone()).unwrap(),
            &GenerationPolicy::default(),
        )
        .unwrap();
        let req = &requests[0];
        assert_eq!(req.rid.client_facing(), "router");
        assert_eq!(req.bootstrap_host.as_deref(), Some("10.0.0.1"));
        let ints = [
            req.bootstrap_port,
            req.bootstrap_room,
            req.routed_dp_rank,
            req.disagg_prefill_dp_rank,
            req.priority,
        ];
        assert_eq!(
            ints,
            [Some(8998), Some(i64::MAX), Some(3), Some(1), Some(-2)]
        );

        let err = apply_header_overrides(
            &mut body,
            &headers(&[("x-override-bootstrap-port", "eighty")]),
        )
        .unwrap_err();
        assert!(
            err.contains("invalid x-override-bootstrap-port header \"eighty\""),
            "{err}"
        );
    }
}

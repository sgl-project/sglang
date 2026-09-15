//! Router overrides are applied before request normalization, so a scalar
//! request id or bootstrap room has the same batch semantics as a body field.

use axum::http::HeaderMap;

use crate::message::request::GenerateBody;
use crate::message::types::OneOrMany;

pub(super) fn apply_overrides(body: &mut GenerateBody, headers: &HeaderMap) -> Result<(), String> {
    if let Some(value) = string(headers, "x-override-rid")? {
        body.rid = Some(OneOrMany::One(value));
    }
    if let Some(value) = string(headers, "x-override-bootstrap-host")? {
        body.bootstrap_host = Some(OneOrMany::One(Some(value)));
    }
    if let Some(value) = integer(headers, "x-override-bootstrap-port")? {
        body.bootstrap_port = Some(OneOrMany::One(Some(value)));
    }
    if let Some(value) = integer(headers, "x-override-bootstrap-room")? {
        body.bootstrap_room = Some(OneOrMany::One(Some(value)));
    }
    if let Some(value) = string(headers, "x-override-conversation-id")? {
        body.conversation_id = Some(value);
    }
    if let Some(value) = integer(headers, "x-override-routed-dp-rank")? {
        body.routed_dp_rank = Some(value);
    }
    if let Some(value) = integer(headers, "x-override-disagg-prefill-dp-rank")? {
        body.disagg_prefill_dp_rank = Some(value);
    }
    if let Some(value) = integer(headers, "x-override-priority")? {
        body.priority = Some(value);
    }
    if let Some(value) = string(headers, "x-override-disagg-prefill-serve-addr")? {
        body.disagg_prefill_serve_addr = Some(OneOrMany::One(Some(value)));
    }
    Ok(())
}

fn string(headers: &HeaderMap, name: &str) -> Result<Option<String>, String> {
    headers
        .get(name)
        .map(|value| {
            value
                .to_str()
                .map(str::to_owned)
                .map_err(|error| format!("invalid {name} header: {error}"))
        })
        .transpose()
}

pub(super) fn integer(headers: &HeaderMap, name: &str) -> Result<Option<i64>, String> {
    string(headers, name)?
        .map(|value| {
            // Python int accepts whitespace, a sign, and underscores between
            // decimal digits. Validate separators before removing them.
            let trimmed = value.trim().as_bytes();
            for (index, &byte) in trimmed.iter().enumerate() {
                if byte == b'_'
                    && !(index > 0
                        && trimmed[index - 1].is_ascii_digit()
                        && trimmed.get(index + 1).is_some_and(u8::is_ascii_digit))
                {
                    return Err(format!("invalid {name} header {value:?}: invalid integer"));
                }
            }
            value
                .trim()
                .replace('_', "")
                .parse()
                .map_err(|error| format!("invalid {name} header {value:?}: {error}"))
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_precede_batch_normalization_and_preserve_router_fields() {
        let mut body: GenerateBody = serde_json::from_value(serde_json::json!({
            "text": ["first", "second"], "rid": ["duplicate", "duplicate"],
            "bootstrap_host": "body", "bootstrap_port": 1, "bootstrap_room": [2, 2],
            "conversation_id": "body", "routed_dp_rank": 1, "disagg_prefill_dp_rank": 2,
            "priority": 1, "disagg_prefill_serve_addr": "http://body"
        }))
        .unwrap();
        let mut headers = HeaderMap::new();
        for (name, value) in [
            ("x-override-rid", "router"),
            ("x-override-bootstrap-host", "::1"),
            ("x-override-bootstrap-port", "8998"),
            ("x-override-bootstrap-room", " +9_000 "),
            ("x-override-conversation-id", "conversation"),
            ("x-override-routed-dp-rank", "3"),
            ("x-override-disagg-prefill-dp-rank", "4"),
            ("x-override-priority", "-5"),
            ("x-override-disagg-prefill-serve-addr", "http://[::1]:30000"),
        ] {
            headers.insert(name, value.parse().unwrap());
        }
        apply_overrides(&mut body, &headers).unwrap();
        let (requests, batch) = body.into_requests().unwrap();
        assert!(batch);
        for (index, request) in requests.iter().enumerate() {
            assert_eq!(request.rid.client_facing(), format!("router_{index}"));
            assert_eq!(request.bootstrap_host.as_deref(), Some("::1"));
            assert_eq!(request.bootstrap_port, Some(8998));
            assert_eq!(request.bootstrap_room, Some(9000 + index as i64));
            assert_eq!(request.conversation_id.as_deref(), Some("conversation"));
            assert_eq!(request.routed_dp_rank, Some(3));
            assert_eq!(request.disagg_prefill_dp_rank, Some(4));
            assert_eq!(request.priority, Some(-5));
            assert_eq!(
                request.disagg_prefill_serve_addr.as_deref(),
                Some("http://[::1]:30000")
            );
        }
        for value in ["", "1.0", "_1", "1_", "1__2", "+_2", "9223372036854775808"] {
            headers.insert("x-override-bootstrap-room", value.parse().unwrap());
            assert!(
                apply_overrides(&mut GenerateBody::default(), &headers).is_err(),
                "{value}"
            );
        }
    }
}

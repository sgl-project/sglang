//! Minimal "what the router needs to read" views over the chat/generate
//! request body.
//!
//! Mirrors the design of the SGLang gRPC service's OpenAI-compatible
//! RPCs (see `proto/sglang/runtime/v1/sglang.proto`):
//!
//! ```proto
//! message OpenAIRequest {
//!   bytes json_body = 1;
//!   map<string, string> trace_headers = 2;
//! }
//!
//! rpc ChatComplete(OpenAIRequest) returns (stream OpenAIStreamChunk);
//! ```
//!
//! The proto deliberately treats the OpenAI-shaped JSON as opaque
//! `bytes` and lets the receiving server unmarshal it. The HTTP gateway
//! adopts the same shape: only parse the few fields the router itself
//! consumes for routing decisions; forward the rest of the body
//! verbatim.
//!
//! Compared to deserialising the full `openai-protocol::ChatCompletionRequest`,
//! this:
//! - Doesn't allocate a `String` for the multimodal `image_url.url`
//!   (which can be 100 KB+ on real traffic). Unknown fields are
//!   skipped at the JSON tokenizer level.
//! - Doesn't pull in the openai-protocol crate on the routing path —
//!   the gateway becomes protocol-version-agnostic for forwarding.
//! - Forward-compatible by design: any future SGLang RL extension
//!   field rides through as bytes; only fields the router branches on
//!   need to be added here.

use serde::Deserialize;

/// Routing-decision view over a `/v1/chat/completions` request.
///
/// Captures only the fields the gateway *itself* reads. Everything
/// else stays in the original `Bytes` body and flows through to the
/// backend verbatim.
#[derive(Debug, Default, Deserialize)]
pub struct ChatRoutingView {
    /// Required. Used for model-based worker selection and metrics
    /// labels.
    pub model: String,

    /// Whether the response is streamed (SSE).
    #[serde(default)]
    pub stream: bool,

    /// Number of completions per prompt. PD batch-size estimation
    /// branches on this for `n > 1`.
    #[serde(default)]
    pub n: Option<u32>,

    /// `logprobs: true/false` on the chat-completions endpoint. PD
    /// merges prefill+decode logprobs only when this is true.
    #[serde(default)]
    pub logprobs: bool,

    /// SGLang RL extension. PD merges prefill+decode `routed_experts`
    /// only when this is true. PD streaming rejects this combination
    /// up front (the SSE path has no merge step).
    #[serde(default)]
    pub return_routed_experts: bool,
}

/// Routing-decision view over a `/generate` request. SGLang's native
/// generate endpoint, which uses snake_case field names and supports
/// batch via `input_ids`.
#[derive(Debug, Default, Deserialize)]
pub struct GenerateRoutingView {
    /// Optional on `/generate` — model is inferred from the worker
    /// pool when absent.
    #[serde(default)]
    pub model: Option<String>,

    /// Whether the response is streamed.
    #[serde(default)]
    pub stream: bool,

    /// SGLang's `return_logprob` flag (snake_case on /generate, vs
    /// `logprobs` bool on /v1/chat/completions). PD merges
    /// prefill+decode logprobs only when this is true.
    #[serde(default)]
    pub return_logprob: Option<bool>,

    /// SGLang RL extension. Same semantics as on chat.
    #[serde(default)]
    pub return_routed_experts: bool,

    /// First-position user text. Captured here for cache-aware
    /// routing policies that hash on the prompt prefix.
    #[serde(default)]
    pub text: Option<String>,

    /// Raw `input_ids`. SGLang accepts either a single `Vec<u32>` or
    /// a `Vec<Vec<u32>>` (batch). Captured as a generic JSON Value so
    /// `batch_size()` can disambiguate without committing to a typed
    /// shape (avoids serde-untagged-enum ambiguity for flat-int
    /// arrays).
    #[serde(default)]
    pub input_ids: Option<serde_json::Value>,
}

impl ChatRoutingView {
    /// Number of completions requested (`n` field). Returns `None`
    /// for the default case `n == 1`.
    pub fn batch_size(&self) -> Option<usize> {
        match self.n {
            Some(n) if n > 1 => Some(n as usize),
            _ => None,
        }
    }
}

impl GenerateRoutingView {
    /// Number of items in a batch request, or `None` when single-shot.
    /// SGLang's `input_ids` is either `Vec<u32>` (single — its
    /// length is the token count, NOT the batch size) or
    /// `Vec<Vec<u32>>` (batch — outer length IS the batch size).
    /// Disambiguate by inspecting the first element: if it's an
    /// array, the outer is a batch.
    pub fn batch_size(&self) -> Option<usize> {
        let arr = self.input_ids.as_ref()?.as_array()?;
        match arr.first() {
            Some(serde_json::Value::Array(_)) => Some(arr.len()),
            _ => None,
        }
    }
}

/// Classifier output for a failed routing-view parse. Lets the HTTP
/// entrypoint return a structured 400 with the right error code.
pub enum ParseErrorKind {
    /// The bytes weren't even valid JSON (truncated body, garbled
    /// content-type, etc.). Surface as `json_parse_error`.
    Json,
    /// The bytes parsed as JSON but a known SGLang extension field
    /// (e.g. `return_routed_experts`) had the wrong value type.
    /// Surface as `invalid_sglang_extension` with the offending field
    /// named — matches the contract the older `body_raw` design had,
    /// so a mistyped flag isn't lumped into a generic JSON error.
    InvalidSglangExtension(&'static str),
    /// The bytes parsed as JSON but a routing-view field the gateway
    /// itself reads (e.g. `model: 42`) had the wrong type. This is a
    /// shape error on a non-extension field, so it stays a
    /// `json_parse_error`. Kept distinct so future code can choose a
    /// different code for it without entangling the extension case.
    InvalidViewField,
}

/// Inspect the original request bytes and the failing routing-view
/// parse error, and pick the right [`ParseErrorKind`].
///
/// Strategy: re-parse the body as a generic `serde_json::Value`. If
/// that succeeds, the bytes are valid JSON but a typed field on the
/// view didn't match — walk the known SGLang extension keys and check
/// whether one of them is present with the wrong type. If yes, it's
/// `InvalidSglangExtension(field)`. Otherwise it's a non-extension
/// shape mismatch (`InvalidViewField`). If the second parse also fails,
/// the bytes truly aren't JSON: `Json`.
///
/// The cost is one extra `from_slice::<Value>` on the failure path —
/// happy path is unchanged.
pub fn classify_parse_error<V: ViewSchema>(body: &[u8]) -> ParseErrorKind {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) else {
        return ParseErrorKind::Json;
    };
    let Some(obj) = value.as_object() else {
        // Top-level wasn't an object — every routing view requires
        // one. Treat as a shape error on the view, not as malformed
        // JSON.
        return ParseErrorKind::InvalidViewField;
    };
    for &field in V::EXTENSION_FIELDS {
        if let Some(slot) = obj.get(field) {
            if !V::extension_type_matches(field, slot) {
                return ParseErrorKind::InvalidSglangExtension(field);
            }
        }
    }
    ParseErrorKind::InvalidViewField
}

/// Glue trait so `classify_parse_error` can drive both `ChatRoutingView`
/// and `GenerateRoutingView` without duplicating the field-list /
/// type-check tables. Each view enumerates the SGLang extension keys it
/// strictly types and validates a candidate `Value` against the
/// expected type.
pub trait ViewSchema {
    /// SGLang extension keys this view types strictly. A value present
    /// under one of these names with the wrong JSON type is reported
    /// as `invalid_sglang_extension` rather than the generic JSON
    /// parse error.
    const EXTENSION_FIELDS: &'static [&'static str];

    /// Returns `true` when `value` matches the expected type for
    /// `field`. `field` is guaranteed to be one of `EXTENSION_FIELDS`.
    fn extension_type_matches(field: &str, value: &serde_json::Value) -> bool;
}

impl ViewSchema for ChatRoutingView {
    const EXTENSION_FIELDS: &'static [&'static str] = &["return_routed_experts"];
    fn extension_type_matches(field: &str, value: &serde_json::Value) -> bool {
        match field {
            "return_routed_experts" => value.is_boolean(),
            _ => true,
        }
    }
}

impl ViewSchema for GenerateRoutingView {
    const EXTENSION_FIELDS: &'static [&'static str] = &["return_routed_experts"];
    fn extension_type_matches(field: &str, value: &serde_json::Value) -> bool {
        match field {
            "return_routed_experts" => value.is_boolean(),
            _ => true,
        }
    }
}

/// Walks an already-parsed JSON value and returns the first SGLang
/// extension field that violates its expected type, or `None` when
/// every present extension field is correctly typed. Used by routers
/// (e.g. PD) that parse the body to `serde_json::Value` directly and
/// therefore can't rely on a typed-deserialize failure to surface
/// extension type errors.
pub fn validate_extensions_in_value<V: ViewSchema>(
    value: &serde_json::Value,
) -> Option<&'static str> {
    let obj = value.as_object()?;
    for &field in V::EXTENSION_FIELDS {
        if let Some(slot) = obj.get(field) {
            if !V::extension_type_matches(field, slot) {
                return Some(field);
            }
        }
    }
    None
}

/// Convert a failed routing-view parse into a structured 400. Promotes
/// known-extension type mismatches (e.g. `return_routed_experts: "yes"`)
/// to `invalid_sglang_extension` instead of folding them into the
/// generic `json_parse_error`. Routers that parse a typed view (HTTP
/// unified, gRPC) call this on `serde_json::Error`.
pub fn view_parse_error<V: ViewSchema>(
    body: &bytes::Bytes,
    err: serde_json::Error,
) -> axum::response::Response {
    use crate::routers::error::bad_request;
    match classify_parse_error::<V>(body) {
        ParseErrorKind::InvalidSglangExtension(field) => bad_request(
            "invalid_sglang_extension",
            format!("Invalid SGLang extension field `{field}`: {err}"),
        ),
        ParseErrorKind::Json | ParseErrorKind::InvalidViewField => {
            bad_request("json_parse_error", format!("Invalid JSON data: {err}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_view_parses_minimal() {
        let body = br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#;
        let v: ChatRoutingView = serde_json::from_slice(body).unwrap();
        assert_eq!(v.model, "x");
        assert!(!v.stream);
        assert!(!v.return_routed_experts);
    }

    #[test]
    fn chat_view_skips_giant_image_string_without_allocating_it() {
        // The whole point of the proto-pattern: a 100KB image_url.url
        // string lives in the bytes but never gets materialised into a
        // typed String during routing.
        let blob: String = "A".repeat(100 * 1024);
        let body = format!(
            r#"{{
                "model":"x",
                "messages":[{{
                    "role":"user",
                    "content":[
                        {{"type":"text","text":"describe"}},
                        {{"type":"image_url","image_url":{{"url":"data:image/png;base64,{blob}"}}}}
                    ]
                }}],
                "return_routed_experts":true
            }}"#
        );
        let v: ChatRoutingView = serde_json::from_slice(body.as_bytes()).unwrap();
        assert_eq!(v.model, "x");
        assert!(v.return_routed_experts);
    }

    #[test]
    fn chat_view_rejects_bad_extension_type() {
        // `return_routed_experts: "yes"` should 400 — no separate
        // strict-type pass needed, serde does it.
        let body = br#"{"model":"x","return_routed_experts":"yes"}"#;
        let err = serde_json::from_slice::<ChatRoutingView>(body).unwrap_err();
        assert!(
            err.to_string().contains("return_routed_experts")
                || err.to_string().contains("boolean"),
            "error should name the offending field or expected type: {err}"
        );
    }

    #[test]
    fn generate_view_batch_size_via_input_ids() {
        let body = br#"{"input_ids":[[1,2,3],[4,5,6],[7,8,9]],"stream":false}"#;
        let v: GenerateRoutingView = serde_json::from_slice(body).unwrap();
        assert_eq!(v.batch_size(), Some(3));
    }

    #[test]
    fn generate_view_single_input_ids_not_a_batch() {
        let body = br#"{"input_ids":[1,2,3,4]}"#;
        let v: GenerateRoutingView = serde_json::from_slice(body).unwrap();
        assert_eq!(v.batch_size(), None);
    }

    #[test]
    fn generate_view_text_extraction() {
        let body = br#"{"text":"hello world","return_routed_experts":true}"#;
        let v: GenerateRoutingView = serde_json::from_slice(body).unwrap();
        assert_eq!(v.text.as_deref(), Some("hello world"));
        assert!(v.return_routed_experts);
    }

    /// Routers that take a permissive `Value` parse (PD, gRPC) rely
    /// on this helper to surface the same `invalid_sglang_extension`
    /// 400 the typed-view path would produce. These tests pin that
    /// contract so a future router added on the same shape stays
    /// honest.
    mod validate_extensions {
        use serde_json::json;

        use super::super::*;

        #[test]
        fn flags_wrong_typed_chat_extension() {
            let value = json!({
                "model": "x",
                "messages": [],
                "return_routed_experts": "yes"
            });
            assert_eq!(
                validate_extensions_in_value::<ChatRoutingView>(&value),
                Some("return_routed_experts")
            );
        }

        #[test]
        fn flags_wrong_typed_generate_extension() {
            let value = json!({"text": "hi", "return_routed_experts": 1});
            assert_eq!(
                validate_extensions_in_value::<GenerateRoutingView>(&value),
                Some("return_routed_experts")
            );
        }

        #[test]
        fn passes_correctly_typed_extension() {
            let value = json!({
                "model": "x",
                "messages": [],
                "return_routed_experts": true
            });
            assert_eq!(
                validate_extensions_in_value::<ChatRoutingView>(&value),
                None
            );
        }

        #[test]
        fn passes_when_extension_absent() {
            let value = json!({"model": "x", "messages": []});
            assert_eq!(
                validate_extensions_in_value::<ChatRoutingView>(&value),
                None
            );
        }

        #[test]
        fn null_extension_value_is_a_type_mismatch() {
            // `null` is not a boolean — surface as
            // `invalid_sglang_extension` so the client sees it
            // instead of having the field silently dropped.
            let value = json!({"return_routed_experts": null});
            assert_eq!(
                validate_extensions_in_value::<GenerateRoutingView>(&value),
                Some("return_routed_experts")
            );
        }

        #[test]
        fn non_object_top_level_returns_none() {
            // Top-level shape mismatch isn't this helper's concern;
            // typed parse will reject downstream.
            let value = json!([1, 2, 3]);
            assert_eq!(
                validate_extensions_in_value::<ChatRoutingView>(&value),
                None
            );
        }
    }
}

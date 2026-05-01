//! SGLang server extension fields.
//!
//! Request-body keys SGLang servers accept that other OpenAI-compatible
//! backends (vendor APIs like OpenAI/Anthropic/etc.) reject as 400. Some
//! of these names also happen to be defined on the upstream
//! `openai-protocol` types we deserialize against and so do round-trip
//! through the typed structs (e.g. `lora_path`, `bootstrap_host`); the
//! list here is the *union* — every key that needs to be either stripped
//! before forwarding to a non-SGLang backend or read out of the original
//! request bytes by an SGLang-aware router.
//!
//! Two consumers:
//!
//! 1. Routers that proxy to non-SGLang backends (e.g.
//!    [`crate::routers::openai::provider`]): they must strip every entry
//!    in [`EXTENSION_FIELD_NAMES`] before forwarding, otherwise the
//!    backend 400s.
//!
//! 2. Routers that need to *read* a flag's value (e.g. the PD HTTP
//!    router branches on `return_routed_experts` to decide whether to
//!    merge prefill/decode responses): the typed
//!    `ChatCompletionRequest` / `GenerateRequest` drop fields that
//!    aren't on the upstream types, so reading happens via
//!    [`SglangExtensions`] from the original request bytes.
//!
//! Adding a new SGLang field:
//! - If routers only need to forward/strip it, add the name to
//!   [`EXTENSION_FIELD_NAMES`] so the OpenAI provider strips it.
//! - If a router needs to read its value, also add a typed field to
//!   [`SglangExtensions`]. The reflective test
//!   `every_typed_extension_field_is_in_strip_list` enforces both lists
//!   stay in sync.

use serde::{Deserialize, Serialize};

/// Every request-body key recognised as an SGLang extension. Source of
/// truth for both stripping (OpenAI passthrough) and typed parsing.
pub const EXTENSION_FIELD_NAMES: &[&str] = &[
    "request_id",
    "priority",
    "top_k",
    "min_p",
    "min_tokens",
    "regex",
    "ebnf",
    "json_schema",
    "stop_token_ids",
    "no_stop_trim",
    "ignore_eos",
    "continue_final_message",
    "skip_special_tokens",
    "lora_path",
    "session_params",
    "separate_reasoning",
    "stream_reasoning",
    "chat_template",
    "chat_template_kwargs",
    "return_hidden_states",
    "return_routed_experts",
    "return_cached_tokens_details",
    "return_prompt_token_ids",
    "return_meta_info",
    "input_ids",
    "stop_regex",
    "custom_logit_processor",
    "custom_params",
    "max_dynamic_patch",
    "min_dynamic_patch",
    "use_audio_in_video",
    "rid",
    "extra_key",
    "cache_salt",
    "bootstrap_host",
    "bootstrap_port",
    "bootstrap_room",
    "routed_dp_rank",
    "disagg_prefill_dp_rank",
    "data_parallel_rank",
    "repetition_penalty",
    "sampling_seed",
    "custom_labels",
    "backend_url",
];

/// Typed view over the SGLang extension fields the gateway *reads*.
///
/// Most extension fields are passthrough — the gateway forwards them to
/// the backend without inspecting their values. Only the fields a router
/// branches on are listed here. Add a field when a code path needs typed
/// access; passthrough-only fields just need to be in
/// [`EXTENSION_FIELD_NAMES`].
///
/// Deserialization is strict on type: `return_routed_experts: "yes"`
/// returns a `serde_json::Error` rather than silently defaulting to
/// `false`. The HTTP routers wrap this via
/// [`crate::routers::http::parse_sglang_extensions`], which surfaces
/// the error as a 400 `invalid_sglang_extension` response.
#[derive(Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(default)]
pub struct SglangExtensions {
    pub return_routed_experts: bool,
}

impl SglangExtensions {
    /// Parse extensions from the original request bytes. Missing fields
    /// take their `Default` value; type-mismatched fields return `Err`.
    pub fn parse(body_raw: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(body_raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_returns_default_when_field_absent() {
        let bytes = br#"{"model": "x", "messages": []}"#;
        let ext = SglangExtensions::parse(bytes).unwrap();
        assert!(!ext.return_routed_experts);
    }

    #[test]
    fn parse_reads_return_routed_experts_when_present() {
        let bytes = br#"{"return_routed_experts": true}"#;
        let ext = SglangExtensions::parse(bytes).unwrap();
        assert!(ext.return_routed_experts);
    }

    #[test]
    fn parse_rejects_wrong_type_for_return_routed_experts() {
        // Strict type check is the whole point of this struct: a typo
        // like passing a string should surface as 400, not silently
        // default to false.
        let bytes = br#"{"return_routed_experts": "yes"}"#;
        let err = SglangExtensions::parse(bytes).unwrap_err();
        assert!(
            err.to_string().contains("return_routed_experts")
                || err.to_string().contains("boolean"),
            "error should mention the offending field or expected type: {err}"
        );
    }

    #[test]
    fn parse_ignores_unknown_extension_fields() {
        // Forward-compat: a future SGLang field we don't yet read
        // shouldn't fail extension parsing — it just rides through via
        // the raw bytes the caller still holds.
        let bytes = br#"{"some_future_field": 42, "return_routed_experts": true}"#;
        let ext = SglangExtensions::parse(bytes).unwrap();
        assert!(ext.return_routed_experts);
    }

    #[test]
    fn extension_field_names_includes_known_sglang_keys() {
        // Spot-check a few entries to catch accidental deletions.
        for required in [
            "return_routed_experts",
            "return_hidden_states",
            "bootstrap_host",
            "data_parallel_rank",
            "sampling_seed",
        ] {
            assert!(
                EXTENSION_FIELD_NAMES.contains(&required),
                "{required} missing from EXTENSION_FIELD_NAMES"
            );
        }
    }

    #[test]
    fn extension_field_names_are_unique() {
        // `EXTENSION_FIELD_NAMES` is morally a set, expressed as a slice
        // for `pub const` ergonomics. Duplicates wouldn't break behaviour
        // (`contains` would still answer correctly) but would hide a
        // careless rebase merge that re-added an entry. Cheap to check.
        let mut seen = std::collections::HashSet::new();
        for name in EXTENSION_FIELD_NAMES {
            assert!(seen.insert(*name), "duplicate entry: {name}");
        }
    }

    #[test]
    fn every_typed_extension_field_is_in_strip_list() {
        // Cross-invariant: every field on `SglangExtensions` (the typed
        // read-list) MUST also live in `EXTENSION_FIELD_NAMES` (the strip
        // list). Otherwise a router that forwards to OpenAI would leak
        // the extension and trigger a 400 — even though we read it
        // server-side. Catches "added a typed field, forgot the
        // strip-list" foot-gun without any extra code at the call sites.
        let serialized =
            serde_json::to_value(SglangExtensions::default()).expect("SglangExtensions serialises");
        let object = serialized
            .as_object()
            .expect("SglangExtensions serialises to a JSON object");
        for key in object.keys() {
            assert!(
                EXTENSION_FIELD_NAMES.contains(&key.as_str()),
                "typed SglangExtensions field {key} missing from EXTENSION_FIELD_NAMES",
            );
        }
    }
}

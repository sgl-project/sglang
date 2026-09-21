// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Engine-parity request shaping for dynamo-render's native Kimi-K3 formatter.
//!
//! Mirrors `_prepare_kimi_k3_messages` and the Kimi branch of `_encode_messages`
//! in the engine's `serving_chat.py`, so router token IDs equal the engine's.
//! Dynamo owns rendering and encoding; this module only shapes the request and
//! refuses the shapes where the pinned Dynamo output is known to differ from
//! the engine, leaving those requests on engine-side tokenization.

use anyhow::Result;
use dynamo_tokenizers::EncodeSegment;
use minijinja::Value;
use serde_json::Value as JsonValue;

use super::chat_formatter::{ChatTemplateKwargs, RequestFields};

/// A reserved spelling the engine escapes everywhere in request text, even
/// without image inputs.
const IMAGE_PLACEHOLDER: &str = "<|kimi_image_placeholder|>";
const IMAGE_PLACEHOLDER_ESCAPED: &str = "<| kimi_image_placeholder |>";

/// Shape `messages` and `kwargs` as the engine does before Kimi rendering and
/// resolve the request fields dynamo-render reads through `OAIChatLikeRequest`.
pub(super) fn normalize(
    request: &JsonValue,
    messages: &mut [JsonValue],
    kwargs: &mut ChatTemplateKwargs,
) -> Result<RequestFields> {
    // The engine forwards top-level `reasoning_effort` as `thinking_effort` only
    // for the values Kimi accepts, and never hands `reasoning_effort` itself to
    // the encoder.
    if let Some(effort) = kwargs
        .remove("reasoning_effort")
        .filter(|v| matches!(v.as_str(), Some("low" | "high" | "max")))
    {
        kwargs.entry("thinking_effort".into()).or_insert(effort);
    }
    // Parity gap in dynamo-renderer (5.1.2 through 5.3.1): a null effort renders
    // as `max`, while the engine omits the effort preamble. Refuse rather than
    // patch rendered output; the engine tokenizes the original request.
    anyhow::ensure!(
        !kwargs
            .get("thinking_effort")
            .is_some_and(JsonValue::is_null),
        "Kimi null thinking effort requires engine-side tokenization"
    );
    for message in messages.iter_mut() {
        neutralize_image_placeholder(&mut message["content"]);
        if message["role"] == "assistant" {
            if let Some(reasoning) = message.get_mut("reasoning_content") {
                neutralize_image_placeholder(reasoning);
            }
            for call in message["tool_calls"].as_array_mut().into_iter().flatten() {
                if let Some(args) = call
                    .get_mut("function")
                    .and_then(|f| f.get_mut("arguments"))
                {
                    neutralize_image_placeholder(args);
                }
            }
        }
    }
    // Tools go to the native formatter unmodified: no named-tool filtering and
    // no schema fixing, as in the engine's Kimi path.
    let tools = request
        .get("tools")
        .filter(|t| t.as_array().is_some_and(|t| !t.is_empty()));
    // The engine forwards `tool_choice` only when some tool is declared, on the
    // request or on a system/developer message, and only for these two values.
    // An explicit `chat_template_kwargs.tool_choice` passes through as-is.
    let has_tools = tools.is_some()
        || messages.iter().any(|m| {
            matches!(m["role"].as_str(), Some("system" | "developer"))
                && m["tools"].as_array().is_some_and(|t| !t.is_empty())
        });
    let tool_choice = kwargs.get("tool_choice").or_else(|| {
        request
            .get("tool_choice")
            .filter(|c| has_tools && matches!(c.as_str(), Some("required" | "none")))
    });
    // Unlike the DeepSeek formatters, the engine lets the Kimi encoder render
    // `response_format`.
    let response_format = kwargs
        .get("response_format")
        .or_else(|| request.get("response_format"));
    Ok(RequestFields {
        tools: tools.map(Value::from_serialize),
        tool_choice: tool_choice.map(Value::from_serialize),
        response_format: response_format.map(Value::from_serialize),
    })
}

fn neutralize_image_placeholder(value: &mut JsonValue) {
    match value {
        JsonValue::String(text) => {
            *text = text.replace(IMAGE_PLACEHOLDER, IMAGE_PLACEHOLDER_ESCAPED)
        }
        JsonValue::Array(values) => values.iter_mut().for_each(neutralize_image_placeholder),
        JsonValue::Object(values) => values.values_mut().for_each(neutralize_image_placeholder),
        _ => {}
    }
}

/// Parity gap in dynamo-tokenizers (1.8.1 through 1.8.2): the tiktoken backend
/// does not split long text before BPE the way the Python encoder does, so its
/// IDs can differ past these thresholds. Detect that and leave the request on
/// engine-side tokenization rather than reimplement the chunker here.
pub(super) fn validate_native_segments(segments: &[EncodeSegment<'_>]) -> Result<()> {
    for segment in segments {
        let (mut run, mut was_space) = (0, false);
        for (count, ch) in segment.text.chars().enumerate() {
            anyhow::ensure!(
                count < 400_000,
                "Kimi segment requires engine-side chunking"
            );
            // Python str.isspace() also includes these four control characters.
            let space = ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch);
            run = if space == was_space { run + 1 } else { 1 };
            was_space = space;
            anyhow::ensure!(run <= 25_000, "Kimi text run requires engine-side chunking");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::tokenizer::{adapter, chat_formatter::ChatFormatter};
    use serde_json::json;

    const VOCAB: &str = "tests/fixtures/kimi_k3/tiktoken.model";

    fn formatter() -> ChatFormatter {
        ChatFormatter::load("served-alias", VOCAB).unwrap().unwrap()
    }

    /// Golden IDs from the Python reference encoder over the synthetic fixture
    /// vocabulary; `no_effort` records the known null-effort gap.
    #[test]
    fn reference_prompt_token_ids() {
        let tokenizer = adapter::load(VOCAB).unwrap();
        let formatter = formatter();
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("../../tests/fixtures/kimi_k3/prompts.json"))
                .unwrap();
        for case in cases {
            let encoded = formatter.encode(&tokenizer, &case["request"]);
            if case["name"] == "no_effort" {
                assert!(encoded.is_err());
                continue;
            }
            let expected: Vec<u32> = serde_json::from_value(case["token_ids"].clone()).unwrap();
            assert_eq!(encoded.unwrap(), expected, "{}", case["name"]);
        }
    }

    #[test]
    fn long_segments_fall_back_to_engine_tokenization() {
        let tokenizer = adapter::load(VOCAB).unwrap();
        let formatter = formatter();
        // Boundaries count Unicode characters, and Python treats U+001C as whitespace.
        for (text, supported) in [
            ("界".repeat(25_000), true),
            ("界".repeat(25_001), false),
            (
                format!("{}\u{1c}{}", "x".repeat(20_000), "y".repeat(20_000)),
                true,
            ),
            (format!("{}x", "x ".repeat(200_000)), false),
        ] {
            let request = json!({"messages": [{"role": "user", "content": text}]});
            assert_eq!(formatter.encode(&tokenizer, &request).is_ok(), supported);
        }
    }

    /// The engine forwards `tool_choice` only alongside declared tools.
    #[test]
    fn tool_choice_without_tools_is_not_rendered() {
        let formatter = formatter();
        let bare = json!({"messages": [{"role": "user", "content": "hi"}]});
        let with_choice =
            json!({"messages": [{"role": "user", "content": "hi"}], "tool_choice": "none"});
        assert_eq!(
            formatter.render(&bare).unwrap(),
            formatter.render(&with_choice).unwrap()
        );
    }
}

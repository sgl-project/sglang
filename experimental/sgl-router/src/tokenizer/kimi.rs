// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Kimi-K3 request semantics from `serving_chat.py` and the checkpoint's
//! `tokenization_kimi.py`, applied around dynamo-render's native formatter.

use anyhow::{ensure, Result};
use dynamo_tokenizers::{EncodeSegment, Tokenizer};
use serde_json::{json, Value};

use super::chat_formatter::ChatTemplateKwargs;

pub(super) fn normalize(
    request: &Value,
    messages: &mut [Value],
    kwargs: &mut ChatTemplateKwargs,
) -> Result<()> {
    // Dynamo reads `reasoning_effort` and treats a non-bool `thinking` as true;
    // the checkpoint ignores the former and uses Python truthiness for the latter.
    kwargs.remove("reasoning_effort");
    let thinking = kwargs
        .get("thinking")
        .is_none_or(|v| minijinja::Value::from_serialize(v).is_true());
    kwargs.insert("thinking".into(), thinking.into());
    ensure!(
        !thinking || !kwargs.get("thinking_effort").is_some_and(Value::is_null),
        "Kimi null thinking_effort requires engine-side rendering"
    );
    for message in messages.iter_mut() {
        if message["role"] == "developer" {
            message["role"] = "system".into();
        }
        if let Some(parts) = message["content"].as_array_mut() {
            parts.retain(|part| matches!(part["type"].as_str(), Some("text" | "image_url")));
        }
        for call in message["tool_calls"].as_array_mut().into_iter().flatten() {
            let args = &mut call["function"]["arguments"];
            if let Some(parsed) = args
                .as_str()
                .and_then(|s| serde_json::from_str::<Value>(s).ok())
                .filter(Value::is_object)
            {
                *args = parsed;
            }
            neutralize(args);
        }
        neutralize(&mut message["content"]);
        if let Some(reasoning) = message.get_mut("reasoning_content") {
            neutralize(reasoning);
        }
    }
    let has_tools = std::iter::once(request)
        .chain(messages.iter().filter(|m| m["role"] == "system"))
        .any(|m| m["tools"].as_array().is_some_and(|t| !t.is_empty()));
    if has_tools && matches!(request["tool_choice"].as_str(), Some("none" | "required")) {
        kwargs
            .entry("tool_choice".into())
            .or_insert_with(|| request["tool_choice"].clone());
    }
    // The checkpoint renders only these; a named choice is constrained decoding.
    if !matches!(
        kwargs.get("tool_choice").and_then(Value::as_str),
        Some("none" | "required")
    ) {
        kwargs.remove("tool_choice");
    }
    if let Some(mut format) = request
        .get("response_format")
        .filter(|v| !v.is_null())
        .cloned()
    {
        // protocol.py lifts a legacy top-level `schema` into `json_schema`.
        if format["type"] == "json_schema" && format["json_schema"].is_null() {
            if let Some(mut schema) = format.as_object_mut().and_then(|f| f.remove("schema")) {
                if let Some(props) = schema.get_mut("properties").and_then(Value::as_object_mut) {
                    props.remove("strict");
                }
                format["json_schema"] = json!({"schema": schema});
            }
        }
        kwargs.entry("response_format".into()).or_insert(format);
    }
    if let Some(schema) = kwargs.get("response_schema").cloned() {
        if let Some(format) = kwargs
            .get_mut("response_format")
            .filter(|f| f["type"] == "json_schema")
        {
            format["json_schema"] = json!({"schema": schema});
        }
    }
    Ok(())
}

fn neutralize(value: &mut Value) {
    match value {
        Value::String(text) => {
            *text = text.replace("<|kimi_image_placeholder|>", "<| kimi_image_placeholder |>")
        }
        Value::Array(values) => values.iter_mut().for_each(neutralize),
        Value::Object(values) => values.values_mut().for_each(neutralize),
        _ => {}
    }
}

/// `tokenization_kimi.py` encodes 400k-char windows, each split after 25k
/// consecutive (non-)whitespace chars, and BPE is not chunk-invariant.
pub(super) fn encode(tokenizer: &Tokenizer, segments: &[EncodeSegment<'_>]) -> Result<Vec<u32>> {
    let mut chunks = Vec::new();
    for segment in segments {
        let (mut start, mut run, mut was_space) = (0, 0, false);
        for (count, (offset, ch)) in segment.text.char_indices().enumerate() {
            // Python's `str.isspace` also covers U+001C..U+001F.
            let space = ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch);
            run = if space == was_space { run + 1 } else { 1 };
            if (count > 0 && count % 400_000 == 0) || run > 25_000 {
                chunks.push(EncodeSegment::new(
                    &segment.text[start..offset],
                    segment.allow_special,
                ));
                (start, run) = (offset, 1);
            }
            was_space = space;
        }
        chunks.push(EncodeSegment::new(
            &segment.text[start..],
            segment.allow_special,
        ));
    }
    Ok(tokenizer.encode_segments(&chunks)?.token_ids().to_vec())
}

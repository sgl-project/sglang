// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Result};
use dynamo_tokenizers::EncodeSegment;
use serde_json::{json, Value};

use super::chat_formatter::ChatTemplateKwargs;

// Match serving_chat.py before invoking Dynamo's native Kimi formatter.
pub(super) fn normalize(
    request: &Value,
    messages: &mut [Value],
    kwargs: &mut ChatTemplateKwargs,
) -> Result<()> {
    kwargs.remove("reasoning_effort");
    let thinking = kwargs.entry("thinking".into()).or_insert(true.into());
    *thinking = minijinja::Value::from_serialize(&*thinking)
        .is_true()
        .into();
    ensure!(
        kwargs["thinking"] == false || !kwargs.get("thinking_effort").is_some_and(Value::is_null),
        "Kimi null thinking_effort requires engine-side rendering"
    );
    for message in messages.iter_mut() {
        if message["role"] == "developer" {
            message["role"] = "system".into();
        }
        if let Some(parts) = message["content"].as_array_mut() {
            parts.retain(|part| matches!(part["type"].as_str(), Some("text" | "image_url")));
            for part in parts {
                if part["type"] == "text" {
                    neutralize(&mut part["text"]);
                }
            }
        } else {
            neutralize(&mut message["content"]);
        }
        if message["role"] == "assistant" {
            if let Some(reasoning) = message.get_mut("reasoning_content") {
                neutralize(reasoning);
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
        }
    }
    let has_tools = std::iter::once(&request["tools"])
        .chain(
            messages
                .iter()
                .filter(|m| m["role"] == "system")
                .map(|m| &m["tools"]),
        )
        .any(|tools| tools.as_array().is_some_and(|tools| !tools.is_empty()));
    if has_tools && matches!(request["tool_choice"].as_str(), Some("none" | "required")) {
        kwargs
            .entry("tool_choice".into())
            .or_insert_with(|| request["tool_choice"].clone());
    }
    // The checkpoint only consumes string choices; named choice constrains decoding.
    if !matches!(
        kwargs.get("tool_choice").and_then(Value::as_str),
        Some("none" | "required")
    ) {
        kwargs.remove("tool_choice");
    }
    if let Some(format) = request.get("response_format").filter(|v| !v.is_null()) {
        let mut format = format.clone();
        if format["type"] == "json_schema" && format["json_schema"].is_null() {
            if let Some(mut schema) = format.get("schema").cloned() {
                if let Some(properties) =
                    schema.get_mut("properties").and_then(Value::as_object_mut)
                {
                    properties.remove("strict");
                }
                format["json_schema"] = json!({"schema": schema});
            }
        }
        kwargs.entry("response_format".into()).or_insert(format);
    }
    if let (Some(schema), Some(format)) = (
        kwargs.get("response_schema").cloned(),
        kwargs.get_mut("response_format"),
    ) {
        if format["type"] == "json_schema" {
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

// Python's Kimi tokenizer splits by character count before BPE, including prefixes.
pub(super) fn split_segments<'a>(segments: &[EncodeSegment<'a>]) -> Vec<EncodeSegment<'a>> {
    let mut chunks = Vec::new();
    for segment in segments {
        let (mut start, mut count, mut run, mut was_space) = (0, 0, 0, false);
        for (offset, ch) in segment.text.char_indices() {
            let space = ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch);
            if count == 400_000 || (space == was_space && run == 25_000) {
                chunks.push(EncodeSegment::new(
                    &segment.text[start..offset],
                    segment.allow_special,
                ));
                start = offset;
                if count == 400_000 {
                    count = 0;
                }
                run = 0;
            }
            run = if space == was_space { run + 1 } else { 1 };
            was_space = space;
            count += 1;
        }
        chunks.push(EncodeSegment::new(
            &segment.text[start..],
            segment.allow_special,
        ));
    }
    chunks
}

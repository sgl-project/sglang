// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Modified by SGLang: comment-only provenance or spelling metadata; behavior is unchanged.

//! Shared DeepSeek native prompt-formatting helpers.

use anyhow::{Context, Result};
use serde_json::Value as JsonValue;

/// Special tokens for DeepSeek prompt formatting.
pub mod tokens {
    pub const BOS: &str = "<｜begin▁of▁sentence｜>";
    pub const EOS: &str = "<｜end▁of▁sentence｜>";
    pub const THINKING_START: &str = "<think>";
    pub const THINKING_END: &str = "</think>";
    pub const DSML_TOKEN: &str = "｜DSML｜";
    pub const USER_START: &str = "<｜User｜>";
    pub const ASSISTANT_START: &str = "<｜Assistant｜>";
    pub const LATEST_REMINDER: &str = "<｜latest_reminder｜>";

    // Quick-instruction task tokens
    pub const TASK_ACTION: &str = "<｜action｜>";
    pub const TASK_QUERY: &str = "<｜query｜>";
    pub const TASK_AUTHORITY: &str = "<｜authority｜>";
    pub const TASK_DOMAIN: &str = "<｜domain｜>";
    pub const TASK_TITLE: &str = "<｜title｜>";
    pub const TASK_READ_URL: &str = "<｜read_url｜>";
}

pub(crate) const TOOL_CALLS_BLOCK_NAME: &str = "tool_calls";

pub(crate) const RESPONSE_FORMAT_TEMPLATE: &str =
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}";

pub(crate) const TOOLS_TEMPLATE: &str = r#"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"#;

/// System message template for tools.
pub(crate) const TOOLS_SYSTEM_TEMPLATE: &str = r#"## Tools

You have access to a set of tools you can use to answer the user's question.
You can invoke functions by writing a "<{dsml_token}function_calls>" block like the following as part of your reply to the user:
<{dsml_token}function_calls>
<{dsml_token}invoke name="$FUNCTION_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$FUNCTION_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}function_calls>

String and scalar parameters should be specified as is without any escaping or quotes, while lists and objects should use JSON format. The "string" attribute should be set to "true" for string type parameters and "false" for other types (numbers, booleans, arrays, objects).

If the thinking_mode is enabled, then after function results you should strongly consider outputting a thinking block. Here is an example:

<{dsml_token}function_calls>
...
</{dsml_token}function_calls>

<function_results>
...
</function_results>

{thinking_start_token}...thinking about results{thinking_end_token}

Here are the functions available in JSONSchema format:
<functions>
{tool_schemas}
</functions>
"#;

pub(crate) const TOOL_CALL_TEMPLATE: &str =
    "<{dsml_token}invoke name=\"{name}\">\n{arguments}\n</{dsml_token}invoke>";

pub(crate) const TOOL_OUTPUT_TEMPLATE: &str = "\n<result>{content}</result>";

pub(crate) const REASONING_EFFORT_HIGH: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

pub(crate) const REASONING_EFFORT_MAX: &str = "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\nYou MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\nDo not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

/// Thinking mode for the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    Chat,
    Thinking,
}

impl ThinkingMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ThinkingMode::Chat => "chat",
            ThinkingMode::Thinking => "thinking",
        }
    }
}

/// Reasoning effort level. `None` conveyed as `Option<ReasoningEffort>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Max,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NormalizeNonText {
    SerializeJson,
    LeaveUntouched,
}

// Serialize a JSON value to match Python's `json.dumps(ensure_ascii=False)` spacing.
// Python's default separators are `(', ', ': ')`; we use a custom `Formatter`
// so escape sequences inside strings can't confuse state tracking.
pub(crate) fn to_json(value: &JsonValue) -> String {
    use serde::Serialize;
    use serde_json::ser::Formatter; // codespell:ignore ser
    use std::io;

    struct PythonFormatter;

    impl Formatter for PythonFormatter {
        fn begin_array_value<W: ?Sized + io::Write>(
            &mut self,
            writer: &mut W,
            first: bool,
        ) -> io::Result<()> {
            if first {
                Ok(())
            } else {
                writer.write_all(b", ")
            }
        }

        fn begin_object_key<W: ?Sized + io::Write>(
            &mut self,
            writer: &mut W,
            first: bool,
        ) -> io::Result<()> {
            if first {
                Ok(())
            } else {
                writer.write_all(b", ")
            }
        }

        fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
            writer.write_all(b": ")
        }
    }

    // Serializing a JsonValue into Vec<u8> is infallible; the output is always UTF-8.
    let mut buf = Vec::with_capacity(64);
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PythonFormatter); // codespell:ignore ser
    value
        .serialize(&mut ser) // codespell:ignore ser
        .expect("JsonValue serialization to Vec<u8> is infallible");
    String::from_utf8(buf).expect("serde_json output is always valid UTF-8")
}

pub(crate) fn render_tools(template: &str, tools: &[JsonValue]) -> String {
    let tools_json: Vec<String> = tools
        .iter()
        .filter_map(|tool| tool.get("function"))
        .map(to_json)
        .collect();

    // Always do the tool_schemas last because they are user controlled.
    // See test_render_tools_preserves_placeholder_text_inside_tool_schema.
    template
        .replace("{dsml_token}", tokens::DSML_TOKEN)
        .replace("{thinking_start_token}", tokens::THINKING_START)
        .replace("{thinking_end_token}", tokens::THINKING_END)
        .replace("{tool_schemas}", &tools_json.join("\n"))
}

pub(crate) fn find_last_user_index(messages: &[JsonValue]) -> Option<usize> {
    messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, msg)| {
            msg.get("role")
                .and_then(|r| r.as_str())
                .map(|r| r == "user" || r == "developer")
                .unwrap_or(false)
        })
        .map(|(idx, _)| idx)
}

pub(crate) fn extract_visible_text(content: &JsonValue) -> String {
    match content {
        JsonValue::String(text) => text.clone(),
        JsonValue::Array(items) => items
            .iter()
            .filter_map(|item| {
                if let Some(text) = item.as_str() {
                    return Some(text.to_string());
                }
                let item_type = item.get("type").and_then(|v| v.as_str());
                if item_type == Some("text") {
                    return item
                        .get("text")
                        .and_then(|v| v.as_str())
                        .map(|text| text.to_string());
                }
                tracing::warn!(
                    chunk_type = item_type.unwrap_or("unknown"),
                    "DeepSeek formatter dropped non-text content chunk while normalizing message content",
                );
                None
            })
            .collect::<String>(),
        _ => to_json(content),
    }
}

pub(crate) fn normalize_message_contents(messages: &mut [JsonValue], non_text: NormalizeNonText) {
    for msg in messages {
        let Some(content) = msg.get("content") else {
            continue;
        };
        if !content.is_string()
            && !content.is_array()
            && non_text == NormalizeNonText::LeaveUntouched
        {
            continue;
        }
        let normalized = extract_visible_text(content);
        if let Some(obj) = msg.as_object_mut() {
            obj.insert("content".to_string(), JsonValue::String(normalized));
        }
    }
}

pub(crate) fn encode_arguments_to_dsml(tool_call: &JsonValue) -> Result<String> {
    let arguments_str = tool_call
        .get("arguments")
        .and_then(|a| a.as_str())
        .context("Missing or invalid 'arguments' field")?;

    // Python falls back to `{"arguments": raw_string}` on parse failure.
    let arguments: JsonValue = match serde_json::from_str(arguments_str) {
        Ok(v) => v,
        Err(_) => serde_json::json!({ "arguments": arguments_str }),
    };

    let arguments_obj = arguments
        .as_object()
        .context("Arguments must be a JSON object")?;

    let mut params = Vec::new();
    for (key, value) in arguments_obj {
        let value_str = if let Some(vs) = value.as_str() {
            vs.to_string()
        } else {
            to_json(value)
        };
        params.push(format!(
            "<{}parameter name=\"{}\" string=\"{}\">{}</{}parameter>",
            tokens::DSML_TOKEN,
            key,
            if value.is_string() { "true" } else { "false" },
            value_str,
            tokens::DSML_TOKEN
        ));
    }

    Ok(params.join("\n"))
}

pub(crate) fn task_token(task: &str) -> Option<&'static str> {
    match task {
        "action" => Some(tokens::TASK_ACTION),
        "query" => Some(tokens::TASK_QUERY),
        "authority" => Some(tokens::TASK_AUTHORITY),
        "domain" => Some(tokens::TASK_DOMAIN),
        "title" => Some(tokens::TASK_TITLE),
        "read_url" => Some(tokens::TASK_READ_URL),
        _ => None,
    }
}

const USER_FIELDS_TO_PRESERVE: [&str; 3] = ["task", "wo_eos", "mask"];

fn preserve_user_fields(target: &mut JsonValue, source: &JsonValue) {
    if let Some(obj) = target.as_object_mut() {
        for key in USER_FIELDS_TO_PRESERVE {
            if let Some(v) = source.get(key) {
                obj.insert(key.to_string(), v.clone());
            }
        }
    }
}

// Merge `tool` role messages into preceding user `content_blocks` and collapse
// consecutive user turns, matching Python's `merge_tool_messages`.
pub(crate) fn merge_tool_messages(messages: &[JsonValue]) -> Vec<JsonValue> {
    let mut merged: Vec<JsonValue> = Vec::with_capacity(messages.len());

    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");

        if role == "tool" {
            let tool_block = serde_json::json!({
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id").cloned().unwrap_or_else(|| JsonValue::String(String::new())),
                "content": msg.get("content").cloned().unwrap_or_else(|| JsonValue::String(String::new())),
            });

            let can_merge = merged
                .last()
                .map(|m| {
                    m.get("role").and_then(|r| r.as_str()) == Some("user")
                        && m.get("content_blocks").is_some()
                })
                .unwrap_or(false);

            if can_merge {
                let last = merged.last_mut().unwrap();
                if let Some(blocks) = last
                    .as_object_mut()
                    .and_then(|o| o.get_mut("content_blocks"))
                    .and_then(|v| v.as_array_mut())
                {
                    blocks.push(tool_block);
                }
            } else {
                merged.push(serde_json::json!({
                    "role": "user",
                    "content_blocks": [tool_block],
                }));
            }
        } else if role == "user" {
            let text = msg
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            let text_block = serde_json::json!({ "type": "text", "text": text });

            let can_merge = merged
                .last()
                .map(|m| {
                    m.get("role").and_then(|r| r.as_str()) == Some("user")
                        && m.get("content_blocks").is_some()
                        && m.get("task").map(|v| v.is_null()).unwrap_or(true)
                })
                .unwrap_or(false);

            if can_merge {
                let last = merged.last_mut().unwrap();
                let appended = last
                    .as_object_mut()
                    .and_then(|o| o.get_mut("content_blocks"))
                    .and_then(|v| v.as_array_mut())
                    .map(|blocks| {
                        blocks.push(text_block);
                    })
                    .is_some();
                if appended {
                    preserve_user_fields(last, msg);
                }
            } else {
                let mut new_msg = serde_json::json!({
                    "role": "user",
                    "content": text,
                    "content_blocks": [text_block],
                });
                preserve_user_fields(&mut new_msg, msg);
                merged.push(new_msg);
            }
        } else {
            merged.push(msg.clone());
        }
    }

    merged
}

// Sort `tool_result` blocks within user messages by the `tool_calls[].id` order
// of the preceding assistant message.
pub(crate) fn sort_tool_results_by_call_order(mut messages: Vec<JsonValue>) -> Vec<JsonValue> {
    use std::collections::HashMap;
    let mut last_order: HashMap<String, usize> = HashMap::new();

    for msg in &mut messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "assistant" {
            if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                last_order.clear();
                for (idx, tc) in tcs.iter().enumerate() {
                    let id = tc
                        .get("id")
                        .and_then(|v| v.as_str())
                        .or_else(|| {
                            tc.get("function")
                                .and_then(|f| f.get("id"))
                                .and_then(|v| v.as_str())
                        })
                        .unwrap_or("");
                    if !id.is_empty() {
                        last_order.insert(id.to_string(), idx);
                    }
                }
            }
        } else if role == "user" && !last_order.is_empty() {
            let Some(blocks) = msg
                .as_object_mut()
                .and_then(|o| o.get_mut("content_blocks"))
                .and_then(|v| v.as_array_mut())
            else {
                continue;
            };

            // Collect tool_result blocks with their positions.
            let tool_positions: Vec<usize> = blocks
                .iter()
                .enumerate()
                .filter(|(_, b)| b.get("type").and_then(|v| v.as_str()) == Some("tool_result"))
                .map(|(i, _)| i)
                .collect();

            if tool_positions.len() > 1 {
                let start = *tool_positions
                    .first()
                    .expect("tool_positions has length > 1");
                let end = *tool_positions
                    .last()
                    .expect("tool_positions has length > 1");
                let is_contiguous = end - start + 1 == tool_positions.len();

                if is_contiguous {
                    // Fast path: sort the contiguous slice in place
                    blocks[start..=end].sort_by_key(|b| {
                        let id = b.get("tool_use_id").and_then(|v| v.as_str()).unwrap_or("");
                        *last_order.get(id).unwrap_or(&0)
                    });
                } else {
                    // Fallback: extract, sort, and replace for non-contiguous blocks
                    let mut tool_blocks: Vec<JsonValue> = tool_positions
                        .iter()
                        .map(|&i| std::mem::take(&mut blocks[i]))
                        .collect();

                    tool_blocks.sort_by_key(|b| {
                        let id = b.get("tool_use_id").and_then(|v| v.as_str()).unwrap_or("");
                        *last_order.get(id).unwrap_or(&0)
                    });

                    for (sorted_idx, &pos) in tool_positions.iter().enumerate() {
                        blocks[pos] = std::mem::take(&mut tool_blocks[sorted_idx]);
                    }
                }
            }
        }
    }

    messages
}

// Drop reasoning and non-essential messages before the last user message.
pub(crate) fn drop_thinking_messages(messages: Vec<JsonValue>) -> Vec<JsonValue> {
    let last_user_idx = find_last_user_index(&messages);
    let mut out = Vec::with_capacity(messages.len());
    const KEEP: &[&str] = &[
        "user",
        "system",
        "tool",
        "latest_reminder",
        "direct_search_results",
    ];

    for (idx, mut msg) in messages.into_iter().enumerate() {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if KEEP.contains(&role) || last_user_idx.is_none_or(|u| idx >= u) {
            out.push(msg);
        } else if role == "assistant" {
            if let Some(obj) = msg.as_object_mut() {
                obj.remove("reasoning_content");
            }
            out.push(msg);
        }
        // developer and other roles before last_user_idx are dropped.
    }
    out
}

pub(crate) fn resolve_thinking_mode(
    args: Option<&std::collections::HashMap<String, serde_json::Value>>,
    default_mode: ThinkingMode,
) -> ThinkingMode {
    if let Some(enabled) = crate::thinking_bool_from_args(args) {
        return if enabled {
            ThinkingMode::Thinking
        } else {
            ThinkingMode::Chat
        };
    }
    if let Some(args) = args
        && let Some(mode) = args.get("thinking_mode").and_then(|v| v.as_str())
    {
        match mode {
            "chat" => return ThinkingMode::Chat,
            "thinking" => return ThinkingMode::Thinking,
            _ => {}
        }
    }
    default_mode
}

pub(crate) fn inject_tools_and_response_format(
    messages_array: &mut Vec<JsonValue>,
    req: &dyn crate::OAIChatLikeRequest,
) -> Result<()> {
    let tools_json = req
        .tools()
        .map(|t| serde_json::to_value(&t))
        .transpose()
        .context("Failed to convert tools to JSON")?;

    // OpenAI semantics for `tool_choice: "none"`: the model must not call
    // tools. Strip the tool definitions from the prompt so the model never
    // sees the DSML tool instructions and cannot emit raw tool markup.
    // response_format is kept intact. Mirrors the jinja path's
    // exclude_tools_when_tool_choice_none handling (template/oai.rs).
    let tools_json = match req.tool_choice() {
        Some(ref tc) if tc.as_str() == Some("none") => None,
        _ => tools_json,
    };

    let response_format_json = req
        .response_format()
        .map(|rf| serde_json::to_value(&rf))
        .transpose()
        .context("Failed to convert response_format to JSON")?;

    if tools_json.is_some() || response_format_json.is_some() {
        let system_idx = messages_array
            .iter()
            .position(|msg| msg.get("role").and_then(|r| r.as_str()) == Some("system"));

        if let Some(idx) = system_idx {
            if let Some(msg) = messages_array.get_mut(idx)
                && let Some(obj) = msg.as_object_mut()
            {
                if let Some(tools) = tools_json {
                    obj.insert("tools".to_string(), tools);
                }
                if let Some(rf) = response_format_json {
                    obj.insert("response_format".to_string(), rf);
                }
            }
        } else {
            let mut system_msg = serde_json::json!({
                "role": "system",
                "content": ""
            });
            if let Some(obj) = system_msg.as_object_mut() {
                if let Some(tools) = tools_json {
                    obj.insert("tools".to_string(), tools);
                }
                if let Some(rf) = response_format_json {
                    obj.insert("response_format".to_string(), rf);
                }
            }
            messages_array.insert(0, system_msg);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_extract_visible_text_from_content_array() {
        let content = json!([
            {"type": "text", "text": "who "},
            {"type": "text", "text": "are "},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
            {"type": "text", "text": "you?"}
        ]);
        assert_eq!(extract_visible_text(&content), "who are you?");
    }

    #[test]
    fn test_render_tools_preserves_placeholder_text_inside_tool_schema() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "placeholder_tool",
                "description": "literal {dsml_token} {thinking_start_token} {thinking_end_token}",
                "parameters": {"type": "object", "properties": {}}
            }
        }]);
        let rendered = render_tools(
            "static {dsml_token} {thinking_start_token} {thinking_end_token}\n{tool_schemas}",
            tools.as_array().unwrap(),
        );

        assert!(rendered.starts_with(&format!(
            "static {} {} {}\n",
            tokens::DSML_TOKEN,
            tokens::THINKING_START,
            tokens::THINKING_END
        )));
        assert!(
            rendered.contains("literal {dsml_token} {thinking_start_token} {thinking_end_token}")
        );
    }
}

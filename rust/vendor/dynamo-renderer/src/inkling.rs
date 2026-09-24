// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Inkling chat renderer.
//!
//! Inkling does not publish a Hugging Face `chat_template`. Its wire format is
//! a sequence of role and content-kind special tokens, so rendering it through
//! Jinja is both unnecessary and lossy. This formatter emits those special
//! token spellings as text; the consumer's tokenizer turns them into the exact
//! token IDs before dispatching the request to the backend.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use minijinja::Value;
use serde_json::{Map, Value as JsonValue, json};

use crate::{OAIChatLikeRequest, OAIPromptFormatter};

const MESSAGE_USER: &str = "<|message_user|>";
const MESSAGE_MODEL: &str = "<|message_model|>";
const MESSAGE_SYSTEM: &str = "<|message_system|>";
const MESSAGE_TOOL: &str = "<|message_tool|>";
const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_IMAGE: &str = "<|content_image|>";
const CONTENT_MODEL_END_SAMPLING: &str = "<|content_model_end_sampling|>";
const CONTENT_AUDIO_INPUT: &str = "<|content_audio_input|>";
const CONTENT_THINKING: &str = "<|content_thinking|>";
const CONTENT_XML: &str = "<|content_xml|>";
const CONTENT_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
const END_MESSAGE: &str = "<|end_message|>";
const AUDIO_END: &str = "<|audio_end|>";
const MAX_REASONING_EFFORT: f64 = 0.99;

#[derive(Debug, Clone, Copy, Default)]
pub struct InklingFormatter;

impl OAIPromptFormatter for InklingFormatter {
    fn supports_add_generation_prompt(&self) -> bool {
        true
    }

    fn render(&self, req: &dyn OAIChatLikeRequest) -> Result<String> {
        let mut messages = json_value(req.messages()).context("serialize Inkling messages")?;
        let messages = messages
            .as_array_mut()
            .context("Inkling messages must be an array")?;

        let args = req.chat_template_args();
        if args
            .and_then(|args| args.get("continue_final_message"))
            .and_then(JsonValue::as_bool)
            == Some(true)
        {
            bail!("Inkling renderer does not support continue_final_message");
        }

        let tools_enabled = req.tool_choice().as_ref().and_then(Value::as_str) != Some("none");
        let request_tools = if tools_enabled {
            req.tools()
                .map(json_value)
                .transpose()
                .context("serialize Inkling tools")?
                .unwrap_or_else(|| JsonValue::Array(Vec::new()))
        } else {
            JsonValue::Array(Vec::new())
        };

        let mut output = String::new();
        let all_tools = if tools_enabled {
            collect_tools(messages, &request_tools)?
        } else {
            Vec::new()
        };
        write_tool_declarations(&mut output, &all_tools)?;

        let effort_value = req
            .reasoning_effort()
            .map(json_value)
            .transpose()
            .context("serialize Inkling reasoning_effort")?
            .or_else(|| args.and_then(|args| args.get("reasoning_effort").cloned()));
        let mut reasoning_effort = resolve_reasoning_effort(effort_value.as_ref());
        let mut tool_call_id_to_name = HashMap::new();

        for message in messages {
            let role = message
                .get("role")
                .and_then(JsonValue::as_str)
                .context("Inkling message is missing a string role")?;

            if !matches!(role, "system" | "developer")
                && let Some(effort) = reasoning_effort.take()
            {
                write_reasoning_effort(&mut output, effort)?;
            }

            match role {
                "system" | "developer" => write_content(
                    &mut output,
                    MESSAGE_SYSTEM,
                    message.get("content").unwrap_or(&JsonValue::Null),
                )?,
                "user" => write_content(
                    &mut output,
                    MESSAGE_USER,
                    message.get("content").unwrap_or(&JsonValue::Null),
                )?,
                "assistant" => write_assistant(&mut output, message, &mut tool_call_id_to_name)?,
                "tool" => write_tool_response(&mut output, message, &tool_call_id_to_name)?,
                other => bail!(
                    "unsupported Inkling message role {other:?}; expected system, developer, user, assistant, or tool"
                ),
            }
        }

        if let Some(effort) = reasoning_effort {
            write_reasoning_effort(&mut output, effort)?;
        }
        if req.should_add_generation_prompt() {
            output.push_str(MESSAGE_MODEL);
        }
        Ok(output)
    }
}

fn json_value(value: Value) -> Result<JsonValue> {
    serde_json::to_value(value).context("convert minijinja value to JSON")
}

fn collect_tools(messages: &[JsonValue], request_tools: &JsonValue) -> Result<Vec<JsonValue>> {
    let mut tools = request_tools
        .as_array()
        .context("Inkling tools must be an array")?
        .clone();
    for message in messages {
        if message.get("role").and_then(JsonValue::as_str) == Some("developer")
            && let Some(local_tools) = message.get("tools")
        {
            tools.extend(
                local_tools
                    .as_array()
                    .context("developer message tools must be an array")?
                    .iter()
                    .cloned(),
            );
        }
    }
    Ok(tools)
}

fn write_tool_declarations(output: &mut String, tools: &[JsonValue]) -> Result<()> {
    if tools.is_empty() {
        return Ok(());
    }

    let mut specs = Vec::with_capacity(tools.len());
    for tool in tools {
        let tool = tool
            .as_object()
            .context("Inkling tool declaration must be an object")?;
        let function = tool
            .get("function")
            .and_then(JsonValue::as_object)
            .context("Inkling tool declaration is missing function")?;
        let name = function
            .get("name")
            .and_then(JsonValue::as_str)
            .context("Inkling tool function is missing name")?;
        specs.push(json!({
            "description": function
                .get("description")
                .and_then(JsonValue::as_str)
                .unwrap_or(""),
            "name": name,
            "parameters": function
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| JsonValue::Object(Map::new())),
            "type": tool
                .get("type")
                .and_then(JsonValue::as_str)
                .unwrap_or("function"),
        }));
    }
    let payload = canonical_json(&JsonValue::Array(specs))?;
    write_block(
        output,
        MESSAGE_SYSTEM,
        Some("tool_declare"),
        CONTENT_XML,
        &payload,
    );
    Ok(())
}

fn write_content(output: &mut String, role_token: &str, content: &JsonValue) -> Result<()> {
    match content {
        JsonValue::Null => {}
        JsonValue::String(text) => {
            if !text.is_empty() {
                write_block(output, role_token, None, CONTENT_TEXT, text);
            }
        }
        JsonValue::Array(parts) => {
            for part in parts {
                if let Some(text) = part.as_str() {
                    write_block(output, role_token, None, CONTENT_TEXT, text);
                    continue;
                }
                let part = part
                    .as_object()
                    .context("Inkling content part must be an object")?;
                let part_type = part
                    .get("type")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("text");
                match part_type {
                    "text" | "input_text" => write_block(
                        output,
                        role_token,
                        None,
                        CONTENT_TEXT,
                        part.get("text").and_then(JsonValue::as_str).unwrap_or(""),
                    ),
                    "image" | "input_image" | "image_url" => {
                        write_block(output, role_token, None, CONTENT_IMAGE, "")
                    }
                    "audio" | "input_audio" | "audio_url" => {
                        output.push_str(role_token);
                        output.push_str(CONTENT_AUDIO_INPUT);
                        output.push_str(AUDIO_END);
                        output.push_str(END_MESSAGE);
                    }
                    "video" | "input_video" | "video_url" => {
                        bail!("Inkling does not support video content")
                    }
                    other => bail!("unsupported Inkling content part type {other:?}"),
                }
            }
        }
        _ => bail!("Inkling message content must be a string or array"),
    }
    Ok(())
}

fn write_assistant(
    output: &mut String,
    message: &JsonValue,
    tool_call_id_to_name: &mut HashMap<String, String>,
) -> Result<()> {
    let tool_calls = message
        .get("tool_calls")
        .and_then(JsonValue::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let reasoning = message
        .get("reasoning_content")
        .or_else(|| message.get("reasoning"));

    match reasoning {
        Some(JsonValue::Array(segments)) => {
            for (index, tool_call) in tool_calls.iter().enumerate() {
                if let Some(text) = segments.get(index).and_then(JsonValue::as_str) {
                    write_reasoning_block(output, text);
                }
                write_tool_call(output, tool_call, tool_call_id_to_name)?;
            }
            for segment in segments.iter().skip(tool_calls.len()) {
                let text = segment
                    .as_str()
                    .context("Inkling reasoning_content segments must be strings")?;
                write_reasoning_block(output, text);
            }
            write_content(
                output,
                MESSAGE_MODEL,
                message.get("content").unwrap_or(&JsonValue::Null),
            )?;
        }
        Some(JsonValue::String(text)) => {
            write_reasoning_block(output, text);
            write_content(
                output,
                MESSAGE_MODEL,
                message.get("content").unwrap_or(&JsonValue::Null),
            )?;
            for tool_call in tool_calls {
                write_tool_call(output, tool_call, tool_call_id_to_name)?;
            }
        }
        None | Some(JsonValue::Null) => {
            write_content(
                output,
                MESSAGE_MODEL,
                message.get("content").unwrap_or(&JsonValue::Null),
            )?;
            for tool_call in tool_calls {
                write_tool_call(output, tool_call, tool_call_id_to_name)?;
            }
        }
        Some(_) => bail!("Inkling reasoning_content must be a string or array of strings"),
    }

    output.push_str(CONTENT_MODEL_END_SAMPLING);
    Ok(())
}

fn write_reasoning_block(output: &mut String, text: &str) {
    if !text.is_empty() {
        write_block(output, MESSAGE_MODEL, None, CONTENT_THINKING, text);
    }
}

fn write_tool_call(
    output: &mut String,
    tool_call: &JsonValue,
    tool_call_id_to_name: &mut HashMap<String, String>,
) -> Result<()> {
    let tool_call = tool_call
        .as_object()
        .context("Inkling tool call must be an object")?;
    let function = tool_call
        .get("function")
        .and_then(JsonValue::as_object)
        .context("Inkling tool call is missing function")?;
    let name = function
        .get("name")
        .and_then(JsonValue::as_str)
        .context("Inkling tool call function is missing name")?;
    if let Some(id) = tool_call.get("id").and_then(JsonValue::as_str)
        && !id.is_empty()
    {
        tool_call_id_to_name.insert(id.to_string(), name.to_string());
    }

    let arguments = match function.get("arguments") {
        None | Some(JsonValue::Null) => JsonValue::Object(Map::new()),
        Some(JsonValue::String(arguments)) if arguments.trim().is_empty() => {
            JsonValue::Object(Map::new())
        }
        Some(JsonValue::String(arguments)) => serde_json::from_str(arguments)
            .context("Inkling tool call arguments must be valid JSON")?,
        Some(arguments) => arguments.clone(),
    };
    if !arguments.is_object() {
        bail!("Inkling tool call arguments must decode to a JSON object");
    }

    let name_json = serde_json::to_string(name)?;
    let args_json = canonical_json(&arguments)?;
    let payload = format!("{{\"name\":{name_json},\"args\":{args_json}}}");
    write_block(
        output,
        MESSAGE_MODEL,
        Some(name),
        CONTENT_INVOKE_TOOL_JSON,
        &payload,
    );
    Ok(())
}

fn write_tool_response(
    output: &mut String,
    message: &JsonValue,
    tool_call_id_to_name: &HashMap<String, String>,
) -> Result<()> {
    let tool_call_id = message
        .get("tool_call_id")
        .and_then(JsonValue::as_str)
        .unwrap_or("");
    let name = message
        .get("name")
        .and_then(JsonValue::as_str)
        .or_else(|| tool_call_id_to_name.get(tool_call_id).map(String::as_str))
        .unwrap_or("");
    let text = flatten_text_content(message.get("content").unwrap_or(&JsonValue::Null))?;
    write_block(output, MESSAGE_TOOL, Some(name), CONTENT_TEXT, &text);
    Ok(())
}

fn flatten_text_content(content: &JsonValue) -> Result<String> {
    match content {
        JsonValue::Null => Ok(String::new()),
        JsonValue::String(text) => Ok(text.clone()),
        JsonValue::Array(parts) => {
            let mut output = String::new();
            for part in parts {
                if let Some(text) = part.as_str() {
                    output.push_str(text);
                    continue;
                }
                let part = part
                    .as_object()
                    .context("Inkling tool response part must be an object")?;
                let kind = part
                    .get("type")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("text");
                if !matches!(kind, "text" | "input_text") {
                    bail!("Inkling tool response content must be text, got {kind:?}");
                }
                output.push_str(part.get("text").and_then(JsonValue::as_str).unwrap_or(""));
            }
            Ok(output)
        }
        _ => bail!("Inkling tool response content must be text"),
    }
}

fn write_reasoning_effort(output: &mut String, effort: f64) -> Result<()> {
    if !(0.0..=MAX_REASONING_EFFORT).contains(&effort) {
        bail!("Inkling reasoning_effort must be in [0.0, 0.99], got {effort}");
    }
    let formatted = format!("{effort:.2}");
    let effort = formatted.trim_end_matches('0').trim_end_matches('.');
    let effort = if matches!(effort, "0" | "-0") {
        "0.0"
    } else {
        effort
    };
    write_block(
        output,
        MESSAGE_SYSTEM,
        None,
        CONTENT_TEXT,
        &format!("Thinking effort level: {effort}"),
    );
    Ok(())
}

fn resolve_reasoning_effort(value: Option<&JsonValue>) -> Option<f64> {
    let Some(value) = value else {
        return Some(0.9);
    };
    match value {
        JsonValue::String(name) => match name.as_str() {
            "none" => Some(0.0),
            "minimal" => Some(0.1),
            "low" => Some(0.2),
            "medium" => Some(0.7),
            "high" => Some(0.9),
            "xhigh" | "max" => Some(0.99),
            _ => None,
        },
        JsonValue::Number(number) => number.as_f64(),
        _ => None,
    }
}

fn write_block(
    output: &mut String,
    role_token: &str,
    author_name: Option<&str>,
    content_token: &str,
    text: &str,
) {
    output.push_str(role_token);
    if let Some(author_name) = author_name
        && !author_name.is_empty()
    {
        output.push_str(author_name);
    }
    output.push_str(content_token);
    output.push_str(text);
    output.push_str(END_MESSAGE);
}

fn canonical_json(value: &JsonValue) -> Result<String> {
    serde_json::to_string(&sort_json(value)).context("serialize Inkling JSON payload")
}

fn sort_json(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Array(items) => JsonValue::Array(items.iter().map(sort_json).collect()),
        JsonValue::Object(map) => {
            let mut sorted = Map::new();
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            for key in keys {
                sorted.insert(key.clone(), sort_json(&map[key]));
            }
            JsonValue::Object(sorted)
        }
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OAIChatLikeRequest, PromptFormatter};
    use std::collections::HashMap;

    #[derive(Default)]
    struct Request {
        messages: JsonValue,
        tools: Option<JsonValue>,
        tool_choice: Option<JsonValue>,
        args: Option<HashMap<String, JsonValue>>,
        reasoning_effort: Option<JsonValue>,
        add_generation_prompt: bool,
    }

    impl Request {
        fn new(messages: JsonValue) -> Self {
            Self {
                messages,
                add_generation_prompt: true,
                ..Default::default()
            }
        }
    }

    impl OAIChatLikeRequest for Request {
        fn model(&self) -> String {
            "thinkingmachines/Inkling-NVFP4".to_string()
        }

        fn messages(&self) -> Value {
            Value::from_serialize(&self.messages)
        }

        fn tools(&self) -> Option<Value> {
            self.tools.as_ref().map(Value::from_serialize)
        }

        fn tool_choice(&self) -> Option<Value> {
            self.tool_choice.as_ref().map(Value::from_serialize)
        }

        fn response_format(&self) -> Option<Value> {
            None
        }

        fn reasoning_effort(&self) -> Option<Value> {
            self.reasoning_effort.as_ref().map(Value::from_serialize)
        }

        fn should_add_generation_prompt(&self) -> bool {
            self.add_generation_prompt
        }

        fn chat_template_args(&self) -> Option<&HashMap<String, JsonValue>> {
            self.args.as_ref()
        }
    }

    #[test]
    fn renders_text_and_image_like_vllm_fixture() {
        let request = Request::new(json!([{
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}
            ]
        }]));
        assert_eq!(
            InklingFormatter.render(&request).unwrap(),
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>look<|end_message|><|message_user|><|content_image|><|end_message|><|message_model|>"
        );
    }

    #[test]
    fn renders_audio_markers_like_vllm_fixture() {
        let request = Request::new(json!([{
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe"},
                {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}},
                {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,"}}
            ]
        }]));
        assert_eq!(
            InklingFormatter.render(&request).unwrap(),
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>transcribe<|end_message|><|message_user|><|content_audio_input|><|audio_end|><|end_message|><|message_user|><|content_audio_input|><|audio_end|><|end_message|><|message_model|>"
        );
    }

    #[test]
    fn renders_tool_declaration_and_round_trip_like_vllm_fixtures() {
        let mut request = Request::new(json!([
            {
                "role": "developer",
                "content": "rules",
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "local_tool",
                        "parameters": {"z": 1, "a": {"b": 2}}
                    }
                }]
            },
            {"role": "user", "content": "hi"}
        ]));
        request.tools = Some(json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "required": ["city"],
                    "properties": {"city": {"type": "string"}}
                }
            }
        }]));
        assert_eq!(
            InklingFormatter.render(&request).unwrap(),
            "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Get weather information\",\"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"},\"type\":\"function\"},{\"description\":\"\",\"name\":\"local_tool\",\"parameters\":{\"a\":{\"b\":2},\"z\":1},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>rules<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>hi<|end_message|><|message_model|>"
        );

        let mut round_trip = Request::new(json!([
            {
                "role": "assistant",
                "reasoning_content": "think",
                "content": "answer",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
        ]));
        round_trip.add_generation_prompt = false;
        assert_eq!(
            InklingFormatter.render(&round_trip).unwrap(),
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_model|><|content_thinking|>think<|end_message|><|message_model|><|content_text|>answer<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"SF\"}}<|end_message|><|content_model_end_sampling|><|message_tool|>get_weather<|content_text|>sunny<|end_message|>"
        );
    }

    #[test]
    fn interleaves_segmented_reasoning_with_tool_calls() {
        let mut request = Request::new(json!([{
            "role": "assistant",
            "reasoning_content": ["first", "second", "after"],
            "content": "done",
            "tool_calls": [
                {"id": "a", "function": {"name": "one", "arguments": "{\"b\":2,\"a\":1}"}},
                {"id": "b", "function": {"name": "two", "arguments": "{}"}}
            ]
        }]));
        request.reasoning_effort = Some(json!("none"));
        request.add_generation_prompt = false;
        assert_eq!(
            InklingFormatter.render(&request).unwrap(),
            "<|message_system|><|content_text|>Thinking effort level: 0.0<|end_message|><|message_model|><|content_thinking|>first<|end_message|><|message_model|>one<|content_invoke_tool_json|>{\"name\":\"one\",\"args\":{\"a\":1,\"b\":2}}<|end_message|><|message_model|><|content_thinking|>second<|end_message|><|message_model|>two<|content_invoke_tool_json|>{\"name\":\"two\",\"args\":{}}<|end_message|><|message_model|><|content_thinking|>after<|end_message|><|message_model|><|content_text|>done<|end_message|><|content_model_end_sampling|>"
        );
    }

    #[test]
    fn ignores_unsupported_reasoning_effort_values() {
        for value in [json!(true), json!("invalid"), json!(null)] {
            let mut request = Request::new(json!([{
                "role": "user",
                "content": "test"
            }]));
            request.reasoning_effort = Some(value);

            assert_eq!(
                InklingFormatter.render(&request).unwrap(),
                "<|message_user|><|content_text|>test<|end_message|><|message_model|>"
            );
        }
    }

    #[test]
    fn tool_choice_none_suppresses_declarations() {
        let mut request = Request::new(json!([
            {
                "role": "developer",
                "content": "rules",
                "tools": [{
                    "type": "function",
                    "function": {"name": "also_hidden", "parameters": {}}
                }]
            },
            {"role": "user", "content": "hi"}
        ]));
        request.tools = Some(json!([{
            "type": "function",
            "function": {"name": "hidden", "parameters": {}}
        }]));
        request.tool_choice = Some(json!("none"));
        let rendered = InklingFormatter.render(&request).unwrap();
        assert!(!rendered.contains("tool_declare"));
        assert!(!rendered.contains("hidden"));
        assert!(!rendered.contains("also_hidden"));
    }

    #[test]
    fn native_selection_uses_exact_model_type_not_display_name() {
        assert!(matches!(
            crate::native_formatter_for(&Some("inkling_mm_model".to_string()), "renamed"),
            Some(PromptFormatter::OAI(_))
        ));
        assert!(crate::native_formatter_for(&None, "inkling-nvfp4").is_none());
    }
}

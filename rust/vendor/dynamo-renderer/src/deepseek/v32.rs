// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek V3.2 native prompt formatting
//!
//! This module provides native Rust implementation of DeepSeek V3.2's chat template,
//! based on their official Python code: encoding_dsv32.py
//!
//! Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main/encoding

use anyhow::{Context, Result};
use serde_json::Value as JsonValue;

use super::common::{
    NormalizeNonText, RESPONSE_FORMAT_TEMPLATE, TOOL_CALL_TEMPLATE, TOOL_OUTPUT_TEMPLATE,
    TOOLS_SYSTEM_TEMPLATE, encode_arguments_to_dsml, find_last_user_index,
    normalize_message_contents, render_tools, to_json,
};

pub use super::common::{ThinkingMode, tokens};

/// Render a single message
fn render_message(
    index: usize,
    messages: &[JsonValue],
    thinking_mode: ThinkingMode,
    last_user_idx: Option<usize>,
) -> Result<String> {
    let msg = &messages[index];
    let role = msg
        .get("role")
        .and_then(|r| r.as_str())
        .context("Missing 'role' field")?;

    let mut prompt = String::new();

    match role {
        "system" => {
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            prompt.push_str(content);

            if let Some(tools) = msg.get("tools").and_then(|t| t.as_array()) {
                prompt.push_str("\n\n");
                prompt.push_str(&render_tools(TOOLS_SYSTEM_TEMPLATE, tools));
            }

            if let Some(response_format) = msg.get("response_format") {
                prompt.push_str("\n\n");
                prompt.push_str(
                    &RESPONSE_FORMAT_TEMPLATE.replace("{schema}", &to_json(response_format)),
                );
            }
        }

        "user" => {
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            prompt.push_str(tokens::USER_START);
            prompt.push_str(content);
            prompt.push_str(tokens::ASSISTANT_START);

            if Some(index) == last_user_idx && thinking_mode == ThinkingMode::Thinking {
                prompt.push_str(tokens::THINKING_START);
            } else {
                prompt.push_str(tokens::THINKING_END);
            }
        }

        "developer" => {
            let content = msg
                .get("content")
                .and_then(|c| c.as_str())
                .context("Developer role requires content")?;

            let mut content_developer = String::new();

            if let Some(tools) = msg.get("tools").and_then(|t| t.as_array()) {
                content_developer.push_str("\n\n");
                content_developer.push_str(&render_tools(TOOLS_SYSTEM_TEMPLATE, tools));
            }

            if let Some(response_format) = msg.get("response_format") {
                content_developer.push_str("\n\n");
                content_developer.push_str(
                    &RESPONSE_FORMAT_TEMPLATE.replace("{schema}", &to_json(response_format)),
                );
            }

            content_developer.push_str(&format!("\n\n# The user's message is: {}", content));

            prompt.push_str(tokens::USER_START);
            prompt.push_str(&content_developer);
            prompt.push_str(tokens::ASSISTANT_START);

            if Some(index) == last_user_idx && thinking_mode == ThinkingMode::Thinking {
                prompt.push_str(tokens::THINKING_START);
            } else {
                prompt.push_str(tokens::THINKING_END);
            }
        }

        "assistant" => {
            // Handle reasoning content
            // NOTE: If this assistant comes after last user message, the opening <think>
            // was already added in the user message. We only need to add content and closing tag.
            //
            // Handle reasoning_content which may be a plain string or an array of segments.
            // DeepSeek V3.2 always places its <think> block before all tool calls, so
            // joining segments produces the correct flat form here.
            if thinking_mode == ThinkingMode::Thinking
                && last_user_idx.is_some_and(|idx| index > idx)
            {
                let reasoning = msg.get("reasoning_content").and_then(|v| match v {
                    serde_json::Value::String(s) => {
                        if s.is_empty() {
                            None
                        } else {
                            Some(s.clone())
                        }
                    }
                    serde_json::Value::Array(arr) => {
                        let joined = arr
                            .iter()
                            .filter_map(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                            .collect::<Vec<_>>()
                            .join("\n");
                        if joined.is_empty() {
                            None
                        } else {
                            Some(joined)
                        }
                    }
                    _ => None,
                });

                if let Some(reasoning) = reasoning {
                    // DON'T add THINKING_START - it was already added in user message
                    prompt.push_str(&reasoning);
                    prompt.push_str(tokens::THINKING_END);
                }
            }

            // Handle content
            if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                prompt.push_str(content);
            }

            // Handle tool calls
            if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array())
                && !tool_calls.is_empty()
            {
                prompt.push_str("\n\n");
                prompt.push_str(&format!("<{}function_calls>\n", tokens::DSML_TOKEN));

                for tool_call in tool_calls {
                    let name = tool_call
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .context("Missing tool call name")?;

                    let arguments = encode_arguments_to_dsml(
                        tool_call.get("function").context("Missing function")?,
                    )?;

                    let invoke = TOOL_CALL_TEMPLATE
                        .replace("{dsml_token}", tokens::DSML_TOKEN)
                        .replace("{name}", name)
                        .replace("{arguments}", &arguments);

                    prompt.push_str(&invoke);
                    prompt.push('\n');
                }

                prompt.push_str(&format!("</{}function_calls>", tokens::DSML_TOKEN));
            }

            prompt.push_str(tokens::EOS);
        }

        "tool" => {
            // Find the previous assistant message
            let mut prev_assistant_idx = None;
            let mut tool_count = 0;

            for i in (0..index).rev() {
                let prev_role = messages[i].get("role").and_then(|r| r.as_str());
                if prev_role == Some("tool") {
                    tool_count += 1;
                } else if prev_role == Some("assistant") {
                    prev_assistant_idx = Some(i);
                    break;
                }
            }

            let tool_call_order = tool_count + 1;

            // Add opening tag for first tool result
            if tool_call_order == 1 {
                prompt.push_str("\n\n<function_results>");
            }

            // Add result
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            prompt.push_str(&TOOL_OUTPUT_TEMPLATE.replace("{content}", content));

            // Check if this is the last tool result
            if let Some(prev_idx) = prev_assistant_idx {
                let tool_calls_count = messages[prev_idx]
                    .get("tool_calls")
                    .and_then(|t| t.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);

                if tool_call_order == tool_calls_count {
                    prompt.push_str("\n</function_results>");

                    if last_user_idx.is_some_and(|idx| index >= idx)
                        && thinking_mode == ThinkingMode::Thinking
                    {
                        prompt.push_str("\n\n");
                        prompt.push_str(tokens::THINKING_START);
                    } else {
                        prompt.push_str("\n\n");
                        prompt.push_str(tokens::THINKING_END);
                    }
                }
            }
        }

        _ => anyhow::bail!("Unknown role: {}", role),
    }

    Ok(prompt)
}

/// Encode messages to prompt string
///
/// # Arguments
/// * `messages` - Array of messages in OpenAI format
/// * `thinking_mode` - Whether to use thinking mode
/// * `add_bos_token` - Whether to add BOS token at start
///
/// # Returns
/// Formatted prompt string ready for tokenization
pub fn encode_messages(
    messages: &[JsonValue],
    thinking_mode: ThinkingMode,
    add_bos_token: bool,
) -> Result<String> {
    let mut prompt = String::new();

    if add_bos_token {
        prompt.push_str(tokens::BOS);
    }

    let last_user_idx = find_last_user_index(messages);

    for (index, _) in messages.iter().enumerate() {
        let msg_prompt = render_message(index, messages, thinking_mode, last_user_idx)?;
        prompt.push_str(&msg_prompt);
    }

    Ok(prompt)
}

/// DeepSeek V3.2 Prompt Formatter
///
/// Implements OAIPromptFormatter for DeepSeek V3.2 models using native Rust implementation
#[derive(Debug)]
pub struct DeepSeekV32Formatter {
    thinking_mode: ThinkingMode,
}

impl DeepSeekV32Formatter {
    pub fn new(thinking_mode: ThinkingMode) -> Self {
        Self { thinking_mode }
    }

    /// Create formatter with thinking mode enabled (default for DSV3.2)
    pub fn new_thinking() -> Self {
        Self::new(ThinkingMode::Thinking)
    }

    /// Create formatter with chat mode
    pub fn new_chat() -> Self {
        Self::new(ThinkingMode::Chat)
    }
}

impl crate::OAIPromptFormatter for DeepSeekV32Formatter {
    fn supports_add_generation_prompt(&self) -> bool {
        true
    }

    fn render(&self, req: &dyn crate::OAIChatLikeRequest) -> Result<String> {
        let thinking_mode =
            super::common::resolve_thinking_mode(req.chat_template_args(), self.thinking_mode);

        // Get messages from request
        let messages_value = req.messages();

        // Convert minijinja Value to serde_json Value
        let messages_json =
            serde_json::to_value(&messages_value).context("Failed to convert messages to JSON")?;

        let mut messages_array = messages_json
            .as_array()
            .context("Messages is not an array")?
            .clone();

        // DeepSeek V3.2 native formatter expects text content in each message.
        // Normalize OpenAI content arrays (e.g. [{type: "text", text: "..."}]) to strings.
        normalize_message_contents(&mut messages_array, NormalizeNonText::SerializeJson);

        // Inject tools and response_format from request into the first system message
        // DeepSeek V3.2 expects these to be part of the system message for prompt rendering
        super::common::inject_tools_and_response_format(&mut messages_array, req)?;

        // Encode with native implementation
        encode_messages(
            &messages_array,
            thinking_mode,
            true, // always add BOS token
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_simple_conversation() {
        let messages = json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]);

        let result =
            encode_messages(messages.as_array().unwrap(), ThinkingMode::Thinking, true).unwrap();

        assert!(result.starts_with(tokens::BOS));
        assert!(result.contains("You are a helpful assistant."));
        assert!(result.contains(tokens::USER_START));
        assert!(result.contains("Hello!"));
        assert!(result.contains(tokens::ASSISTANT_START));
        assert!(result.contains(tokens::THINKING_START));
    }

    #[test]
    fn test_formatter_handles_user_content_array() {
        use crate::OAIPromptFormatter;

        let request = MockRequest::new(json!([
            {"role": "user", "content": [
                {"type": "text", "text": "who are you?"}
            ]}
        ]));

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        assert!(result.contains("who are you?"));
        assert!(result.contains(tokens::USER_START));
        assert!(result.contains(tokens::ASSISTANT_START));
    }

    #[test]
    fn test_formatter_serializes_non_text_content() {
        use crate::OAIPromptFormatter;

        let request = MockRequest::new(json!([
            {"role": "user", "content": {"foo": "bar"}}
        ]));

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        assert!(result.contains(r#"{"foo": "bar"}"#));
    }

    #[test]
    fn test_tools_rendering() {
        let messages = json!([
            {
                "role": "system",
                "content": "You are helpful.",
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            }
                        }
                    }
                }]
            },
            {"role": "user", "content": "What's the weather?"}
        ]);

        let result =
            encode_messages(messages.as_array().unwrap(), ThinkingMode::Thinking, true).unwrap();

        assert!(result.contains("## Tools"));
        assert!(result.contains("get_weather"));
        assert!(result.contains("<functions>"));
    }

    // Mock request for testing OAIPromptFormatter implementation
    struct MockRequest {
        messages: JsonValue,
        tools: Option<JsonValue>,
        response_format: Option<JsonValue>,
        chat_template_args: Option<std::collections::HashMap<String, JsonValue>>,
    }

    impl MockRequest {
        fn new(messages: JsonValue) -> Self {
            Self {
                messages,
                tools: None,
                response_format: None,
                chat_template_args: None,
            }
        }

        fn with_tools(mut self, tools: JsonValue) -> Self {
            self.tools = Some(tools);
            self
        }

        fn with_response_format(mut self, response_format: JsonValue) -> Self {
            self.response_format = Some(response_format);
            self
        }

        fn with_chat_template_args(
            mut self,
            args: std::collections::HashMap<String, JsonValue>,
        ) -> Self {
            self.chat_template_args = Some(args);
            self
        }
    }

    impl crate::OAIChatLikeRequest for MockRequest {
        fn model(&self) -> String {
            "deepseek-v3.2".to_string()
        }

        fn messages(&self) -> minijinja::value::Value {
            minijinja::value::Value::from_serialize(&self.messages)
        }

        fn tools(&self) -> Option<minijinja::value::Value> {
            self.tools
                .as_ref()
                .map(minijinja::value::Value::from_serialize)
        }

        fn response_format(&self) -> Option<minijinja::value::Value> {
            self.response_format
                .as_ref()
                .map(minijinja::value::Value::from_serialize)
        }

        fn should_add_generation_prompt(&self) -> bool {
            true
        }

        fn chat_template_args(
            &self,
        ) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
            self.chat_template_args.as_ref()
        }
    }

    #[test]
    fn test_formatter_injects_tools_into_existing_system_message() {
        use crate::OAIPromptFormatter;

        let tools = json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Moscow?"}
        ]))
        .with_tools(tools);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // Verify tools were injected into the prompt
        assert!(
            result.contains("## Tools"),
            "Should contain Tools section header"
        );
        assert!(
            result.contains("get_weather"),
            "Should contain function name"
        );
        assert!(
            result.contains("<functions>"),
            "Should contain functions block"
        );
        assert!(
            result.contains("</functions>"),
            "Should contain closing functions tag"
        );
        assert!(
            result.contains("You are a helpful assistant."),
            "Should preserve original system content"
        );
        assert!(
            result.contains(&format!("<{}function_calls>", tokens::DSML_TOKEN)),
            "Should contain DSML format instructions"
        );
    }

    #[test]
    fn test_formatter_creates_system_message_for_tools_when_missing() {
        use crate::OAIPromptFormatter;

        let tools = json!([{
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get current time in a timezone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string"}
                    },
                    "required": ["timezone"]
                }
            }
        }]);

        // Request without system message
        let request = MockRequest::new(json!([
            {"role": "user", "content": "What time is it in Tokyo?"}
        ]))
        .with_tools(tools);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // Verify tools were injected via auto-created system message
        assert!(
            result.contains("## Tools"),
            "Should contain Tools section even without explicit system message"
        );
        assert!(
            result.contains("get_current_time"),
            "Should contain function name"
        );
        assert!(
            result.contains("<functions>"),
            "Should contain functions block"
        );
    }

    #[test]
    fn test_formatter_without_tools_does_not_add_tools_section() {
        use crate::OAIPromptFormatter;

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]));

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // Verify no tools section was added
        assert!(
            !result.contains("## Tools"),
            "Should not contain Tools section when no tools provided"
        );
        assert!(
            !result.contains("<functions>"),
            "Should not contain functions block when no tools provided"
        );
        assert!(
            result.contains("You are a helpful assistant."),
            "Should preserve system content"
        );
    }

    #[test]
    fn test_formatter_with_multiple_tools() {
        use crate::OAIPromptFormatter;

        let tools = json!([
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get current time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {"type": "string"}
                        }
                    }
                }
            }
        ]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Weather and time in Moscow?"}
        ]))
        .with_tools(tools);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // Verify both tools are present
        assert!(
            result.contains("get_weather"),
            "Should contain first function"
        );
        assert!(
            result.contains("get_current_time"),
            "Should contain second function"
        );
    }

    // ==================== Structured Output Tests ====================

    #[test]
    fn test_formatter_injects_response_format_into_existing_system_message() {
        use crate::OAIPromptFormatter;

        let response_format = json!({
            "type": "json_schema",
            "json_schema": {
                "name": "city_info",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "country": {"type": "string"},
                        "population": {"type": "number"}
                    },
                    "required": ["city", "country", "population"]
                }
            }
        });

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about Moscow."}
        ]))
        .with_response_format(response_format);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // Verify response format was injected into the prompt
        assert!(
            result.contains("## Response Format:"),
            "Should contain Response Format section header"
        );
        assert!(
            result.contains("json_schema"),
            "Should contain json_schema type"
        );
        assert!(result.contains("city_info"), "Should contain schema name");
        assert!(
            result.contains("You are a helpful assistant."),
            "Should preserve original system content"
        );
    }

    #[test]
    fn test_formatter_creates_system_message_for_response_format_when_missing() {
        use crate::OAIPromptFormatter;

        let response_format = json!({
            "type": "json_schema",
            "json_schema": {
                "name": "weather_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "temperature": {"type": "number"},
                        "conditions": {"type": "string"}
                    }
                }
            }
        });

        // Request without system message
        let request = MockRequest::new(json!([
            {"role": "user", "content": "What's the weather?"}
        ]))
        .with_response_format(response_format);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // Verify response format was injected via auto-created system message
        assert!(
            result.contains("## Response Format:"),
            "Should contain Response Format section even without explicit system message"
        );
        assert!(
            result.contains("weather_response"),
            "Should contain schema name"
        );
    }

    #[test]
    fn test_formatter_with_both_tools_and_response_format() {
        use crate::OAIPromptFormatter;

        let tools = json!([{
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            }
        }]);

        let response_format = json!({
            "type": "json_schema",
            "json_schema": {
                "name": "search_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "results": {"type": "array"},
                        "total_count": {"type": "number"}
                    }
                }
            }
        });

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a search assistant."},
            {"role": "user", "content": "Find documents about Rust."}
        ]))
        .with_tools(tools)
        .with_response_format(response_format);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // Verify both tools and response format are present
        assert!(result.contains("## Tools"), "Should contain Tools section");
        assert!(
            result.contains("search_database"),
            "Should contain function name"
        );
        assert!(
            result.contains("## Response Format:"),
            "Should contain Response Format section"
        );
        assert!(
            result.contains("search_result"),
            "Should contain schema name"
        );
        assert!(
            result.contains("You are a search assistant."),
            "Should preserve original system content"
        );
    }

    #[test]
    fn test_formatter_without_response_format_does_not_add_response_format_section() {
        use crate::OAIPromptFormatter;

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]));

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // Verify no response format section was added
        assert!(
            !result.contains("## Response Format:"),
            "Should not contain Response Format section when not provided"
        );
    }

    // ==================== Thinking Mode Override Tests ====================

    #[test]
    fn test_chat_mode_via_thinking_false() {
        use crate::OAIPromptFormatter;

        let args = std::collections::HashMap::from([("thinking".to_string(), json!(false))]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]))
        .with_chat_template_args(args);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // In chat mode, the last user message should be followed by </think> (closing tag)
        // rather than <think> (opening tag)
        assert!(
            result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_END
            )),
            "Chat mode should end with </think> after Assistant token, got: ...{}",
            &result[result.len().saturating_sub(80)..],
        );
        assert!(
            !result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_START
            )),
            "Chat mode should NOT end with <think>",
        );
    }

    #[test]
    fn test_explicit_thinking_true_via_args() {
        use crate::OAIPromptFormatter;

        let args = std::collections::HashMap::from([("thinking".to_string(), json!(true))]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]))
        .with_chat_template_args(args);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        assert!(
            result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_START
            )),
            "Thinking mode should end with <think> after Assistant token",
        );
    }

    #[test]
    fn test_chat_mode_via_thinking_mode_string() {
        use crate::OAIPromptFormatter;

        let args = std::collections::HashMap::from([("thinking_mode".to_string(), json!("chat"))]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]))
        .with_chat_template_args(args);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        assert!(
            result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_END
            )),
            "thinking_mode='chat' should produce chat mode (ends with </think>)",
        );
    }

    #[test]
    fn test_thinking_mode_string_thinking() {
        use crate::OAIPromptFormatter;

        let args =
            std::collections::HashMap::from([("thinking_mode".to_string(), json!("thinking"))]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]))
        .with_chat_template_args(args);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        assert!(
            result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_START
            )),
            "thinking_mode='thinking' should produce thinking mode (ends with <think>)",
        );
    }

    #[test]
    fn test_default_thinking_mode_without_args() {
        use crate::OAIPromptFormatter;

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]));

        // No chat_template_args — should default to formatter's thinking mode
        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        assert!(
            result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_START
            )),
            "Default (new_thinking) should produce thinking mode",
        );

        // Verify new_chat() default also works
        let formatter_chat = DeepSeekV32Formatter::new_chat();
        let result_chat = formatter_chat.render(&request).unwrap();

        assert!(
            result_chat.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_END
            )),
            "Default (new_chat) should produce chat mode",
        );
    }

    #[test]
    fn test_thinking_false_overrides_default_thinking() {
        use crate::OAIPromptFormatter;

        let args = std::collections::HashMap::from([("thinking".to_string(), json!(false))]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]))
        .with_chat_template_args(args);

        // Formatter defaults to thinking, but request overrides to chat
        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        assert!(
            result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_END
            )),
            "Per-request thinking=false should override new_thinking() default",
        );
    }

    #[test]
    fn test_thinking_true_overrides_default_chat() {
        use crate::OAIPromptFormatter;

        let args = std::collections::HashMap::from([("thinking".to_string(), json!(true))]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]))
        .with_chat_template_args(args);

        // Formatter defaults to chat, but request overrides to thinking
        let formatter = DeepSeekV32Formatter::new_chat();
        let result = formatter.render(&request).unwrap();

        assert!(
            result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_START
            )),
            "Per-request thinking=true should override new_chat() default",
        );
    }

    #[test]
    fn test_thinking_bool_takes_precedence_over_thinking_mode_string() {
        use crate::OAIPromptFormatter;

        let args = std::collections::HashMap::from([
            ("thinking".to_string(), json!(false)),
            ("thinking_mode".to_string(), json!("thinking")),
        ]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]))
        .with_chat_template_args(args);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // "thinking": false should win over "thinking_mode": "thinking"
        assert!(
            result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_END
            )),
            "Boolean 'thinking' key should take precedence over 'thinking_mode' string",
        );
    }
}

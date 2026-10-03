// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Buffered chat completion → Anthropic Message.

use serde_json::{json, Value};

use super::{new_id, stop_reason, tool_input, usage_from_chat, EchoContext};

pub fn chat_to_message(chat: &Value, echo: &EchoContext) -> Value {
    let choice = chat.pointer("/choices/0");
    let message = choice.and_then(|c| c.get("message"));
    let str_field = |k: &str| {
        message
            .and_then(|m| m.get(k))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
    };
    let mut content = Vec::new();
    if let Some(r) = str_field("reasoning_content") {
        content.push(json!({"type": "thinking", "thinking": r, "signature": ""}));
    }
    if let Some(t) = str_field("content") {
        content.push(json!({"type": "text", "text": t}));
    }
    for tc in message
        .and_then(|m| m.get("tool_calls"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        content.push(json!({
            "type": "tool_use",
            "id": tc.get("id").and_then(Value::as_str).map(str::to_owned)
                .unwrap_or_else(|| new_id("toolu")),
            "name": tc.pointer("/function/name").and_then(Value::as_str).unwrap_or(""),
            "input": tool_input(tc.pointer("/function/arguments").and_then(Value::as_str).unwrap_or("")),
        }));
    }
    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }
    let (reason, sequence) = stop_reason(
        choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(Value::as_str),
        choice.and_then(|c| c.get("matched_stop")),
        echo,
    );
    json!({
        "id": new_id("msg"),
        "type": "message",
        "role": "assistant",
        "model": echo.model,
        "content": content,
        "stop_reason": reason,
        "stop_sequence": sequence,
        "usage": usage_from_chat(chat.get("usage")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn echo() -> EchoContext {
        EchoContext {
            model: "m".into(),
            stop_sequences: vec![],
        }
    }

    #[test]
    fn thinking_text_and_usage() {
        let m = chat_to_message(
            &json!({"choices": [{"finish_reason": "stop", "message": {"role": "assistant",
                    "content": "OK", "reasoning_content": "hm"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2}}),
            &echo(),
        );
        assert!(m["id"].as_str().unwrap().starts_with("msg_"));
        assert_eq!(m["type"], "message");
        assert_eq!(m["role"], "assistant");
        assert_eq!(m["model"], "m");
        assert_eq!(
            m["content"],
            json!([{"type": "thinking", "thinking": "hm", "signature": ""},
                   {"type": "text", "text": "OK"}])
        );
        assert_eq!(m["stop_reason"], "end_turn");
        assert_eq!(m["stop_sequence"], Value::Null);
        assert_eq!(m["usage"]["input_tokens"], 5);
        assert_eq!(m["usage"]["output_tokens"], 2);
    }

    #[test]
    fn tool_use_and_truncation() {
        let m = chat_to_message(
            &json!({"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant",
                    "content": null, "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"bj\"}"}}]}}]}),
            &echo(),
        );
        assert_eq!(m["stop_reason"], "tool_use");
        assert_eq!(
            m["content"],
            json!([{"type": "tool_use", "id": "call_1", "name": "get_weather",
                    "input": {"city": "bj"}}])
        );
        let m = chat_to_message(
            &json!({"choices": [{"finish_reason": "length", "message": {"role": "assistant",
                    "content": null, "reasoning_content": "long"}}]}),
            &echo(),
        );
        assert_eq!(m["stop_reason"], "max_tokens");
        assert_eq!(m["content"].as_array().unwrap().len(), 1);
        assert_eq!(m["content"][0]["type"], "thinking");
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Buffered chat completion → Response object.

use serde_json::Value;

use super::{
    call_item, message_item, new_id, now_secs, reasoning_item, response_object, usage_from_chat,
    EchoContext, Finish,
};

/// Output order: reasoning, message, function calls.
pub fn chat_to_response(chat: &Value, echo: &EchoContext) -> Value {
    let choice = chat.pointer("/choices/0");
    let message = choice.and_then(|c| c.get("message"));
    let finish = Finish::from_chat(
        choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(Value::as_str),
    );
    let item_status = finish.item_status();

    let mut output = Vec::new();
    let str_field = |k: &str| {
        message
            .and_then(|m| m.get(k))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
    };
    if let Some(r) = str_field("reasoning_content") {
        output.push(reasoning_item(&new_id("rs"), r, "completed"));
    }
    let tool_calls = message
        .and_then(|m| m.get("tool_calls"))
        .and_then(Value::as_array)
        .filter(|t| !t.is_empty());
    let content = str_field("content");
    // Emit a message item unless there are tool calls or only reasoning.
    if content.is_some() || (tool_calls.is_none() && output.is_empty()) {
        output.push(message_item(
            &new_id("msg"),
            content.unwrap_or(""),
            item_status,
        ));
    }
    for tc in tool_calls.into_iter().flatten() {
        let call_id = tc
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| new_id("call"));
        let name = tc
            .pointer("/function/name")
            .and_then(Value::as_str)
            .unwrap_or("");
        let args = tc
            .pointer("/function/arguments")
            .and_then(Value::as_str)
            .unwrap_or("");
        output.push(call_item(
            &new_id("fc"),
            &call_id,
            name,
            args,
            "completed",
            &echo.tools,
        ));
    }

    let created_at = chat
        .get("created")
        .and_then(Value::as_u64)
        .unwrap_or_else(now_secs);
    let mut resp = response_object(
        echo,
        &new_id("resp"),
        created_at,
        finish.status(),
        output,
        usage_from_chat(chat.get("usage")),
    );
    finish.annotate(&mut resp);
    resp
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::responses::to_chat;
    use serde_json::json;

    fn echo() -> EchoContext {
        to_chat(json!({"model": "m", "input": "x", "max_output_tokens": 512}))
            .unwrap()
            .echo
    }

    #[test]
    fn text_reply() {
        let r = chat_to_response(
            &json!({"id": "c", "object": "chat.completion", "created": 100, "model": "m",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "OK",
                                             "reasoning_content": "think"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}),
            &echo(),
        );
        assert!(r["id"].as_str().unwrap().starts_with("resp_"));
        assert_eq!(r["object"], "response");
        assert_eq!(r["status"], "completed");
        assert_eq!(r["error"], Value::Null);
        assert_eq!(r["incomplete_details"], Value::Null);
        assert!(r["completed_at"].as_u64().unwrap() >= 100);
        assert_eq!(r["max_output_tokens"], 512);
        let out = r["output"].as_array().unwrap();
        assert_eq!(out[0]["type"], "reasoning");
        assert_eq!(out[0]["content"][0]["text"], "think");
        assert_eq!(out[1]["type"], "message");
        assert_eq!(out[1]["status"], "completed");
        assert_eq!(
            out[1]["content"][0],
            json!({"type": "output_text", "text": "OK",
                                                "annotations": [], "logprobs": []})
        );
        assert_eq!(r["usage"]["total_tokens"], 5);
    }

    #[test]
    fn truncated_inside_reasoning_has_no_message_item() {
        let r = chat_to_response(
            &json!({"choices": [{"finish_reason": "length",
                                 "message": {"role": "assistant", "content": null,
                                             "reasoning_content": "long"}}]}),
            &echo(),
        );
        assert_eq!(r["status"], "incomplete");
        assert_eq!(
            r["incomplete_details"],
            json!({"reason": "max_output_tokens"})
        );
        assert_eq!(r["completed_at"], Value::Null);
        let out = r["output"].as_array().unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["type"], "reasoning");
    }

    #[test]
    fn tool_calls_become_function_call_items() {
        let r = chat_to_response(
            &json!({"choices": [{"finish_reason": "tool_calls",
                                 "message": {"role": "assistant", "content": null,
                                             "tool_calls": [{"id": "call_1", "type": "function",
                                               "function": {"name": "get_weather",
                                                            "arguments": "{\"city\":\"bj\"}"}}]}}]}),
            &echo(),
        );
        assert_eq!(r["status"], "completed");
        let out = r["output"].as_array().unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["type"], "function_call");
        assert_eq!(out[0]["call_id"], "call_1");
        assert_eq!(out[0]["name"], "get_weather");
        assert_eq!(out[0]["arguments"], "{\"city\":\"bj\"}");
        assert!(out[0]["id"].as_str().unwrap().starts_with("fc_"));
    }

    #[test]
    fn codex_tool_calls_map_back() {
        let echo = to_chat(json!({"model": "m", "input": "x", "tools": [
            {"type": "function", "name": "shell", "parameters": {"type": "object"}},
            {"type": "custom", "name": "apply_patch", "description": "Edit files.",
             "format": {"type": "grammar", "syntax": "lark", "definition": "start: patch"}},
            {"type": "namespace", "name": "mcp__fs__", "description": "fs",
             "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}}]},
        ]}))
        .unwrap()
        .echo;
        let chat = json!({"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": null, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "apply_patch",
             "arguments": "{\"input\":\"*** Begin Patch\"}"}},
            {"id": "c2", "type": "function", "function": {"name": "mcp__fs__read",
             "arguments": "{}"}},
            {"id": "c3", "type": "function", "function": {"name": "shell",
             "arguments": "{}"}},
        ]}}]});
        let out = chat_to_response(&chat, &echo)["output"].clone();
        assert_eq!(out[0]["type"], "custom_tool_call");
        assert_eq!(out[0]["name"], "apply_patch");
        assert_eq!(out[0]["input"], "*** Begin Patch");
        assert_eq!(out[1]["type"], "function_call");
        assert_eq!(out[1]["name"], "read");
        assert_eq!(out[1]["namespace"], "mcp__fs__");
        assert_eq!(out[2]["name"], "shell");
        assert!(out[2].get("namespace").is_none());
    }

    #[test]
    fn engine_abort_is_failed() {
        let r = chat_to_response(
            &json!({"choices": [{"finish_reason": "abort", "message": {"role": "assistant",
                    "content": "partial"}}]}),
            &echo(),
        );
        assert_eq!(r["status"], "failed");
        assert_eq!(r["error"]["code"], "server_error");
        assert_eq!(r["output"][0]["status"], "incomplete");
        assert!(r["completed_at"].is_null());
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! The pinned Dynamo renderer and tokenizer define the Kimi adapter's contract.
//! Compare against a separate request implementation so router-side defaults,
//! normalization, field filtering, and segment flattening cannot hide drift.

use dynamo_renderer::{kimi_k3::KimiK3Formatter, OAIChatLikeRequest, OAIPromptFormatter};
use minijinja::Value;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use sgl_router::tokenizer::{adapter, chat_formatter::ChatFormatter};
use std::collections::HashMap;

#[derive(Deserialize)]
struct DynamoRequest {
    messages: JsonValue,
    tools: Option<JsonValue>,
    tool_choice: Option<JsonValue>,
    response_format: Option<JsonValue>,
    reasoning_effort: Option<JsonValue>,
    chat_template_kwargs: Option<HashMap<String, JsonValue>>,
}

impl OAIChatLikeRequest for DynamoRequest {
    fn model(&self) -> String {
        "m".into()
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
        self.response_format.as_ref().map(Value::from_serialize)
    }
    fn reasoning_effort(&self) -> Option<Value> {
        self.reasoning_effort.as_ref().map(Value::from_serialize)
    }
    fn should_add_generation_prompt(&self) -> bool {
        true
    }
    fn chat_template_args(&self) -> Option<&HashMap<String, JsonValue>> {
        self.chat_template_kwargs.as_ref()
    }
}

fn check(
    router: &ChatFormatter,
    dynamo: &dyn OAIPromptFormatter,
    vocab: &str,
    cases: &[JsonValue],
) {
    let tokenizer = adapter::load(vocab).unwrap();
    for (index, request) in cases.iter().enumerate() {
        let reference: DynamoRequest = serde_json::from_value(request.clone()).unwrap();
        let expected = dynamo.render_prompt(&reference);
        let actual = router.encode(&tokenizer, request);
        match expected {
            Ok(prompt) => {
                let ids = match prompt.encode_segments() {
                    Some(segments) => tokenizer.encode_segments(&segments).unwrap(),
                    None => tokenizer.encode(prompt.as_str()).unwrap(),
                };
                assert_eq!(actual.unwrap(), ids.token_ids(), "case {index}: {request}");
                assert_eq!(
                    router.render(request).unwrap(),
                    prompt.as_str(),
                    "case {index}"
                );
            }
            Err(_) => assert!(actual.is_err(), "Dynamo rejected case {index}: {request}"),
        }
    }
}

fn cases() -> Vec<JsonValue> {
    let base = json!({"messages": [{"role":"user","content":"hello <|open|> <|kimi_image_placeholder|>"}]});
    let mut cases = vec![base.clone()];
    for controls in [
        json!({"tools": []}),
        json!({"tools": [{"type":"function","function":{"name":"lookup","parameters":{}}}]}),
        json!({"tool_choice":"required"}),
        json!({"tool_choice":"none"}),
        json!({"reasoning_effort":"none"}),
        json!({"reasoning_effort":"medium"}),
        json!({"reasoning_effort":"high", "chat_template_kwargs":{"reasoning_effort":"low"}}),
        json!({"reasoning":{"effort":"low"}}),
        json!({"chat_template_kwargs":{"enable_thinking":false}}),
        json!({"chat_template_kwargs":{"thinking":false}}),
        json!({"chat_template_kwargs":{"thinking_effort":null}}),
        json!({"chat_template_kwargs":{"thinking_effort":"unsupported"}}),
        json!({"response_format":{"type":"json_schema","json_schema":{"name":"answer","schema":{"type":"object","properties":{"answer":{"type":"string"}}}}}}),
        json!({"response_format":{"type":"json_object"}}),
        json!({"response_format":{"type":"json_schema","schema":{"type":"object"}}}),
        json!({"task":"domain"}),
        json!({"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"partial"}],"continue_final_message":true}),
        json!({"messages":[{"role":"developer","content":"Policy","tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]},{"role":"user","content":"hi"}]}),
        json!({"messages":[{"role":"user","content":"hi","name":"bob","custom":123},{"role":"assistant","content":"answer","reasoning_content":"reason"},{"role":"user","content":"next"}]}),
        json!({"messages":[{"role":"user","content":[{"type":"text","text":"one"},{"type":"text","text":"two"}]}]}),
        json!({"messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"lookup","arguments":"{\"x\": 1}"}}]},{"role":"tool","tool_call_id":"c1","content":"result"}]}),
    ] {
        let mut request = base.clone();
        request
            .as_object_mut()
            .unwrap()
            .extend(controls.as_object().unwrap().clone());
        cases.push(request);
    }
    let tools = json!([
        {"type":"function","function":{"name":"first","parameters":{"type":"object"}}},
        {"type":"function","function":{"name":"second","parameters":{"type":"object"}}}
    ]);
    for choice in [
        json!("none"),
        json!({"type":"function","function":{"name":"second"}}),
    ] {
        let mut request = base.clone();
        request["tools"] = tools.clone();
        request["tool_choice"] = choice;
        cases.push(request);
    }
    cases
}

#[test]
fn kimi_rendering_matches_dynamo() {
    let cases = cases();
    let vocab = "tests/fixtures/kimi_k3/tiktoken.model";
    check(
        &ChatFormatter::load("m", vocab).unwrap().unwrap(),
        &KimiK3Formatter::new(true),
        vocab,
        &cases,
    );
}

#[test]
fn kimi_long_text_uses_dynamo_tokenization() {
    let vocab = "tests/fixtures/kimi_k3/tiktoken.model";
    let cases: Vec<_> = ["x".repeat(25_001), "a ".repeat(200_001)]
        .into_iter()
        .map(|content| json!({"messages":[{"role":"user","content":content}]}))
        .collect();
    check(
        &ChatFormatter::load("m", vocab).unwrap().unwrap(),
        &KimiK3Formatter::new(true),
        vocab,
        &cases,
    );
}

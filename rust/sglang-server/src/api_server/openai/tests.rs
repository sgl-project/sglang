//! Focused unit tests for the OpenAI endpoint adapters.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use dynamo_parsers::{ToolChoice as DynamoToolChoice, ToolDefinition};
use dynamo_protocols::types::responses::{CreateResponse, Status};
use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionToolChoiceOption, Choice,
    CreateChatCompletionRequest, CreateCompletionRequest, CreateCompletionResponse, Prompt,
};
use futures::StreamExt;
use serde_json::json;
use tokio::sync::mpsc;
use tower::util::ServiceExt;

use super::super::AppState;
use super::super::guard::AbortGuard;
use super::chat::{
    SamplingDefaults, chat_event_stream, chat_logprobs, chat_sampling_params, unary_chat,
};
use super::completions::{
    ChoiceExtensions, SubmittedChoice, completion_event_stream, completion_logprobs,
    completion_response_value, unary_completion,
};
use super::new_response_store;
use super::response_stream::{response_object, responses_event_stream};
use super::responses::{responses_chat_request, sse_frame, unary_responses};
use super::tools::{apply_tool_constraint, parse_chat_tool_calls};
use super::{StoredResponse, routes, unix_seconds};
use crate::ids::Rid;
use crate::message::{ChunkEvent, ChunkExtras, EgressItem, SamplingParams};
use crate::runtime::ServerArgs;
use crate::tokenizer_manager::{AbortSource, Senders};

fn senders() -> Senders {
    Senders {
        tm: flume::unbounded().0,
        abort: flume::unbounded().0,
        tok: flume::unbounded().0,
        detok: vec![],
    }
}

fn chunk(rid: &str, text: &str, done: bool) -> EgressItem {
    let output = ChunkEvent {
        rid: rid.into(),
        text: text.into(),
        token_ids: vec![1],
        prompt_tokens: 5,
        completion_tokens: 1,
        finish_reason: done.then(|| {
            serde_json::from_value(serde_json::json!({
                "type": "stop",
                "matched": "</s>"
            }))
            .unwrap()
        }),
        ..Default::default()
    };
    if done {
        EgressItem::Done(output)
    } else {
        EgressItem::Frame(output)
    }
}

fn submitted(
    index: usize,
    prompt_index: usize,
    rid: &str,
) -> (SubmittedChoice, mpsc::Sender<EgressItem>) {
    let (tx, rx) = mpsc::channel(8);
    (
        SubmittedChoice {
            index,
            prompt_index,
            rid: rid.into(),
            echo: String::new(),
            rx,
        },
        tx,
    )
}

fn chat_submitted(
    index: usize,
    rid: &str,
) -> (
    (usize, Rid, mpsc::Receiver<EgressItem>),
    mpsc::Sender<EgressItem>,
) {
    let (tx, rx) = mpsc::channel(8);
    ((index, rid.into(), rx), tx)
}

#[test]
fn dynamo_completion_request_deserializes_directly() {
    let request: CreateCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "m",
        "prompt": ["a", "b"],
        "max_tokens": 8,
        "n": 2,
        "stream_options": {
            "include_usage": true,
            "continuous_usage_stats": true
        }
    }))
    .unwrap();
    assert!(matches!(request.prompt, Prompt::StringArray(_)));
    assert_eq!(request.n, Some(2));
    assert!(request.stream_options.unwrap().continuous_usage_stats);
}

#[test]
fn max_tokens_zero_is_rejected_before_submission() {
    let request: CreateCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "m",
        "prompt": "hello",
        "max_tokens": 0
    }))
    .unwrap();
    assert_eq!(request.max_tokens, Some(0));
}

#[test]
fn zero_top_logprobs_keeps_selected_token_and_empty_top_map() {
    let extras = ChunkExtras {
        out_lp_val: vec![-0.25],
        out_lp_idx: vec![7],
        out_lp_txt: vec!["x".into()],
        out_top_lens: vec![0],
        ..Default::default()
    };
    let logprobs = completion_logprobs(Some(&extras), false);
    assert_eq!(logprobs.tokens, ["x"]);
    assert_eq!(logprobs.token_logprobs, [Some(-0.25)]);
    assert_eq!(logprobs.top_logprobs, [serde_json::Value::Null]);

    let value = completion_response_value(
        CreateCompletionResponse {
            id: "cmpl-test".into(),
            choices: vec![Choice {
                text: "x".into(),
                index: 0,
                logprobs: Some(logprobs),
                finish_reason: None,
            }],
            created: 1,
            model: "model".into(),
            system_fingerprint: None,
            object: "text_completion".into(),
            usage: None,
        },
        &[ChoiceExtensions::default()],
    );
    assert_eq!(
        value["choices"][0]["logprobs"]["text_offset"],
        serde_json::json!([-1])
    );
}

#[test]
fn chat_logprobs_use_dynamo_wire_types() {
    let extras = ChunkExtras {
        out_lp_val: vec![-0.25],
        out_lp_idx: vec![7],
        out_lp_txt: vec!["x".into()],
        out_top_val: vec![-0.25, -1.0],
        out_top_idx: vec![7, 8],
        out_top_lens: vec![2],
        out_top_txt: vec!["x".into(), "y".into()],
        ..Default::default()
    };
    let logprobs = chat_logprobs(Some(&extras));
    let token = &logprobs.content.unwrap()[0];
    assert_eq!(token.token, "x");
    assert_eq!(token.top_logprobs.len(), 2);
    assert_eq!(token.top_logprobs[1].token, "y");
}

#[tokio::test]
async fn unary_fold_orders_choices_and_counts_each_prompt_once() {
    let (choice0, tx0) = submitted(0, 0, "r0");
    let (choice1, tx1) = submitted(1, 0, "r1");
    tx0.send(chunk("r0", "a", false)).await.unwrap();
    tx0.send(chunk("r0", "b", true)).await.unwrap();
    tx1.send(chunk("r1", "x", false)).await.unwrap();
    tx1.send(chunk("r1", "y", true)).await.unwrap();

    let response = unary_completion(
        vec![choice0, choice1],
        AbortGuard::new_empty(senders()),
        "cmpl-test".into(),
        "model".into(),
        1,
        false,
        false,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["choices"][0]["text"], "ab");
    assert_eq!(value["choices"][1]["text"], "xy");
    assert_eq!(value["choices"][0]["matched_stop"], "</s>");
    assert_eq!(value["usage"]["prompt_tokens"], 5);
    assert_eq!(value["usage"]["completion_tokens"], 4);
}

#[tokio::test]
async fn stream_uses_deltas_then_usage_and_done() {
    let (choice, tx) = submitted(0, 0, "r0");
    tx.send(chunk("r0", "a", false)).await.unwrap();
    tx.send(chunk("r0", "b", true)).await.unwrap();

    let stream = completion_event_stream(
        vec![choice],
        AbortGuard::new_empty(senders()),
        "cmpl-test".into(),
        "model".into(),
        1,
        false,
        false,
        true,
        false,
    );
    futures::pin_mut!(stream);
    let frames: Vec<String> = stream.collect().await;
    assert_eq!(frames.len(), 4);
    let first: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
    let terminal: serde_json::Value = serde_json::from_str(&frames[1]).unwrap();
    let usage: serde_json::Value = serde_json::from_str(&frames[2]).unwrap();
    assert_eq!(first["choices"][0]["text"], "a");
    assert_eq!(terminal["choices"][0]["text"], "b");
    assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
    assert!(usage["choices"].as_array().unwrap().is_empty());
    assert_eq!(usage["usage"]["prompt_tokens"], 5);
    assert_eq!(usage["usage"]["completion_tokens"], 2);
    assert_eq!(frames[3], "[DONE]");
}

#[tokio::test]
async fn unary_chat_fans_in_choices_and_usage() {
    let (choice0, tx0) = chat_submitted(0, "r0");
    let (choice1, tx1) = chat_submitted(1, "r1");
    tx0.send(chunk("r0", "Paris", true)).await.unwrap();
    tx1.send(chunk("r1", "Paris", true)).await.unwrap();

    let response = unary_chat(
        vec![choice0, choice1],
        AbortGuard::new_empty(senders()),
        "chatcmpl-test".into(),
        "model".into(),
        1,
        false,
        None,
        None,
        None,
        true,
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["choices"][0]["message"]["role"], "assistant");
    assert_eq!(value["choices"][0]["message"]["content"], "Paris");
    assert_eq!(value["choices"][1]["index"], 1);
    assert_eq!(value["usage"]["prompt_tokens"], 5);
    assert_eq!(value["usage"]["completion_tokens"], 2);
}

#[tokio::test]
async fn unary_chat_separates_reasoning_content_with_parser_configured() {
    let (choice, tx) = chat_submitted(0, "r0");
    tx.send(chunk(
        "r0",
        "<think>because Paris is famous</think>Paris",
        true,
    ))
    .await
    .unwrap();

    let response = unary_chat(
        vec![choice],
        AbortGuard::new_empty(senders()),
        "chatcmpl-test".into(),
        "model".into(),
        1,
        false,
        None,
        Some("deepseek-r1".into()),
        None,
        true,
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        value["choices"][0]["message"]["reasoning_content"],
        "because Paris is famous"
    );
    assert_eq!(value["choices"][0]["message"]["content"], "Paris");
    assert!(value["choices"][0]["message"]["reasoning_content"].is_string());
}

#[tokio::test]
async fn streaming_chat_separates_reasoning_into_own_deltas() {
    let (choice, tx) = chat_submitted(0, "r0");
    // Force mode starts in reasoning, so the opener is stripped and the first
    // reasoning fragment streams immediately.
    tx.send(chunk("r0", "<think>be", false)).await.unwrap();
    tx.send(chunk("r0", "cause</think>Par", false))
        .await
        .unwrap();
    tx.send(chunk("r0", "is", true)).await.unwrap();

    let stream = chat_event_stream(
        vec![choice],
        AbortGuard::new_empty(senders()),
        "chatcmpl-test".into(),
        "model".into(),
        1,
        false,
        true,
        None,
        Some("deepseek-r1".into()),
        None,
        None,
        false,
        true,
        None,
    );
    futures::pin_mut!(stream);
    let frames: Vec<String> = stream.collect().await;
    let role: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
    let first_reasoning: serde_json::Value = serde_json::from_str(&frames[1]).unwrap();
    let second_reasoning: serde_json::Value = serde_json::from_str(&frames[2]).unwrap();
    let content: serde_json::Value = serde_json::from_str(&frames[3]).unwrap();
    let terminal: serde_json::Value = serde_json::from_str(&frames[4]).unwrap();
    assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(
        first_reasoning["choices"][0]["delta"]["reasoning_content"],
        "be"
    );
    assert!(first_reasoning["choices"][0]["delta"]["content"].is_null());
    assert_eq!(
        second_reasoning["choices"][0]["delta"]["reasoning_content"],
        "cause"
    );
    assert_eq!(content["choices"][0]["delta"]["content"], "Par");
    assert!(content["choices"][0]["delta"]["reasoning_content"].is_null());
    assert_eq!(terminal["choices"][0]["delta"]["content"], "is");
    assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
    assert_eq!(frames.len(), 7);
}

#[tokio::test]
async fn streaming_chat_emits_role_deltas_usage_and_done() {
    let (choice, tx) = chat_submitted(0, "r0");
    tx.send(chunk("r0", "Par", false)).await.unwrap();
    tx.send(chunk("r0", "is", true)).await.unwrap();

    let stream = chat_event_stream(
        vec![choice],
        AbortGuard::new_empty(senders()),
        "chatcmpl-test".into(),
        "model".into(),
        1,
        false,
        true,
        None,
        None,
        None,
        None,
        false,
        true,
        None,
    );
    futures::pin_mut!(stream);
    let frames: Vec<String> = stream.collect().await;
    assert_eq!(frames.len(), 5);
    let role: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
    let delta: serde_json::Value = serde_json::from_str(&frames[1]).unwrap();
    let terminal: serde_json::Value = serde_json::from_str(&frames[2]).unwrap();
    let usage: serde_json::Value = serde_json::from_str(&frames[3]).unwrap();
    assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
    assert!(role["choices"][0]["delta"]["reasoning_content"].is_null());
    assert_eq!(delta["choices"][0]["delta"]["content"], "Par");
    assert!(delta["choices"][0]["delta"]["reasoning_content"].is_null());
    assert_eq!(terminal["choices"][0]["delta"]["content"], "is");
    assert!(terminal["choices"][0]["delta"]["reasoning_content"].is_null());
    assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
    assert_eq!(usage["usage"]["completion_tokens"], 2);
    assert_eq!(frames[4], "[DONE]");
}

#[tokio::test]
async fn streaming_chat_buffers_and_parses_function_calls() {
    let (choice, tx) = chat_submitted(0, "r0");
    tx.send(chunk(
        "r0",
        r#"<|python_tag|>{"name":"get_weather","parameters":{"location":"Paris"}}"#,
        true,
    ))
    .await
    .unwrap();

    let stream = chat_event_stream(
        vec![choice],
        AbortGuard::new_empty(senders()),
        "chatcmpl-test".into(),
        "model".into(),
        1,
        false,
        false,
        Some("llama3".into()),
        None,
        None,
        None,
        false,
        true,
        None,
    );
    futures::pin_mut!(stream);
    let frames: Vec<String> = stream.collect().await;
    assert_eq!(frames.len(), 3);
    let terminal: serde_json::Value = serde_json::from_str(&frames[1]).unwrap();
    assert!(terminal["choices"][0]["delta"]["reasoning_content"].is_null());
    assert_eq!(terminal["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        terminal["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    assert_eq!(frames[2], "[DONE]");
}

#[tokio::test]
async fn streaming_chat_with_tools_does_not_buffer_normal_text() {
    let (choice, tx) = chat_submitted(0, "r0");
    tx.send(chunk("r0", "Par", false)).await.unwrap();
    tx.send(chunk("r0", "is", true)).await.unwrap();

    let stream = chat_event_stream(
        vec![choice],
        AbortGuard::new_empty(senders()),
        "chatcmpl-test".into(),
        "model".into(),
        1,
        false,
        false,
        Some("llama3".into()),
        None,
        None,
        None,
        false,
        true,
        None,
    );
    futures::pin_mut!(stream);
    let frames: Vec<String> = stream.collect().await;
    let deltas = frames
        .iter()
        .filter_map(|frame| serde_json::from_str::<serde_json::Value>(frame).ok())
        .filter_map(|frame| {
            frame["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect::<Vec<_>>();
    assert_eq!(deltas, ["Par", "is"]);
}

#[tokio::test]
async fn streaming_chat_holds_only_a_split_tool_marker() {
    let (choice, tx) = chat_submitted(0, "r0");
    tx.send(chunk("r0", "Before <|python_", false))
        .await
        .unwrap();
    tx.send(chunk(
        "r0",
        r#"tag|>{"name":"get_weather","parameters":{"city":"Paris"}}"#,
        true,
    ))
    .await
    .unwrap();

    let stream = chat_event_stream(
        vec![choice],
        AbortGuard::new_empty(senders()),
        "chatcmpl-test".into(),
        "model".into(),
        1,
        false,
        false,
        Some("llama3".into()),
        None,
        None,
        None,
        false,
        true,
        None,
    );
    futures::pin_mut!(stream);
    let frames: Vec<String> = stream.collect().await;
    let values = frames
        .iter()
        .filter_map(|frame| serde_json::from_str::<serde_json::Value>(frame).ok())
        .collect::<Vec<_>>();
    assert!(
        values
            .iter()
            .any(|frame| { frame["choices"][0]["delta"]["content"].as_str() == Some("Before ") })
    );
    assert!(values.iter().any(|frame| {
        frame["choices"][0]["delta"]["tool_calls"][0]["function"]["name"].as_str()
            == Some("get_weather")
    }));
}

#[tokio::test]
async fn streaming_chat_releases_an_incomplete_marker_at_done() {
    let (choice, tx) = chat_submitted(0, "r0");
    tx.send(chunk("r0", "Before <|python_", false))
        .await
        .unwrap();
    tx.send(chunk("r0", "", true)).await.unwrap();

    let stream = chat_event_stream(
        vec![choice],
        AbortGuard::new_empty(senders()),
        "chatcmpl-test".into(),
        "model".into(),
        1,
        false,
        false,
        Some("llama3".into()),
        None,
        None,
        None,
        false,
        true,
        None,
    );
    futures::pin_mut!(stream);
    let frames: Vec<String> = stream.collect().await;
    let text = frames
        .iter()
        .filter_map(|frame| serde_json::from_str::<serde_json::Value>(frame).ok())
        .filter_map(|frame| {
            frame["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect::<String>();
    assert_eq!(text, "Before <|python_");
}

#[tokio::test]
async fn streaming_chat_emits_a_complete_tool_call_before_done() {
    let (choice, tx) = chat_submitted(0, "r0");
    tx.send(chunk(
        "r0",
        r#"<|python_tag|>{"name":"get_weather","parameters":{"city":"Paris"}}"#,
        false,
    ))
    .await
    .unwrap();
    tx.send(chunk("r0", "", true)).await.unwrap();

    let stream = chat_event_stream(
        vec![choice],
        AbortGuard::new_empty(senders()),
        "chatcmpl-test".into(),
        "model".into(),
        1,
        false,
        false,
        Some("llama3".into()),
        None,
        None,
        None,
        false,
        true,
        None,
    );
    futures::pin_mut!(stream);
    let frames: Vec<String> = stream.collect().await;
    let tool_position = frames
        .iter()
        .position(|frame| frame.contains("\"tool_calls\":[{"))
        .expect("tool call chunk");
    let terminal_position = frames
        .iter()
        .position(|frame| frame.contains("\"finish_reason\":\"tool_calls\""))
        .expect("terminal chunk");
    assert!(tool_position < terminal_position);
}

#[test]
fn required_tool_choice_builds_python_compatible_constraint() {
    let tools = vec![ToolDefinition {
        name: "get_weather".into(),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        })),
        strict: Some(true),
    }];
    let mut sampling = SamplingParams::default();
    apply_tool_constraint(
        &mut sampling,
        "llama3",
        &DynamoToolChoice::Required,
        &tools,
        Some(false),
    )
    .unwrap();
    let schema: serde_json::Value =
        serde_json::from_str(sampling.json_schema.as_deref().unwrap()).unwrap();
    assert_eq!(schema["type"], "array");
    assert_eq!(schema["minItems"], 1);
    assert_eq!(schema["maxItems"], 1);
    assert_eq!(
        schema["items"]["properties"]["name"]["enum"][0],
        "get_weather"
    );
}

#[test]
fn chat_without_a_token_limit_stays_unbounded() {
    let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap();
    assert_eq!(
        chat_sampling_params(&request, &SamplingDefaults::CHAT)
            .unwrap()
            .max_new_tokens,
        None
    );
}

#[test]
fn strict_auto_llama_tool_uses_python_compatible_constraint() {
    let tools = vec![ToolDefinition {
        name: "get_weather".into(),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        })),
        strict: Some(true),
    }];
    let mut sampling = SamplingParams::default();
    apply_tool_constraint(
        &mut sampling,
        "llama3",
        &DynamoToolChoice::Auto,
        &tools,
        None,
    )
    .unwrap();
    let schema: serde_json::Value =
        serde_json::from_str(sampling.structural_tag.as_deref().unwrap()).unwrap();
    assert_eq!(schema["type"], "structural_tag");
    assert_eq!(schema["format"]["type"], "triggered_tags");
    assert_eq!(schema["format"]["at_least_one"], false);
    assert_eq!(
        schema["format"]["tags"][0]["content"]["json_schema"]["required"][0],
        "city"
    );
}

#[tokio::test]
async fn canonical_qwen_parser_name_uses_dynamo_qwen25() {
    let (content, calls) = parse_chat_tool_calls(
        r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#.into(),
        Some("qwen"),
        None,
        true,
    )
    .await;
    assert!(content.is_empty());
    assert_eq!(calls.unwrap()[0].function.name, "get_weather");
}

#[test]
fn structured_responses_input_reuses_chat_history_and_function_tools() {
    let request: CreateResponse = serde_json::from_value(serde_json::json!({
        "model": "model",
        "input": [
            {"role": "user", "content": "What is the weather?"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{\"city\":\"Paris\"}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "sunny"
            }
        ],
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "parameters": {"type": "object"}
        }],
        "tool_choice": "required"
    }))
    .unwrap();
    let chat = responses_chat_request(&request, "model").unwrap();
    assert_eq!(chat.messages.len(), 3);
    assert!(matches!(
        chat.messages[0],
        ChatCompletionRequestMessage::User(_)
    ));
    assert!(matches!(
        chat.messages[1],
        ChatCompletionRequestMessage::Assistant(_)
    ));
    assert!(matches!(
        chat.messages[2],
        ChatCompletionRequestMessage::Tool(_)
    ));
    assert_eq!(chat.tools.unwrap()[0].function.name, "get_weather");
    assert!(matches!(
        chat.tool_choice,
        Some(ChatCompletionToolChoiceOption::Required)
    ));
}

fn response_request(stream: bool) -> CreateResponse {
    serde_json::from_value(serde_json::json!({
        "model": "model",
        "input": "The capital of France is",
        "stream": stream,
        "temperature": 0.0,
        "max_output_tokens": 8
    }))
    .unwrap()
}

#[tokio::test]
async fn unary_responses_uses_standard_output_items() {
    let (tx, rx) = mpsc::channel(8);
    tx.send(chunk("r0", "Paris", true)).await.unwrap();
    let response = unary_responses(
        rx,
        AbortGuard::new_empty(senders()),
        "r0".into(),
        "resp_test".into(),
        1_234_567_890,
        "model".into(),
        response_request(false),
        None,
        None,
        new_response_store(),
        vec![],
    )
    .await;
    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["object"], "response");
    assert_eq!(value["created_at"], 1_234_567_890);
    assert_eq!(value["status"], "completed");
    assert_eq!(value["output"][0]["type"], "message");
    assert_eq!(value["output"][0]["content"][0]["text"], "Paris");
    assert_eq!(value["usage"]["input_tokens"], 5);
    assert_eq!(value["usage"]["output_tokens"], 1);
    assert_eq!(value["usage"]["prompt_tokens"], 5);
    assert_eq!(value["usage"]["completion_tokens"], 1);
}

#[tokio::test]
async fn streaming_responses_emits_lifecycle_and_text_deltas() {
    let (tx, rx) = mpsc::channel(8);
    tx.send(chunk("r0", "Par", false)).await.unwrap();
    tx.send(chunk("r0", "is", true)).await.unwrap();
    let stream = responses_event_stream(
        rx,
        AbortGuard::new_empty(senders()),
        "r0".into(),
        "resp_test".into(),
        1_234_567_890,
        "model".into(),
        response_request(true),
        None,
        None,
        None,
        false,
        new_response_store(),
        vec![],
    );
    futures::pin_mut!(stream);
    let frames: Vec<String> = stream.collect().await;
    let event_types = frames[..frames.len() - 1]
        .iter()
        .map(|frame| {
            serde_json::from_str::<serde_json::Value>(frame).unwrap()["type"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect::<Vec<_>>();
    assert_eq!(event_types[0], "response.created");
    assert_eq!(event_types[1], "response.in_progress");
    assert_eq!(
        event_types
            .iter()
            .filter(|event| *event == "response.output_text.delta")
            .count(),
        2
    );
    assert_eq!(event_types.last().unwrap(), "response.completed");
    let created: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
    let completed: serde_json::Value = serde_json::from_str(&frames[frames.len() - 2]).unwrap();
    assert_eq!(created["response"]["created_at"], 1_234_567_890);
    assert_eq!(completed["response"]["created_at"], 1_234_567_890);
    assert_eq!(completed["response"]["usage"]["output_tokens"], 2);
    assert_eq!(frames.last().unwrap(), "[DONE]");
}

/// Python `_send_event` frames each event as `event: {type}\ndata: {payload}`.
/// Assert the actual SSE wire bytes: lifecycle frames carry their event name;
/// `[DONE]` and error frames stay data-only.
#[tokio::test]
async fn responses_sse_frames_carry_event_names() {
    use axum::response::IntoResponse;
    use std::convert::Infallible;

    let payload = r#"{"type":"response.created","sequence_number":0,"response":{}}"#.to_string();
    let stream = futures::stream::iter(vec![
        Ok::<_, Infallible>(sse_frame(payload)),
        Ok::<_, Infallible>(sse_frame("[DONE]".into())),
        Ok::<_, Infallible>(sse_frame(r#"{"error":{"message":"boom"}}"#.into())),
    ]);
    let response = axum::response::Sse::new(stream).into_response();
    let body = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    let mut frames = text.split("\n\n");
    let first = frames.next().unwrap();
    assert!(
        first.contains("event: response.created"),
        "missing event name in {text:?}"
    );
    assert!(
        first.contains("response.created"),
        "missing payload in {text:?}"
    );
    assert!(!frames.next().unwrap().contains("event:"));
    assert!(!frames.next().unwrap().contains("event:"));
}

#[tokio::test]
async fn streaming_responses_with_tools_emits_normal_text_deltas() {
    let (tx, rx) = mpsc::channel(8);
    tx.send(chunk("r0", "Par", false)).await.unwrap();
    tx.send(chunk("r0", "is", true)).await.unwrap();
    let stream = responses_event_stream(
        rx,
        AbortGuard::new_empty(senders()),
        "r0".into(),
        "resp_test".into(),
        1_234_567_890,
        "model".into(),
        response_request(true),
        Some("llama3".into()),
        None,
        None,
        false,
        new_response_store(),
        vec![],
    );
    futures::pin_mut!(stream);
    let frames: Vec<String> = stream.collect().await;
    let deltas = frames
        .iter()
        .filter_map(|frame| serde_json::from_str::<serde_json::Value>(frame).ok())
        .filter(|frame| frame["type"] == "response.output_text.delta")
        .filter_map(|frame| frame["delta"].as_str().map(str::to_owned))
        .collect::<Vec<_>>();
    assert_eq!(deltas, ["Par", "is"]);
}

#[tokio::test]
async fn streaming_responses_emits_function_call_events() {
    let (tx, rx) = mpsc::channel(8);
    tx.send(chunk(
        "r0",
        r#"<|python_tag|>{"name":"get_weather","parameters":{"city":"Paris"}}"#,
        true,
    ))
    .await
    .unwrap();
    let stream = responses_event_stream(
        rx,
        AbortGuard::new_empty(senders()),
        "r0".into(),
        "resp_test".into(),
        1_234_567_890,
        "model".into(),
        response_request(true),
        Some("llama3".into()),
        None,
        None,
        false,
        new_response_store(),
        vec![],
    );
    futures::pin_mut!(stream);
    let frames = stream.collect::<Vec<_>>().await;
    let event_types = frames
        .iter()
        .filter_map(|frame| serde_json::from_str::<serde_json::Value>(frame).ok())
        .filter_map(|event| event["type"].as_str().map(str::to_owned))
        .collect::<Vec<_>>();
    assert!(event_types.contains(&"response.output_item.added".into()));
    assert!(event_types.contains(&"response.function_call_arguments.delta".into()));
    assert!(event_types.contains(&"response.function_call_arguments.done".into()));
    assert!(event_types.contains(&"response.output_item.done".into()));
    assert!(event_types.contains(&"response.completed".into()));
}

// ---------------------------------------------------------------------
// Handler-level tests: full router, real extractors, no scheduler. A
// request that reaches `submit` with an OPEN tm lane would wait on the
// egress receiver forever, so submission-reaching cases use `senders_closed`
// (503) and everything else fails validation before submit.
// ---------------------------------------------------------------------

fn server_args() -> Arc<ServerArgs> {
    Arc::new(
        serde_json::from_value(serde_json::json!({ "served_model_name": "model" }))
            .expect("ServerArgs must deserialize"),
    )
}

fn app_state(senders: Senders) -> AppState {
    AppState {
        senders,
        egress_buf: 8,
        server_args: server_args(),
        tokenizer: None,
        chat_formatter: None,
        chat_tokenizer: None,
        response_store: new_response_store(),
        egress_activity: Default::default(),
    }
}

fn senders_closed() -> Senders {
    // Dropping the receivers disconnects the channels; the senders stay
    // valid (moveable) but every send reports `Err`, the shutdown state
    // `submit` surfaces as a 503.
    let (tm_tx, tm_rx) = flume::unbounded();
    drop(tm_rx);
    let (abort_tx, abort_rx) = flume::unbounded();
    drop(abort_rx);
    let (tok_tx, tok_rx) = flume::unbounded();
    drop(tok_rx);
    Senders {
        tm: tm_tx,
        abort: abort_tx,
        tok: tok_tx,
        detok: vec![],
    }
}

fn senders_with_abort_rx() -> (Senders, flume::Receiver<AbortSource>) {
    let (tm_tx, _tm_rx) = flume::unbounded();
    let (abort_tx, abort_rx) = flume::unbounded();
    let (tok_tx, _tok_rx) = flume::unbounded();
    (
        Senders {
            tm: tm_tx,
            abort: abort_tx,
            tok: tok_tx,
            detok: vec![],
        },
        abort_rx,
    )
}

fn request(method: &str, path: &str) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(path)
        .body(Body::empty())
        .unwrap()
}

/// Serve one request through the full router (extractors, auth, routing).
/// `with_state` consumes the state into a `Router<()>`, which is what
/// implements `tower::Service`.
async fn oneshot(app: Router<()>, req: Request<Body>) -> Response {
    app.oneshot(req).await.unwrap()
}

async fn post_json(app: Router<()>, path: &str, body: serde_json::Value) -> Response {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    oneshot(app, req).await
}

async fn body_json(response: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// The common StatusCode→error helper follows `pre_submit_error`'s shape:
/// unary requests get the JSON error with its status; a committed stream gets
/// 200 + one SSE error frame + `[DONE]`, and the frame carries the OpenAI
/// error fields (`type`, `param`, `code`) that the SDKs dispatch on.
#[tokio::test]
async fn openai_error_response_covers_unary_and_sse() {
    let unary = super::openai_error_response(StatusCode::BAD_REQUEST, "bad input", false);
    assert_eq!(unary.status(), StatusCode::BAD_REQUEST);
    let value = body_json(unary).await;
    assert_eq!(value["error"]["message"], "bad input");
    assert_eq!(value["error"]["type"], "BadRequestError");
    assert_eq!(value["error"]["code"], 400);
    assert!(value["error"]["param"].is_null());

    let streamed = super::openai_error_response(StatusCode::BAD_REQUEST, "bad input", true);
    assert_eq!(streamed.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(streamed.into_body(), 64 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let frame = text
        .split("\n\n")
        .next()
        .unwrap()
        .strip_prefix("data: ")
        .unwrap();
    let frame: serde_json::Value = serde_json::from_str(frame).unwrap();
    assert_eq!(frame["error"]["message"], "bad input");
    assert_eq!(frame["error"]["type"], "BadRequestError");
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn completions_handler_validates_before_submit() {
    let app = routes().with_state(app_state(senders()));
    let cases = [
        (json!({"model": "other", "prompt": "hi"}), "unknown model"),
        (json!({"model": "model", "prompt": "hi", "n": 0}), "n=0"),
        (
            json!({"model": "model", "prompt": "hi", "max_tokens": 0}),
            "max_tokens=0",
        ),
        (json!({"model": "model", "prompt": ""}), "empty prompt"),
        (
            json!({"model": "model", "prompt": "hi", "best_of": 2}),
            "best_of>1",
        ),
        (
            json!({"model": "model", "prompt": "hi", "suffix": "x"}),
            "suffix",
        ),
        (
            json!({"model": "model", "prompt": "hi", "prompt_embeds": [[1.0]]}),
            "prompt_embeds",
        ),
    ];
    for (body, label) in cases {
        let response = post_json(app.clone(), "/v1/completions", body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{label}");
    }
    // Malformed JSON → 400 (JsonRejection path).
    let req = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from("not json"))
        .unwrap();
    let response = oneshot(app.clone(), req).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    // A closed tm inbox (shutdown) surfaces as 503.
    let app = routes().with_state(app_state(senders_closed()));
    let response = post_json(
        app.clone(),
        "/v1/completions",
        json!({"model": "model", "prompt": "hi"}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn chat_handler_validates_before_submit() {
    let app = routes().with_state(app_state(senders()));
    let cases = [
        (
            json!({"model": "other", "messages": [{"role": "user", "content": "hi"}]}),
            "unknown model",
        ),
        (json!({"model": "model", "messages": []}), "empty messages"),
        (
            json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "n": 0}),
            "n=0",
        ),
        (
            json!({"model": "model", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://example.com/x.png"}}]}]}),
            "media content",
        ),
        (
            json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "function_call": "auto"}),
            "deprecated function_call",
        ),
        (
            json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "audio": {"input_audio": {"data": "x", "format": "wav"}}}),
            "audio",
        ),
        (
            json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 0}),
            "max_completion_tokens=0",
        ),
    ];
    for (body, label) in cases {
        let response = post_json(app.clone(), "/v1/chat/completions", body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{label}");
    }
    // A valid request with no loaded chat template → 400 (template gate).
    let response = post_json(
        app.clone(),
        "/v1/chat/completions",
        json!({"model": "model", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn responses_handler_validates_before_submit() {
    let app = routes().with_state(app_state(senders()));
    let cases = [
        (json!({"model": "other", "input": "hi"}), "unknown model"),
        (
            json!({"input": "hi", "max_output_tokens": 0}),
            "max_output_tokens=0",
        ),
        (json!({"input": "hi", "conversation": {}}), "conversation"),
        (json!({"input": "hi", "prompt": "x"}), "prompt template"),
        (json!({"input": "hi", "include": ["reasoning"]}), "include"),
        (
            json!({"input": "hi", "max_tool_calls": 3}),
            "max_tool_calls",
        ),
        (
            json!({"input": "hi", "truncation": "auto"}),
            "truncation auto",
        ),
        (
            json!({"input": "hi", "reasoning": {"summary": "x"}}),
            "reasoning summary",
        ),
        (
            json!({"input": "hi", "previous_response_id": "nope"}),
            "bad previous_response_id",
        ),
        (
            json!({"input": [{"type": "item_reference", "item_id": "x"}]}),
            "item reference",
        ),
        (json!({"input": []}), "empty input"),
        (
            json!({"input": "hi", "background": true, "stream": true}),
            "background+stream",
        ),
    ];
    for (body, label) in cases {
        let response = post_json(app.clone(), "/v1/responses", body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{label}");
    }
    // Unknown previous_response_id → 404 (store lookup).
    let response = post_json(
        app.clone(),
        "/v1/responses",
        json!({"input": "hi", "previous_response_id": "resp_missing"}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    // A valid request without a chat template → 400 (template gate).
    let response = post_json(app.clone(), "/v1/responses", json!({"input": "hi"})).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

/// A closed tm inbox with a *streaming* request must answer inside the
/// committed stream: 200 + one OpenAI-shaped SSE error frame + `[DONE]` (the
/// same rule `pre_submit_error` applies to the native API), not a unary 503.
#[tokio::test]
async fn streaming_submit_failure_answers_inside_the_stream() {
    let app = routes().with_state(app_state(senders_closed()));
    let response = post_json(
        app,
        "/v1/completions",
        json!({"model": "model", "prompt": "hi", "stream": true}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let frame = text
        .split("\n\n")
        .next()
        .unwrap()
        .strip_prefix("data: ")
        .unwrap();
    let frame: serde_json::Value = serde_json::from_str(frame).unwrap();
    assert_eq!(frame["error"]["message"], "service unavailable");
    assert_eq!(frame["error"]["type"], "InternalServerError");
    assert_eq!(frame["error"]["code"], 503);
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn response_retrieve_and_cancel_lifecycle() {
    let (senders, abort_rx) = senders_with_abort_rx();
    let state = app_state(senders);
    let app = routes().with_state(state.clone());

    // Unknown / malformed ids.
    let response = oneshot(app.clone(), request("GET", "/v1/responses/resp_missing")).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let response = oneshot(app.clone(), request("GET", "/v1/responses/nope")).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    // Seed an in-progress response directly: a background request would
    // need a real scheduler behind the tm lane. `Rid::from_client` mints a
    // fresh uniquified rid per call, so capture the one the store holds.
    let seeded_rid = Rid::from_client("resp_seeded");
    state.response_store.write().await.insert(
        "resp_seeded".into(),
        StoredResponse {
            response: response_object(
                "resp_seeded",
                "model",
                &response_request(false),
                unix_seconds(),
                Status::InProgress,
                vec![],
                None,
            ),
            messages: vec![],
            rid: Some(seeded_rid.clone()),
        },
    );

    // Retrieve returns the stored object.
    let response = oneshot(app.clone(), request("GET", "/v1/responses/resp_seeded")).await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = body_json(response).await;
    assert_eq!(value["object"], "response");
    assert_eq!(value["status"], "in_progress");

    // Cancel → Cancelled, and the abort reaches the scheduler lane.
    let response = oneshot(
        app.clone(),
        request("POST", "/v1/responses/resp_seeded/cancel"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = body_json(response).await;
    assert_eq!(value["status"], "cancelled");
    assert!(matches!(abort_rx.try_recv(), Ok(AbortSource::Guard(rid)) if rid == seeded_rid));

    // The store reflects the cancellation; cancelling again is a 400.
    let response = oneshot(app.clone(), request("GET", "/v1/responses/resp_seeded")).await;
    let value = body_json(response).await;
    assert_eq!(value["status"], "cancelled");
    let response = oneshot(
        app.clone(),
        request("POST", "/v1/responses/resp_seeded/cancel"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

use axum::extract::ws::Message;
use serde_json::json;
use tokio::sync::mpsc;

use crate::{
    protocols::responses::{
        ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseOutputItem,
        ResponseStatus, ResponsesRequest, ResponsesResponse, ServiceTier, StringOrContentParts,
        Truncation,
    },
    routers::{
        grpc::{
            common::responses::streaming::{
                OutputItemType, ResponseEventSink, ResponseStreamEventEmitter,
                SseResponseEventSink, WsResponseEventSink,
            },
            regular::responses::normalize_request_input_items,
        },
        ws_responses::CachedWsResponse,
    },
};

fn repeated_text(size: usize) -> String {
    "x".repeat(size)
}

fn bench_request() -> ResponsesRequest {
    ResponsesRequest {
        background: Some(false),
        include: None,
        input: ResponseInput::Text("bench request".to_string()),
        instructions: None,
        max_output_tokens: Some(256),
        max_tool_calls: None,
        metadata: None,
        model: "bench-model".to_string(),
        parallel_tool_calls: Some(true),
        previous_response_id: None,
        reasoning: None,
        service_tier: Some(ServiceTier::Auto),
        store: Some(true),
        stream: Some(true),
        temperature: Some(0.0),
        tool_choice: None,
        tools: None,
        top_logprobs: Some(0),
        top_p: None,
        truncation: Some(Truncation::Disabled),
        text: None,
        user: None,
        request_id: Some("resp_bench".to_string()),
        priority: 0,
        frequency_penalty: Some(0.0),
        presence_penalty: Some(0.0),
        stop: None,
        top_k: -1,
        min_p: 0.0,
        repetition_penalty: 1.0,
        conversation: None,
    }
}

fn bench_message_item(text: &str, item_id: &str) -> serde_json::Value {
    json!({
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": text
        }]
    })
}

fn build_cached_response(history_turns: usize, text_bytes: usize) -> CachedWsResponse {
    let mut input_items = Vec::with_capacity(history_turns);
    let mut output_items = Vec::with_capacity(history_turns);
    let text = repeated_text(text_bytes);

    for turn in 0..history_turns {
        input_items.push(ResponseInputOutputItem::Message {
            id: format!("msg_user_{turn}"),
            role: "user".to_string(),
            content: vec![ResponseContentPart::InputText { text: text.clone() }],
            status: Some("completed".to_string()),
        });
        output_items.push(ResponseOutputItem::Message {
            id: format!("msg_assistant_{turn}"),
            role: "assistant".to_string(),
            content: vec![ResponseContentPart::OutputText {
                text: text.clone(),
                annotations: vec![],
                logprobs: None,
            }],
            status: "completed".to_string(),
        });
    }

    let response = ResponsesResponse::builder("resp_cached_bench", "bench-model")
        .copy_from_request(&bench_request())
        .status(ResponseStatus::Completed)
        .output(output_items)
        .build();

    CachedWsResponse {
        response,
        input_items,
    }
}

fn build_incremental_items(item_count: usize, text_bytes: usize) -> Vec<ResponseInputOutputItem> {
    let mut items = Vec::with_capacity(item_count);
    let text = repeated_text(text_bytes);

    for index in 0..item_count {
        if index % 2 == 0 {
            items.push(ResponseInputOutputItem::SimpleInputMessage {
                role: "user".to_string(),
                content: StringOrContentParts::Array(vec![ResponseContentPart::InputText {
                    text: text.clone(),
                }]),
                r#type: None,
            });
        } else {
            items.push(ResponseInputOutputItem::FunctionCallOutput {
                id: None,
                call_id: format!("call_{index}"),
                output: text.clone(),
                status: None,
            });
        }
    }

    items
}

fn make_incremental_request(item_count: usize, text_bytes: usize) -> ResponsesRequest {
    ResponsesRequest {
        input: ResponseInput::Items(build_incremental_items(item_count, text_bytes)),
        previous_response_id: Some("resp_prev_bench".to_string()),
        ..bench_request()
    }
}

fn emit_text_stream_batch(
    sink: &impl ResponseEventSink,
    delta_events: usize,
    delta_bytes: usize,
) -> usize {
    let delta = repeated_text(delta_bytes);
    let mut emitter = ResponseStreamEventEmitter::new(
        "resp_bench_stream".to_string(),
        "bench-model".to_string(),
        0,
    );
    emitter.set_original_request(bench_request());

    let created = emitter.emit_created();
    emitter.send_event(&created, sink).unwrap();
    let in_progress = emitter.emit_in_progress();
    emitter.send_event(&in_progress, sink).unwrap();

    let (output_index, item_id) = emitter.allocate_output_index(OutputItemType::Message);
    let added_item = json!({
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "content": []
    });
    let output_added = emitter.emit_output_item_added(output_index, &added_item);
    emitter.send_event(&output_added, sink).unwrap();

    let content_added = emitter.emit_content_part_added(output_index, &item_id, 0);
    emitter.send_event(&content_added, sink).unwrap();

    for _ in 0..delta_events {
        let delta_event = emitter.emit_text_delta(&delta, output_index, &item_id, 0);
        emitter.send_event(&delta_event, sink).unwrap();
    }

    let text_done = emitter.emit_text_done(output_index, &item_id, 0);
    emitter.send_event(&text_done, sink).unwrap();
    let content_done = emitter.emit_content_part_done(output_index, &item_id, 0);
    emitter.send_event(&content_done, sink).unwrap();

    let done_item = bench_message_item(&delta.repeat(delta_events), &item_id);
    let output_done = emitter.emit_output_item_done(output_index, &done_item);
    emitter.send_event(&output_done, sink).unwrap();
    emitter.complete_output_item(output_index);

    let completed = emitter.emit_completed(None);
    emitter.send_event(&completed, sink).unwrap();

    delta_events * delta_bytes
}

#[doc(hidden)]
pub fn bench_emit_ws_text_stream(delta_events: usize, delta_bytes: usize) -> usize {
    let (tx, mut rx) = mpsc::channel::<Message>(4096);
    let sink = WsResponseEventSink::new(tx);
    let payload_bytes = emit_text_stream_batch(&sink, delta_events, delta_bytes);
    let mut drained = 0usize;
    while let Ok(message) = rx.try_recv() {
        drained += match message {
            Message::Text(text) => text.len(),
            Message::Binary(payload) => payload.len(),
            Message::Ping(payload) | Message::Pong(payload) => payload.len(),
            Message::Close(_) => 0,
        };
    }
    payload_bytes + drained
}

#[doc(hidden)]
pub fn bench_emit_sse_text_stream(delta_events: usize, delta_bytes: usize) -> usize {
    let (tx, mut rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    let sink = SseResponseEventSink::new(tx);
    let payload_bytes = emit_text_stream_batch(&sink, delta_events, delta_bytes);
    let mut drained = 0usize;
    while let Ok(item) = rx.try_recv() {
        drained += item.unwrap().len();
    }
    payload_bytes + drained
}

#[doc(hidden)]
pub fn bench_cached_response_to_items(history_turns: usize, text_bytes: usize) -> usize {
    let cached = build_cached_response(history_turns, text_bytes);
    let items = cached.to_conversation_items();
    items.len()
        + items
            .iter()
            .map(|item| serde_json::to_string(item).unwrap().len())
            .sum::<usize>()
}

#[doc(hidden)]
pub fn bench_cached_continuation_shape(
    history_turns: usize,
    history_text_bytes: usize,
    new_items: usize,
    new_text_bytes: usize,
) -> usize {
    let cached = build_cached_response(history_turns, history_text_bytes);
    let mut items = cached.to_conversation_items();
    let request = make_incremental_request(new_items, new_text_bytes);
    items.extend(normalize_request_input_items(&request));
    items.len()
        + items
            .iter()
            .map(|item| serde_json::to_string(item).unwrap().len())
            .sum::<usize>()
}

#[doc(hidden)]
pub fn bench_normalize_incremental_request(item_count: usize, text_bytes: usize) -> usize {
    let request = make_incremental_request(item_count, text_bytes);
    let items = normalize_request_input_items(&request);
    items.len()
        + items
            .iter()
            .map(|item| serde_json::to_string(item).unwrap().len())
            .sum::<usize>()
}

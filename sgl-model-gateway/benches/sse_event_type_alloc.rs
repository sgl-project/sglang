use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use serde_json::{json, Value};

const ADDED: &str = "response.output_item.added";
const DONE: &str = "response.output_item.done";
const ARGUMENTS_DONE: &str = "response.function_call_arguments.done";
const CREATED: &str = "response.created";
const COMPLETED: &str = "response.completed";
const TEXT_DELTA: &str = "response.output_text.delta";

#[derive(PartialEq)]
enum EventAction {
    AddedOrDone,
    ArgsDone,
    Other,
}

#[inline(always)]
fn classify_event(event_type: &str) -> EventAction {
    match event_type {
        s if s == ADDED || s == DONE => EventAction::AddedOrDone,
        s if s == ARGUMENTS_DONE => EventAction::ArgsDone,
        _ => EventAction::Other,
    }
}

// Current approach: allocates a String per token
fn current_approach(parsed_data: &Value) -> usize {
    let event_type = parsed_data
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    // THE ALLOCATION: one heap String per token in the LLM stream
    let event_type = event_type.to_string();

    match event_type.as_str() {
        s if s == ADDED || s == DONE => 1,
        s if s == ARGUMENTS_DONE => 2,
        _ => 0,
    }
}

// Proposed approach: classify once, no allocation

fn proposed_approach(parsed_data: &Value) -> usize {
    let event_type = parsed_data
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let action = classify_event(event_type);

    match action {
        EventAction::AddedOrDone => 1,
        EventAction::ArgsDone => 2,
        EventAction::Other => 0,
    }
}

//
// Benchmark groups
//

/// Benchmark over a representative mix of SSE event types seen during
/// a real LLM streaming response (mostly text_delta, occasional added/done).
fn bench_event_classification(c: &mut Criterion) {
    // Build sample JSON values upfront — we're benchmarking the classification
    // logic, not JSON construction.
    let events: Vec<(&str, Value)> = vec![
        (
            "text_delta (hot path)",
            json!({"type": TEXT_DELTA,     "delta": {"type": "text", "text": "Hello"}}),
        ),
        (
            "output_item.added",
            json!({"type": ADDED,          "output_index": 0, "item": {"type": "message"}}),
        ),
        (
            "output_item.done",
            json!({"type": DONE,           "output_index": 0, "item": {"type": "message"}}),
        ),
        (
            "arguments_done",
            json!({"type": ARGUMENTS_DONE, "item_id": "fc_abc123", "call_id": "call_xyz"  }),
        ),
        (
            "response.created",
            json!({"type": CREATED,        "response": {"id": "resp_001"}}),
        ),
        (
            "response.completed",
            json!({"type": COMPLETED,      "response": {"id": "resp_001"}}),
        ),
    ];

    let mut group = c.benchmark_group("sse_event_classification");

    for (label, event) in &events {
        group.bench_with_input(
            BenchmarkId::new("current (to_string alloc)", label),
            event,
            |b, ev| b.iter(|| black_box(current_approach(ev))),
        );
        group.bench_with_input(
            BenchmarkId::new("proposed (enum, zero alloc)", label),
            event,
            |b, ev| b.iter(|| black_box(proposed_approach(ev))),
        );
    }

    group.finish();
}

/// Throughput simulation: N tokens classified in sequence (like a real stream).
fn bench_token_stream_throughput(c: &mut Criterion) {
    // A realistic token stream: mostly text_delta, 2 boundary events per response.
    let mut stream: Vec<Value> = Vec::with_capacity(202);
    stream.push(json!({"type": ADDED, "output_index": 0, "item": {"type": "message"}}));
    for i in 0..200u32 {
        stream.push(
            json!({"type": TEXT_DELTA, "delta": {"type": "text", "text": format!("tok{i}")}}),
        );
    }
    stream.push(json!({"type": DONE, "output_index": 0, "item": {"type": "message"}}));

    let mut group = c.benchmark_group("sse_token_stream_200_tokens");

    group.bench_function("current (to_string alloc per token)", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for ev in &stream {
                sum += current_approach(black_box(ev));
            }
            black_box(sum)
        })
    });

    group.bench_function("proposed (enum, zero alloc)", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for ev in &stream {
                sum += proposed_approach(black_box(ev));
            }
            black_box(sum)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_event_classification,
    bench_token_stream_throughput
);
criterion_main!(benches);

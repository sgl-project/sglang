use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde_json::{json, Value};

const ADDED: &str = "response.output_item.added";
const DONE: &str = "response.output_item.done";
const ARGUMENTS_DONE: &str = "response.function_call_arguments.done";
const CREATED: &str = "response.created";
const COMPLETED: &str = "response.completed";
const TEXT_DELTA: &str = "response.output_text.delta";
const ARGS_DELTA: &str = "response.function_call_arguments.delta";
const OUTPUT_DONE: &str = "response.output.done";

// Zero-allocation approach: classify into an enum before any mutable borrow
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

/// Current code path: allocates a `String` on every token.
#[inline(never)]
fn current_approach(parsed_data: &Value) -> usize {
    let event_type = parsed_data
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    // heap allocation — fires for every single generated token
    let event_type = event_type.to_string();
    match event_type.as_str() {
        s if s == ADDED || s == DONE => 1,
        s if s == ARGUMENTS_DONE => 2,
        _ => 0,
    }
}

/// Proposed code path: zero allocation.
#[inline(never)]
fn proposed_approach(parsed_data: &Value) -> usize {
    let event_type = parsed_data
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    match classify_event(event_type) {
        EventAction::AddedOrDone => 1,
        EventAction::ArgsDone => 2,
        EventAction::Other => 0,
    }
}

// Stream builders

/// Standard response stream: 1 added + N text_delta tokens + 1 done.
fn make_text_stream(n_tokens: usize) -> Vec<Value> {
    let mut stream = Vec::with_capacity(n_tokens + 2);
    stream.push(json!({"type": ADDED, "output_index": 0, "item": {"type": "message"}}));
    for i in 0..n_tokens {
        stream.push(json!({"type": TEXT_DELTA, "output_index": 0,
                           "delta": {"type": "text", "text": format!("t{i}")}}));
    }
    stream.push(json!({"type": DONE, "output_index": 0, "item": {"type": "message"}}));
    stream
}

/// Tool-call stream: added → N argument delta tokens → arguments_done → done.
/// Exercises the `ArgsDone` arm and the boundary events on every iteration.
fn make_tool_call_stream(n_arg_tokens: usize) -> Vec<Value> {
    let mut stream = Vec::with_capacity(n_arg_tokens + 4);
    stream.push(json!({"type": ADDED, "output_index": 0,
                       "item": {"type": "function_call", "call_id": "call_abc"}}));
    for i in 0..n_arg_tokens {
        stream.push(json!({"type": ARGS_DELTA, "output_index": 0,
                           "delta": format!("\"arg{i}\"")}));
    }
    stream.push(
        json!({"type": ARGUMENTS_DONE, "item_id": "fc_abc", "call_id": "call_abc",
                       "arguments": "{}"}),
    );
    stream.push(json!({"type": OUTPUT_DONE, "output_index": 0}));
    stream.push(json!({"type": DONE, "output_index": 0,
                       "item": {"type": "function_call", "call_id": "call_abc"}}));
    stream
}

// Group 1 – Per-event-type single-call latency
fn bench_single_event(c: &mut Criterion) {
    let events: &[(&str, Value)] = &[
        (
            "text_delta [hot-path]",
            json!({"type": TEXT_DELTA, "delta": {"type": "text", "text": "Hello"}}),
        ),
        (
            "output_item.added",
            json!({"type": ADDED, "output_index": 0, "item": {"type": "message"}}),
        ),
        (
            "output_item.done",
            json!({"type": DONE, "output_index": 0, "item": {"type": "message"}}),
        ),
        (
            "arguments_done",
            json!({"type": ARGUMENTS_DONE, "item_id": "fc_abc123", "call_id": "call_xyz"}),
        ),
        (
            "response.created",
            json!({"type": CREATED, "response": {"id": "resp_001"}}),
        ),
        (
            "response.completed",
            json!({"type": COMPLETED, "response": {"id": "resp_001"}}),
        ),
    ];

    let mut group = c.benchmark_group("sse_single_event");
    for (label, event) in events {
        group.bench_with_input(
            BenchmarkId::new("current/to_string_alloc", label),
            event,
            |b, ev| b.iter(|| black_box(current_approach(ev))),
        );
        group.bench_with_input(
            BenchmarkId::new("proposed/enum_zero_alloc", label),
            event,
            |b, ev| b.iter(|| black_box(proposed_approach(ev))),
        );
    }
    group.finish();
}

// Group 2 – Varying response length (10, 50, 200, 500, 1000, 2000 tokens)
// Shows how savings scale with response length.
fn bench_stream_varying_length(c: &mut Criterion) {
    let lengths: &[usize] = &[10, 50, 200, 500, 1000, 2000];

    let mut group = c.benchmark_group("sse_stream_varying_length");
    for &n in lengths {
        let stream = make_text_stream(n);
        // Report throughput in events/s so the chart is easy to compare.
        group.throughput(Throughput::Elements(stream.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("current/to_string_alloc", n),
            &stream,
            |b, s| {
                b.iter(|| {
                    let mut acc = 0usize;
                    for ev in s {
                        acc += current_approach(black_box(ev));
                    }
                    black_box(acc)
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("proposed/enum_zero_alloc", n),
            &stream,
            |b, s| {
                b.iter(|| {
                    let mut acc = 0usize;
                    for ev in s {
                        acc += proposed_approach(black_box(ev));
                    }
                    black_box(acc)
                })
            },
        );
    }
    group.finish();
}

// Group 3 – Allocator pressure: 10k tight-loop calls (simulates concurrent
// streams fighting over the allocator).
fn bench_allocation_pressure(c: &mut Criterion) {
    let hot_event = json!({"type": TEXT_DELTA, "delta": {"type": "text", "text": "x"}});
    const REPS: usize = 10_000;

    let mut group = c.benchmark_group("sse_allocation_pressure");
    group.throughput(Throughput::Elements(REPS as u64));

    group.bench_function("current/to_string_alloc", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for _ in 0..REPS {
                acc += current_approach(black_box(&hot_event));
            }
            black_box(acc)
        })
    });

    group.bench_function("proposed/enum_zero_alloc", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for _ in 0..REPS {
                acc += proposed_approach(black_box(&hot_event));
            }
            black_box(acc)
        })
    });

    group.finish();
}

// Group 4 – Mixed tool-call stream (exercises all three match arms)

fn bench_mixed_tool_call_stream(c: &mut Criterion) {
    // lengths: (text_tokens, arg_tokens)
    let scenarios: &[(&str, usize, usize)] = &[
        ("small (10t + 5a)", 10, 5),
        ("medium (100t + 20a)", 100, 20),
        ("large (500t + 50a)", 500, 50),
    ];

    let mut group = c.benchmark_group("sse_mixed_tool_call_stream");
    for &(label, n_text, n_args) in scenarios {
        let mut stream = make_text_stream(n_text);
        stream.extend(make_tool_call_stream(n_args));
        group.throughput(Throughput::Elements(stream.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("current/to_string_alloc", label),
            &stream,
            |b, s| {
                b.iter(|| {
                    let mut acc = 0usize;
                    for ev in s {
                        acc += current_approach(black_box(ev));
                    }
                    black_box(acc)
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("proposed/enum_zero_alloc", label),
            &stream,
            |b, s| {
                b.iter(|| {
                    let mut acc = 0usize;
                    for ev in s {
                        acc += proposed_approach(black_box(ev));
                    }
                    black_box(acc)
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_single_event,
    bench_stream_varying_length,
    bench_allocation_pressure,
    bench_mixed_tool_call_stream,
);
criterion_main!(benches);

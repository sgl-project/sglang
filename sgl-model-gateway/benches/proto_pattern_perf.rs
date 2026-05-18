//! Bench: proto-pattern routing-view + bytes vs. typed-deserialise on
//! the chat/generate hot path.
//!
//! The proto-pattern (this worktree's `Router::route_chat` /
//! `route_generate`) parses only a tiny `ChatRoutingView` /
//! `GenerateRoutingView` from the request body and forwards the
//! bytes verbatim. The pre-existing typed-deserialise design parses
//! the full `openai-protocol::ChatCompletionRequest` (which
//! materialises every field, including the multimodal image_url.url
//! string) and then re-serialises it back to bytes for the wire.
//!
//! Three groups, each across five payload shapes (small chat,
//! medium chat with extension fields, multimodal 50KB / 200KB
//! uniform-`A`, multimodal 200KB realistic-base64):
//!
//! 1. `routing_view_only` — just parse the routing view. The
//!    proto-pattern's *full* per-request cost on the unified non-
//!    dp-aware path (after which the bytes forward via
//!    `Bytes::clone`).
//! 2. `typed_full_pipeline` — `serde_json::from_slice::<ChatCompletionRequest>`
//!    + `serde_json::to_vec(&typed)`. What the typed-deserialise
//!    design does on every chat request.
//! 3. `proto_pattern_full_pipeline` — routing view parse + Bytes
//!    clone, end-to-end. What this worktree's
//!    `Router::route_chat` actually runs.

use bytes::Bytes;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde_json::json;
use smg::{protocols::chat::ChatCompletionRequest, routers::http::routing_view::ChatRoutingView};

fn small_chat() -> Bytes {
    Bytes::from(
        serde_json::to_vec(&json!({
            "model": "deepseek-v2-lite",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "What is 2 + 2?"}
            ],
            "max_tokens": 64,
            "temperature": 0.7,
        }))
        .unwrap(),
    )
}

fn medium_chat() -> Bytes {
    let mut messages = vec![json!({
        "role": "system",
        "content": "You are a helpful assistant with extensive knowledge.",
    })];
    for i in 0..30 {
        messages.push(json!({
            "role": "user",
            "content": format!(
                "Question {}: explain topic {} in detail with multiple paragraphs.",
                i, i
            ),
        }));
        messages.push(json!({
            "role": "assistant",
            "content": format!(
                "Answer {}: here is a thorough multi-paragraph explanation of topic {}.",
                i, i
            ),
        }));
    }
    Bytes::from(
        serde_json::to_vec(&json!({
            "model": "deepseek-v2-lite",
            "messages": messages,
            "max_tokens": 1024,
            "return_routed_experts": true,
        }))
        .unwrap(),
    )
}

fn large_multimodal(payload_kb: usize, realistic_base64: bool) -> Bytes {
    let blob: String = if realistic_base64 {
        // Real base64 alphabet (no escape chars but non-uniform).
        "AB+/=09az".repeat((payload_kb * 1024) / 9)
    } else {
        // Best-case uniform payload.
        "A".repeat(payload_kb * 1024)
    };
    Bytes::from(
        serde_json::to_vec(&json!({
            "model": "deepseek-v2-lite",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": format!("data:image/png;base64,{blob}")},
                    },
                ],
            }],
            "max_tokens": 256,
            "return_routed_experts": false,
        }))
        .unwrap(),
    )
}

fn bench_proto_vs_typed(c: &mut Criterion) {
    let mut group = c.benchmark_group("proto_vs_typed");

    let inputs: [(&str, Bytes); 5] = [
        ("small_chat_~250B", small_chat()),
        ("medium_chat_~10KB", medium_chat()),
        ("multimodal_50KB", large_multimodal(50, false)),
        ("multimodal_200KB", large_multimodal(200, false)),
        ("multimodal_200KB_base64ish", large_multimodal(200, true)),
    ];

    for (label, body) in inputs.iter() {
        group.throughput(Throughput::Bytes(body.len() as u64));

        // Proto-pattern routing-view-only parse (the bulk of the work
        // this worktree's chat/generate path does per request).
        group.bench_with_input(
            BenchmarkId::new("routing_view_only", label),
            body,
            |b, body| {
                b.iter(|| {
                    let view: ChatRoutingView = serde_json::from_slice(black_box(body)).unwrap();
                    black_box(view);
                });
            },
        );

        // Typed-deserialise-and-reserialise (what the gateway used to
        // do on chat/generate before the proto-pattern refactor; what
        // upstream smg's `model_gateway` still does today).
        group.bench_with_input(
            BenchmarkId::new("typed_full_pipeline", label),
            body,
            |b, body| {
                b.iter(|| {
                    let typed: ChatCompletionRequest =
                        serde_json::from_slice(black_box(body)).unwrap();
                    let bytes = serde_json::to_vec(&typed).unwrap();
                    black_box((typed, bytes));
                });
            },
        );

        // Proto-pattern end-to-end: parse routing view + Bytes::clone
        // for forwarding. Mirrors `Router::route_chat` →
        // `send_bytes_request`'s non-dp-aware branch.
        group.bench_with_input(
            BenchmarkId::new("proto_pattern_full_pipeline", label),
            body,
            |b, body| {
                b.iter(|| {
                    let view: ChatRoutingView = serde_json::from_slice(black_box(body)).unwrap();
                    let cloned = body.clone();
                    black_box((view, cloned));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_proto_vs_typed);
criterion_main!(benches);

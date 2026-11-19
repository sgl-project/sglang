//! Chat completions through the typed handler: how reasoning is separated
//! from content, unary and streamed, and what a stream's frames are.

use std::net::SocketAddr;

use serde_json::{Value, json};
use sglang_server::ServerArgs;
use sglang_server_test::{OutputDetails, SchedulerlessServer};

use super::server_with;
use crate::common::{Sse, fused_dir, http, next_scheduler_request, piece_id};

/// A ChatML server whose output text is exactly the pieces it is sent, after
/// `change` to its launch arguments.
fn chat_server(change: impl FnOnce(&mut ServerArgs)) -> (SchedulerlessServer, SocketAddr) {
    server_with(|args| {
        args.tokenizer_path = fused_dir().into();
        args.chat_template = Some("chatml".into());
        change(args);
    })
}

fn body(stream: bool) -> Value {
    json!({
        "model": "model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
        "stream_options": stream.then(|| json!({"include_usage": true})),
    })
}

/// More output for a request still running.
fn frame() -> OutputDetails {
    OutputDetails {
        prompt_tokens: 2,
        ..Default::default()
    }
}

fn stop() -> OutputDetails {
    OutputDetails {
        finish_reason: Some(json!({"type": "stop"})),
        ..frame()
    }
}

/// Stream a chat whose output arrives as `pieces`, one event read before the
/// next piece is sent, and return every event.
fn streamed(app: &(SchedulerlessServer, SocketAddr), pieces: &[&str]) -> Vec<String> {
    let (rt, addr) = app;
    let mut events = Sse::post(*addr, "/v1/chat/completions", &body(true));
    let rid = next_scheduler_request(rt).rid().expect("rid");
    // The role comes before any output.
    let mut frames = vec![events.next().expect("the role event")];
    for (sent, piece) in pieces.iter().enumerate() {
        let details = if sent + 1 == pieces.len() {
            stop()
        } else {
            frame()
        };
        rt.push_output_with(&rid, &[piece_id(piece)], details)
            .unwrap();
        if sent + 1 < pieces.len() {
            frames.push(events.next().expect("an output event"));
        }
    }
    while let Some(data) = events.next() {
        frames.push(data);
    }
    frames
}

fn parse(frame: &str) -> Value {
    serde_json::from_str(frame).unwrap_or_else(|_| panic!("not JSON: {frame}"))
}

#[test]
fn unary_chat_separates_reasoning_content_with_parser_configured() {
    let (rt, addr) = chat_server(|args| args.reasoning_parser = Some("deepseek-r1".into()));
    let client = std::thread::spawn(move || {
        http(
            addr,
            "POST",
            "/v1/chat/completions",
            &[],
            &body(false).to_string(),
        )
    });
    let rid = next_scheduler_request(&rt).rid().expect("rid");
    let piece = piece_id("<think>because Paris is famous</think>Paris");
    rt.push_output_with(&rid, &[piece], stop()).unwrap();
    let response = client.join().unwrap();
    assert_eq!(response.status, 200, "{}", response.body);
    let value = response.json();
    assert_eq!(
        value["choices"][0]["message"]["reasoning_content"],
        "because Paris is famous"
    );
    assert_eq!(value["choices"][0]["message"]["content"], "Paris");
}

#[test]
fn streaming_chat_separates_reasoning_into_own_deltas() {
    let app = chat_server(|args| args.reasoning_parser = Some("deepseek-r1".into()));
    // Force mode starts in reasoning, so the opener is stripped and the first
    // reasoning fragment streams immediately. The middle piece closes the
    // reasoning and opens the content: two events.
    let (rt, addr) = &app;
    let mut events = Sse::post(*addr, "/v1/chat/completions", &body(true));
    let rid = next_scheduler_request(rt).rid().expect("rid");
    let role = events.next_json();
    rt.push_output_with(&rid, &[piece_id("<think>be")], frame())
        .unwrap();
    let first_reasoning = events.next_json();
    rt.push_output_with(&rid, &[piece_id("cause</think>Par")], frame())
        .unwrap();
    let second_reasoning = events.next_json();
    let content = events.next_json();
    rt.push_output_with(&rid, &[piece_id("is")], stop())
        .unwrap();
    let terminal = events.next_json();
    let mut rest = Vec::new();
    while let Some(data) = events.next() {
        rest.push(data);
    }
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
    // Then the usage chunk and the end: seven events in all.
    assert_eq!(rest.len(), 2, "{rest:?}");
    assert_eq!(rest[1], "[DONE]");
}

#[test]
fn streaming_chat_emits_role_deltas_usage_and_done() {
    for cache_report in [false, true] {
        let app = chat_server(|args| args.enable_cache_report = cache_report);
        let frames = streamed(&app, &["Par", "is"]);
        assert_eq!(frames.len(), 5, "{frames:?}");
        let role = parse(&frames[0]);
        let delta = parse(&frames[1]);
        let terminal = parse(&frames[2]);
        let usage = parse(&frames[3]);
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert!(role["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(delta["choices"][0]["delta"]["content"], "Par");
        assert!(delta["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(terminal["choices"][0]["delta"]["content"], "is");
        assert!(terminal["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
        assert_eq!(usage["usage"]["completion_tokens"], 2);
        assert_eq!(frames[4], "[DONE]");
        // Python reports a zero cache hit rather than omitting it.
        let details = &usage["usage"]["prompt_tokens_details"];
        match cache_report {
            true => assert_eq!(details["cached_tokens"], 0, "{usage}"),
            false => assert!(details.is_null(), "{usage}"),
        }
    }
}

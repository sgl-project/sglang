//! Chat completions as the server lowers them to the native request: what
//! the lowering refuses, how `n` choices fan out, and how a scheduler's
//! failure, abort and cached-token count are answered.

use std::net::SocketAddr;
use std::time::Duration;

use serde_json::{Value, json};
use sglang_server_test::{OutputDetails, SchedulerlessServer};

use sglang_server::ServerArgs;

use super::{post, server_with};
use crate::common::{
    HttpResponse, byte_level_deepseek_dir, deepseek_dir, http, next_scheduler_work, word_id,
};

/// A server on a DeepSeek V4.1 checkpoint, whose chat the processor lowers,
/// after `change` to its launch arguments.
fn lowered_server_with(change: impl FnOnce(&mut ServerArgs)) -> (SchedulerlessServer, SocketAddr) {
    server_with(|args| {
        args.tokenizer_path = deepseek_dir().into();
        args.reasoning_parser = Some("deepseek-v41".into());
        change(args);
    })
}

fn chat_server() -> (SchedulerlessServer, SocketAddr) {
    lowered_server_with(|_| {})
}

fn messages() -> Value {
    json!([{"role": "user", "content": "hi"}])
}

fn chat_body() -> Value {
    json!({"model": "model", "messages": messages()})
}

/// One finished `ok` for a request.
fn ok() -> OutputDetails {
    OutputDetails {
        prompt_tokens: 1,
        finish_reason: Some(json!({"type": "stop"})),
        ..Default::default()
    }
}

/// Send `body` to the chat route with `headers`, answer each of the
/// `choices` generations it submits with `answer`, and return the response.
fn chat(
    (rt, addr): &(SchedulerlessServer, SocketAddr),
    headers: &'static [&'static str],
    body: Value,
    choices: usize,
    answer: impl Fn(&SchedulerlessServer, &str),
) -> HttpResponse {
    let addr = *addr;
    let client = std::thread::spawn(move || {
        http(
            addr,
            "POST",
            "/v1/chat/completions",
            headers,
            &body.to_string(),
        )
    });
    for _ in 0..choices {
        let rid = next_scheduler_work(rt).rid().expect("rid");
        answer(rt, &rid);
    }
    client.join().unwrap()
}

fn answer_ok(rt: &SchedulerlessServer, rid: &str) {
    rt.push_output_with(rid, &[word_id("ok")], ok()).unwrap();
}

fn nothing_submitted(rt: &SchedulerlessServer, what: &str) {
    assert!(
        rt.next_request(Duration::from_millis(200)).is_none(),
        "{what} reached the scheduler"
    );
}

/// The events of a streamed response's body, as JSON, without `[DONE]`.
fn events(response: &HttpResponse) -> Vec<Value> {
    response
        .body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect()
}

/// An `n` past the batch cap is a 400 before the prompt is copied, not a
/// capacity-overflow panic.
#[test]
fn oversized_n_is_rejected_before_copying_the_prompt() {
    let (rt, addr) = chat_server();
    let mut body = chat_body();
    body["n"] = json!(i64::MAX);
    body["stream"] = json!(true);
    let response = post(addr, "/v1/chat/completions", &body);
    // Unary, as Python answers a failure before the first chunk.
    assert_eq!(response.status, 400);
    let value = response.json();
    assert_eq!(value["type"], "BadRequestError");
    let message = value["message"].as_str().unwrap();
    assert!(message.contains("exceeds the maximum"), "{message}");
    nothing_submitted(&rt, "an oversized n");
}

/// Chat lowers to the native request, so it refuses what `/generate`
/// refuses: a custom logit processor Python would run, but not an empty
/// name, which Python reads as none.
#[test]
fn chat_refuses_a_custom_logit_processor() {
    let app = chat_server();
    let mut body = chat_body();
    body["custom_logit_processor"] = json!("");
    assert_eq!(chat(&app, &[], body.clone(), 1, answer_ok).status, 200);

    body["custom_logit_processor"] = json!("cls");
    let response = post(app.1, "/v1/chat/completions", &body);
    assert_eq!(response.status, 400);
    assert_eq!(
        response.json()["message"],
        "custom_logit_processor is not supported by the Rust server"
    );
}

/// With `n` choices, a prompt's one-element lists cover every choice, as
/// Python's parallel sampling repeats them.
#[test]
fn n_choices_repeat_the_prompts_list_fields() {
    let app = chat_server();
    let mut body = chat_body();
    body["n"] = json!(2);
    body["rid"] = json!(["x"]);
    body["bootstrap_host"] = json!(["h"]);
    body["bootstrap_room"] = json!([5]);
    let response = chat(&app, &[], body, 2, answer_ok);
    assert_eq!(response.status, 200, "{}", response.body);
    assert_eq!(response.json()["choices"].as_array().unwrap().len(), 2);
}

/// A large list field counts toward the copy budget, so `n` choices cannot
/// clone it past the limit.
#[test]
fn n_choices_bound_the_copied_list_fields() {
    let (rt, addr) = chat_server();
    let mut body = chat_body();
    body["n"] = json!(128);
    body["cache_salt"] = json!(["s".repeat(1 << 20)]);
    let response = post(addr, "/v1/chat/completions", &body);
    assert_eq!(response.status, 400);
    let value = response.json();
    let message = value["message"].as_str().unwrap();
    assert!(message.contains("byte limit"), "{message}");
    nothing_submitted(&rt, "an over-budget batch");
}

/// A scheduler abort in a stream, whatever its status, is Python's error
/// event under that status, then the stream's end, not a unary 400.
#[test]
fn streamed_abort_keeps_its_status() {
    let app = chat_server();
    for (status, name) in [
        (503, "SERVICE_UNAVAILABLE"),
        (400, "BAD_REQUEST"),
        (412, "PRECONDITION_FAILED"),
    ] {
        let mut body = chat_body();
        body["stream"] = json!(true);
        let response = chat(&app, &[], body, 1, |rt, rid| {
            let abort = json!({"type": "abort", "message": "aborted", "status_code": status});
            let details = OutputDetails {
                finish_reason: Some(abort),
                ..ok()
            };
            rt.push_output_with(rid, &[], details).unwrap();
        });
        assert_eq!(response.status, 200);
        assert_eq!(
            response.header("content-type"),
            Some("text/event-stream; charset=utf-8")
        );
        let error = events(&response)
            .into_iter()
            .find_map(|event| event.get("error").cloned())
            .unwrap_or_else(|| panic!("no error event: {}", response.body));
        assert_eq!(
            (error["type"].as_str(), error["code"].as_u64()),
            (Some(name), Some(status))
        );
        assert!(response.body.contains("data: [DONE]"), "{}", response.body);
    }
}

/// The router's override headers reach the request, as they reach
/// Python's `adapted_request`.
#[test]
fn override_headers_reach_the_request() {
    let app = lowered_server_with(|args| args.enable_request_header_overrides = true);
    let response = chat(&app, &["x-override-rid: routed"], chat_body(), 1, answer_ok);
    assert_eq!(response.json()["id"], "routed");
    let response = http(
        app.1,
        "POST",
        "/v1/chat/completions",
        &["x-override-priority: high"],
        &chat_body().to_string(),
    );
    assert_eq!(response.status, 400);
    // `HTTPException(400)`, typed by its status.
    assert_eq!(response.json()["type"], "400");
}

/// A failed choice answers at once, without waiting on an unfinished sibling.
#[test]
fn failed_choice_does_not_wait_for_siblings() {
    let app = chat_server();
    let mut body = chat_body();
    body["n"] = json!(2);
    let answered = std::sync::atomic::AtomicUsize::new(0);
    let response = chat(&app, &[], body, 2, |rt, rid| {
        match answered.fetch_add(1, std::sync::atomic::Ordering::Relaxed) {
            0 => rt.push_error(rid, "boom").unwrap(),
            _ => rt
                .push_output_with(rid, &[word_id("ok")], OutputDetails::default())
                .unwrap(),
        }
    });
    assert_eq!(response.status, 500, "{}", response.body);
}

/// This server's own failure answers as Python's generic exception does,
/// unary and streamed before the first frame alike.
#[test]
fn failed_choice_keeps_its_status() {
    let app = chat_server();
    for stream in [false, true] {
        let mut body = chat_body();
        body["stream"] = json!(stream);
        let response = chat(&app, &[], body, 1, |rt, rid| {
            rt.push_error(rid, "boom").unwrap()
        });
        assert_eq!(response.status, 500, "stream={stream}: {}", response.body);
        let value = response.json();
        assert_eq!(
            (value["type"].as_str(), value["code"].as_u64()),
            (Some("InternalServerError"), Some(500)),
            "stream={stream}"
        );
        let message = value["message"].as_str().unwrap();
        assert!(
            message.starts_with("Internal server error: ") && message.contains("boom"),
            "stream={stream}: {value}"
        );
    }
}

/// `--enable-cache-report` reports the prompt's cached tokens, unary and
/// streamed, and a zero count rather than none, as Python does.
#[test]
fn cache_report_reports_cached_tokens() {
    for enable_cache_report in [false, true] {
        let app = lowered_server_with(|args| args.enable_cache_report = enable_cache_report);
        for cached in [0, 3] {
            let answer = |rt: &SchedulerlessServer, rid: &str| {
                let details = OutputDetails {
                    prompt_tokens: 5,
                    cached_tokens: cached,
                    ..ok()
                };
                rt.push_output_with(rid, &[word_id("ok")], details).unwrap();
            };
            let expected = enable_cache_report.then(|| json!({"cached_tokens": cached}));
            let value = chat(&app, &[], chat_body(), 1, answer).json();
            let details =
                Some(value["usage"]["prompt_tokens_details"].clone()).filter(|d| !d.is_null());
            assert_eq!(details, expected, "unary: {value}");

            let mut body = chat_body();
            body["stream"] = json!(true);
            body["stream_options"] = json!({"include_usage": true});
            let response = chat(&app, &[], body, 1, answer);
            let usage = events(&response)
                .into_iter()
                .find_map(|chunk| chunk.get("usage").filter(|u| !u.is_null()).cloned())
                .unwrap_or_else(|| panic!("no usage chunk: {}", response.body));
            let details = Some(usage["prompt_tokens_details"].clone()).filter(|d| !d.is_null());
            assert_eq!(details, expected, "streamed: {}", response.body);
        }
    }
}

/// A served name with a colon, requested exactly, is not a LoRA adapter.
#[test]
fn served_name_with_a_colon_is_the_base_model() {
    let app = lowered_server_with(|args| args.served_model_name = "model:tag".into());
    let body = json!({"model": "model:tag", "messages": messages()});
    assert_eq!(chat(&app, &[], body, 1, answer_ok).status, 200);
}

/// `/tokenize` takes what chat takes, such as a mixed-case role Dynamo's
/// typed request rejects.
#[test]
fn tokenize_takes_what_chat_takes() {
    let (_rt, addr) = chat_server();
    let messages =
        json!([{"role": "System", "content": "Be brief."}, {"role": "user", "content": "hi"}]);
    let response = post(addr, "/v1/tokenize", &json!({"messages": messages}));
    assert_eq!(response.status, 200, "{}", response.body);
}

/// A request the processor does not reproduce reaches the typed handler.
#[test]
fn required_tool_choice_falls_back() {
    let (rt, addr) = chat_server();
    let tool = json!({"type": "function", "function": {"name": "f", "parameters": {}}});
    let mut body = chat_body();
    body["tools"] = json!([tool]);
    body["tool_choice"] = json!("required");
    let response = post(addr, "/v1/chat/completions", &body);
    assert_eq!(
        response.json()["error"]["message"],
        "tool calls require --tool-call-parser"
    );
    nothing_submitted(&rt, "a refused tool call");
}

/// Launch defaults and request overrides must reach both rendering and response parsing.
#[test]
fn processor_honors_server_thinking_defaults() {
    let app = lowered_server_with(|args| {
        args.default_chat_template_kwargs = [("thinking".into(), Value::Bool(false))].into();
    });
    let value = chat(&app, &[], chat_body(), 1, answer_ok).json();
    assert_eq!(value["choices"][0]["message"]["content"], "ok");
    assert!(value["choices"][0]["message"]["reasoning_content"].is_null());
    let mut body = chat_body();
    body["chat_template_kwargs"] = json!({"thinking": true});
    let value = chat(&app, &[], body, 1, answer_ok).json();
    assert_eq!(value["choices"][0]["message"]["reasoning_content"], "ok");
}

/// Logprobs are served by the processor: a partial character's `bytes`
/// are its byte-level token's, as on Python, not the replacement text's.
#[test]
fn processor_serves_logprobs() {
    // Token 0 is the lone byte 0xC3, half of a two-byte character; 1 is `!`.
    let server = |incremental: bool| {
        server_with(|args| {
            args.tokenizer_path = byte_level_deepseek_dir().into();
            args.reasoning_parser = Some("deepseek-v41".into());
            args.incremental_streaming_output = incremental;
        })
    };
    let mut body = chat_body();
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(2);
    let value = chat(&server(false), &[], body, 1, |rt, rid| {
        let details = OutputDetails {
            output_token_logprobs: vec![(-0.25, 0)],
            output_top_logprobs: vec![vec![(-0.25, 0), (-2.0, 1)]],
            ..ok()
        };
        rt.push_output_with(rid, &[0], details).unwrap();
    })
    .json();
    let token = &value["choices"][0]["logprobs"]["content"][0];
    assert_eq!(token["token"], "\u{FFFD}");
    assert_eq!(token["logprob"], -0.25);
    assert_eq!(token["bytes"], json!([0xC3]));
    assert_eq!(token["top_logprobs"][1]["token"], "!");
    assert_eq!(token["top_logprobs"][1]["bytes"], json!([b'!']));
    assert!(value["usage"].get("reasoning_tokens").is_some());

    // Streamed, cumulative or incremental.
    for incremental in [false, true] {
        let mut body = chat_body();
        body["logprobs"] = json!(true);
        body["stream"] = json!(true);
        let response = chat(&server(incremental), &[], body, 1, |rt, rid| {
            let details = OutputDetails {
                output_token_logprobs: vec![(-0.5, 1)],
                ..ok()
            };
            rt.push_output_with(rid, &[1], details).unwrap();
        });
        assert!(
            response.body.contains(r#""logprob":-0.5"#),
            "incremental={incremental}: {}",
            response.body
        );
    }
}

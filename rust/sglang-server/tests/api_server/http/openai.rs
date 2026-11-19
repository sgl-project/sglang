//! The OpenAI routes over HTTP: what is refused before anything reaches the
//! scheduler, how chat completions and completions are answered, unary and
//! streamed, and how `n` choices, scheduler failures and parsers are served.

use std::net::SocketAddr;
use std::time::Duration;

use serde_json::{Value, json};
use sglang_server::ServerArgs;
use sglang_server_test::{OutputDetails, SchedulerlessServer, free_loopback_addrs};

use crate::common::{
    HttpResponse, PIECES, Sse, byte_level_deepseek_dir, deepseek_dir, fused_dir, http,
    next_scheduler_request, next_scheduler_work, piece_id, push_terminal, take_startup_freeze_gc,
    tokenizer_dir, word_id,
};

/// A server named `model` with the word tokenizer of `tokenizer_dir`, after
/// `change` to its launch arguments.
fn server_with(change: impl FnOnce(&mut ServerArgs)) -> (SchedulerlessServer, SocketAddr) {
    let [addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(addr)
        .server_args(|args| {
            args.skip_tokenizer_init = false;
            args.tokenizer_path = tokenizer_dir().into();
            args.served_model_name = "model".into();
            change(args);
        })
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);
    (rt, addr)
}

/// A server that renders chat with the built-in ChatML template.
fn chatml_server() -> (SchedulerlessServer, SocketAddr) {
    server_with(|args| args.chat_template = Some("chatml".into()))
}

fn post(addr: SocketAddr, path: &str, body: &Value) -> HttpResponse {
    http(addr, "POST", path, &[], &body.to_string())
}

/// Send `body` to `path` and answer the generation it submits with one
/// finished `ok`.
fn answered(
    (rt, addr): &(SchedulerlessServer, SocketAddr),
    path: &'static str,
    body: Value,
) -> HttpResponse {
    let addr = *addr;
    let client = std::thread::spawn(move || post(addr, path, &body));
    let rid = next_scheduler_request(rt).rid().expect("rid");
    assert!(push_terminal(rt, &rid, &[word_id("ok")]));
    client.join().unwrap()
}

#[test]
fn completions_handler_validates_before_submit() {
    let (rt, addr) = server_with(|_| {});
    let cases = [
        (json!({"model": "model", "prompt": "hi", "n": 0}), "n=0"),
        (
            json!({"model": "model", "prompt": "hi", "max_tokens": 0}),
            "max_tokens=0",
        ),
        (json!({"model": "model", "prompt": ""}), "empty prompt"),
    ];
    for (body, label) in cases {
        assert_eq!(post(addr, "/v1/completions", &body).status, 400, "{label}");
    }
    // Malformed JSON is a 400 too.
    assert_eq!(
        http(addr, "POST", "/v1/completions", &[], "not json").status,
        400
    );
    assert!(
        rt.next_request(Duration::from_millis(200)).is_none(),
        "a refused completion reached the scheduler"
    );
}

#[test]
fn chat_handler_validates_before_submit() {
    let (rt, addr) = server_with(|_| {});
    let user = json!([{"role": "user", "content": "hi"}]);
    let cases = [
        (json!({"model": "model", "messages": []}), "empty messages"),
        (json!({"model": "model", "messages": user, "n": 0}), "n=0"),
        (
            json!({"model": "model", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://example.com/x.png"}}]}]}),
            "media content",
        ),
        (
            json!({"model": "model", "messages": user, "function_call": "auto"}),
            "deprecated function_call",
        ),
        (
            json!({"model": "model", "messages": user, "audio": {"input_audio": {"data": "x", "format": "wav"}}}),
            "audio",
        ),
        (
            json!({"model": "model", "messages": user, "max_completion_tokens": 0}),
            "max_completion_tokens=0",
        ),
    ];
    for (body, label) in cases {
        assert_eq!(
            post(addr, "/v1/chat/completions", &body).status,
            400,
            "{label}"
        );
    }
    // A valid request with no loaded chat template is a 400 (template gate).
    let valid = json!({"model": "model", "messages": user});
    assert_eq!(post(addr, "/v1/chat/completions", &valid).status, 400);
    assert!(
        rt.next_request(Duration::from_millis(200)).is_none(),
        "a refused chat reached the scheduler"
    );
}

/// Python's chat and completions handlers serve any `model` name and echo it
/// back. A `base-model:adapter` name asks for that LoRA adapter, refused
/// here because none is loaded.
#[test]
fn openai_handlers_serve_any_model_name() {
    let app = chatml_server();
    let messages = json!([{"role": "user", "content": "hi"}]);
    for (path, body) in [
        ("/v1/completions", json!({"prompt": "hi"})),
        ("/v1/chat/completions", json!({"messages": messages})),
    ] {
        for model in ["model", "other-model"] {
            let mut body = body.clone();
            body["model"] = json!(model);
            let response = answered(&app, path, body);
            assert_eq!(response.status, 200, "{path} {model}: {}", response.body);
            assert_eq!(response.json()["model"], model, "{path}");
        }
        let mut body = body.clone();
        body["model"] = json!("model:adapter");
        let response = post(app.1, path, &body);
        assert_eq!(response.status, 400, "{path}");
        let message = response.json()["message"].clone();
        assert!(message.as_str().unwrap().contains("'adapter'"), "{message}");
    }
}

/// Both OpenAI streams declare UTF-8, as Python's `StreamingResponse` does.
#[test]
fn openai_streams_declare_utf8_event_stream() {
    let app = chatml_server();
    let messages = json!([{"role": "user", "content": "hi"}]);
    for (path, body) in [
        (
            "/v1/completions",
            json!({"model": "model", "prompt": "hi", "stream": true}),
        ),
        (
            "/v1/chat/completions",
            json!({"model": "model", "messages": messages, "stream": true}),
        ),
    ] {
        let response = answered(&app, path, body);
        assert_eq!(response.status, 200, "{path}");
        assert_eq!(
            response.header("content-type"),
            Some("text/event-stream; charset=utf-8"),
            "{path}"
        );
    }
}

/// A malformed `/v1/tokenize` or `/v1/detokenize` body is Python's flat
/// error object; a chat prompt with no messages is the chat route's error.
#[test]
fn invalid_tokenize_requests_get_the_flat_python_error() {
    let (rt, addr) = chatml_server();
    for (path, body) in [
        ("/v1/tokenize", json!({"model": "m"})),
        (
            "/v1/tokenize",
            json!({"prompt": "a", "messages": [{"role": "user", "content": "hi"}]}),
        ),
        ("/v1/tokenize", json!({"prompt": 3})),
        ("/v1/detokenize", json!({"tokens": [1, [2]]})),
        ("/v1/detokenize", json!({})),
    ] {
        let response = post(addr, path, &body);
        assert_eq!(response.status, 400, "{body}");
        let error = response.json();
        assert_eq!(error["object"], "error", "{body}");
        assert_eq!(error["type"], "Bad Request", "{body}");
        assert_eq!(error["code"], 400, "{body}");
        assert!(error["param"].is_null(), "{body}");
    }
    let response = post(addr, "/v1/tokenize", &json!({"messages": []}));
    assert_eq!(response.status, 400);
    assert_eq!(response.json()["type"], "BadRequestError");
    assert!(
        rt.next_request(Duration::from_millis(200)).is_none(),
        "a refused request reached the scheduler"
    );
}

// ---- Chat completions on a ChatML server: reasoning, tools, stream frames ----

/// A ChatML server whose output text is exactly the pieces it is sent, after
/// `change` to its launch arguments.
fn fused_chat_server(change: impl FnOnce(&mut ServerArgs)) -> (SchedulerlessServer, SocketAddr) {
    server_with(|args| {
        args.tokenizer_path = fused_dir().into();
        args.chat_template = Some("chatml".into());
        change(args);
    })
}

fn chat_request(stream: bool) -> Value {
    json!({
        "model": "model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
        "stream_options": stream.then(|| json!({"include_usage": true})),
    })
}

/// More output for a request still running.
fn chat_frame() -> OutputDetails {
    OutputDetails {
        prompt_tokens: 2,
        ..Default::default()
    }
}

fn chat_stop() -> OutputDetails {
    OutputDetails {
        finish_reason: Some(json!({"type": "stop"})),
        ..chat_frame()
    }
}

/// Whether a stream event carries output text.
fn has_text(frame: &str) -> bool {
    let delta = &parse(frame)["choices"][0]["delta"];
    [&delta["content"], &delta["reasoning_content"]]
        .iter()
        .any(|text| text.as_str().is_some_and(|text| !text.is_empty()))
}

/// Stream a chat whose output arrives as `pieces`, each piece's text read
/// before the next is sent, and return every event.
fn streamed(app: &(SchedulerlessServer, SocketAddr), pieces: &[&str]) -> Vec<String> {
    let (rt, addr) = (&app.0, app.1);
    // The response starts with the first output, as Python's does.
    let post =
        std::thread::spawn(move || Sse::post(addr, "/v1/chat/completions", &chat_request(true)));
    let rid = next_scheduler_request(rt).rid().expect("rid");
    let (last, earlier) = pieces.split_last().expect("a piece");
    let mut pushed = earlier.iter().map(|piece| {
        rt.push_output_with(&rid, &[piece_id(piece)], chat_frame())
            .unwrap();
    });
    let first = pushed.next();
    if first.is_none() {
        rt.push_output_with(&rid, &[piece_id(last)], chat_stop())
            .unwrap();
    }
    let mut events = post.join().unwrap();
    let mut frames: Vec<String> = Vec::new();
    let read_text = |events: &mut Sse, frames: &mut Vec<String>| loop {
        frames.push(events.next().expect("an output event"));
        if has_text(frames.last().unwrap()) {
            break;
        }
    };
    if first.is_some() {
        read_text(&mut events, &mut frames);
        for () in pushed {
            read_text(&mut events, &mut frames);
        }
        rt.push_output_with(&rid, &[piece_id(last)], chat_stop())
            .unwrap();
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
    let (rt, addr) = fused_chat_server(|args| args.reasoning_parser = Some("deepseek-r1".into()));
    let client = std::thread::spawn(move || {
        http(
            addr,
            "POST",
            "/v1/chat/completions",
            &[],
            &chat_request(false).to_string(),
        )
    });
    let rid = next_scheduler_request(&rt).rid().expect("rid");
    let piece = piece_id("<think>because Paris is famous</think>Paris");
    rt.push_output_with(&rid, &[piece], chat_stop()).unwrap();
    let response = client.join().unwrap();
    assert_eq!(response.status, 200, "{}", response.body);
    let value = response.json();
    assert_eq!(
        value["choices"][0]["message"]["reasoning_content"],
        "because Paris is famous"
    );
    assert_eq!(value["choices"][0]["message"]["content"], "Paris");
}

/// The delta of the first choice of each event that has one.
fn deltas(frames: &[String]) -> Vec<Value> {
    let choices = frames.iter().filter(|frame| *frame != "[DONE]");
    choices
        .map(|frame| parse(frame)["choices"][0].clone())
        .filter(|choice| !choice.is_null())
        .collect()
}

#[test]
fn streaming_chat_separates_reasoning_into_own_deltas() {
    let app = fused_chat_server(|args| args.reasoning_parser = Some("deepseek-r1".into()));
    // The middle piece closes the reasoning and opens the content: two events.
    let frames = streamed(&app, &["<think>be", "cause</think>Par", "is"]);
    let choices = deltas(&frames);
    let reasoning: Vec<_> = choices
        .iter()
        .map(|choice| &choice["delta"]["reasoning_content"])
        .collect();
    let content: Vec<_> = choices
        .iter()
        .map(|choice| &choice["delta"]["content"])
        .collect();
    let null = Value::Null;
    assert_eq!(
        reasoning,
        [&null, &json!("be"), &json!("cause"), &null, &null, &null]
    );
    assert_eq!(
        content,
        [&json!(""), &null, &null, &json!("Par"), &json!("is"), &null]
    );
    assert_eq!(choices[0]["delta"]["role"], "assistant");
    assert_eq!(choices[5]["finish_reason"], "stop");
    // Then the usage chunk and the end.
    assert_eq!(frames.len(), 8, "{frames:?}");
    assert_eq!(frames[7], "[DONE]");
}

#[test]
fn streaming_chat_emits_role_deltas_usage_and_done() {
    for cache_report in [false, true] {
        let app = fused_chat_server(|args| args.enable_cache_report = cache_report);
        let frames = streamed(&app, &["Par", "is"]);
        assert_eq!(frames.len(), 6, "{frames:?}");
        let choices = deltas(&frames);
        let usage = parse(&frames[4]);
        assert_eq!(choices[0]["delta"]["role"], "assistant");
        assert_eq!(choices[1]["delta"]["content"], "Par");
        assert_eq!(choices[2]["delta"]["content"], "is");
        assert!(choices[2]["finish_reason"].is_null());
        // The finish reason is a chunk of its own, as on Python.
        assert!(choices[3]["delta"]["content"].is_null());
        assert_eq!(choices[3]["finish_reason"], "stop");
        assert!(usage["choices"].as_array().unwrap().is_empty());
        assert_eq!(usage["usage"]["completion_tokens"], 2);
        assert_eq!(frames[5], "[DONE]");
        // Python reports a zero cache hit rather than omitting it.
        let details = &usage["usage"]["prompt_tokens_details"];
        match cache_report {
            true => assert_eq!(details["cached_tokens"], 0, "{usage}"),
            false => assert!(details.is_null(), "{usage}"),
        }
    }
}

/// A tool parser the processor has no port of parses through Dynamo's, whole
/// and streamed, for a free and a forced choice.
#[test]
fn unported_tool_parser_returns_tool_calls() {
    let [head, tail] = [PIECES[9], PIECES[10]].map(piece_id);
    let cases = [false, true]
        .into_iter()
        .flat_map(|stream| [(stream, Value::Null), (stream, json!("required"))]);
    for (stream, tool_choice) in cases {
        let (rt, addr) = fused_chat_server(|args| args.tool_call_parser = Some("hermes".into()));
        let mut body = chat_request(stream);
        body["tool_choice"] = tool_choice;
        body["tools"] = json!([{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }}]);
        let client = std::thread::spawn(move || {
            http(addr, "POST", "/v1/chat/completions", &[], &body.to_string())
        });
        let rid = next_scheduler_request(&rt).rid().expect("rid");
        rt.push_output_with(&rid, &[head], chat_frame()).unwrap();
        rt.push_output_with(&rid, &[tail], chat_stop()).unwrap();
        let response = client.join().unwrap();
        assert_eq!(response.status, 200, "{}", response.body);
        let (name, arguments, finish_reason) = if stream {
            let events: Vec<Value> = response
                .body
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter(|data| *data != "[DONE]")
                .map(parse)
                .collect();
            let choices = events.iter().map(|event| &event["choices"][0]);
            let calls: Vec<&Value> = choices
                .clone()
                .map(|choice| &choice["delta"]["tool_calls"][0]["function"])
                .collect();
            let text = |key: &str| -> String {
                calls.iter().filter_map(|call| call[key].as_str()).collect()
            };
            let mut finish = choices.filter_map(|choice| choice["finish_reason"].as_str());
            (
                text("name"),
                text("arguments"),
                finish.next_back().map(str::to_owned),
            )
        } else {
            let choice = &response.json()["choices"][0];
            let call = &choice["message"]["tool_calls"][0]["function"];
            (
                call["name"].as_str().unwrap_or_default().to_owned(),
                call["arguments"].as_str().unwrap_or_default().to_owned(),
                choice["finish_reason"].as_str().map(str::to_owned),
            )
        };
        assert_eq!(name, "get_weather", "{}", response.body);
        assert_eq!(parse(&arguments), json!({"city": "Paris"}));
        assert_eq!(finish_reason.as_deref(), Some("tool_calls"));
    }
}

// ---- Completions: how choices fold and how a stream is framed ----

/// A server whose output text is exactly the pieces it is sent.
fn completions_server(cache_report: bool) -> (SchedulerlessServer, SocketAddr) {
    server_with(|args| {
        args.tokenizer_path = fused_dir().into();
        args.enable_cache_report = cache_report;
    })
}

fn completion_frame() -> OutputDetails {
    OutputDetails {
        prompt_tokens: 5,
        ..Default::default()
    }
}

fn completion_stop(cached_tokens: u32) -> OutputDetails {
    OutputDetails {
        cached_tokens,
        finish_reason: Some(json!({"type": "stop", "matched": "</s>"})),
        ..completion_frame()
    }
}

#[test]
fn unary_fold_orders_choices_and_counts_each_prompt_once() {
    for cache_report in [false, true] {
        let (rt, addr) = completions_server(cache_report);
        let body = json!({"model": "model", "prompt": "hi", "n": 2, "rid": "r"});
        let client = std::thread::spawn(move || {
            http(addr, "POST", "/v1/completions", &[], &body.to_string())
        });
        // A prompt's choices are its rid numbered in choice order.
        let mut rids = [next_scheduler_request(&rt), next_scheduler_request(&rt)]
            .map(|request| request.rid().expect("rid"));
        rids.sort();
        let [rid0, rid1] = rids;
        rt.push_output_with(&rid0, &[piece_id("Par")], completion_frame())
            .unwrap();
        rt.push_output_with(&rid0, &[piece_id("is")], completion_stop(3))
            .unwrap();
        rt.push_output_with(&rid1, &[piece_id("is")], completion_frame())
            .unwrap();
        rt.push_output_with(&rid1, &[piece_id("Par")], completion_stop(4))
            .unwrap();

        let response = client.join().unwrap();
        assert_eq!(response.status, 200, "{}", response.body);
        let value = response.json();
        assert_eq!(value["choices"][0]["text"], "Paris");
        assert_eq!(value["choices"][1]["text"], "isPar");
        assert_eq!(value["choices"][0]["matched_stop"], "</s>");
        assert_eq!(value["usage"]["prompt_tokens"], 5);
        assert_eq!(value["usage"]["completion_tokens"], 4);
        assert_eq!(
            value["usage"]["completion_tokens_details"]["reasoning_tokens"],
            0
        );
        // A prompt's cached tokens are its first choice's, as on Python.
        let details = &value["usage"]["prompt_tokens_details"];
        match cache_report {
            true => assert_eq!(details["cached_tokens"], 3, "{value}"),
            false => assert!(details.is_null(), "{value}"),
        }
    }
}

#[test]
fn stream_uses_deltas_then_usage_and_done() {
    for cache_report in [false, true] {
        let (rt, addr) = completions_server(cache_report);
        let body = json!({
            "model": "model",
            "prompt": "hi",
            "stream": true,
            "stream_options": {"include_usage": true, "continuous_usage_stats": cache_report},
        });
        // The response starts with the first output, as Python's does.
        let post = std::thread::spawn(move || Sse::post(addr, "/v1/completions", &body));
        let rid = next_scheduler_request(&rt).rid().expect("rid");
        rt.push_output_with(&rid, &[piece_id("Par")], completion_frame())
            .unwrap();
        let mut events = post.join().unwrap();
        let first = events.next_json();
        rt.push_output_with(&rid, &[piece_id("is")], completion_stop(0))
            .unwrap();
        let mut frames = Vec::new();
        while let Some(data) = events.next() {
            frames.push(data);
        }
        assert_eq!(frames.len(), 3, "{frames:?}");
        let terminal: Value = serde_json::from_str(&frames[0]).unwrap();
        let usage: Value = serde_json::from_str(&frames[1]).unwrap();
        assert_eq!(first["choices"][0]["text"], "Par");
        assert_eq!(terminal["choices"][0]["text"], "is");
        assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
        assert!(usage["choices"].as_array().unwrap().is_empty());
        assert_eq!(usage["usage"]["prompt_tokens"], 5);
        assert_eq!(usage["usage"]["completion_tokens"], 2);
        assert_eq!(frames[2], "[DONE]");
        // Python reports a zero cache hit rather than omitting it, but not
        // in per-chunk usage.
        let details = &usage["usage"]["prompt_tokens_details"];
        match cache_report {
            true => {
                assert_eq!(details["cached_tokens"], 0, "{usage}");
                assert_eq!(first["usage"]["prompt_tokens"], 5);
                assert!(first["usage"]["prompt_tokens_details"].is_null());
            }
            false => assert!(details.is_null(), "{usage}"),
        }
    }
}

// ---- Chat on a DeepSeek V4.1 checkpoint: choices, failures, usage ----

/// A server on a DeepSeek V4.1 checkpoint, whose chat the processor lowers,
/// after `change` to its launch arguments.
fn deepseek_server_with(change: impl FnOnce(&mut ServerArgs)) -> (SchedulerlessServer, SocketAddr) {
    server_with(|args| {
        args.tokenizer_path = deepseek_dir().into();
        args.reasoning_parser = Some("deepseek-v41".into());
        change(args);
    })
}

fn deepseek_server() -> (SchedulerlessServer, SocketAddr) {
    deepseek_server_with(|_| {})
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
    let (rt, addr) = deepseek_server();
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
    let app = deepseek_server();
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
    let app = deepseek_server();
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
    let (rt, addr) = deepseek_server();
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
    let app = deepseek_server();
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
    let app = deepseek_server_with(|args| args.enable_request_header_overrides = true);
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
    let app = deepseek_server();
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
    let app = deepseek_server();
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
        let app = deepseek_server_with(|args| args.enable_cache_report = enable_cache_report);
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
    let app = deepseek_server_with(|args| args.served_model_name = "model:tag".into());
    let body = json!({"model": "model:tag", "messages": messages()});
    assert_eq!(chat(&app, &[], body, 1, answer_ok).status, 200);
}

/// `/tokenize` takes what chat takes, such as a mixed-case role Dynamo's
/// typed request rejects.
#[test]
fn tokenize_takes_what_chat_takes() {
    let (_rt, addr) = deepseek_server();
    let messages =
        json!([{"role": "System", "content": "Be brief."}, {"role": "user", "content": "hi"}]);
    let response = post(addr, "/v1/tokenize", &json!({"messages": messages}));
    assert_eq!(response.status, 200, "{}", response.body);
}

/// A request the processor declines is refused, naming the reason.
#[test]
fn declined_request_is_refused() {
    let (rt, addr) = deepseek_server();
    let tool = json!({"type": "function", "function": {"name": "f", "parameters": {}}});
    let mut body = chat_body();
    body["tools"] = json!([tool]);
    body["tool_choice"] = json!("required");
    let response = post(addr, "/v1/chat/completions", &body);
    assert_eq!(response.status, 400, "{}", response.body);
    assert_eq!(response.json()["message"], "Invalid request (tool_choice)");
    nothing_submitted(&rt, "a refused tool call");
}

/// Launch defaults and request overrides must reach both rendering and response parsing.
#[test]
fn processor_honors_server_thinking_defaults() {
    let app = deepseek_server_with(|args| {
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

//! The OpenAI completions and chat routes over HTTP: what is refused before
//! anything reaches the scheduler, and how a served request is answered.

mod chat;
mod completions;
mod lowered;

use std::net::SocketAddr;
use std::time::Duration;

use serde_json::{Value, json};
use sglang_server::ServerArgs;
use sglang_server_test::{SchedulerlessServer, free_loopback_addrs};

use crate::common::{
    HttpResponse, http, next_scheduler_request, push_terminal, take_startup_freeze_gc,
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
fn chat_server() -> (SchedulerlessServer, SocketAddr) {
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
/// back. Only a `base-model:adapter` name is refused: it asks for a LoRA
/// adapter, which this server does not support.
#[test]
fn openai_handlers_serve_any_model_name() {
    let app = chat_server();
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
        let message = response.json()["error"]["message"].clone();
        assert!(message.as_str().unwrap().contains("`adapter`"), "{message}");
    }
}

/// Both OpenAI streams declare UTF-8, as Python's `StreamingResponse` does.
#[test]
fn openai_streams_declare_utf8_event_stream() {
    let app = chat_server();
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
    let (rt, addr) = chat_server();
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

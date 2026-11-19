//! The native `/generate` and health routes over HTTP: the frames a stream
//! and a unary call carry for what the scheduler sends, in Python's shapes.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use sglang_server::{GenerateRequest, GenerateValidator, HttpLayer, RequestStart, ServerArgs};
use sglang_server_test::{
    OutputDetails, PreferredSamplingParams, SchedulerlessServer, free_loopback_addrs,
};

use crate::common::{
    Sse, header_json, http, next_scheduler_request, take_startup_freeze_gc, tokenizer_dir, word_id,
};

/// A server with the word tokenizer of `tokenizer_dir`, after `change` to its
/// launch arguments.
fn server_with(change: impl FnOnce(&mut ServerArgs)) -> (SchedulerlessServer, SocketAddr) {
    serve(change, None)
}

fn server() -> (SchedulerlessServer, SocketAddr) {
    server_with(|_| {})
}

fn serve(
    change: impl FnOnce(&mut ServerArgs),
    layer: Option<HttpLayer>,
) -> (SchedulerlessServer, SocketAddr) {
    let [addr] = free_loopback_addrs();
    let mut builder = SchedulerlessServer::builder(addr).server_args(|args| {
        args.skip_tokenizer_init = false;
        args.tokenizer_path = tokenizer_dir().into();
        change(args);
    });
    if let Some(layer) = layer {
        builder = builder.http_layer(layer);
    }
    let rt = builder.start().expect("start runtime");
    take_startup_freeze_gc(&rt);
    (rt, addr)
}

fn ids(words: &[&str]) -> Vec<i64> {
    words.iter().map(|word| word_id(word)).collect()
}

/// More tokens for a request still running.
fn frame() -> OutputDetails {
    OutputDetails::default()
}

/// The request's last tokens, ending on Python's length finish.
fn done() -> OutputDetails {
    OutputDetails {
        finish_reason: Some(json!({"type": "length", "length": 1})),
        ..Default::default()
    }
}

/// Start a unary `/generate` on its own thread, and return the rid it
/// reaches the scheduler under.
fn unary(
    rt: &SchedulerlessServer,
    addr: SocketAddr,
    body: Value,
) -> (std::thread::JoinHandle<crate::common::HttpResponse>, String) {
    let client =
        std::thread::spawn(move || http(addr, "POST", "/generate", &[], &body.to_string()));
    let rid = next_scheduler_request(rt).rid().expect("rid");
    (client, rid)
}

/// The rid the client sent is the one every frame reports, and an HTTP
/// layer's [`RequestStart`] starts the request's timing, so `e2e_latency`
/// covers the layer's work too.
fn assert_native_http_rid_echo(client_rid: &str, stream: bool) {
    let layer_work = Duration::from_secs(5);
    let layer: HttpLayer = Arc::new(move |app: axum::Router| {
        app.layer(axum::middleware::from_fn(
            move |mut request: axum::extract::Request, next: axum::middleware::Next| async move {
                let start = Instant::now().checked_sub(layer_work).unwrap();
                request.extensions_mut().insert(RequestStart(start));
                next.run(request).await
            },
        ))
    });
    let (rt, addr) = serve(|_| {}, Some(layer));
    let body = json!({
        "rid": client_rid,
        "input_ids": [1, 2],
        "sampling_params": {"max_new_tokens": 2},
        "stream": stream,
    });
    let with_prompt = |details: OutputDetails| OutputDetails {
        prompt_tokens: 2,
        ..details
    };
    let values: Vec<Value> = if stream {
        let mut events = Sse::post(addr, "/generate", &body);
        assert_eq!(events.status, 200);
        let request = next_scheduler_request(&rt);
        assert_eq!(request.buffer_shape("input_ids"), Some(&[2][..]));
        let rid = request.rid().expect("rid");
        rt.push_output_with(&rid, &ids(&["Hel"]), with_prompt(frame()))
            .unwrap();
        // Read the first event before the end, avoiding cumulative coalescing.
        let mut values = vec![events.next_json()];
        rt.push_output_with(&rid, &ids(&["lo"]), with_prompt(done()))
            .unwrap();
        while let Some(data) = events.next() {
            if data != "[DONE]" {
                values.push(serde_json::from_str(&data).unwrap());
            }
        }
        values
    } else {
        let (client, rid) = unary(&rt, addr, body);
        rt.push_output_with(&rid, &ids(&["Hel", "lo"]), with_prompt(done()))
            .unwrap();
        let response = client.join().unwrap();
        assert_eq!(response.status, 200, "{}", response.body);
        vec![response.json()]
    };
    if stream {
        assert!(
            values.len() >= 2,
            "stream must carry intermediate and terminal frames"
        );
    }
    for value in &values {
        assert_eq!(value["meta_info"]["id"], client_rid);
    }
    let value = values.last().unwrap();
    assert_eq!(value["text"], "Hel lo");
    assert_eq!(value["meta_info"]["prompt_tokens"], 2);
    assert_eq!(value["meta_info"]["completion_tokens"], 2);
    let e2e_latency = value["meta_info"]["e2e_latency"].as_f64().unwrap();
    assert!(e2e_latency >= layer_work.as_secs_f64(), "{e2e_latency}");
    assert!(
        rt.next_request(Duration::from_millis(200)).is_none(),
        "a finished request was aborted"
    );
}

#[test]
fn unary_http_generate_round_trips_through_frontend_handle() {
    assert_native_http_rid_echo("client-rid", false);
}

#[test]
fn long_rids_round_trip_through_unary_and_streaming_http() {
    for client_rid in ["x".repeat(200), "é".repeat(128)] {
        for stream in [false, true] {
            assert_native_http_rid_echo(&client_rid, stream);
        }
    }
}

#[test]
fn unary_terminal_meta_info_matches_python_semantics() {
    let (rt, addr) = server();
    let body =
        json!({"rid": "client-rid", "input_ids": [1], "sampling_params": {"max_new_tokens": 2}});
    let (client, rid) = unary(&rt, addr, body);
    std::thread::sleep(Duration::from_millis(20));
    let details = OutputDetails {
        prompt_tokens: 5,
        finish_reason: Some(json!({"type": "length", "length": 2})),
        ..Default::default()
    };
    rt.push_output_with(&rid, &ids(&["ok", "hi"]), details)
        .unwrap();
    let response = client.join().unwrap();
    let value = response.json();
    assert_eq!(response.status, 200);
    assert_eq!(value["meta_info"]["id"], "client-rid");
    assert_eq!(value["meta_info"]["prompt_tokens"], 5);
    assert_eq!(value["meta_info"]["completion_tokens"], 2);
    assert_eq!(
        value["meta_info"]["finish_reason"],
        json!({"type": "length", "length": 2})
    );
    assert!(
        value["meta_info"]["e2e_latency"].as_f64().unwrap() >= 0.020,
        "latency is expressed in seconds from request creation"
    );
    assert!(
        value["meta_info"].get("ttft").is_none()
            && value["meta_info"].get("time_to_first_token").is_none(),
        "TTFT is recorded internally but is not part of the API"
    );
}

/// Two sub-requests' frames interleave into one stream, each tagged with its
/// batch `index`; text accumulates per item; `[DONE]` comes only after both
/// terminate, then the stream ends.
#[test]
fn interleaves_indexes_and_accumulates() {
    let (rt, addr) = server();
    let body =
        json!({"input_ids": [[1], [2]], "stream": true, "sampling_params": {"max_new_tokens": 4}});
    let mut events = Sse::post(addr, "/generate", &body);
    let rids = [next_scheduler_request(&rt), next_scheduler_request(&rt)]
        .map(|request| request.rid().expect("rid"));
    let [rid0, rid1] = rids;
    std::thread::sleep(Duration::from_millis(10));

    // Drive deterministically: exactly one item has data before each read.
    rt.push_output_with(&rid0, &ids(&["a"]), frame()).unwrap();
    let v = events.next_json();
    let first = v["index"].as_u64().unwrap();
    assert_eq!(v["text"], "a");

    rt.push_output_with(&rid1, &ids(&["b"]), frame()).unwrap();
    let v = events.next_json();
    assert_eq!(v["index"], 1 - first);
    assert_eq!(v["text"], "b");

    rt.push_output_with(&rid0, &ids(&["c"]), done()).unwrap();
    let v = events.next_json();
    assert_eq!(v["index"], first);
    assert_eq!(v["text"], "a c", "cumulative per item");
    assert_eq!(v["meta_info"]["finish_reason"]["type"], "length");
    assert!(v["meta_info"]["e2e_latency"].as_f64().unwrap() >= 0.010);

    rt.push_output_with(&rid1, &ids(&["hi"]), done()).unwrap();
    let v = events.next_json();
    assert_eq!(v["index"], 1 - first);
    assert_eq!(v["text"], "b hi");
    assert!(v["meta_info"]["e2e_latency"].as_f64().unwrap() >= 0.010);

    assert_eq!(events.next().as_deref(), Some("[DONE]"));
    assert!(events.next().is_none());
}

#[test]
fn dropping_stream_aborts_only_unfinished_calls() {
    let (rt, addr) = server();
    let body =
        json!({"input_ids": [[1], [2]], "stream": true, "sampling_params": {"max_new_tokens": 4}});
    let mut events = Sse::post(addr, "/generate", &body);
    let rids = [next_scheduler_request(&rt), next_scheduler_request(&rt)]
        .map(|request| request.rid().expect("rid"));

    rt.push_output_with(&rids[0], &ids(&["ok"]), done())
        .unwrap();
    let finished = events.next_json()["index"].clone();
    rt.push_output_with(&rids[1], &ids(&["a"]), frame())
        .unwrap();
    assert_ne!(events.next_json()["index"], finished);

    drop(events);
    let abort = next_scheduler_request(&rt);
    assert_eq!(abort.kind().as_deref(), Some("AbortReq"));
    assert_eq!(abort.rid().as_deref(), Some(rids[1].as_str()));
    assert!(rt.next_request(Duration::from_millis(200)).is_none());
}

#[test]
fn scheduler_abort_ends_a_stream_on_a_frame_and_fails_unary() {
    let (rt, addr) = server();
    let finish_reason = json!({
        "type": "abort",
        "message": "media decode failed",
        "status_code": 432,
        "err_type": "BadRequestError"
    });
    let abort = || OutputDetails {
        prompt_tokens: 5,
        finish_reason: Some(finish_reason.clone()),
        ..Default::default()
    };

    let body = json!({"input_ids": [1], "stream": true, "sampling_params": {"max_new_tokens": 4}});
    let mut events = Sse::post(addr, "/generate", &body);
    let rid = next_scheduler_request(&rt).rid().expect("rid");
    std::thread::sleep(Duration::from_millis(10));
    rt.push_output_with(&rid, &ids(&["partial"]), frame())
        .unwrap();
    assert_eq!(events.next_json()["text"], "partial");
    rt.push_output_with(&rid, &[], abort()).unwrap();
    let v = events.next_json();
    assert!(v.get("error").is_none(), "no error item: {v}");
    assert_eq!(v["text"], "partial");
    assert_eq!(v["meta_info"]["prompt_tokens"], 5);
    assert_eq!(v["meta_info"]["completion_tokens"], 1);
    assert_eq!(v["meta_info"]["finish_reason"], finish_reason);
    assert!(v["meta_info"]["e2e_latency"].as_f64().unwrap() >= 0.010);
    assert_eq!(events.next().as_deref(), Some("[DONE]"));

    let (client, rid) = unary(&rt, addr, json!({"input_ids": [1]}));
    rt.push_output_with(&rid, &[], abort()).unwrap();
    let response = client.join().unwrap();
    assert_eq!(response.status, 432);
    let v = response.json();
    assert_eq!(v["error"]["code"], 432);
    assert_eq!(v["error"]["message"], "media decode failed");
}

/// With incremental streaming each frame carries this step's **delta**
/// text, but `meta_info.completion_tokens` stays cumulative (matching
/// Python), and so does the cached-token split, which a scheduler abort does
/// not carry.
#[test]
fn incremental_emits_deltas_with_cumulative_count() {
    let (rt, addr) = server_with(|args| args.incremental_streaming_output = true);
    let body = json!({"input_ids": [1], "stream": true, "sampling_params": {"max_new_tokens": 4}});
    let mut events = Sse::post(addr, "/generate", &body);
    let rid = next_scheduler_request(&rt).rid().expect("rid");
    std::thread::sleep(Duration::from_millis(10));
    let split = json!({"device": 1, "host": 2});

    let hello = OutputDetails {
        cached_tokens: 3,
        cached_tokens_split: Some(json!({"device": 1, "host": 2, "storage": 0})),
        ..frame()
    };
    rt.push_output_with(&rid, &ids(&["Hel"]), hello).unwrap();
    let v = events.next_json();
    assert_eq!(v["text"], "Hel");
    assert_eq!(v["meta_info"]["completion_tokens"], 1);
    assert_eq!(v["meta_info"]["cached_tokens_details"], split);
    assert!(v["meta_info"].get("e2e_latency").is_none());

    rt.push_output_with(&rid, &ids(&["lo"]), frame()).unwrap();
    let v = events.next_json();
    assert_eq!(v["text"], " lo", "delta, not cumulative 'Hel lo'");
    assert_eq!(
        v["meta_info"]["completion_tokens"], 2,
        "count stays cumulative"
    );
    assert!(v["meta_info"].get("e2e_latency").is_none());

    let aborted = OutputDetails {
        cached_tokens: 3,
        abort: true,
        ..done()
    };
    rt.push_output_with(&rid, &ids(&["hi"]), aborted).unwrap();
    let v = events.next_json();
    assert_eq!(v["text"], " hi");
    assert_eq!(v["meta_info"]["completion_tokens"], 3);
    assert_eq!(v["meta_info"]["finish_reason"]["type"], "length");
    assert_eq!(v["meta_info"]["cached_tokens_details"], split);
    assert!(v["meta_info"]["e2e_latency"].as_f64().unwrap() >= 0.010);

    assert_eq!(events.next().as_deref(), Some("[DONE]"));
}

/// The single-request shape omits the `index` field entirely, and still
/// terminates with `[DONE]`.
#[test]
fn single_shape_omits_index() {
    let (rt, addr) = server();
    let body = json!({"input_ids": [1], "stream": true, "sampling_params": {"max_new_tokens": 4}});
    let mut events = Sse::post(addr, "/generate", &body);
    let rid = next_scheduler_request(&rt).rid().expect("rid");
    std::thread::sleep(Duration::from_millis(10));

    rt.push_output_with(&rid, &ids(&["hi"]), done()).unwrap();
    let v = events.next_json();
    assert_eq!(v["text"], "hi");
    assert!(v.get("index").is_none(), "single response has no index");
    assert!(v["meta_info"]["e2e_latency"].as_f64().unwrap() >= 0.010);

    assert_eq!(events.next().as_deref(), Some("[DONE]"));
}

#[test]
fn generate_takes_put_like_post() {
    let (_rt, addr) = server();
    for method in ["POST", "PUT"] {
        let response = http(addr, method, "/generate", &[], r#"{"text": 5}"#);
        assert_eq!(response.status, 400, "{method}");
    }
}

/// A probe the scheduler answers is healthy, and `/health_generate` carries
/// the configured headers while `/health` does not.
#[test]
fn healthy_probe_carries_configured_headers() {
    let (rt, addr) = server_with(|args| {
        args.health_generate_headers = [
            ("x-custom-header".to_string(), "v".to_string()),
            ("x-rust-server".to_string(), "health".to_string()),
        ]
        .into();
    });
    let probe = |path: &'static str| {
        let client = std::thread::spawn(move || http(addr, "GET", path, &[], ""));
        // The probe is a generation; its output is the scheduler's sign of life.
        let rid = next_scheduler_request(&rt).rid().expect("probe rid");
        rt.push_output_with(&rid, &[1], done()).unwrap();
        client.join().unwrap()
    };
    let response = probe("/health_generate");
    assert_eq!(response.status, 200, "{}", response.body);
    assert_eq!(response.header("x-custom-header"), Some("v"));
    assert_eq!(response.header("x-rust-server"), Some("health"));

    let response = probe("/health");
    assert_eq!(response.status, 200, "{}", response.body);
    assert_eq!(response.header("x-custom-header"), None);
}

/// Refuses requests asking for more than one top logprob.
#[derive(Debug)]
struct TopLogprobsCap;

impl GenerateValidator for TopLogprobsCap {
    fn validate(&self, request: &GenerateRequest) -> Result<(), String> {
        if request.top_logprobs_num > 1 {
            return Err(format!("top_logprobs_num {}", request.top_logprobs_num));
        }
        Ok(())
    }
}

/// The launch's generate validator refuses at every ingress before
/// anything is submitted, a batch's valid items included: native
/// `/generate` with its 400 (a stream's error frame), and the OpenAI
/// routes with their 400, unary even for a stream, as Python answers the
/// tokenizer manager's `ValueError` before the stream starts.
#[test]
fn generate_validator_refuses_every_ingress_before_submission() {
    let [addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(addr)
        .server_args(|args| {
            args.skip_tokenizer_init = false;
            args.tokenizer_path = tokenizer_dir().into();
            args.served_model_name = "m".into();
        })
        .generate_validator(Arc::new(TopLogprobsCap))
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);
    let native = json!({"code": 400, "message": "top_logprobs_num 2"});
    let native_stream = json!({
        "code": 400,
        "message": "top_logprobs_num 2",
        "type": "invalid_request_error",
        "retryable": false,
    });
    let openai = json!({
        "object": "error",
        "message": "top_logprobs_num 2",
        "type": "BadRequestError",
        "param": null,
        "code": 400,
    });
    let generate = |text: Value, top_logprobs_num: Value, stream: bool| {
        json!({
            "text": text,
            "stream": stream,
            "return_logprob": true,
            "top_logprobs_num": top_logprobs_num,
        })
    };
    let completion = |stream: bool| json!({"model": "m", "prompt": [1, 2], "logprobs": 2, "max_tokens": 1, "stream": stream});
    for (uri, body, status, error) in [
        (
            "/generate",
            generate(json!(["a", "b"]), json!([1, 2]), false),
            400,
            &native,
        ),
        (
            "/generate",
            generate(json!(["a", "b"]), json!([1, 2]), true),
            200,
            &native_stream,
        ),
        (
            "/generate",
            generate(json!("a"), json!(2), false),
            400,
            &native,
        ),
        ("/v1/completions", completion(false), 400, &openai),
        ("/v1/completions", completion(true), 400, &openai),
    ] {
        let case = format!("{uri} {body}");
        let response = http(addr, "POST", uri, &[], &body.to_string());
        assert_eq!(response.status, status, "{case}");
        // A streamed refusal is chunked: its event is the first `data:` line.
        let first = response
            .body
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .unwrap_or(&response.body);
        let first: Value =
            serde_json::from_str(first).unwrap_or_else(|_| panic!("{case}: {first}"));
        // The OpenAI route's error is the whole body, as Python's is.
        let got = match uri {
            "/generate" => &first["error"],
            _ => &first,
        };
        assert_eq!(got, error, "{case}");
    }
    assert!(
        rt.next_request(Duration::from_millis(200)).is_none(),
        "an item reached the scheduler"
    );
}

/// A scheduler abort that carried a status fails the request with that
/// status, and the scheduler's message names the request by the client's
/// rid, not the suffixed one this server sent the scheduler: raw, or as
/// Python's `repr` (`{rid=}`) with the client's escaping and quoting kept.
#[test]
fn status_abort_names_the_client_rid() {
    let (rt, addr) = server();
    // (client rid, Python's `repr` of it)
    for (client_rid, client_repr) in [
        ("client-rid", "'client-rid'"),
        (r"back\slash", r"'back\\slash'"),
        ("new\nline", r"'new\nline'"),
        ("it's", r#""it's""#),
    ] {
        let (client, internal) = unary(&rt, addr, json!({"rid": client_rid, "input_ids": [1]}));
        assert_ne!(internal, client_rid, "the scheduler sees a suffixed rid");
        let suffix = &internal[client_rid.len()..];
        // `repr` of the suffixed rid: the suffix lands inside the quotes.
        let (body, quote) = client_repr.split_at(client_repr.len() - 1);
        let internal_repr = format!("{body}{suffix}{quote}");
        let details = OutputDetails {
            prompt_tokens: 3,
            finish_reason: Some(json!({
                "type": "abort",
                "message": format!("Request {internal} failed: req.rid={internal_repr}"),
                "status_code": 500,
                "err_type": null
            })),
            ..Default::default()
        };
        rt.push_output_with(&internal, &[], details).unwrap();
        let response = client.join().unwrap();
        let expected = format!("Request {client_rid} failed: req.rid={client_repr}");
        assert_eq!(response.status, 500, "client rid {client_rid:?}");
        assert_eq!(
            response.json()["error"]["message"],
            expected,
            "client rid {client_rid:?}"
        );
    }
}

/// What a hook is told of each request: every frame it annotated, and how
/// the request ended.
#[derive(Clone, Debug, Default)]
struct Recorded(Arc<std::sync::Mutex<Vec<Value>>>);

impl Recorded {
    fn log(&self) -> Vec<Value> {
        self.0.lock().unwrap().clone()
    }
}

impl sglang_server::GenerateHook for Recorded {
    fn start(
        &self,
        _: &GenerateRequest,
        _: &axum::http::HeaderMap,
        _: Option<&serde_json::Map<String, Value>>,
    ) -> Box<dyn sglang_server::GenerateFrameHook> {
        Box::new(self.clone())
    }
}

impl sglang_server::GenerateFrameHook for Recorded {
    fn annotate(
        &mut self,
        frame: &sglang_server::GenerateFrameInfo<'_>,
        keys: &mut sglang_server::GenerateFrameKeys,
    ) {
        let mut log = self.0.lock().unwrap();
        keys.meta_info.insert("hook_frame".into(), json!(log.len()));
        log.push(json!({
            "frame": {
                "finished": frame.finished,
                "decode_latency": frame.decode_latency.map(|latency| latency.as_secs_f64()),
            }
        }));
    }

    fn finished(&mut self, terminal: &sglang_server::GenerateTerminal<'_>) {
        let completion_tokens = terminal.meta_info.completion_tokens;
        self.0
            .lock()
            .unwrap()
            .push(json!({"completion_tokens": completion_tokens}));
    }

    fn failed(&mut self, status: axum::http::StatusCode, message: &str) {
        self.0
            .lock()
            .unwrap()
            .push(json!({"failed": [status.as_u16(), message]}));
    }
}

/// A server with the word tokenizer and `hook` on native generation.
fn hooked_server(hook: &Recorded) -> (SchedulerlessServer, SocketAddr) {
    let [addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(addr)
        .server_args(|args| {
            args.skip_tokenizer_init = false;
            args.tokenizer_path = tokenizer_dir().into();
        })
        .extensions(sglang_server_test::Extensions {
            generate_hook: Some(Arc::new(hook.clone())),
            abort_hook: None,
        })
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);
    (rt, addr)
}

/// A per-item error is surfaced with its `index` and doesn't end the batch;
/// `[DONE]` still waits for the other item. A package's hook hears of the
/// failure, and of the other item's end.
#[test]
fn per_item_error_carries_index() {
    let hook = Recorded::default();
    let (rt, addr) = hooked_server(&hook);
    // A client rid, so the items reach the scheduler as `r_0` and `r_1`.
    let body = json!({
        "rid": "r",
        "input_ids": [[1], [2]],
        "stream": true,
        "sampling_params": {"max_new_tokens": 4},
    });
    let mut events = Sse::post(addr, "/generate", &body);
    let mut rids = [next_scheduler_request(&rt), next_scheduler_request(&rt)]
        .map(|request| request.rid().expect("rid"));
    rids.sort();

    rt.push_error(&rids[0], "bad").unwrap();
    let v = events.next_json();
    assert_eq!(v["index"], 0);
    // A scheduler's error is this server's own failure, and worth a retry.
    assert_eq!(v["error"]["code"], 500);
    assert_eq!(v["error"]["type"], "internal_error");
    assert_eq!(v["error"]["retryable"], true);

    rt.push_output_with(&rids[1], &ids(&["ok"]), done())
        .unwrap();
    let v = events.next_json();
    assert_eq!(v["index"], 1);

    assert_eq!(events.next().as_deref(), Some("[DONE]"));
    let log = hook.log();
    let failed = log
        .iter()
        .find_map(|entry| entry.get("failed"))
        .expect("the failure");
    assert_eq!(failed[0], 500);
    assert!(
        failed[1]
            .as_str()
            .is_some_and(|message| message.contains("bad")),
        "{failed}"
    );
    assert!(log.contains(&json!({"completion_tokens": 1})), "{log:?}");
}

/// The scheduler's decode latency reaches the hook on a unary request's
/// terminal frame, in place of this frontend's own measure.
#[test]
fn scheduler_decode_latency_reaches_the_hook() {
    let hook = Recorded::default();
    let (rt, addr) = hooked_server(&hook);
    let (client, rid) = unary(&rt, addr, json!({"rid": "client-rid", "input_ids": [1]}));
    let details = OutputDetails {
        decode_latency: Some(0.25),
        ..done()
    };
    rt.push_output_with(&rid, &ids(&["ok"]), details).unwrap();
    let response = client.join().unwrap();
    assert_eq!(response.status, 200, "{}", response.body);
    assert_eq!(response.json()["meta_info"]["hook_frame"], 0);

    let frames: Vec<Value> = hook
        .log()
        .iter()
        .filter_map(|entry| entry.get("frame").cloned())
        .collect();
    assert_eq!(frames, [json!({"finished": true, "decode_latency": 0.25})]);
}

/// An omitted `max_new_tokens` reaches the scheduler as the launch default,
/// beneath the request's own value and the preferred sampling params; with
/// no default it is no limit.
#[test]
fn omitted_max_new_tokens_takes_the_launch_default() {
    let omitted = json!({"input_ids": [1, 2]});
    let explicit = json!({"input_ids": [1, 2], "sampling_params": {"max_new_tokens": 3}});
    for (default_max_new_tokens, preferred, body, expected) in [
        (Some(4096), None, &omitted, json!(4096)),
        (None, None, &omitted, Value::Null),
        (
            Some(4096),
            Some(json!({"max_new_tokens": 7})),
            &omitted,
            json!(7),
        ),
        (Some(4096), None, &explicit, json!(3)),
    ] {
        let (rt, addr) = server_with(|args| {
            args.default_max_new_tokens = default_max_new_tokens;
            args.preferred_sampling_params = preferred.map(PreferredSamplingParams);
        });
        let sent = body.clone();
        let client =
            std::thread::spawn(move || http(addr, "POST", "/generate", &[], &sent.to_string()));
        let request = next_scheduler_request(&rt);
        // The sampling params' first field on the ring is `max_new_tokens`.
        assert_eq!(header_json(&request)[8][0], expected, "{body}");
        rt.push_output_with(&request.rid().expect("rid"), &ids(&["ok"]), done())
            .unwrap();
        assert_eq!(client.join().unwrap().status, 200, "{body}");
    }
}

/// A scheduler abort that carries no status is the request's terminal
/// output, not a failure: a unary call answers 200 with the abort as its
/// finish reason, and nothing is aborted in return.
#[test]
fn plain_scheduler_abort_remains_a_terminal_output() {
    let (rt, addr) = server();
    let (client, rid) = unary(&rt, addr, json!({"rid": "plain-abort", "input_ids": [1]}));
    let abort =
        json!({"type": "abort", "message": "Aborted", "status_code": null, "err_type": null});
    let details = OutputDetails {
        finish_reason: Some(abort),
        ..Default::default()
    };
    rt.push_output_with(&rid, &[], details).unwrap();
    let response = client.join().unwrap();
    assert_eq!(response.status, 200, "{}", response.body);
    assert_eq!(
        response.json()["meta_info"]["finish_reason"]["type"],
        "abort"
    );
    assert!(
        rt.next_request(Duration::from_millis(200)).is_none(),
        "the scheduler was told to abort a request it ended"
    );
}

/// A package hook that tags each native frame with the request's `top_p`,
/// the scheduler's or this frontend's decode latency on a terminal frame,
/// and `first` on a request's first frame. It logs what it saw of each
/// request's start and end.
#[derive(Clone, Debug, Default)]
struct TopPHook {
    log: Arc<std::sync::Mutex<Vec<Value>>>,
}

struct TopPFrames {
    top_p: f64,
    first: bool,
    /// The rid, a header, the submitted `top_k` and the client's own
    /// sampling params, as the hook saw them at the start.
    started: Value,
    log: Arc<std::sync::Mutex<Vec<Value>>>,
}

impl sglang_server::GenerateHook for TopPHook {
    fn start(
        &self,
        request: &GenerateRequest,
        headers: &axum::http::HeaderMap,
        client_sampling_params: Option<&serde_json::Map<String, Value>>,
    ) -> Box<dyn sglang_server::GenerateFrameHook> {
        Box::new(TopPFrames {
            top_p: request.sampling_params.top_p,
            first: true,
            started: json!({
                "rid": request.rid.client_facing(),
                "header": headers.get("x-test").map(|value| value.to_str().unwrap()),
                "top_k": request.sampling_params.top_k,
                "client": client_sampling_params,
            }),
            log: self.log.clone(),
        })
    }
}

impl sglang_server::GenerateFrameHook for TopPFrames {
    fn annotate(
        &mut self,
        frame: &sglang_server::GenerateFrameInfo<'_>,
        keys: &mut sglang_server::GenerateFrameKeys,
    ) {
        assert_eq!(frame.finished, frame.decode_latency.is_some());
        keys.meta_info.insert("top_p".into(), json!(self.top_p));
        if let Some(latency) = frame.decode_latency {
            keys.meta_info
                .insert("decode_s".into(), json!(latency.as_secs_f64()));
        }
        if std::mem::take(&mut self.first) {
            keys.frame.insert("first".into(), json!(true));
        }
    }

    fn finished(&mut self, terminal: &sglang_server::GenerateTerminal<'_>) {
        self.log.lock().unwrap().push(json!({
            "started": self.started,
            "completion_tokens": terminal.meta_info.completion_tokens,
            "queue_time": terminal.request_stats.map(|stats| stats["queue_time"].clone()),
            "decode_latency": terminal.decode_latency.map(|latency| latency.as_secs_f64()),
            "output_token_logprobs": terminal.output_token_logprobs,
        }));
    }
}

/// A hook's keys reach every native frame of every request — unary,
/// cumulative and incremental streams, single and batch — each request's
/// hook started from its own sampling params, with the scheduler's decode
/// latency for a unary request and this frontend's for a streamed one;
/// without a hook the frames are the schema's. A finished request's stats
/// reach its terminal frame only, after the schema's keys and before the
/// hook's, with the decode throughput when metrics are on; a request whose
/// terminal output carries no stats reports neither. The hook starts from
/// each item's request, the HTTP headers and the sampling params the client
/// sent for it (without the launch-time preferred ones), and sees each
/// terminal frame once. An HTTP layer's earlier [`RequestStart`] lengthens
/// every terminal `e2e_latency` but neither decode latency. Every frame
/// carries the listener's DP rank (null without data parallelism), a status
/// abort's terminal stream frame included.
#[test]
fn generate_hook_keys_reach_every_frame() {
    let metrics_dir = std::env::temp_dir().join(format!("sglang-it-hook-{}", std::process::id()));
    std::fs::create_dir_all(&metrics_dir).unwrap();
    let layer_work = Duration::from_secs(5);
    for (stream, incremental, hook, batch, metrics) in [
        (false, false, true, true, true),
        (false, false, false, false, false),
        (true, false, true, true, false),
        (true, false, false, false, true),
        (true, true, true, true, true),
        (true, true, false, false, true),
    ] {
        let case = format!(
            "stream={stream} incremental={incremental} hook={hook} batch={batch} metrics={metrics}"
        );
        // A hooked case also runs as DP rank 1, which frames report; the
        // rest without DP, whose frames report null.
        let dp_rank: u32 = if hook { 1 } else { 0 };
        let top_p_hook = TopPHook::default();
        let layer: HttpLayer = Arc::new(move |app: axum::Router| {
            app.layer(axum::middleware::from_fn(
                move |mut request: axum::extract::Request, next: axum::middleware::Next| async move {
                    let start = Instant::now().checked_sub(layer_work).unwrap();
                    request.extensions_mut().insert(RequestStart(start));
                    next.run(request).await
                },
            ))
        });
        let [addr] = free_loopback_addrs();
        let dir = metrics_dir.to_string_lossy().into_owned();
        let rt = SchedulerlessServer::builder(addr)
            .server_args(|args| {
                args.skip_tokenizer_init = false;
                args.tokenizer_path = tokenizer_dir().into();
                args.incremental_streaming_output = incremental;
                args.metrics = metrics.then(|| sglang_server::MetricsConfig {
                    multiproc_dir: dir,
                    labels: Default::default(),
                    priority_label: false,
                    bucket_time_to_first_token: None,
                    bucket_inter_token_latency: None,
                    bucket_e2e_request_latency: None,
                    bucket_prompt_tokens: None,
                    bucket_generation_tokens: None,
                });
                args.preferred_sampling_params = Some(PreferredSamplingParams(json!({"top_k": 7})));
                args.deployment_info.dp_size = if hook { 2 } else { 1 };
                args.deployment_info.dp_rank = Some(dp_rank);
            })
            .extensions(sglang_server_test::Extensions {
                generate_hook: hook
                    .then(|| Arc::new(top_p_hook.clone()) as Arc<dyn sglang_server::GenerateHook>),
                abort_hook: None,
            })
            .http_layer(layer)
            .start()
            .expect("start runtime");
        take_startup_freeze_gc(&rt);

        let items = if batch { 2 } else { 1 };
        let (text, sampling_params) = if batch {
            (json!(["a", "b"]), json!([{"top_p": 0.5}, {"top_p": 0.25}]))
        } else {
            (json!("a"), json!({"top_p": 0.5}))
        };
        let body = json!({
            "rid": "r",
            "text": text,
            "stream": stream,
            "sampling_params": sampling_params,
        });
        // Every item's delta, then every terminal: a stream emits two frames
        // per item. One output token logprob per output; the first item's
        // terminal output carries its stats.
        let delta = || OutputDetails {
            output_token_logprobs: vec![(-0.5, 1)],
            ..Default::default()
        };
        let terminal = |item: usize| {
            // A streamed batch's second item ends on a status abort, still a frame.
            let finish_reason = if stream && item == 1 {
                json!({"type": "abort", "message": "Aborted", "status_code": 500, "err_type": null})
            } else {
                json!({"type": "length", "length": 1})
            };
            OutputDetails {
                finish_reason: Some(finish_reason),
                // The scheduler times non-streaming requests only.
                decode_latency: (!stream).then_some(0.25),
                request_stats: (item == 0)
                    .then(|| json!({"num_retractions": 1, "queue_time": 0.5})),
                ..delta()
            }
        };
        // A client rid `r` reaches the scheduler as `r`, or `r_0` and `r_1`.
        let submitted = |rt: &SchedulerlessServer| {
            let mut rids: Vec<String> = (0..items)
                .map(|_| next_scheduler_request(rt).rid().expect("rid"))
                .collect();
            rids.sort();
            rids
        };
        let frames: Vec<Value> = if stream {
            let mut events = Sse::post_with(addr, "/generate", &["x-test: v"], &body);
            assert_eq!(
                events.content_type.as_deref(),
                Some("text/event-stream; charset=utf-8"),
                "{case}"
            );
            let rids = submitted(&rt);
            for rid in &rids {
                rt.push_output_with(rid, &ids(&["a"]), delta()).unwrap();
            }
            // A streamed item's decode latency runs from this frontend's
            // first output, so let the stream take every delta first.
            let mut frames: Vec<Value> = (0..items).map(|_| events.next_json()).collect();
            for (item, rid) in rids.iter().enumerate() {
                rt.push_output_with(rid, &ids(&["b"]), terminal(item))
                    .unwrap();
            }
            while let Some(data) = events.next() {
                if data != "[DONE]" {
                    frames.push(serde_json::from_str(&data).unwrap());
                }
            }
            frames
        } else {
            let sent = body.clone();
            let client = std::thread::spawn(move || {
                http(addr, "POST", "/generate", &["x-test: v"], &sent.to_string())
            });
            let rids = submitted(&rt);
            for (item, rid) in rids.iter().enumerate() {
                rt.push_output_with(rid, &ids(&["a"]), delta()).unwrap();
                rt.push_output_with(rid, &ids(&["b"]), terminal(item))
                    .unwrap();
            }
            let response = client.join().unwrap();
            assert_eq!(response.status, 200, "{case}: {}", response.body);
            match response.json() {
                Value::Array(frames) => frames,
                frame => vec![frame],
            }
        };

        // A stream takes each delta before its terminal output arrives.
        assert_eq!(frames.len(), items * if stream { 2 } else { 1 }, "{case}");
        let mut seen = [0usize; 2];
        for (position, frame) in frames.iter().enumerate() {
            // A unary batch is an array in request order, without `index`;
            // a single request has none either.
            let index = frame["index"]
                .as_u64()
                .map_or(if batch { position } else { 0 }, |i| i as usize);
            let meta = frame["meta_info"].as_object().unwrap();
            let terminal = !meta["finish_reason"].is_null();
            assert_eq!(
                meta.get("dp_rank"),
                Some(&json!(hook.then_some(dp_rank))),
                "{case}"
            );
            let keys: Vec<&str> = meta.keys().map(String::as_str).collect();
            let e2e_latency = meta.get("e2e_latency").and_then(Value::as_f64);
            assert_eq!(e2e_latency.is_some(), terminal, "{case}");
            if let Some(e2e_latency) = e2e_latency {
                assert!(
                    e2e_latency >= layer_work.as_secs_f64(),
                    "{case}: {e2e_latency}"
                );
            }
            let stats = terminal && index == 0;
            assert_eq!(meta.get("num_retractions").is_some(), stats, "{case}");
            assert_eq!(
                meta.get("decode_throughput").is_some(),
                stats && metrics,
                "{case}"
            );
            if stats {
                let at = keys
                    .iter()
                    .position(|key| *key == "num_retractions")
                    .unwrap();
                assert_eq!(
                    keys[at..at + 2],
                    ["num_retractions", "queue_time"],
                    "{case}"
                );
                assert_eq!(meta["queue_time"], 0.5, "{case}");
                if metrics {
                    assert_eq!(keys[at + 2], "decode_throughput", "{case}");
                    // Two tokens, 0.25 s apart by the scheduler's clock.
                    if !stream {
                        assert_eq!(meta["decode_throughput"], 4.0, "{case}");
                    }
                }
            }
            if !hook {
                assert!(meta.get("top_p").is_none() && frame.get("first").is_none());
                continue;
            }
            assert_eq!(meta["top_p"], [0.5, 0.25][index], "{case}");
            assert_eq!(frame.get("first").is_some(), seen[index] == 0, "{case}");
            assert_eq!(meta.get("decode_s").is_some(), terminal, "{case}");
            if terminal && !stream {
                assert_eq!(meta["decode_s"], 0.25, "{case}");
            }
            // A streamed request's decode latency runs from its first
            // output, which the layer's work came before.
            if terminal && stream {
                let decode_s = meta["decode_s"].as_f64().unwrap();
                assert!(
                    e2e_latency.unwrap() - decode_s >= layer_work.as_secs_f64(),
                    "{case}: {e2e_latency:?} - {decode_s}"
                );
            }
            if stats {
                let stats_at = keys.iter().position(|key| *key == "num_retractions");
                let hook_at = keys.iter().position(|key| *key == "top_p");
                assert!(stats_at < hook_at, "{case}");
            }
            seen[index] += 1;
        }
        let mut log = std::mem::take(&mut *top_p_hook.log.lock().unwrap());
        if !hook {
            assert!(log.is_empty(), "{case}");
            continue;
        }
        log.sort_by_key(|entry| entry["started"]["rid"].as_str().unwrap().to_owned());
        assert_eq!(log.len(), items, "{case}");
        for (item, entry) in log.iter().enumerate() {
            let rid = if batch {
                format!("r_{item}")
            } else {
                "r".into()
            };
            let client = if batch {
                sampling_params[item].clone()
            } else {
                sampling_params.clone()
            };
            let expected = json!({
                "started": {"rid": rid, "header": "v", "top_k": 7, "client": client},
                // Two tokens, one per output.
                "completion_tokens": 2,
                "queue_time": (item == 0).then_some(0.5),
                "decode_latency": entry["decode_latency"],
                // Both outputs', though an incremental terminal frame
                // carries only its own.
                "output_token_logprobs": 2,
            });
            assert_eq!(entry, &expected, "{case}");
            if !stream {
                assert_eq!(entry["decode_latency"], 0.25, "{case}");
            }
        }
    }
    let _ = std::fs::remove_dir_all(&metrics_dir);
}

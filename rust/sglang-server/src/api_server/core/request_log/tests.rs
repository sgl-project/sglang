use super::*;
use crate::message::finish_reason::{AbortReason, FinishKind, FinishReason};
use crate::message::ids::Rid;
use crate::message::request::{GenerationPolicy, into_requests};
use crate::message::response::ChunkEvent;
use sglang_api_types::api::v1 as api;

/// Python's lines for the same requests, from
/// `test/registered/unit/utils/test_rust_request_log.py`, which checks this
/// file against Python's `RequestLogger`.
const FIXTURE: &str = include_str!("../../../../testdata/request_log.json");

fn config(settings: &Value) -> RequestLogConfig {
    let strings = |key: &str| -> Vec<String> {
        serde_json::from_value(settings[key].clone()).expect("a string list")
    };
    RequestLogConfig {
        max_length: settings["max_length"].as_u64().unwrap() as usize,
        skip_names: strings("skip_names"),
        out_skip_names: strings("out_skip_names"),
        class_name: settings["class_name"].as_str().unwrap().into(),
        fields: serde_json::from_value(settings["fields"].clone()).unwrap(),
        batch_item_fields: strings("batch_item_fields"),
        batch_shared_fields: strings("batch_shared_fields"),
        headers: strings("headers"),
        exceeded_ms: settings["exceeded_ms"].as_i64().unwrap(),
        omit_custom_params: Vec::new(),
    }
}

fn header_map(headers: &Value) -> http::HeaderMap {
    let mut map = http::HeaderMap::new();
    for (name, value) in headers.as_object().unwrap() {
        map.insert(
            http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
            http::HeaderValue::from_str(value.as_str().unwrap()).unwrap(),
        );
    }
    map
}

/// A body through the native `/generate` path: snapshot, then normalization.
fn receive_body(
    log: &Arc<RequestLog>,
    body: &Value,
    headers: &http::HeaderMap,
) -> Vec<RequestRecord> {
    let snapshot = log.snapshot_body(body, headers);
    let decoded: api::GenerateRequest = serde_json::from_value(body.clone()).unwrap();
    let (requests, batch) = into_requests(decoded, &GenerationPolicy::default()).unwrap();
    log.receive(&requests, ReceivedAs::Body(snapshot), batch)
}

/// Every fixture case renders exactly Python's `Receive` and `Finish` lines at
/// every level it lists.
#[test]
fn lines_match_python_request_logger() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let mut checked = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        for (level, expected) in case["expected"].as_object().unwrap() {
            let log = RequestLog::for_test(config(&fixture["levels"][level]));
            let records = receive_body(&log, &case["body"], &header_map(&case["headers"]));
            let item = case["batch_item"].as_u64().unwrap_or(0) as usize;
            let record = &records[item];
            let out = match case.get("core_output") {
                // The Rust server's own frame of that output.
                Some(output) => python_out(
                    core_output(output),
                    &record.rid,
                    output["e2e_latency"].as_f64().unwrap(),
                ),
                None => case["out"].as_object().unwrap().clone(),
            };
            assert_eq!(Value::Object(out.clone()), case["out"], "{name}: out");
            log.logger.finished(&record.obj, &record.headers, &out, 1.0);
            let lines = log.logged_lines();
            // A batch's `Receive` is best effort and not compared.
            if !expected["receive"].is_null() {
                assert_eq!(
                    lines[0], expected["receive"],
                    "{name} at level {level}: Receive"
                );
            }
            assert_eq!(
                lines[1], expected["finish"],
                "{name} at level {level}: Finish"
            );
            checked += 1;
        }
    }
    assert!(checked >= 21, "the fixture lost cases: {checked}");
}

fn core_output(output: &Value) -> CoreOutput {
    CoreOutput {
        text: output["text"].as_str().unwrap().into(),
        token_ids: serde_json::from_value(output["token_ids"].clone()).unwrap(),
        prompt_tokens: output["prompt_tokens"].as_u64().unwrap() as u32,
        completion_tokens: output["completion_tokens"].as_u64().unwrap(),
        cached_tokens: output["cached_tokens"].as_u64().unwrap() as u32,
        finish_reason: serde_json::from_value(output["finish_reason"].clone()).unwrap(),
        weight_version: Some(Arc::from("default")),
        ..Default::default()
    }
}

/// Python `repr(float)` for the values a request or `meta_info` carries.
#[test]
fn floats_print_as_python_repr() {
    for (value, python) in [
        (0.0, "0.0"),
        (-0.0, "-0.0"),
        (1.0, "1.0"),
        (0.1, "0.1"),
        (1.0 / 3.0, "0.3333333333333333"),
        (123.456, "123.456"),
        (0.0001, "0.0001"),
        (1e-05, "1e-05"),
        (1.5e-07, "1.5e-07"),
        (1e15, "1000000000000000.0"),
        (1e16, "1e+16"),
        (2.5e20, "2.5e+20"),
        (1234567890.1234567, "1234567890.1234567"),
        (-1e300, "-1e+300"),
    ] {
        let mut out = String::new();
        write_float(&mut out, value);
        assert_eq!(out, python, "{value:e}");
    }
}

fn chunk(rid: &Rid, text: &str, ids: &[i64], finish: Option<FinishReason>) -> ChunkEvent {
    ChunkEvent {
        rid: rid.clone(),
        token_ids: ids.to_vec(),
        text: text.into(),
        completion_tokens: ids.len() as u64,
        prompt_tokens: 3,
        finish_reason: finish,
        ..Default::default()
    }
}

fn level1(fixture: &Value) -> RequestLogConfig {
    config(&fixture["levels"]["1"])
}

/// The record follows the call: `Finish` (abort included) comes from the
/// scheduler's terminal output.
#[test]
fn terminal_output_logs_finish() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let log = RequestLog::for_test(level1(&fixture));
    let body = serde_json::json!({"rid": "r-abort", "text": "hi"});
    let mut record = receive_body(&log, &body, &http::HeaderMap::new()).remove(0);
    let rid = Rid::from_client("r-abort");
    record.observe(&ResponseItem::Frame(chunk(&rid, "a", &[1], None)));
    let abort = FinishReason::from(FinishKind::Abort(Box::new(AbortReason {
        message: Some("Abort in waiting queue".into()),
        status_code: Some(503),
        err_type: None,
    })));
    record.observe(&ResponseItem::Done(chunk(&rid, "b", &[2], Some(abort))));

    let lines = log.logged_lines();
    assert!(lines[0].starts_with("Receive: obj=GenerateReqInput(rid='r-abort', "));
    let finish = &lines[1];
    assert!(finish.starts_with("Finish: obj=GenerateReqInput(rid='r-abort', "));
    assert!(
        finish.contains(", out={'meta_info': {'id': 'r-abort', ")
            && finish.contains(
                "'finish_reason': {'type': 'abort', 'message': 'Abort in waiting queue', \
                 'status_code': 503, 'err_type': None}"
            ),
        "{finish}"
    );
}

/// With `--incremental-streaming-output`, a stream's `Finish` shows its
/// terminal frame: the last delta, with the cumulative completion count.
#[test]
fn incremental_stream_logs_last_delta() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let log = Arc::new(RequestLog {
        logger: RequestLogger::new(
            config(&fixture["levels"]["3"]),
            LogSink::Memory(Mutex::default()),
        ),
        default_priority: None,
        weight_version: Some(Arc::from("default")),
        incremental_streaming_output: true,
    });
    let body = serde_json::json!({"rid": "r-inc", "text": "hi", "stream": true});
    let mut record = receive_body(&log, &body, &http::HeaderMap::new()).remove(0);
    let rid = Rid::from_client("r-inc");
    record.observe(&ResponseItem::Frame(chunk(&rid, "a", &[1], None)));
    record.observe(&ResponseItem::Done(chunk(&rid, "b", &[2], None)));

    let finish = log.logged_lines().remove(1);
    assert!(
        finish.contains(", out={'text': 'b', 'output_ids': [2], 'meta_info': {'id': 'r-inc', ")
            && finish.contains("'completion_tokens': 2, "),
        "{finish}"
    );
}

/// `SGLANG_LOG_REQUEST_EXCEEDED_MS` logs only the slow finishes.
#[test]
fn exceeded_ms_skips_fast_finishes() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let log = RequestLog::for_test(RequestLogConfig {
        exceeded_ms: 500,
        ..level1(&fixture)
    });
    let logger = &log.logger;
    let out = Map::new();
    logger.finished(&PyObj::default(), &[], &out, 0.499);
    logger.finished(&PyObj::default(), &[], &out, 0.5);
    let lines = log.logged_lines();
    assert_eq!(lines.len(), 1);
    assert!(lines[0].starts_with("Finish: "));
}

/// A key the server itself adds to `custom_params` never reaches a line; the
/// client's own custom params still do.
#[test]
fn server_added_custom_params_are_not_logged() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let log = RequestLog::for_test(RequestLogConfig {
        omit_custom_params: vec!["server_only".into()],
        ..level1(&fixture)
    });
    let body = serde_json::json!({
        "rid": ["a", "b"],
        "text": ["x", "y"],
        "sampling_params": [
            {"custom_params": {"server_only": "srv-added-1", "client": 1}},
            {"custom_params": {"server_only": "srv-added-2"}},
        ],
    });
    let mut records = receive_body(&log, &body, &http::HeaderMap::new());
    let receive = log.logged_lines().remove(0);
    assert!(!receive.contains("server_only"), "{receive}");
    assert!(!receive.contains("srv-added"), "{receive}");
    assert!(receive.contains("'client': 1"), "{receive}");
    let rid = Rid::from_client("a");
    records[0].observe(&ResponseItem::Done(chunk(&rid, "z", &[3], None)));
    let finish = log.logged_lines().remove(0);
    assert!(!finish.contains("srv-added"), "{finish}");
    assert!(finish.contains("'client': 1"), "{finish}");
}

/// A request an HTTP layer held before `/generate` (work an `HttpLayer` does
/// before the handler runs) counts that wait in its `Finish` latency, so
/// `SGLANG_LOG_REQUEST_EXCEEDED_MS` doesn't drop it as fast.
#[test]
fn finish_latency_counts_the_wait_before_generate() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let log = RequestLog::for_test(RequestLogConfig {
        exceeded_ms: 1000,
        ..level1(&fixture)
    });
    let body = serde_json::json!({"rid": "a", "text": "x"});
    let snapshot = log
        .snapshot_body(&body, &http::HeaderMap::new())
        .map(|snapshot| {
            snapshot.started_at(Some(
                std::time::Instant::now() - std::time::Duration::from_secs(2),
            ))
        });
    let decoded: api::GenerateRequest = serde_json::from_value(body).unwrap();
    let (requests, batch) = into_requests(decoded, &GenerationPolicy::default()).unwrap();
    let mut records = log.receive(&requests, ReceivedAs::Body(snapshot), batch);
    log.logged_lines();
    records[0].observe(&ResponseItem::Done(chunk(
        &Rid::from_client("a"),
        "z",
        &[3],
        None,
    )));
    let lines = log.logged_lines();
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].starts_with("Finish: "), "{lines:?}");
}

/// A batch logs one `Receive` with every rid and one `Finish` per item.
#[test]
fn batch_items_log_separately() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let log = RequestLog::for_test(level1(&fixture));
    let body = serde_json::json!({
        "rid": ["a", "b"],
        "text": ["x", "y"],
        "sampling_params": [{"max_new_tokens": 1}, {"max_new_tokens": 2}],
    });
    let mut records = receive_body(&log, &body, &http::HeaderMap::new());
    let receive = log.logged_lines().remove(0);
    assert!(receive.contains("rid=['a', 'b'], "), "{receive}");
    assert!(
        receive.contains("sampling_params=[{'max_new_tokens': 1}, {'max_new_tokens': 2}], "),
        "{receive}"
    );
    let rid = Rid::from_client("b");
    records[1].observe(&ResponseItem::Done(chunk(&rid, "z", &[3], None)));
    let finish = log.logged_lines().remove(0);
    assert!(finish.contains("(rid='b', "), "{finish}");
    assert!(
        finish.contains("sampling_params={'max_new_tokens': 2}, "),
        "{finish}"
    );
}

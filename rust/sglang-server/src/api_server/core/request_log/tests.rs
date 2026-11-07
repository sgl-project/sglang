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
    let snapshot = log.snapshot_body(body, None, headers);
    let decoded: api::GenerateRequest = serde_json::from_value(body.clone()).unwrap();
    let (requests, batch) = into_requests(decoded, &GenerationPolicy::default()).unwrap();
    log.receive(&requests, ReceivedAs::Body(snapshot), batch, true)
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
            let log = RequestLog::for_test(Some(config(&fixture["levels"][level])), false);
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
            log.logger
                .as_ref()
                .unwrap()
                .finished(&record.obj, &record.headers, &out, 1.0);
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

/// The record follows the call: `Finish` (abort included) and the crash-dump
/// entry come from the scheduler's terminal output.
#[test]
fn terminal_output_logs_finish_and_keeps_the_request() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let log = RequestLog::for_test(Some(level1(&fixture)), true);
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

    let snapshot: Value = serde_json::from_slice(&log.crash_dump_snapshot().unwrap()).unwrap();
    assert_eq!(snapshot["unfinished"], serde_json::json!([]));
    let finished = &snapshot["finished"][0];
    assert_eq!(snapshot["bodies"][finished["body"].to_string()], body);
    assert_eq!(finished["batch_index"], Value::Null);
    assert_eq!(finished["rid"], "r-abort");
    assert_eq!(finished["out"]["text"], "ab");
    assert_eq!(finished["out"]["output_ids"], serde_json::json!([1, 2]));
    assert!(finished["finished_time"].as_f64() >= finished["created_time"].as_f64());
}

/// With `--incremental-streaming-output`, a stream's `Finish` shows its
/// terminal frame: the last delta, with the cumulative completion count.
#[test]
fn incremental_stream_logs_last_delta() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let log = Arc::new(RequestLog {
        logger: Some(RequestLogger::new(
            config(&fixture["levels"]["3"]),
            LogSink::Memory(Mutex::default()),
        )),
        recorder: None,
        server_added_custom_params: Vec::new(),
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
    let log = RequestLog::for_test(
        Some(RequestLogConfig {
            exceeded_ms: 500,
            ..level1(&fixture)
        }),
        false,
    );
    let logger = log.logger.as_ref().unwrap();
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
    let log = RequestLog::for_test_omitting(Some(level1(&fixture)), false, &["server_only"]);
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

/// A crash dump leaves the server-added `custom_params` keys out of a body
/// that carries them, whole or per batch item, and keeps the client's own;
/// a body without them is still written as it came.
#[test]
fn server_added_custom_params_are_not_dumped() {
    let log = RequestLog::for_test_omitting(None, true, &["server_only"]);
    let recorder = log.recorder.as_ref().unwrap();
    let dump = |raw: &'static [u8], rid: &str| {
        let body: Value = serde_json::from_slice(raw).unwrap();
        let raw = RawBody {
            bytes: Bytes::from_static(raw),
            overrides: None,
        };
        let snapshot = log.snapshot_body(&body, Some(raw), &http::HeaderMap::new());
        let request = DumpRequest {
            body: snapshot.unwrap().dump.unwrap(),
            batch_index: None,
            rid: rid.into(),
        };
        recorder.register(request, 1.0, Arc::default());
    };
    dump(
        br#"{"rid": ["a", "b"], "text": ["x", "y"], "sampling_params": [
            {"custom_params": {"server_only": "srv-added-1", "client": 1}},
            {"custom_params": {"server_only": "srv-added-2"}}]}"#,
        "a",
    );
    let untouched: &'static [u8] =
        br#"{ "rid" : "c", "sampling_params": {"custom_params": {"client": 2}} }"#;
    dump(untouched, "c");
    let snapshot = recorder.snapshot(2.0);
    let text = String::from_utf8(snapshot.clone()).unwrap();
    assert!(!text.contains("server_only"), "{text}");
    assert!(!text.contains("srv-added"), "{text}");
    let parsed: Value = serde_json::from_slice(&snapshot).unwrap();
    assert_eq!(
        parsed["bodies"]["0"]["sampling_params"][0]["custom_params"],
        serde_json::json!({"client": 1})
    );
    assert!(
        text.contains(std::str::from_utf8(untouched).unwrap()),
        "{text}"
    );
}

/// A request an HTTP layer held before `/generate` (work an `HttpLayer` does
/// before the handler runs) counts that wait in its `Finish` latency, so
/// `SGLANG_LOG_REQUEST_EXCEEDED_MS` doesn't drop it as fast.
#[test]
fn finish_latency_counts_the_wait_before_generate() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let log = RequestLog::for_test(
        Some(RequestLogConfig {
            exceeded_ms: 1000,
            ..level1(&fixture)
        }),
        false,
    );
    let body = serde_json::json!({"rid": "a", "text": "x"});
    let snapshot = log
        .snapshot_body(&body, None, &http::HeaderMap::new())
        .map(|snapshot| {
            snapshot.started_at(Some(
                std::time::Instant::now() - std::time::Duration::from_secs(2),
            ))
        });
    let decoded: api::GenerateRequest = serde_json::from_value(body).unwrap();
    let (requests, batch) = into_requests(decoded, &GenerationPolicy::default()).unwrap();
    let mut records = log.receive(&requests, ReceivedAs::Body(snapshot), batch, true);
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

/// A batch logs one `Receive` with every rid and one `Finish` per item, and
/// each item's dump names its index in the shared body.
#[test]
fn batch_items_log_and_dump_separately() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let log = RequestLog::for_test(Some(level1(&fixture)), true);
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

    // Both items name one copy of the body.
    let snapshot: Value = serde_json::from_slice(&log.crash_dump_snapshot().unwrap()).unwrap();
    assert_eq!(snapshot["bodies"].as_object().unwrap().len(), 1);
    let (finished, unfinished) = (&snapshot["finished"][0], &snapshot["unfinished"][0]);
    assert_eq!(finished["batch_index"], 1);
    assert_eq!(unfinished["batch_index"], 0);
    assert_eq!(unfinished["rid"], "a");
    assert_eq!(finished["body"], unfinished["body"]);
    assert_eq!(snapshot["bodies"][finished["body"].to_string()], body);
}

fn recorded(recorder: &CrashRecorder) -> (Vec<String>, Vec<Value>) {
    let snapshot: Value = serde_json::from_slice(&recorder.snapshot(0.0)).unwrap();
    let rids = |key: &str| {
        snapshot[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|record| record["rid"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    };
    (
        rids("finished"),
        snapshot["unfinished"].as_array().unwrap().clone(),
    )
}

fn dump_request(recorder: &CrashRecorder, rid: &str) -> DumpRequest {
    DumpRequest {
        body: recorder.body(RawBody {
            bytes: Bytes::from_static(b"{}"),
            overrides: None,
        }),
        batch_index: None,
        rid: rid.into(),
    }
}

/// Python's window: a finished request stays five minutes after it finished;
/// the health probe (`log_metrics=False`) is never kept; an unfinished one
/// carries what it generated so far, and leaves when its call is dropped.
#[test]
fn recorder_keeps_five_minutes_and_the_unfinished() {
    let recorder = CrashRecorder::default();
    let output = || Arc::new(Mutex::new(CoreOutput::default()));
    let first = recorder.register(dump_request(&recorder, "first"), 1.0, output());
    let probe = recorder.register(dump_request(&recorder, "probe"), 1.0, output());
    let partial = output();
    lock(&partial).text = "so far".into();
    lock(&partial).token_ids = vec![4, 5];
    let _running = recorder.register(dump_request(&recorder, "running"), 1.0, partial);
    let dropped = recorder.register(dump_request(&recorder, "dropped"), 1.0, output());

    recorder.finish(first, true, &Map::new(), 10.0);
    recorder.finish(probe, false, &Map::new(), 10.0);
    recorder.unregister(dropped);
    let (finished, unfinished) = recorded(&recorder);
    assert_eq!(finished, ["first"]);
    assert_eq!(unfinished.len(), 1);
    assert_eq!(unfinished[0]["rid"], "running");
    assert_eq!(
        unfinished[0]["out"],
        serde_json::json!({"text": "so far", "output_ids": [4, 5]})
    );

    // Just inside the window, then out of it once a later request finishes.
    let second = recorder.register(dump_request(&recorder, "second"), 2.0, output());
    recorder.finish(second, true, &Map::new(), 309.9);
    assert_eq!(recorded(&recorder).0, ["first", "second"]);
    let third = recorder.register(dump_request(&recorder, "third"), 3.0, output());
    recorder.finish(third, true, &Map::new(), 310.0);
    assert_eq!(recorded(&recorder).0, ["second", "third"]);
}

/// A call dropped unfinished after admission (the client went away) is kept
/// with what it generated, as Python records an aborted request; one dropped
/// before the scheduler saw it (preprocessing failed) is not.
#[test]
fn dropped_records_leave_the_unfinished_set() {
    let log = RequestLog::for_test(None, true);
    let snapshot = |log: &RequestLog| -> Value {
        serde_json::from_slice(&log.crash_dump_snapshot().unwrap()).unwrap()
    };
    let body = serde_json::json!({"rid": "gone", "text": "hi"});
    let mut admitted = receive_body(&log, &body, &http::HeaderMap::new()).remove(0);
    admitted.admit();
    admitted.observe(&ResponseItem::Frame(chunk(
        &Rid::from_client("gone"),
        "so",
        &[4],
        None,
    )));
    let rejected = receive_body(
        &log,
        &serde_json::json!({"rid": "rejected", "text": "hi"}),
        &http::HeaderMap::new(),
    )
    .remove(0);
    assert_eq!(snapshot(&log)["unfinished"].as_array().unwrap().len(), 2);

    drop(admitted);
    drop(rejected);
    let snapshot = snapshot(&log);
    assert_eq!(snapshot["unfinished"], serde_json::json!([]));
    let finished = snapshot["finished"].as_array().unwrap();
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0]["rid"], "gone");
    assert_eq!(finished[0]["out"]["text"], "so");
    assert_eq!(finished[0]["out"]["output_ids"], serde_json::json!([4]));
    assert_eq!(
        finished[0]["out"]["meta_info"]["finish_reason"],
        Value::Null
    );
}

/// A crash dump keeps the body's bytes as they came. With header overrides
/// it applies them only when written, and the dump is byte for byte the one
/// serializing the overridden body on every request gave.
#[test]
fn dump_keeps_the_raw_body() {
    // Spacing, escapes and number spellings that serializing changes.
    let raw = Bytes::from_static(
        br#"{ "rid" : "client", "text": "\u00e9", "input_ids": [1, 2,3],
            "sampling_params": {"temperature": 7E-1, "top_p": 1.0e0}, "stream": false }"#,
    );
    let dump = |body: &Value, raw: Option<RawBody>| -> Vec<u8> {
        let log = RequestLog::for_test(None, true);
        let snapshot = log.snapshot_body(body, raw, &http::HeaderMap::new());
        let recorder = log.recorder.as_ref().unwrap();
        let request = DumpRequest {
            body: snapshot.unwrap().dump.unwrap(),
            batch_index: None,
            rid: "r".into(),
        };
        recorder.register(request, 1.0, Arc::default());
        recorder.snapshot(2.0)
    };
    let mut body: Value = serde_json::from_slice(&raw).unwrap();
    let exact = RawBody {
        bytes: raw.clone(),
        overrides: None,
    };
    let expected = [br#"{"bodies":{"0":"#.as_slice(), &raw, b"},"].concat();
    assert!(dump(&body, Some(exact)).starts_with(&expected));

    // The native handler: one override replaces a field, one adds a field.
    let overrides = serde_json::json!({"rid": "router", "bootstrap_room": i64::MAX});
    let overrides = overrides.as_object().unwrap();
    for (field, value) in overrides {
        body.as_object_mut()
            .unwrap()
            .insert(field.clone(), value.clone());
    }
    let overridden = RawBody {
        bytes: raw,
        overrides: Some(overrides.clone()),
    };
    // `None` serializes `body`, as every overridden request used to.
    assert_eq!(
        String::from_utf8(dump(&body, Some(overridden))).unwrap(),
        String::from_utf8(dump(&body, None)).unwrap()
    );
}

/// A request an adapter built keeps its media in the dump, in the fields
/// Python's request takes them in.
#[test]
fn built_request_keeps_its_media() {
    use crate::message::multimodal::MediaHints;
    use crate::message::request::MmData;

    let log = RequestLog::for_test(None, true);
    let request = GenerateRequest {
        rid: Rid::from_client("with-media"),
        text: Some("look".into()),
        mm: Some(Box::new(MmData {
            image_data: vec![MmItem::Source("data:image/png;base64,AAAA".into())],
            video_data: vec![MmItem::Ref {
                url: "file:///v.mp4".into(),
                hints: MediaHints { fps: Some(2.0) },
            }],
            ..Default::default()
        })),
        ..Default::default()
    };
    let _record = log
        .receive(&[request], ReceivedAs::without_headers(), false, true)
        .remove(0);
    let snapshot: Value = serde_json::from_slice(&log.crash_dump_snapshot().unwrap()).unwrap();
    let body = &snapshot["bodies"][snapshot["unfinished"][0]["body"].to_string()];
    assert_eq!(
        body["image_data"],
        serde_json::json!(["data:image/png;base64,AAAA"])
    );
    assert_eq!(
        body["video_data"],
        serde_json::json!([{"url": "file:///v.mp4", "fps": 2.0}])
    );
    assert_eq!(body["audio_data"], Value::Null);
}

use serde_json::json;
use sglang_parity::compare::prepare_comparison;
use sglang_parity::sse::{SseDecoder, SseEvent};

use super::*;

fn config(incremental: bool) -> RunConfig {
    serde_json::from_value(json!({
        "server": {
            "python": "/not/needed/by/describe",
            "model": "model-is-not-loaded",
            "args": if incremental { vec!["--incremental-streaming-output"] } else { vec![] },
        }
    }))
    .unwrap()
}

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/unary.json")).unwrap()
}

fn batch_fixture() -> [Value; 2] {
    let mut values = [fixture(), fixture()];
    for (index, value) in values.iter_mut().enumerate() {
        value["meta_info"]["id"] = json!(format!("batch-{index}"));
    }
    values[1]["text"] = json!("Goodbye moon");
    values[1]["output_ids"] = json!([30, 31]);
    values[1]["meta_info"]["output_token_logprobs"] = json!([[-0.3, 30, null], [-0.4, 31, null]]);
    values[1]["meta_info"]["output_top_logprobs"] = json!([[[-0.3, 30, null]], [[-0.4, 31, null]]]);
    values
}

fn events(incremental: bool) -> Vec<SseEvent> {
    // A trailing comment keeps the DONE event's required blank line intact
    // while allowing repository text-file checks to enforce one final LF.
    let bytes = if incremental {
        include_bytes!("fixtures/incremental.sse").as_slice()
    } else {
        include_bytes!("fixtures/cumulative.sse").as_slice()
    };
    let mut decoder = SseDecoder::default();
    let mut events = Vec::new();
    for bytes in bytes.chunks(7) {
        events.extend(decoder.push(bytes).unwrap());
    }
    events.extend(decoder.finish().unwrap());
    events
}

fn event(value: Value) -> SseEvent {
    SseEvent {
        event: "message".into(),
        data: value.to_string(),
        id: None,
    }
}

fn done() -> SseEvent {
    SseEvent {
        event: "message".into(),
        data: "[DONE]".into(),
        id: None,
    }
}

fn observation(events: Vec<SseEvent>) -> HttpObservation {
    HttpObservation {
        status: Some(200),
        headers: [(
            "content-type".into(),
            "text/event-stream; charset=utf-8".into(),
        )]
        .into(),
        events,
        ..Default::default()
    }
}

fn json_observation(value: Value) -> HttpObservation {
    HttpObservation {
        status: Some(200),
        json: Some(value),
        ..Default::default()
    }
}

fn case(name: &str, incremental: bool) -> (HttpCase, GeneratePolicy) {
    let (suite, policy) = load(DEFAULT_SPEC, &config(incremental)).unwrap();
    (
        suite
            .cases
            .into_iter()
            .find(|case| case.name == name)
            .unwrap(),
        policy,
    )
}

fn modify_event(events: &mut [SseEvent], index: usize, edit: impl FnOnce(&mut Value)) {
    let mut value = serde_json::from_str(&events[index].data).unwrap();
    edit(&mut value);
    events[index].data = value.to_string();
}

#[test]
fn default_spec_has_five_explicit_stream_pairs() {
    let (suite, _) = load(DEFAULT_SPEC, &config(false)).unwrap();
    assert_eq!(suite.cases.len(), 10);
    assert_eq!(suite.comparison.per_result_value_exceptions.len(), 2);
    let mut groups = BTreeSet::new();
    for pair in suite.cases.chunks_exact(2) {
        let group = pair[0].equivalence_group.as_deref().unwrap();
        assert!(!group.trim().is_empty());
        assert!(groups.insert(group), "duplicate equivalence group: {group}");
        assert_eq!(pair[0].equivalence_group, pair[1].equivalence_group);
        assert_eq!(pair[0].capture, CaptureMode::Json);
        assert_eq!(pair[1].capture, CaptureMode::Sse);
        assert_eq!(pair[0].comparison_scope, pair[1].comparison_scope);
        let mut left = pair[0].body.clone();
        let mut right = pair[1].body.clone();
        left.as_object_mut().unwrap().remove("stream");
        right.as_object_mut().unwrap().remove("stream");
        assert_eq!(left, right);
    }
    assert_eq!(
        suite.cases[4].comparison_scope,
        ComparisonScope::TopLevelArrayItems
    );
    assert_eq!(suite.output_mode, "cumulative");
    assert_eq!(
        load(DEFAULT_SPEC, &config(true)).unwrap().0.output_mode,
        "incremental"
    );
}

#[test]
fn suite_configuration_rejects_unknown_fields_and_mixed_contracts() {
    for pointer in [
        "",
        "/http",
        "/cases/0",
        "/comparison",
        "/comparison/per_result_value_exceptions/0",
    ] {
        let mut spec: Value = serde_json::from_str(DEFAULT_SPEC).unwrap();
        spec.pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("unknown_option".into(), json!(true));
        assert!(
            load(&spec.to_string(), &config(false)).is_err(),
            "{pointer}"
        );
    }
    let mut spec: Value = serde_json::from_str(DEFAULT_SPEC).unwrap();
    spec["cases"][0]["expect_status"] = json!(400);
    assert!(load(&spec.to_string(), &config(false)).is_err());
    spec["comparison"]["per_result_value_exceptions"] = json!([]);
    assert!(load(&spec.to_string(), &config(false)).is_err());
}

#[test]
fn request_payload_is_preserved_and_shape_comes_from_request() {
    let mut spec: Value = serde_json::from_str(DEFAULT_SPEC).unwrap();
    spec["cases"] = json!([{
        "name": "one_item_batch", "expect_status": 200,
        "body": {"input_ids": [[1, 2]], "stream": true, "future_request_field": {"keep": null}}
    }]);
    let (suite, _) = load(&spec.to_string(), &config(false)).unwrap();
    assert_eq!(suite.cases[0].body, spec["cases"][0]["body"]);
    assert_eq!(
        suite.cases[0].comparison_scope,
        ComparisonScope::TopLevelArrayItems
    );
    assert_eq!(
        request_shape(&json!({"text":"hello", "sampling_params":{"n":3}}))
            .unwrap()
            .count,
        3
    );
    assert!(request_shape(&json!({"text":[], "stream":true})).is_err());
}

#[test]
fn unary_and_both_stream_modes_preserve_the_complete_fixture() {
    let expected = fixture();
    let (json_case, policy) = case("greedy_json", false);
    let result = policy
        .prepare(&json_case, &json_observation(expected.clone()))
        .unwrap();
    assert_eq!(result, expected);
    for incremental in [false, true] {
        let (case, policy) = case("greedy_stream", incremental);
        assert_eq!(
            policy
                .prepare(&case, &observation(events(incremental)))
                .unwrap(),
            expected
        );
        // A transport backlog may coalesce all data into one event.
        assert_eq!(
            policy
                .prepare(&case, &observation(vec![event(expected.clone()), done()]))
                .unwrap(),
            expected
        );
    }
}

#[test]
fn one_token_without_logprobs_is_valid_in_unary_and_both_stream_modes() {
    let expected = json!({
        "text": "Tokyo", "output_ids": [42],
        "meta_info": {
            "id": "one-token", "prompt_tokens": 3, "completion_tokens": 1,
            "finish_reason": {"type": "length", "length": 1}
        }
    });
    let (json_case, policy) = case("one_token_json", false);
    assert_eq!(
        policy
            .prepare(&json_case, &json_observation(expected.clone()))
            .unwrap(),
        expected
    );
    for incremental in [false, true] {
        let (case, policy) = case("one_token_stream", incremental);
        assert_eq!(
            policy
                .prepare(&case, &observation(vec![event(expected.clone()), done()]))
                .unwrap(),
            expected
        );
    }
}

#[test]
fn batch_interleaving_restores_input_order_and_removes_only_index() {
    let expected = batch_fixture();
    for incremental in [false, true] {
        let (case, policy) = case("batch_stream", incremental);
        let original = events(incremental);
        let mut stream = Vec::new();
        for (index, position) in [(1, 0), (0, 0), (1, 1), (0, 1)] {
            let mut value: Value = serde_json::from_str(&original[position].data).unwrap();
            value["index"] = json!(index);
            value["meta_info"]["id"] = expected[index]["meta_info"]["id"].clone();
            if index == 1 {
                value["text"] = json!(match (position, incremental) {
                    (0, _) => "Goodbye",
                    (_, true) => " moon",
                    (_, false) => "Goodbye moon",
                });
                let start = if incremental { position } else { 0 };
                for path in [
                    "/output_ids",
                    "/meta_info/output_token_logprobs",
                    "/meta_info/output_top_logprobs",
                ] {
                    *value.pointer_mut(path).unwrap() = json!(
                        expected[index].pointer(path).unwrap().as_array().unwrap()
                            [start..=position]
                    );
                }
            }
            stream.push(event(value));
        }
        stream.push(done());
        let actual = policy.prepare(&case, &observation(stream)).unwrap();
        assert_eq!(actual, json!(expected));
    }
}

#[test]
fn incorrect_batch_indices_and_missing_results_fail() {
    let (case, policy) = case("batch_stream", false);
    for index in [
        None,
        Some(json!(-1)),
        Some(json!(2)),
        Some(json!(0.5)),
        Some(json!("0")),
    ] {
        let mut value = fixture();
        if let Some(index) = index {
            value["index"] = index;
        }
        let failure = policy
            .prepare(&case, &observation(vec![event(value), done()]))
            .unwrap_err();
        assert!(failure.iter().any(|violation| violation.path == "/index"));
    }
    let mut value = fixture();
    value["index"] = json!(0);
    assert!(
        policy
            .prepare(&case, &observation(vec![event(value), done()]))
            .is_err()
    );
    let (single, single_policy) = self::case("greedy_stream", false);
    let mut value = fixture();
    value["index"] = json!(0);
    assert!(
        single_policy
            .prepare(&single, &observation(vec![event(value), done()]))
            .is_err()
    );
}

#[test]
fn batch_json_requires_exact_cardinality_and_validates_every_item() {
    let (case, policy) = case("batch_json", false);
    let expected = json!(batch_fixture());
    assert_eq!(
        policy
            .prepare(&case, &json_observation(expected.clone()))
            .unwrap(),
        expected
    );
    for values in [
        vec![],
        vec![fixture()],
        vec![fixture(), fixture(), fixture()],
    ] {
        assert!(
            policy
                .prepare(&case, &json_observation(json!(values)))
                .is_err()
        );
    }
    let mut second = fixture();
    second["meta_info"]["finish_reason"] = Value::Null;
    let errors = policy
        .prepare(&case, &json_observation(json!([fixture(), second])))
        .unwrap_err();
    assert_eq!(errors[0].path, "/1/meta_info/finish_reason");
}

#[test]
fn every_result_and_stream_must_terminate_exactly_once() {
    let (case, policy) = case("greedy_stream", false);
    let base = events(false);
    let variants = [
        vec![],
        base[..2].to_vec(),
        vec![base[0].clone(), done()],
        vec![done(), base[1].clone()],
        vec![base[1].clone(), base[1].clone(), done()],
        vec![base[1].clone(), base[0].clone(), done()],
        vec![base[1].clone(), done(), done()],
        vec![base[1].clone(), done(), base[0].clone()],
    ];
    for stream in variants {
        assert!(policy.prepare(&case, &observation(stream)).is_err());
    }
}

#[test]
fn status_media_type_errors_and_malformed_events_are_failures() {
    let (case, policy) = case("greedy_stream", false);
    let mut wrong_type = observation(events(false));
    wrong_type.headers.clear();
    assert!(policy.prepare(&case, &wrong_type).is_err());
    let mut wrong_status = observation(events(false));
    wrong_status.status = Some(500);
    assert!(policy.prepare(&case, &wrong_status).is_err());
    let mut transport = observation(events(false));
    transport.transport_error = Some("truncated response".into());
    assert!(policy.prepare(&case, &transport).is_err());
    for first in [
        event(json!({"error":{"message":"failed"}})),
        SseEvent {
            event: "error".into(),
            ..event(fixture())
        },
        SseEvent {
            event: "message".into(),
            data: "not json".into(),
            id: None,
        },
    ] {
        let errors = policy
            .prepare(&case, &observation(vec![first, done()]))
            .unwrap_err();
        assert_eq!(errors[0].event, Some(0));
    }
}

#[test]
fn one_result_cannot_change_id_prompt_count_or_reverse_completion_count() {
    let (case, policy) = case("greedy_stream", false);
    for (key, value) in [
        ("id", json!("different")),
        ("prompt_tokens", json!(4)),
        ("completion_tokens", json!(0)),
    ] {
        let mut stream = events(false);
        modify_event(&mut stream, 1, |frame| frame["meta_info"][key] = value);
        let errors = policy.prepare(&case, &observation(stream)).unwrap_err();
        assert_eq!(errors[0].path, format!("/meta_info/{key}"));
        assert_eq!(errors[0].event, Some(1));
    }
}

#[test]
fn cumulative_sequences_must_remain_and_extend_the_previous_prefix() {
    let (case, policy) = case("greedy_stream", false);
    for path in [
        "/text",
        "/output_ids",
        "/meta_info/output_token_logprobs",
        "/meta_info/output_top_logprobs",
        "/meta_info/input_token_logprobs",
    ] {
        for disappear in [false, true] {
            let mut stream = events(false);
            modify_event(&mut stream, 1, |frame| {
                if disappear {
                    let (parent, key) = path.rsplit_once('/').unwrap();
                    frame
                        .pointer_mut(parent)
                        .unwrap()
                        .as_object_mut()
                        .unwrap()
                        .remove(key);
                } else {
                    *frame.pointer_mut(path).unwrap() = match path {
                        "/text" => json!("changed"),
                        "/output_ids" => json!([99, 11]),
                        "/meta_info/output_token_logprobs" => {
                            json!([[-0.9, 10, null], [-0.2, 11, null]])
                        }
                        "/meta_info/output_top_logprobs" => {
                            json!([[[-9.0, 10, null]], [[-0.2, 11, null]]])
                        }
                        _ => json!([]),
                    };
                }
            });
            let errors = policy.prepare(&case, &observation(stream)).unwrap_err();
            assert_eq!(errors[0].path, path, "disappear={disappear}");
            assert_eq!(errors[0].event, Some(1));
        }
    }
}

#[test]
fn incremental_input_logprobs_are_set_once_or_repeated_not_concatenated() {
    let (case, policy) = case("greedy_stream", true);
    let mut stream = events(true);
    modify_event(&mut stream, 1, |frame| {
        frame["meta_info"]["input_token_logprobs"] =
            fixture()["meta_info"]["input_token_logprobs"].clone();
        frame["meta_info"]["input_top_logprobs"] =
            fixture()["meta_info"]["input_top_logprobs"].clone();
    });
    assert_eq!(
        policy.prepare(&case, &observation(stream.clone())).unwrap(),
        fixture()
    );
    modify_event(&mut stream, 1, |frame| {
        frame["meta_info"]["input_token_logprobs"][1][0] = json!(-100.0)
    });
    assert!(policy.prepare(&case, &observation(stream)).is_err());
}

#[test]
fn input_logprobs_may_arrive_after_an_empty_placeholder() {
    let (case, policy) = case("greedy_stream", true);
    let mut stream = events(true);
    modify_event(&mut stream, 0, |frame| {
        frame["meta_info"]["input_token_logprobs"] = json!([]);
        frame["meta_info"]
            .as_object_mut()
            .unwrap()
            .remove("input_top_logprobs");
    });
    modify_event(&mut stream, 1, |frame| {
        frame["meta_info"]["input_token_logprobs"] =
            fixture()["meta_info"]["input_token_logprobs"].clone();
        frame["meta_info"]["input_top_logprobs"] =
            fixture()["meta_info"]["input_top_logprobs"].clone();
    });
    assert_eq!(
        policy.prepare(&case, &observation(stream)).unwrap(),
        fixture()
    );
}

#[test]
fn incremental_logprob_positions_track_cumulative_counts() {
    let (case, policy) = case("greedy_stream", true);
    let mut stream = events(true);
    modify_event(&mut stream, 1, |frame| {
        frame["meta_info"]["output_token_logprobs"] =
            fixture()["meta_info"]["output_token_logprobs"].clone();
    });
    let errors = policy.prepare(&case, &observation(stream)).unwrap_err();
    assert_eq!(errors[0].path, "/meta_info/output_token_logprobs");
}

#[test]
fn incremental_new_token_logprobs_must_match_new_output_ids() {
    let (case, policy) = case("greedy_stream", true);
    let mut stream = events(true);
    modify_event(&mut stream, 1, |frame| {
        frame["meta_info"]["output_token_logprobs"][0][1] = json!(99);
    });
    let errors = policy.prepare(&case, &observation(stream)).unwrap_err();
    assert_eq!(errors[0].path, "/meta_info/output_token_logprobs");
    assert_eq!(errors[0].event, Some(1));
}

#[test]
fn incremental_late_token_ids_or_logprobs_validate_the_entire_new_overlap() {
    let (case, policy) = case("greedy_stream", true);
    for key in ["output_ids", "output_token_logprobs"] {
        let mut stream = events(true);
        modify_event(&mut stream, 0, |frame| {
            let object = if key == "output_ids" {
                frame
            } else {
                &mut frame["meta_info"]
            };
            object.as_object_mut().unwrap().remove(key);
        });
        modify_event(&mut stream, 1, |frame| {
            if key == "output_ids" {
                frame[key] = fixture()[key].clone();
            } else {
                frame["meta_info"][key] = fixture()["meta_info"][key].clone();
            }
        });
        assert_eq!(
            policy.prepare(&case, &observation(stream.clone())).unwrap(),
            fixture()
        );
        modify_event(&mut stream, 1, |frame| {
            if key == "output_ids" {
                frame[key][0] = json!(99);
            } else {
                frame["meta_info"][key][0][1] = json!(99);
            }
        });
        let errors = policy.prepare(&case, &observation(stream)).unwrap_err();
        assert_eq!(errors[0].path, "/meta_info/output_token_logprobs");
        assert_eq!(errors[0].event, Some(1));
    }
}

#[test]
fn trimmed_stop_tokens_still_count_as_generated_tokens() {
    let (case, policy) = case("greedy_stream", true);
    let mut stream = events(true);
    modify_event(&mut stream, 1, |frame| {
        frame["text"] = json!("");
        frame.as_object_mut().unwrap().remove("output_ids");
        frame["meta_info"]["finish_reason"] = json!({"type":"stop","matched":11});
    });
    let value = policy.prepare(&case, &observation(stream)).unwrap();
    assert_eq!(value["output_ids"], json!([10]));
    assert_eq!(value["meta_info"]["completion_tokens"], 2);
}

#[test]
fn missing_incremental_token_ids_are_not_a_trimmed_stop() {
    let (case, policy) = case("greedy_stream", true);
    let mut stream = events(true);
    modify_event(&mut stream, 1, |frame| {
        frame.as_object_mut().unwrap().remove("output_ids");
    });
    let errors = policy.prepare(&case, &observation(stream)).unwrap_err();
    assert_eq!(errors[0].path, "/output_ids");
}

#[test]
fn malformed_logprob_tuples_are_rejected_in_unary_and_streams() {
    for malformed in [
        json!([-0.1, 10]),
        json!(["bad", 10, null]),
        json!([-0.1, -1, null]),
        json!([-0.1, 10, {}]),
    ] {
        for key in ["output_token_logprobs", "output_top_logprobs"] {
            let entry = if key == "output_top_logprobs" {
                json!([malformed])
            } else {
                malformed.clone()
            };
            let (json_case, policy) = case("greedy_json", false);
            let mut value = fixture();
            value["meta_info"][key][0] = entry.clone();
            let errors = policy
                .prepare(&json_case, &json_observation(value))
                .unwrap_err();
            assert_eq!(errors[0].path, format!("/meta_info/{key}/0"));
            for incremental in [false, true] {
                let (case, policy) = case("greedy_stream", incremental);
                let mut stream = events(incremental);
                modify_event(&mut stream, 0, |frame| {
                    frame["meta_info"][key][0] = entry.clone()
                });
                let errors = policy.prepare(&case, &observation(stream)).unwrap_err();
                assert_eq!(errors[0].path, format!("/meta_info/{key}/0"));
                assert_eq!(errors[0].event, Some(0));
            }
        }
    }
}

#[test]
fn requested_logprobs_cannot_be_omitted_by_both_implementations() {
    for (name, incremental) in [
        ("logprobs_json", false),
        ("logprobs_stream", false),
        ("logprobs_stream", true),
    ] {
        let (case, policy) = case(name, incremental);
        for key in [
            "input_token_logprobs",
            "output_token_logprobs",
            "output_top_logprobs",
        ] {
            let response = if case.capture == CaptureMode::Json {
                let mut value = fixture();
                value["meta_info"].as_object_mut().unwrap().remove(key);
                json_observation(value)
            } else {
                let mut stream = events(incremental);
                // Omit the field from every data frame so no accumulation check masks
                // the final response's obligation to include requested logprobs.
                for index in 0..stream.len() - 1 {
                    modify_event(&mut stream, index, |frame| {
                        frame["meta_info"].as_object_mut().unwrap().remove(key);
                    });
                }
                observation(stream)
            };
            let errors = policy.prepare(&case, &response).unwrap_err();
            assert_eq!(
                errors[0].path,
                format!("/meta_info/{key}"),
                "{name}, incremental={incremental}"
            );
            assert_eq!(errors[0].event, None);
        }
    }
}

#[test]
fn declared_exceptions_alone_control_latency_validation_and_comparison() {
    let mut spec: Value = serde_json::from_str(DEFAULT_SPEC).unwrap();
    let mut value = fixture();
    value["meta_info"]
        .as_object_mut()
        .unwrap()
        .remove("e2e_latency");
    let (suite, policy) = load(&spec.to_string(), &config(false)).unwrap();
    let case = &suite.cases[0];
    let prepared = policy
        .prepare(case, &json_observation(value.clone()))
        .unwrap();
    assert_eq!(prepared, value);
    assert!(prepare_comparison(&prepared, case.comparison_scope, &suite.comparison).is_err());
    spec["comparison"]["per_result_value_exceptions"]
        .as_array_mut()
        .unwrap()
        .pop();
    let (suite, _) = load(&spec.to_string(), &config(false)).unwrap();
    assert!(prepare_comparison(&prepared, case.comparison_scope, &suite.comparison).is_ok());
}

#[test]
fn explicit_http_error_suite_uses_json_root_without_success_exceptions() {
    let spec = json!({
        "name":"native_generate", "http":{"method":"POST","path":"/generate"},
        "comparison":{"base":"exact_json","per_result_value_exceptions":[]},
        "cases":[{"name":"invalid","expect_status":400,"body":{"stream":true,"input_ids":[]}}]
    });
    let (suite, policy) = load(&spec.to_string(), &config(false)).unwrap();
    assert_eq!(suite.cases[0].capture, CaptureMode::Json);
    assert_eq!(suite.cases[0].comparison_scope, ComparisonScope::Root);
    let value = json!({"error":{"message":"invalid","code":400},"diagnostic":null});
    assert_eq!(
        policy
            .prepare(
                &suite.cases[0],
                &HttpObservation {
                    status: Some(400),
                    ..json_observation(value.clone())
                }
            )
            .unwrap(),
        value
    );
    assert!(
        policy
            .prepare(
                &suite.cases[0],
                &HttpObservation {
                    status: Some(400),
                    ..json_observation(json!({}))
                }
            )
            .is_err()
    );
    let (success, success_policy) = case("greedy_json", false);
    assert!(
        success_policy
            .prepare(&success, &json_observation(value))
            .is_err()
    );
}

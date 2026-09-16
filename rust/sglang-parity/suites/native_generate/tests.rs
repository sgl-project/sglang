use serde_json::json;
use sglang_parity::compare::prepare_comparison;
use sglang_parity::sse::{SseDecoder, SseEvent};

use super::*;

// Response-policy fixtures do not start services; profile bindings are covered
// separately through the production load_plan entry point.
fn load(spec: &str, config: &RunConfig) -> Result<(HttpSuite, GeneratePolicy), String> {
    compile(
        serde_json::from_str(spec).map_err(|error| error.to_string())?,
        config,
    )
}

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
    let mut spec: Value = serde_json::from_str(DEFAULT_SPEC).unwrap();
    spec["streaming"]["fields"]["/future_field"] = json!("terminal");
    spec["streaming"]["fields"]["/meta_info/unknown_metadata"] = json!("terminal");
    let (suite, policy) = load(&spec.to_string(), &config(incremental)).unwrap();
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
fn default_spec_covers_all_scenarios_with_explicit_stream_pairs() {
    let (suite, _) = load(DEFAULT_SPEC, &config(false)).unwrap();
    assert_eq!(suite.cases.len(), 24);
    assert_eq!(suite.comparison.per_result_value_exceptions.len(), 3);
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
        groups,
        BTreeSet::from([
            "greedy",
            "one_token",
            "batch",
            "logprobs",
            "sampling",
            "input_logprobs",
            "cached",
            "versioned",
            "reasoning",
            "dp_rank_0",
            "dp_rank_1",
            "retractions",
        ])
    );
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
        .unwrap()
        .value;
    assert_eq!(result, expected);
    for incremental in [false, true] {
        let (case, policy) = case("greedy_stream", incremental);
        assert_eq!(
            policy
                .prepare(&case, &observation(events(incremental)))
                .unwrap()
                .value,
            expected
        );
        // A transport backlog may coalesce all data into one event.
        assert_eq!(
            policy
                .prepare(&case, &observation(vec![event(expected.clone()), done()]))
                .unwrap()
                .value,
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
            .unwrap()
            .value,
        expected
    );
    for incremental in [false, true] {
        let (case, policy) = case("one_token_stream", incremental);
        assert_eq!(
            policy
                .prepare(&case, &observation(vec![event(expected.clone()), done()]))
                .unwrap()
                .value,
            expected
        );
    }
}

#[test]
fn batch_interleaving_restores_input_order_and_removes_only_index() {
    let mut expected = batch_fixture();
    for (index, value) in expected.iter_mut().enumerate() {
        value["meta_info"]["response_sent_to_client_ts"] = json!(100 + index);
    }
    for incremental in [false, true] {
        let (case, policy) = case("batch_stream", incremental);
        let original = events(incremental);
        let mut stream = Vec::new();
        for (index, position) in [(1, 0), (0, 0), (1, 1), (0, 1)] {
            let mut value: Value = serde_json::from_str(&original[position].data).unwrap();
            value["index"] = json!(index);
            value["meta_info"]["id"] = expected[index]["meta_info"]["id"].clone();
            if position == 0 {
                value["meta_info"]["response_sent_to_client_ts"] =
                    expected[index]["meta_info"]["response_sent_to_client_ts"].clone();
            }
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
        assert_eq!(actual.value, json!(expected));
        assert_eq!(
            actual.origins["/0/meta_info/response_sent_to_client_ts"],
            [1]
        );
        assert_eq!(
            actual.origins["/1/meta_info/response_sent_to_client_ts"],
            [0]
        );
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
            .unwrap()
            .value,
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
        policy
            .prepare(&case, &observation(stream.clone()))
            .unwrap()
            .value,
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
        policy.prepare(&case, &observation(stream)).unwrap().value,
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
            policy
                .prepare(&case, &observation(stream.clone()))
                .unwrap()
                .value,
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
    let value = policy.prepare(&case, &observation(stream)).unwrap().value;
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
        .unwrap()
        .value;
    assert_eq!(prepared, value);
    assert!(prepare_comparison(&prepared, case.comparison_scope, &suite.comparison).is_err());
    spec["comparison"]["per_result_value_exceptions"]
        .as_array_mut()
        .unwrap()
        .retain(|rule| rule["path"] != "/meta_info/e2e_latency");
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
            .unwrap()
            .value,
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

#[test]
fn lifecycle_reconstruction_survives_coalescing_and_preserves_sources() {
    for incremental in [false, true] {
        let (stream_case, policy) = case("greedy_stream", incremental);
        let mut stream = events(incremental);
        for position in 0..2 {
            modify_event(&mut stream, position, |frame| {
                let meta = &mut frame["meta_info"];
                meta["cached_tokens"] = json!(position);
                meta["dp_rank"] = json!(position);
                meta["output_token_logprobs_length"] = json!(position + 1);
                if position == 0 {
                    meta["response_sent_to_client_ts"] = json!(123.5);
                } else {
                    meta["weight_version"] = json!("v1");
                    meta["weight_versions"] = json!([{"start":0,"end":2,"version":"v1"}]);
                }
            });
        }
        // Snapshot fields are per-event; their values need not be constant.
        modify_event(&mut stream, 0, |v| {
            v["meta_info"]["weight_version"] = json!("v0")
        });
        let prepared = policy.prepare(&stream_case, &observation(stream)).unwrap();
        assert_eq!(
            prepared.value["meta_info"]["response_sent_to_client_ts"],
            123.5
        );
        assert_eq!(
            prepared.origins["/meta_info/response_sent_to_client_ts"],
            [0]
        );
        assert_eq!(prepared.origins["/meta_info/weight_versions"], [1]);
        assert_eq!(
            prepared.origins["/output_ids"],
            if incremental { vec![0, 1] } else { vec![1] }
        );
        let coalesced = policy
            .prepare(
                &stream_case,
                &observation(vec![event(prepared.value.clone()), done()]),
            )
            .unwrap();
        assert_eq!(coalesced.value, prepared.value);
        let (json_case, _) = case("greedy_json", incremental);
        let unary = policy
            .prepare(&json_case, &json_observation(prepared.value.clone()))
            .unwrap();
        assert_eq!(unary.value, prepared.value);
        assert!(unary.origins.is_empty());
    }
}

#[test]
fn lifecycle_violations_identify_the_field_and_event() {
    for (field, first, last, event_index) in [
        ("response_sent_to_client_ts", None, Some(json!(1)), 1),
        (
            "response_sent_to_client_ts",
            Some(json!(1)),
            Some(json!(1)),
            1,
        ),
        ("response_sent_to_client_ts", Some(Value::Null), None, 0),
        ("e2e_latency", Some(json!(0.1)), Some(json!(0.2)), 0),
        (
            "weight_versions",
            Some(json!([{"start":0,"end":1,"version":"v1"}])),
            None,
            0,
        ),
        (
            "weight_versions",
            None,
            Some(json!([{"start":1,"end":2,"version":"v1"}])),
            1,
        ),
        ("cached_tokens", Some(json!(2)), Some(json!(1)), 1),
        ("cached_tokens", Some(json!(0)), None, 1),
        ("cached_tokens", None, Some(json!(0)), 1),
        ("cached_tokens", Some(json!(-1)), Some(json!(0)), 0),
        ("dp_rank", Some(json!("invalid")), None, 0),
        (
            "output_token_logprobs_length",
            Some(json!(2)),
            Some(json!(2)),
            0,
        ),
        ("unclassified", Some(json!(true)), None, 0),
    ] {
        for incremental in [false, true] {
            let (case, policy) = case("greedy_stream", incremental);
            let mut stream = events(incremental);
            for (position, value) in [first.clone(), last.clone()].into_iter().enumerate() {
                modify_event(&mut stream, position, |frame| {
                    let meta = frame["meta_info"].as_object_mut().unwrap();
                    if let Some(value) = value {
                        meta.insert(field.into(), value);
                    } else {
                        meta.remove(field);
                    }
                });
            }
            let errors = policy.prepare(&case, &observation(stream)).unwrap_err();
            assert_eq!(errors[0].path, format!("/meta_info/{field}"));
            assert_eq!(errors[0].event, Some(event_index), "{field}");
        }
    }
}

#[test]
fn streaming_rules_are_explicit_and_extension_values_remain_complete() {
    for (pointer, rule) in [
        ("/meta_info", "snapshot"),
        ("/meta_info/a/b", "first"),
        ("/bad~2key", "terminal"),
        ("/*", "terminal"),
        ("/text", "counter"),
        ("/extension", "text"),
        ("/extension", "unknown_rule"),
    ] {
        let mut spec: Value = serde_json::from_str(DEFAULT_SPEC).unwrap();
        spec["streaming"]["fields"][pointer] = json!(rule);
        assert!(
            load(&spec.to_string(), &config(false)).is_err(),
            "{pointer}: {rule}"
        );
    }
    let mut spec: Value = serde_json::from_str(DEFAULT_SPEC).unwrap();
    spec.as_object_mut().unwrap().remove("streaming");
    assert!(load(&spec.to_string(), &config(false)).is_err());

    let (case, mut policy) = case("greedy_stream", false);
    let mut stream = events(false);
    modify_event(&mut stream, 0, |v| {
        v["new/field~"] = json!({"nested":[null, {"keep":true}]})
    });
    let errors = policy
        .prepare(&case, &observation(stream.clone()))
        .unwrap_err();
    assert_eq!(errors[0].path, "/new~1field~0");
    assert!(errors[0].message.contains("rule not covered"));
    policy
        .streaming
        .fields
        .insert("/new~1field~0".into(), FieldRule::First);
    let prepared = policy.prepare(&case, &observation(stream)).unwrap();
    assert_eq!(
        prepared.value["new/field~"],
        json!({"nested":[null, {"keep":true}]})
    );
    assert_eq!(prepared.origins["/new~1field~0"], [0]);
}

#[test]
fn profiles_bind_explicitly_and_compile_the_actual_streaming_mode() {
    let mut config = config(false);
    config.profiles = serde_json::from_value(json!({
        "incremental":{"server":{"args":["--incremental-streaming-output"]}},
        "unused":{"server":{"model":"not-started"}}
    }))
    .unwrap();
    let mut spec: Value = serde_json::from_str(DEFAULT_SPEC).unwrap();
    for case in spec["cases"].as_array_mut().unwrap() {
        case["profiles"] = json!(["default", "incremental"]);
    }
    let plan = load_plan(&spec.to_string(), &config).unwrap();
    assert_eq!(plan.profiles.len(), 2);
    for (index, entry) in plan.profiles.iter().enumerate() {
        assert_eq!(entry.policy.incremental, index == 1);
        assert_eq!(
            entry.suite.output_mode,
            if index == 1 {
                "incremental"
            } else {
                "cumulative"
            }
        );
        assert_eq!(entry.suite.cases.len(), 24);
    }
    for binding in [json!([]), json!(["missing"]), json!(["default", "default"])] {
        let mut invalid = spec.clone();
        invalid["cases"][0]["profiles"] = binding;
        assert!(load_plan(&invalid.to_string(), &config).is_err());
    }
    // A JSON/SSE equivalence pair must be complete within each profile.
    spec["cases"][0]["profiles"] = json!(["default"]);
    assert!(load_plan(&spec.to_string(), &config).is_err());
}

#[test]
fn default_spec_resolves_all_profiles_for_both_platforms_without_starting_services() {
    for config in [
        include_str!("../../configs/mlx.json"),
        include_str!("../../configs/cuda.json"),
    ] {
        let config: RunConfig = serde_json::from_str(config).unwrap();
        let model_parts = config.server.model.split('/').collect::<Vec<_>>();
        assert_eq!(model_parts.len(), 2, "defaults must use a Hub repository");
        assert!(model_parts.iter().all(|part| !part.is_empty()));
        assert!(config.server.python.is_none());
        assert!(config.server.working_dir.is_none());
        assert!(config.server.env.is_empty());
        assert!(config.output_dir.is_relative());
        let revision = config
            .server
            .args
            .windows(2)
            .find(|pair| pair[0] == "--revision")
            .expect("default model revision must be pinned");
        assert_eq!(revision[1].len(), 40);
        assert!(revision[1].bytes().all(|b| b.is_ascii_hexdigit()));
        let plan = load_plan(DEFAULT_SPEC, &config).unwrap();
        assert_eq!(plan.profiles.len(), 12);
        assert_eq!(
            plan.profiles
                .iter()
                .map(|p| p.suite.cases.len())
                .sum::<usize>(),
            48
        );
        for entry in &plan.profiles {
            // Profile argv replaces the base list; every override must keep the pin.
            assert_eq!(entry.profile.server.model, config.server.model);
            assert!(
                entry
                    .profile
                    .server
                    .args
                    .windows(2)
                    .any(|pair| pair == revision)
            );
            if config.environment.backend == sglang_parity::environment::Backend::Cuda {
                assert!(
                    entry
                        .profile
                        .server
                        .args
                        .windows(2)
                        .any(|pair| pair == ["--attention-backend", "triton"]),
                    "CUDA defaults need deterministic attention with radix-cache support"
                );
            }
            assert_eq!(
                entry.policy.incremental,
                entry.profile.server.incremental_output()
            );
            for case in &entry.suite.cases {
                let group = case.equivalence_group.as_deref().unwrap();
                assert_eq!(
                    entry.policy.expectations.contains_key(&case.name),
                    !matches!(group, "one_token" | "batch" | "sampling"),
                    "{}: scenario assertions must survive consolidation",
                    case.name
                );
                if case.name.starts_with("cached_") {
                    assert!(entry.profile.server.radix_cache);
                    assert_eq!(case.isolation, Isolation::FreshProcess);
                    assert_eq!(case.before_each.len(), 1);
                    assert_eq!(case.body["text"], case.before_each[0].body["text"]);
                }
            }
        }
    }
}

#[test]
fn scenario_assertions_require_real_values_and_do_not_reject_valid_responses() {
    use expectations::Expectation::*;
    let passing = json!({"meta_info":{
        "cached_tokens":128, "cached_tokens_details":{"device":128,"host":0}, "dp_rank":1,
        "reasoning_tokens":1,"num_retractions":1,"completion_tokens":2,"response_sent_to_client_ts":123.5,
        "weight_version":"v1","weight_versions":[{"start":0,"end":2,"version":"v1"}],
        "input_top_logprobs":[null, [[-0.2,42,null]]], "output_token_logprobs_length":2,
        "output_token_logprobs":[[-0.1,42,null],[-0.2,43,null]]
    }});
    let assertions = [
        CachedTokens { positive: true },
        CacheDetails { positive: true },
        DpRank { value: Some(1) },
        ReasoningTokens { positive: true },
        Retractions { positive: true },
        Timestamp,
        WeightVersion { value: "v1".into() },
        InputTopLogprobs { populated: true },
        OutputLogprobsLength,
    ];
    for assertion in &assertions {
        assert!(
            assertion.evaluate(&passing).violations.is_empty(),
            "{assertion:?}"
        );
        assert!(
            !assertion
                .evaluate(&json!({"meta_info":{}}))
                .violations
                .is_empty(),
            "{assertion:?}"
        );
        assert!(!assertion.evaluate(&json!({"meta_info":{
            "cached_tokens":0,"cached_tokens_details":null,"dp_rank":null,"reasoning_tokens":0,
            "num_retractions":0,"response_sent_to_client_ts":0,"weight_version":"default",
            "input_top_logprobs":[null],"output_token_logprobs_length":0
        }})).violations.is_empty(), "{assertion:?}");
    }
    let mut zero = passing.clone();
    zero["meta_info"]["num_retractions"] = json!(0);
    assert!(
        Retractions { positive: true }
            .evaluate(&json!([zero, passing]))
            .violations
            .is_empty()
    );
    let mut spec: Value = serde_json::from_str(DEFAULT_SPEC).unwrap();
    spec["cases"][0]["expectations"] = json!([{"check":"cached_tokens","positive":true}]);
    let (suite, policy) = load(&spec.to_string(), &config(false)).unwrap();
    let response = fixture();
    let prepared = policy
        .prepare(&suite.cases[0], &json_observation(response.clone()))
        .unwrap();
    assert_eq!(prepared.value, response);
    assert_eq!(prepared.assertions.len(), 1);
    assert!(!prepared.assertions[0].violations.is_empty());
    // Preparation requests do not inherit the measured case's assertions.
    let prepared = policy
        .prepare(
            &suite.cases[0].request.as_case(),
            &json_observation(response),
        )
        .unwrap();
    assert!(prepared.assertions.is_empty());
}

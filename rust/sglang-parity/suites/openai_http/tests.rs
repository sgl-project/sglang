use super::*;
use sglang_parity::sse::SseEvent;

fn config() -> RunConfig {
    let mut config: RunConfig =
        serde_json::from_str(include_str!("../../configs/cuda.json")).unwrap();
    config.server.model = "test".into();
    config
}

fn plan() -> ExecutionPlan<OpenAiPolicy> {
    load_plan(DEFAULT_SPEC, &config()).unwrap()
}

fn case(name: &str) -> (HttpCase, OpenAiPolicy) {
    case_for_check(name, CheckTarget::FullResponse)
}

fn case_for_check(name: &str, check: CheckTarget) -> (HttpCase, OpenAiPolicy) {
    let profile = load_plan_for_check(DEFAULT_SPEC, &config(), check)
        .unwrap()
        .profiles
        .into_iter()
        .find(|p| p.suite.cases.iter().any(|c| c.name == name))
        .unwrap();
    (
        profile
            .suite
            .cases
            .into_iter()
            .find(|c| c.name == name)
            .unwrap(),
        profile.policy,
    )
}

fn chunk(chat: bool, choice: Value) -> Value {
    json!({"id":"request", "created":1, "model":"test",
        "object":if chat {"chat.completion.chunk"} else {"text_completion"},
        "choices":[choice], "usage":null})
}

fn capture(mut values: Vec<Value>) -> HttpObservation {
    values.push(json!("[DONE]"));
    HttpObservation {
        events: values
            .into_iter()
            .map(|v| SseEvent {
                event: String::new(),
                id: None,
                data: if v == "[DONE]" {
                    "[DONE]".into()
                } else {
                    v.to_string()
                },
            })
            .collect(),
        ..Default::default()
    }
}

fn usage_event(chat: bool) -> Value {
    let mut value = chunk(chat, Value::Null);
    value["choices"] = json!([]);
    value["usage"] = json!({"prompt_tokens":3,"completion_tokens":2,"total_tokens":5});
    value
}

fn stream(chat: bool) -> Vec<Value> {
    if chat {
        vec![
            chunk(
                true,
                json!({"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null,"logprobs":null}),
            ),
            chunk(
                true,
                json!({"index":0,"delta":{"content":"Hi"},"finish_reason":null,"logprobs":null}),
            ),
            chunk(
                true,
                json!({"index":0,"delta":{},"finish_reason":"length","logprobs":null}),
            ),
            usage_event(true),
        ]
    } else {
        vec![
            chunk(
                false,
                json!({"index":0,"text":"H","finish_reason":null,"logprobs":null}),
            ),
            chunk(
                false,
                json!({"index":0,"text":"i","finish_reason":"length","logprobs":null}),
            ),
            usage_event(false),
        ]
    }
}

fn unary(chat: bool) -> Value {
    let mut value = chunk(
        chat,
        json!({"index":0,"finish_reason":"length","logprobs":null}),
    );
    if chat {
        value["object"] = json!("chat.completion");
        value["choices"][0]["message"] = json!({"role":"assistant","content":"Hi"});
    } else {
        value["choices"][0]["text"] = json!("Hi");
    }
    value["usage"] = usage_event(chat)["usage"].clone();
    value
}

#[test]
fn plans_bind_only_selected_profiles_and_validate_equivalence_pairs() {
    let plan = plan();
    assert_eq!(plan.profiles.len(), 10);
    for profile in plan.profiles {
        assert!(
            profile
                .suite
                .cases
                .iter()
                .all(|c| c.body["model"] == "test")
        );
        assert!(profile.suite.response_policy.is_some());
    }
    let config = config();
    for mutation in [
        "profile",
        "body",
        "endpoint",
        "rules",
        "expectations",
        "warmup",
        "tool_expectation",
    ] {
        let mut spec: Value = serde_json::from_str(DEFAULT_SPEC).unwrap();
        match mutation {
            "profile" => spec["cases"][0]["profiles"] = json!(["missing"]),
            "body" => spec["cases"][0]["body"]["temperature"] = json!(1),
            "endpoint" => spec["cases"][0]["path"] = json!("/v1/responses"),
            "expectations" => {
                spec["cases"][0]["expectations"] = json!([
                {"check":"content","empty":false}, {"check":"content","empty":true}])
            }
            "warmup" => spec["cases"][0]["before_each"] = json!([{"prompt":"hello","stream":true}]),
            "tool_expectation" => {
                spec["cases"][0]["expectations"] = json!([
                {"check":"tool_calls","enabled":false,"arguments":{}}])
            }
            _ => spec["streaming"]["delta"]["content"] = json!("constant"),
        }
        assert!(load_plan(&spec.to_string(), &config).is_err(), "{mutation}");
    }
}

#[test]
fn json_and_stream_have_equal_semantics_without_discarding_wire_fields() {
    for chat in [false, true] {
        let prefix = if chat { "chat" } else { "completion" };
        let (json_case, policy) = case(&format!("{prefix}_greedy_json"));
        let (stream_case, _) = case(&format!("{prefix}_greedy_stream"));
        let mut value = unary(chat);
        value["extra"] = json!({"keep":null});
        let json = policy
            .prepare(
                &json_case,
                &HttpObservation {
                    json: Some(value.clone()),
                    ..Default::default()
                },
            )
            .unwrap();
        let sse = policy
            .prepare(&stream_case, &capture(stream(chat)))
            .unwrap();
        assert_eq!(json.value, value);
        assert_eq!(
            json.equivalence.unwrap().value,
            sse.equivalence.as_ref().unwrap().value
        );
        assert_ne!(json.value, sse.value);
        assert!(!sse.equivalence.unwrap().origins["/choices/0/content"].is_empty());
        assert_eq!(sse.origins["/usage"], vec![stream(chat).len() - 1]);
    }
}

#[test]
fn legal_fragmentation_and_choice_interleaving_do_not_change_results() {
    let (mut c, p) = case("completion_greedy_stream");
    let a = p.prepare(&c, &capture(stream(false))).unwrap();
    let b = p
        .prepare(
            &c,
            &capture(vec![
                chunk(
                    false,
                    json!({"index":0,"text":"Hi","finish_reason":"length","logprobs":null}),
                ),
                usage_event(false),
            ]),
        )
        .unwrap();
    assert_eq!(a.value, b.value);
    c.body["n"] = json!(2);
    c.equivalence_group = None;
    let chunks = [1, 0]
        .into_iter()
        .map(|i| {
            chunk(
                false,
                json!({"index":i,"text":"Hi","finish_reason":"length","logprobs":null}),
            )
        })
        .chain([usage_event(false)])
        .collect();
    let result = p.prepare(&c, &capture(chunks)).unwrap();
    assert_eq!(result.value["choices"][0]["index"], 0);
    assert_eq!(result.origins["/choices/0/text"], vec![1]);
    assert_eq!(result.origins["/choices/1/text"], vec![0]);
}

#[test]
fn stream_id_consistency_is_chat_only_but_every_id_remains_validated() {
    for chat in [false, true] {
        let name = if chat {
            "chat_greedy_stream"
        } else {
            "completion_greedy_stream"
        };
        let (c, p) = case(name);
        for id in [
            None,
            Some(Value::Null),
            Some(json!(3)),
            Some(json!("")),
            Some(json!("changed")),
        ] {
            let mut values = stream(chat);
            if let Some(id) = &id {
                values[1]["id"] = id.clone();
            } else {
                values[1].as_object_mut().unwrap().remove("id");
            }
            let result = p.prepare(&c, &capture(values));
            if !chat && id == Some(json!("changed")) {
                let response = result.unwrap();
                assert_eq!(response.value["id"], "request");
                assert_eq!(response.origins["/id"], vec![0]);
            } else {
                let errors = result.unwrap_err();
                assert_eq!(errors[0].path, "/id");
                assert_eq!(errors[0].event, Some(1));
            }
        }
    }
}

#[test]
fn completion_equivalence_ignores_varying_ids_but_detects_result_changes() {
    for group in ["greedy", "multiple", "batch"] {
        let (json_case, p) = case(&format!("completion_{group}_json"));
        let (stream_case, _) = case(&format!("completion_{group}_stream"));
        let count = choice_count(&stream_case.body, false).unwrap();
        let mut json = unary(false);
        json["id"] = json!("json-request");
        json["choices"] = (0..count)
            .map(|index| {
                json!({"index":index, "text":format!("answer {index}!"),
                "finish_reason":"length", "logprobs":null})
            })
            .collect();
        let mut values = Vec::new();
        for terminal in [false, true] {
            for index in (0..count).rev() {
                let mut value = chunk(
                    false,
                    json!({"index":index,
                    "text":if terminal { "!".into() } else { format!("answer {index}") },
                    "finish_reason":if terminal { json!("length") } else { Value::Null },
                    "logprobs":null}),
                );
                value["id"] = json!(format!("stream-{index}-{terminal}"));
                values.push(value);
            }
        }
        let mut final_usage = usage_event(false);
        final_usage["usage"] =
            json!({"prompt_tokens":3,"completion_tokens":2*count,"total_tokens":3+2*count});
        json["usage"] = final_usage["usage"].clone();
        values.push(final_usage);
        let streamed = p.prepare(&stream_case, &capture(values)).unwrap();
        let semantics = streamed.equivalence.unwrap().value;
        for (path, replacement) in [
            ("", Value::Null),
            ("/choices/0/text", json!("different")),
            ("/choices/0/finish_reason", json!("stop")),
            (
                "/choices/0/logprobs",
                json!({"tokens":["different"],"token_logprobs":[-0.1]}),
            ),
            (
                "/usage",
                json!({"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}),
            ),
        ] {
            let mut value = json.clone();
            if !path.is_empty() {
                *value.pointer_mut(path).unwrap() = replacement;
            }
            let response = p
                .prepare(
                    &json_case,
                    &HttpObservation {
                        json: Some(value),
                        ..Default::default()
                    },
                )
                .unwrap();
            let differences = sglang_parity::compare::compare_json(
                &response.equivalence.unwrap().value,
                &semantics,
            );
            assert_eq!(differences.is_empty(), path.is_empty(), "{group}: {path}");
        }
    }
}

#[test]
fn malformed_streams_have_event_scoped_diagnostics() {
    let (c, p) = case("completion_greedy_stream");
    for mutation in [
        "unknown",
        "created",
        "early_terminal",
        "after_finish",
        "missing_usage",
        "usage_total",
        "index",
        "type",
        "error",
        "done",
    ] {
        let mut values = stream(false);
        match mutation {
            "unknown" => values[0]["extra"] = json!(1),
            "created" => values[1]["created"] = json!(2),
            "early_terminal" => values[0]["choices"][0]["matched_stop"] = json!("stop"),
            "after_finish" => values.insert(2, values[1].clone()),
            "missing_usage" => {
                values.pop();
            }
            "usage_total" => values[2]["usage"]["total_tokens"] = json!(90),
            "index" => values[0]["choices"][0]["index"] = json!(5),
            "type" => values[0]["choices"][0]["text"] = json!(3),
            "error" => values[0] = json!({"error":{"message":"failed"}}),
            _ => {}
        }
        let mut observation = capture(values);
        if mutation == "done" {
            observation.events.pop();
        }
        let errors = p.prepare(&c, &observation).unwrap_err();
        assert!(!errors[0].message.is_empty(), "{mutation}");
        if !matches!(mutation, "missing_usage" | "done") {
            assert!(errors[0].event.is_some(), "{mutation}");
        }
    }
}

#[test]
fn continuous_and_final_usage_are_distinct_requirements() {
    let (c, p) = case("completion_continuous_usage");
    let mut values = stream(false);
    for value in &mut values[..2] {
        value["usage"] = usage_event(false)["usage"].clone();
    }
    assert!(p.prepare(&c, &capture(values.clone())).is_ok());
    values[0]["usage"] = Value::Null;
    assert!(p.prepare(&c, &capture(values)).is_err());
    let (c, p) = case("completion_no_usage");
    let mut values = stream(false);
    values.pop();
    assert!(p.prepare(&c, &capture(values)).is_ok());
    assert!(p.prepare(&c, &capture(stream(false))).is_err());
}

#[test]
fn logprobs_accumulate_and_missing_null_empty_remain_distinct() {
    let (c, p) = case("completion_logprobs_stream");
    let mut values = stream(false);
    for (i, value) in values[..2].iter_mut().enumerate() {
        value["choices"][0]["logprobs"] =
            json!({"tokens":["a"],"token_logprobs":[-0.1],"top_logprobs":[{}],"text_offset":[i]});
    }
    let response = p.prepare(&c, &capture(values.clone())).unwrap();
    assert_eq!(
        response.value["choices"][0]["logprobs"]["tokens"],
        json!(["a", "a"])
    );
    values[0]["choices"][0]["logprobs"]["tokens"] = json!([]);
    assert!(p.prepare(&c, &capture(values)).is_err());
    let (c, p) = case("completion_greedy_stream");
    let mut results = Vec::new();
    for item in [None, Some(Value::Null), Some(json!(""))] {
        let mut values = stream(false);
        if let Some(item) = item {
            values[0]["metadata"] = item.clone();
            values[1]["metadata"] = item.clone();
            values[2]["metadata"] = item;
        }
        results.push(p.prepare(&c, &capture(values)).unwrap().value);
    }
    assert!(results[0].get("metadata").is_none());
    assert!(results[1]["metadata"].is_null());
    assert_eq!(results[2]["metadata"], "");
}

#[test]
fn both_platform_configs_and_model_aliases_resolve_without_side_effects() {
    for text in [
        include_str!("../../configs/mlx.json"),
        include_str!("../../configs/cuda.json"),
    ] {
        let mut config: RunConfig = serde_json::from_str(text).unwrap();
        config
            .server
            .args
            .extend(["--served-model-name".into(), "alias".into()]);
        let plan = load_plan(DEFAULT_SPEC, &config).unwrap();
        assert_eq!(plan.profiles.len(), 10);
        assert!(
            plan.profiles[0]
                .suite
                .cases
                .iter()
                .all(|c| c.body["model"] == "alias")
        );
        let incremental = plan
            .profiles
            .iter()
            .find(|p| p.profile.id == "incremental")
            .unwrap();
        assert_eq!(incremental.suite.output_mode, "incremental");
        assert_eq!(plan.profiles[0].suite.cases.len(), 44);
        assert_eq!(incremental.suite.cases.len(), 42);
    }
}

#[test]
fn logprob_precision_rules_cover_both_apis_and_preserve_original_evidence() {
    use sglang_parity::compare::{prepare_comparison, prepare_numeric_comparison};
    let rules = &plan().profiles[0].suite.comparison;
    for chat in [false, true] {
        let prefix = if chat { "chat" } else { "completion" };
        let (json_case, policy) = case(&format!("{prefix}_logprobs_json"));
        let (stream_case, _) = case(&format!("{prefix}_logprobs_stream"));
        let probabilities = |number| {
            if chat {
                let entry = json!({"token":"Hi","logprob":number,"token_id":16777217,
                    "top_logprobs":[{"token":"Hi","logprob":number}]});
                json!({"content":[entry.clone()],"refusal":[entry]})
            } else {
                json!({"tokens":["Hi"],"token_logprobs":[number],
                    "top_logprobs":[{"a/~":number}],"text_offset":[0]})
            }
        };
        let mut json = unary(chat);
        json["choices"][0]["logprobs"] = probabilities(-0.24555964767932892);
        let original = policy
            .prepare(
                &json_case,
                &HttpObservation {
                    json: Some(json.clone()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(original.value, json);
        let mut event = json.clone();
        event["usage"] = Value::Null;
        event["choices"][0]["logprobs"] = probabilities(-0.24555965);
        if chat {
            event["object"] = json!("chat.completion.chunk");
            let choice = event["choices"][0].as_object_mut().unwrap();
            let message = choice.remove("message").unwrap();
            choice.insert("delta".into(), message);
        }
        let streamed = policy
            .prepare(&stream_case, &capture(vec![event, usage_event(chat)]))
            .unwrap();
        assert_eq!(
            streamed.value["choices"][0]["logprobs"],
            probabilities(-0.24555965)
        );
        let left = prepare_comparison(&original.value, ComparisonScope::Root, rules).unwrap();
        let right = prepare_comparison(&streamed.value, ComparisonScope::Root, rules).unwrap();
        assert_eq!(
            left["choices"][0]["logprobs"],
            right["choices"][0]["logprobs"]
        );
        if chat {
            assert_eq!(
                right["choices"][0]["logprobs"]["content"][0]["token_id"],
                16777217
            );
        }
        let left = original.equivalence.unwrap().value;
        let right = streamed.equivalence.unwrap().value;
        assert_ne!(left, right);
        assert_eq!(
            prepare_numeric_comparison(&left, ComparisonScope::Root, rules).unwrap(),
            prepare_numeric_comparison(&right, ComparisonScope::Root, rules).unwrap()
        );
    }
}

#[test]
fn error_responses_and_malformed_unary_contracts_remain_distinct() {
    let (error_case, p) = case("completion_invalid_max_tokens");
    let error = json!({"error":{"message":"invalid max_tokens","type":"BadRequest","code":400}});
    assert_eq!(
        p.prepare(
            &error_case,
            &HttpObservation {
                json: Some(error.clone()),
                ..Default::default()
            }
        )
        .unwrap()
        .value,
        error
    );
    let (c, p) = case("completion_greedy_json");
    for field in ["id", "created", "model", "object", "usage", "choices"] {
        let mut value = unary(false);
        value.as_object_mut().unwrap().remove(field);
        assert!(
            p.prepare(
                &c,
                &HttpObservation {
                    json: Some(value),
                    ..Default::default()
                }
            )
            .is_err(),
            "{field}"
        );
    }
}

#[test]
fn chat_control_events_need_no_continuous_usage_and_native_errors_are_preserved() {
    let (c, p) = case("chat_continuous_usage");
    let mut values = stream(true);
    values[1]["usage"] = usage_event(true)["usage"].clone();
    assert!(p.prepare(&c, &capture(values)).is_ok());
    let (c, p) = case("chat_invalid_max_tokens");
    let value = json!({"object":"error","message":"invalid tokens","code":400,"param":null});
    let response = p
        .prepare(
            &c,
            &HttpObservation {
                json: Some(value.clone()),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(response.value, value);
}

#[test]
fn scenario_assertions_distinguish_inactive_and_populated_values() {
    let (c, _) = case("chat_greedy_json");
    let rows = [
        (
            json!({"check":"reasoning","enabled":true}),
            "/choices/0/message/reasoning_content",
            json!("Think."),
            json!(null),
        ),
        (
            json!({"check":"reasoning","enabled":false}),
            "/choices/0/message/reasoning_content",
            json!(null),
            json!("Think."),
        ),
        (
            json!({"check":"reasoning_tokens","positive":true}),
            "/usage/reasoning_tokens",
            json!(1),
            json!(0),
        ),
        (
            json!({"check":"reasoning_tokens","positive":false}),
            "/usage/reasoning_tokens",
            json!(0),
            json!(1),
        ),
        (
            json!({"check":"cached_tokens","positive":true}),
            "/usage/prompt_tokens_details",
            json!({"cached_tokens":1}),
            json!(null),
        ),
        (
            json!({"check":"cached_tokens","positive":false}),
            "/usage/prompt_tokens_details",
            json!(null),
            json!({"cached_tokens":1}),
        ),
        (
            json!({"check":"content","empty":true}),
            "/choices/0/message/content",
            json!(""),
            json!("hello"),
        ),
        (
            json!({"check":"content","empty":false}),
            "/choices/0/message/content",
            json!("hello"),
            json!(""),
        ),
        (
            json!({"check":"no_refusal"}),
            "/choices/0/message/refusal",
            json!(null),
            json!("Refused"),
        ),
        (
            json!({"check":"tool_calls","enabled":false}),
            "/choices/0/message/tool_calls",
            json!([]),
            json!([{}]),
        ),
        (
            json!({"check":"logprobs","enabled":false}),
            "/choices/0/logprobs",
            json!(null),
            json!({}),
        ),
        (
            json!({"check":"weight_version","value":"parity-v1"}),
            "/metadata",
            json!({"weight_version":"parity-v1","weight_versions":[{"version":"parity-v1","start":0,"end":2}]}),
            json!({"weight_version":"default"}),
        ),
    ];
    for (spec, path, valid, invalid) in rows {
        let expectation: Expectation = serde_json::from_value(spec.clone()).unwrap();
        let mut response = unary(true);
        // Populate parent objects once; the table varies just the target field.
        response["usage"]["reasoning_tokens"] = Value::Null;
        response["usage"]["prompt_tokens_details"] = Value::Null;
        response["metadata"] = Value::Null;
        for key in ["reasoning_content", "refusal", "tool_calls"] {
            response["choices"][0]["message"][key] = Value::Null;
        }
        for (value, passes) in [(valid, true), (invalid, false)] {
            *response.pointer_mut(path).unwrap() = value;
            let result = expectation.evaluate(&response, &c, CheckTarget::FullResponse);
            assert_eq!(result.violations.is_empty(), passes, "{spec}: {response}");
        }
    }
    let mut cold = unary(true);
    cold["usage"]["prompt_tokens_details"] = json!({"cached_tokens":0});
    assert!(
        Expectation::CachedTokens { positive: false }
            .evaluate(&cold, &c, CheckTarget::FullResponse)
            .violations
            .is_empty()
    );
    assert!(
        !Expectation::CachedTokens { positive: true }
            .evaluate(&cold, &c, CheckTarget::FullResponse)
            .violations
            .is_empty()
    );
    // A missing positive field never counts as exercising the populated state.
    for check in ["reasoning_tokens", "cached_tokens"] {
        let expectation: Expectation =
            serde_json::from_value(json!({"check":check,"positive":true})).unwrap();
        assert!(
            !expectation
                .evaluate(&unary(true), &c, CheckTarget::FullResponse)
                .violations
                .is_empty()
        );
    }
}

#[test]
fn logprob_alternative_scenarios_require_sampled_data_and_preserve_empty_shapes() {
    for chat in [false, true] {
        let prefix = if chat { "chat" } else { "completion" };
        let (c, p) = case(&format!("{prefix}_logprobs_no_alternatives_json"));
        for alternatives in [0, 2] {
            let expectation: Expectation = serde_json::from_value(
                json!({"check":"logprobs","enabled":true,"alternatives":alternatives}),
            )
            .unwrap();
            for populated in [false, true] {
                let mut response = unary(chat);
                response["choices"][0]["logprobs"] = if chat {
                    json!({"content":[{"token":"hi","token_id":0,"logprob":-1.0,
                        "top_logprobs":if populated { json!([{"token":"hi","logprob":-1.0}]) } else {json!([])}}]})
                } else {
                    json!({"tokens":["hi"],"token_logprobs":[-1.0],"text_offset":[-1],
                        "top_logprobs":if populated {json!([{"hi":-1.0}])} else {json!([])}})
                };
                let prepared = p
                    .prepare(
                        &c,
                        &HttpObservation {
                            json: Some(response.clone()),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert_eq!(prepared.value, response);
                assert_eq!(
                    expectation
                        .evaluate(&response, &c, CheckTarget::FullResponse)
                        .violations
                        .is_empty(),
                    populated == (alternatives > 0)
                );
                if chat {
                    let ids = Expectation::TokenIds;
                    assert!(
                        ids.evaluate(&response, &c, CheckTarget::FullResponse)
                            .violations
                            .is_empty()
                    );
                    response["choices"][0]["logprobs"]["content"][0]
                        .as_object_mut()
                        .unwrap()
                        .remove("token_id");
                    assert!(
                        !ids.evaluate(&response, &c, CheckTarget::FullResponse)
                            .violations
                            .is_empty()
                    );
                }
                response["choices"][0]["logprobs"] = json!({});
                assert!(
                    !expectation
                        .evaluate(&response, &c, CheckTarget::FullResponse)
                        .violations
                        .is_empty()
                );
            }
        }
    }
}

#[test]
fn tool_fragments_reconstruct_without_hiding_arguments_or_identity_errors() {
    let (json_case, policy) = case("chat_tools_required_json");
    let (stream_case, _) = case("chat_tools_required_stream");
    let call = json!({"id":"call-json", "type":"function", "function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}});
    let mut response = unary(true);
    response["choices"][0]["message"] =
        json!({"role":"assistant","content":null,"tool_calls":[call]});
    response["choices"][0]["finish_reason"] = json!("tool_calls");
    response["choices"][0]
        .as_object_mut()
        .unwrap()
        .remove("logprobs");
    let json = policy
        .prepare(
            &json_case,
            &HttpObservation {
                json: Some(response),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(json.assertions.iter().all(|a| a.violations.is_empty()));
    let chunks = vec![
        chunk(
            true,
            json!({"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call-stream","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}}]},"finish_reason":null}),
        ),
        chunk(
            true,
            json!({"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]},"finish_reason":"tool_calls"}),
        ),
        usage_event(true),
    ];
    let stream = policy
        .prepare(&stream_case, &capture(chunks.clone()))
        .unwrap();
    assert!(stream.assertions.iter().all(|a| a.violations.is_empty()));
    assert_eq!(
        json.equivalence.unwrap().value,
        stream.equivalence.unwrap().value
    );
    assert_eq!(stream.origins["/choices/0/delta/tool_calls"], vec![0, 1]);
    assert_eq!(
        stream.value["choices"][0]["delta"]["tool_calls"][0]["id"],
        "call-stream"
    );
    for mutation in ["id", "index", "unknown", "arguments", "empty", "wrong_args"] {
        let mut values = chunks.clone();
        let call = &mut values[1]["choices"][0]["delta"]["tool_calls"][0];
        match mutation {
            "id" => call["id"] = json!("changed"),
            "index" => {
                call.as_object_mut().unwrap().remove("index");
            }
            "unknown" => call["new_field"] = json!(1),
            "arguments" => call["function"]["arguments"] = json!(3),
            "empty" => values[0]["choices"][0]["delta"]["tool_calls"][0]["id"] = json!(""),
            _ => call["function"]["arguments"] = json!("\"London\"}"),
        }
        let result = policy.prepare(&stream_case, &capture(values));
        if mutation == "wrong_args" {
            // Valid protocol, but the desired scenario did not occur. Keep the
            // full response and semantic view available for parity comparison.
            let result = result.unwrap();
            assert!(!result.assertions[0].violations.is_empty());
            assert!(result.equivalence.is_some());
        } else {
            assert!(result.is_err(), "{mutation}");
        }
    }
}

#[test]
fn reasoning_only_streams_retain_payload_and_participate_in_equivalence() {
    let (c, p) = case("chat_reasoning_stream");
    let values = vec![
        chunk(
            true,
            json!({"index":0,"delta":{"role":"assistant","reasoning_content":"Let "},"finish_reason":null}),
        ),
        chunk(
            true,
            json!({"index":0,"delta":{"reasoning_content":"me think."},"finish_reason":"length"}),
        ),
        usage_event(true),
    ];
    let result = p.prepare(&c, &capture(values)).unwrap();
    assert_eq!(
        result.value["choices"][0]["delta"]["reasoning_content"],
        "Let me think."
    );
    assert!(result.value["choices"][0]["delta"].get("content").is_none());
    assert_eq!(
        result.equivalence.unwrap().value["choices"][0]["reasoning_content"],
        "Let me think."
    );
    assert!(result.assertions[0].violations.is_empty());
    assert!(!result.assertions[1].violations.is_empty()); // Missing token accounting.
}

#[test]
fn coverage_cases_resolve_assertions_and_warmups_for_every_stream_mode() {
    let plan = plan();
    for (scenario, profile_prefix) in [
        ("cached", "cached"),
        ("versioned", "versioned"),
        ("reasoning", "reasoning"),
        ("tools_required", "tools"),
    ] {
        for mode in ["cumulative", "incremental"] {
            let profile = plan
                .profiles
                .iter()
                .find(|p| p.profile.id == format!("{profile_prefix}_{mode}"))
                .unwrap();
            for capture in [CaptureMode::Json, CaptureMode::Sse] {
                let c = profile
                    .suite
                    .cases
                    .iter()
                    .find(|c| {
                        c.name.starts_with(&format!("chat_{scenario}_")) && c.capture == capture
                    })
                    .unwrap();
                if scenario == "cached" {
                    assert_eq!(c.before_each.len(), 1);
                    assert_eq!(c.before_each[0].body["model"], "test");
                    assert!(profile.profile.server.radix_cache);
                    assert!(
                        profile
                            .profile
                            .server
                            .args
                            .contains(&"--enable-cache-report".into())
                    );
                }
                if scenario != "versioned" || capture == CaptureMode::Json {
                    assert!(!c.assertions.is_empty());
                    assert!(
                        profile.suite.response_policy.as_ref().unwrap()["expectations"]
                            .get(&c.name)
                            .is_some()
                    );
                }
            }
        }
    }
}

#[test]
fn empty_chat_outputs_are_valid_but_keep_null_empty_and_missing_distinct() {
    let (c, p) = case("chat_empty_stop_json");
    let mut values = Vec::new();
    for content in [None, Some(Value::Null), Some(json!(""))] {
        let mut value = unary(true);
        value["choices"][0]["finish_reason"] = json!("stop");
        value["choices"][0]["matched_stop"] = json!("Paris");
        let message = value["choices"][0]["message"].as_object_mut().unwrap();
        if let Some(content) = content {
            message.insert("content".into(), content);
        } else {
            message.remove("content");
        }
        let result = p
            .prepare(
                &c,
                &HttpObservation {
                    json: Some(value.clone()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(result.value, value);
        assert!(result.assertions.iter().all(|a| a.violations.is_empty()));
        values.push(result);
    }
    assert_ne!(values[0].value, values[1].value);
    assert_ne!(values[1].value, values[2].value);
    assert_eq!(
        values[0].equivalence.as_ref().unwrap().value,
        values[2].equivalence.as_ref().unwrap().value
    );
}

#[test]
fn generated_content_selects_generation_cases_and_content_assertions() {
    let plan = load_plan_for_check(DEFAULT_SPEC, &config(), CheckTarget::GeneratedContent).unwrap();
    let full = load_plan(DEFAULT_SPEC, &config()).unwrap();
    for (profile, original) in plan.profiles.iter().zip(&full.profiles) {
        assert_eq!(profile.suite.check, CheckTarget::GeneratedContent);
        assert!(
            profile
                .suite
                .comparison
                .per_result_value_exceptions
                .is_empty()
        );
        assert!(profile.suite.comparison.per_result_numeric_rules.is_empty());
        let excluded = profile.suite.response_policy.as_ref().unwrap()["excluded_cases"]
            .as_array()
            .unwrap();
        assert_eq!(
            profile.suite.cases.len() + excluded.len(),
            original.suite.cases.len()
        );
        for case in &profile.suite.cases {
            assert_eq!(case.expect_status, 200);
            assert!(case.assertions.iter().all(|name| matches!(
                name.as_str(),
                "content" | "reasoning_content" | "tool_calls" | "no_refusal"
            )));
        }
    }
    let mut spec: Value = serde_json::from_str(DEFAULT_SPEC).unwrap();
    spec["generated_content"]["text"] = json!("trim");
    assert!(
        load_plan_for_check(&spec.to_string(), &config(), CheckTarget::GeneratedContent).is_err()
    );
}

#[test]
fn generated_content_ignores_metadata_and_uses_one_view_for_json_and_sse() {
    for chat in [false, true] {
        let prefix = if chat { "chat" } else { "completion" };
        let (json_case, policy) = case_for_check(
            &format!("{prefix}_logprobs_json"),
            CheckTarget::GeneratedContent,
        );
        let (stream_case, _) = case_for_check(
            &format!("{prefix}_logprobs_stream"),
            CheckTarget::GeneratedContent,
        );
        let mut value = unary(chat);
        for key in ["id", "created", "usage", "model", "object"] {
            value.as_object_mut().unwrap().remove(key);
        }
        value["choices"][0]["logprobs"] = json!("irrelevant");
        let json = policy
            .prepare(
                &json_case,
                &HttpObservation {
                    json: Some(value),
                    ..Default::default()
                },
            )
            .unwrap();
        let mut values = stream(chat);
        values.pop();
        for (i, value) in values.iter_mut().enumerate() {
            value["id"] = json!(i);
            value["created"] = json!(-i32::try_from(i).unwrap());
            value["usage"] = json!("irrelevant");
            value["choices"][0]["logprobs"] = json!("irrelevant");
            if chat {
                value["choices"][0]["delta"]["metadata"] = json!({"chunk":i});
                value["choices"][0]["delta"]["role"] = json!(i);
            }
        }
        values.push(json!({"choices":[{"index":0,"usage":{"total_tokens":123}}]}));
        values.push(json!({"choices":[],"sglext":{"input_ids":[1,2,3]}}));
        let mut observation = capture(values);
        let metadata_index = observation.events.len() - 2;
        observation.events[metadata_index].event = "sglext_ids".into();
        let sse = policy.prepare(&stream_case, &observation).unwrap();
        assert_eq!(json.value, sse.value);
        assert!(json.equivalence.is_none() && sse.equivalence.is_none());
        assert!(
            json.assertions
                .iter()
                .chain(&sse.assertions)
                .all(|a| a.violations.is_empty())
        );
        let path = if chat {
            "/choices/0/message/content"
        } else {
            "/choices/0/text"
        };
        assert!(!sse.origins[path].is_empty());
        assert_eq!(json.value.pointer(path), Some(&json!("Hi")));
        let mut changed = json.value.clone();
        *changed.pointer_mut(path).unwrap() = json!("Hi\n");
        assert_eq!(
            sglang_parity::compare::compare_json(&json.value, &changed).len(),
            1
        );
    }
}

#[test]
fn generated_content_retains_stream_integrity_and_required_containers() {
    for chat in [false, true] {
        let prefix = if chat { "chat" } else { "completion" };
        let (case, policy) = case_for_check(
            &format!("{prefix}_greedy_stream"),
            CheckTarget::GeneratedContent,
        );
        for mutation in [
            "index",
            "missing_choice",
            "after_finish",
            "type",
            "missing_content",
            "wrong_container",
            "trailing_wrong_container",
            "abort",
            "error",
            "done",
        ] {
            let mut values = stream(chat);
            let last = values.len() - 2;
            match mutation {
                "index" => values[0]["choices"][0]["index"] = json!(3),
                "missing_choice" => values.retain(|v| v["choices"].as_array().unwrap().is_empty()),
                "after_finish" => values.insert(last + 1, values[1].clone()),
                "type" => {
                    if chat {
                        values[1]["choices"][0]["delta"]["content"] = json!(7);
                    } else {
                        values[1]["choices"][0]["text"] = json!(7);
                    }
                }
                "missing_content" => {
                    for value in &mut values[..=last] {
                        value["choices"][0]
                            .as_object_mut()
                            .unwrap()
                            .remove(if chat { "delta" } else { "text" });
                    }
                }
                "wrong_container" => {
                    values[0]["choices"][0]["message"] = json!({"content":"hidden"})
                }
                "trailing_wrong_container" => {
                    values.push(json!({"choices":[{"index":0,"message":{"content":"hidden"}}]}))
                }
                "abort" => values[last]["choices"][0]["finish_reason"] = json!("abort"),
                "error" => values[0] = json!({"error":{"message":"failed"}}),
                _ => {}
            }
            let mut observation = capture(values);
            if mutation == "done" {
                observation.events.pop();
            }
            assert!(
                policy.prepare(&case, &observation).is_err(),
                "chat={chat}: {mutation}"
            );
        }
        let (case, _) = case_for_check(
            &format!("{prefix}_greedy_json"),
            CheckTarget::GeneratedContent,
        );
        for mutation in ["container", "choice", "content", "abort"] {
            let mut value = unary(chat);
            match mutation {
                "container" => value = json!([]),
                "choice" => value["choices"] = json!([]),
                "content" => {
                    value["choices"][0]
                        .as_object_mut()
                        .unwrap()
                        .remove(if chat { "message" } else { "text" });
                }
                _ => value["choices"][0]["finish_reason"] = json!("abort"),
            }
            assert!(
                policy
                    .prepare(
                        &case,
                        &HttpObservation {
                            json: Some(value),
                            ..Default::default()
                        }
                    )
                    .is_err(),
                "chat={chat}: {mutation}"
            );
        }
    }
}

#[test]
fn generated_chat_content_normalizes_only_empty_payloads() {
    let (case, policy) = case_for_check("chat_empty_stop_json", CheckTarget::GeneratedContent);
    let mut baseline = None;
    for message in [
        json!({}),
        json!({"content":null,"reasoning_content":null,"refusal":null,"tool_calls":null}),
        json!({"content":"","reasoning_content":"","refusal":"","tool_calls":[]}),
    ] {
        let mut value = unary(true);
        value["choices"][0]["message"] = message;
        let prepared = policy
            .prepare(
                &case,
                &HttpObservation {
                    json: Some(value),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(prepared.assertions.iter().all(|a| a.violations.is_empty()));
        if let Some(baseline) = &baseline {
            assert_eq!(&prepared.value, baseline);
        } else {
            baseline = Some(prepared.value);
        }
    }
    for field in ["content", "reasoning_content", "refusal"] {
        let mut value = unary(true);
        value["choices"][0]["message"] = json!({field:"\n"});
        let prepared = policy
            .prepare(
                &case,
                &HttpObservation {
                    json: Some(value),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_ne!(Some(prepared.value), baseline, "{field}");
    }
}

#[test]
fn generated_content_routes_tools_and_choices_without_comparing_ids() {
    let (mut case, policy) = case_for_check("chat_greedy_stream", CheckTarget::GeneratedContent);
    case.body["n"] = json!(2);
    let values = vec![
        json!({"choices":[{"index":1,"delta":{"tool_calls":[{"index":0,"id":"one","type":"function","function":{"name":"weather","arguments":"{\"city\":"}}]}}]}),
        json!({"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}),
        json!({"choices":[{"index":1,"delta":{"tool_calls":[{"index":0,"id":"two","function":{"arguments":" \"Paris\"}"}}]},"finish_reason":"tool_calls"}]}),
    ];
    let stream = policy.prepare(&case, &capture(values.clone())).unwrap();
    assert_eq!(stream.value["choices"][0]["index"], 0);
    let tool_path = "/choices/1/message/tool_calls/0";
    assert_eq!(
        stream.value.pointer(tool_path).unwrap(),
        &json!({"type":"function","function":{"name":"weather","arguments":"{\"city\": \"Paris\"}"}})
    );
    assert_eq!(stream.origins["/choices/1/message/tool_calls"], vec![0, 2]);
    let mut annotated = values.clone();
    let call = &mut annotated[0]["choices"][0]["delta"]["tool_calls"][0];
    call["metadata"] = json!({"trace": 7});
    call["function"]["metadata"] = json!(null);
    assert_eq!(
        policy.prepare(&case, &capture(annotated)).unwrap().value,
        stream.value
    );
    let mut json_case = case.clone();
    json_case.request.capture = CaptureMode::Json;
    let json = json!({"choices":[
        {"index":1,"message":{"tool_calls":[{"type":"function","function":{"name":"weather","arguments":"{\"city\": \"Paris\"}"}}]},"finish_reason":"stop"},
        {"index":0,"message":{"content":"Hi"},"finish_reason":"length"}
    ]});
    assert_eq!(
        stream.value,
        policy
            .prepare(
                &json_case,
                &HttpObservation {
                    json: Some(json),
                    ..Default::default()
                }
            )
            .unwrap()
            .value
    );
    for mutation in ["index", "arguments", "name", "type"] {
        let mut values = values.clone();
        let call = &mut values[0]["choices"][0]["delta"]["tool_calls"][0];
        match mutation {
            "index" => {
                call.as_object_mut().unwrap().remove("index");
            }
            "arguments" => call["function"]["arguments"] = json!(7),
            "name" => call["function"]["name"] = json!(null),
            _ => call["type"] = json!("unsupported"),
        }
        assert!(
            policy.prepare(&case, &capture(values)).is_err(),
            "{mutation}"
        );
    }
}

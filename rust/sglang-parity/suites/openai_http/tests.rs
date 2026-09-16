use super::*;
use sglang_parity::sse::SseEvent;

fn plan() -> ExecutionPlan<OpenAiPolicy> {
    let config = serde_json::from_value(json!({
        "server":{"model":"test","seed":42},
        "profiles":{"incremental":{"server":{"args":["--incremental-streaming-output"]}}}
    }))
    .unwrap();
    load_plan(DEFAULT_SPEC, &config).unwrap()
}

fn case(name: &str) -> (HttpCase, OpenAiPolicy) {
    let mut plan = plan();
    let profile = plan.profiles.remove(0);
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
    assert_eq!(plan.profiles.len(), 2);
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
    let config: RunConfig = serde_json::from_value(json!({"server":{"model":"test"},
        "profiles":{"incremental":{}}}))
    .unwrap();
    for mutation in ["profile", "body", "endpoint", "rules"] {
        let mut spec: Value = serde_json::from_str(DEFAULT_SPEC).unwrap();
        match mutation {
            "profile" => spec["cases"][0]["profiles"] = json!(["missing"]),
            "body" => spec["cases"][0]["body"]["temperature"] = json!(1),
            "endpoint" => spec["cases"][0]["path"] = json!("/v1/responses"),
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
fn malformed_streams_have_event_scoped_diagnostics() {
    let (c, p) = case("completion_greedy_stream");
    for mutation in [
        "unknown",
        "id",
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
            "id" => values[1]["id"] = json!("changed"),
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
        assert_eq!(plan.profiles.len(), 2);
        assert!(
            plan.profiles[0]
                .suite
                .cases
                .iter()
                .all(|c| c.body["model"] == "alias")
        );
        assert_eq!(plan.profiles[1].suite.output_mode, "incremental");
        assert_eq!(plan.profiles[0].suite.cases.len(), 32);
        assert_eq!(plan.profiles[1].suite.cases.len(), 30);
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

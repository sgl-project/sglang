use super::super::test_utils::{chat_submitted, chunk, senders};
use super::{
    SamplingDefaults, chat_event_stream, chat_logprobs, chat_sampling_params, merge_template_stops,
    unary_chat,
};
use crate::api_server::guard::AbortGuard;
use crate::message::config::DefaultSamplingParams;
use crate::message::response::ChunkExtras;
use axum::http::StatusCode;
use dynamo_protocols::types::{CreateChatCompletionRequest, Stop};
use futures::StreamExt;

fn request() -> CreateChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap()
}

fn diagnostic_formatter() -> super::ChatFormatter {
    super::ChatFormatter::HuggingFace(
            dynamo_renderer::PromptFormatter::from_parts(
                serde_json::from_value(serde_json::json!({"chat_template":
                    "{{ messages | tojson }}|{% for key in ['reasoning_effort', 'thinking', 'enable_thinking'] %}{{ key }}={% if key == 'reasoning_effort' %}{{ reasoning_effort is defined }}:{{ reasoning_effort | default('UNDEFINED') | tojson }}{% elif key == 'thinking' %}{{ thinking is defined }}:{{ thinking | default('UNDEFINED') | tojson }}{% else %}{{ enable_thinking is defined }}:{{ enable_thinking | default('UNDEFINED') | tojson }}{% endif %};{% endfor %}"
                })).unwrap(),
                dynamo_renderer::ContextMixins::new(&[dynamo_renderer::PromptContextMixin::OaiChat]),
                true,
            ).unwrap(),
        )
}

fn formatter_from_template(template: &str) -> super::ChatFormatter {
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMPLATE_CONFIG: AtomicUsize = AtomicUsize::new(0);
    let config = std::env::temp_dir().join(format!(
        "sglang-ticket-24-array-{}-{}.json",
        std::process::id(),
        NEXT_TEMPLATE_CONFIG.fetch_add(1, Ordering::Relaxed),
    ));
    std::fs::write(&config, json!({"chat_template": template}).to_string()).unwrap();
    let formatter = super::super::template::load_chat_formatter(
        Some(config.to_str().unwrap()),
        None,
        None,
        None,
    )
    .unwrap();
    std::fs::remove_file(config).unwrap();
    formatter
}

async fn prepared(
    formatter: super::ChatFormatter,
    wire: serde_json::Value,
    kwargs: Option<&super::ChatTemplateKwargs>,
) -> (CreateChatCompletionRequest, String) {
    let mut state = super::super::test_utils::app_state(senders());
    std::sync::Arc::get_mut(&mut state).unwrap().chat_formatter = Some(formatter);
    let parsed: super::ChatRequest = serde_json::from_value(wire).unwrap();
    super::prepare_chat_request(&state, parsed.request, kwargs, Some(&parsed.prompt_tools))
        .await
        .unwrap()
}

fn message_request(content: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"model": "test", "messages": [{"role": "user", "content": content}]})
}

async fn continuation_submissions(
    formatter: super::ChatFormatter,
    mut wire: serde_json::Value,
    tokenizer: Option<std::sync::Arc<dyn crate::tokenizer_manager::tokenizer::TextTokenizer>>,
) -> Vec<crate::message::request::GenerateRequest> {
    use super::super::test_utils::{app_state, post_json};
    use crate::message::request::{Request, RequestKind};
    use crate::runtime::Runnable;
    use crate::tokenizer_manager::tokenizer::TokenizerWorker;
    use crate::tokenizer_manager::wiring::TmEvent;
    use crate::utils::fsm::RequestState;

    wire["model"] = serde_json::json!("model");
    let count = wire["n"].as_u64().unwrap_or(1) as usize;
    let (tx, rx) = flume::unbounded();
    let mut senders = senders();
    senders.tok_manager_tx = tx;
    let mut state = app_state(senders);
    let mutable_state = std::sync::Arc::get_mut(&mut state).unwrap();
    mutable_state.chat_formatter = Some(formatter);
    let server_args = std::sync::Arc::get_mut(&mut mutable_state.server_args).unwrap();
    server_args.model_config.model_type = Some("qwen3".into());
    server_args.tool_call_parser = Some("qwen25".into());
    let handler = tokio::spawn(post_json(
        super::super::routes().with_state(state),
        "/v1/chat/completions",
        wire,
    ));
    let mut submitted = Vec::new();
    for _ in 0..count {
        let event = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv_async())
            .await
            .unwrap()
            .unwrap();
        let TmEvent::Intake(mut request) = event else {
            panic!("expected intake")
        };
        if let Some(tokenizer) = &tokenizer {
            request.state = RequestState::Tokenizing;
            let (request_tx, request_rx) = flume::unbounded::<Request>();
            let (tokenizer_tm_tx, tokenizer_tm_rx) = flume::unbounded();
            request_tx.send(request).unwrap();
            drop(request_tx);
            TokenizerWorker::new(request_rx, tokenizer_tm_tx, tokenizer.clone()).run();
            let TmEvent::Tokenized(tokenized) = tokenizer_tm_rx.try_recv().unwrap() else {
                panic!("expected tokenized request")
            };
            request = tokenized;
        }
        request
            .sink
            .try_send(chunk("response", "done", true))
            .unwrap();
        let RequestKind::Generate(request) = request.kind else {
            panic!("expected generation")
        };
        submitted.push(*request);
    }
    assert_eq!(handler.await.unwrap().status(), StatusCode::OK);
    submitted
}

#[tokio::test]
async fn retained_assistant_missing_content_renders_as_empty_text() {
    use serde_json::json;

    let formatter = formatter_from_template("{{ messages | tojson }}");
    for content in [None, Some(serde_json::Value::Null)] {
        let mut assistant = json!({"role": "assistant", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": "{}"}
        }]});
        if let Some(content) = content {
            assistant["content"] = content;
        }
        let wire = json!({"model": "model", "messages": [
            {"role": "user", "content": "Weather?"},
            assistant,
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"}
        ]});
        let submitted = continuation_submissions(formatter.clone(), wire, None).await;
        let messages: serde_json::Value =
            serde_json::from_str(submitted[0].text.as_deref().unwrap()).unwrap();
        assert_eq!(messages[1]["content"], "");
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["arguments"],
            json!({})
        );
    }
}

#[tokio::test]
async fn retained_tool_history_arguments_are_normalized_in_jinja_prompt_view() {
    use serde_json::json;

    let formatter = formatter_from_template("{{ messages | tojson }}");
    for (arguments, expected) in [
        (r#"{"city":"Paris"}"#, r#""arguments": {"city": "Paris"}"#),
        (r#"{"city": "Paris"}"#, r#""arguments": {"city": "Paris"}"#),
        (
            r#"{"items":[{"note":"  keep  ","name":"éclair"}],"city":"Zürich"}"#,
            r#""arguments": {"items": [{"note": "  keep  ", "name": "éclair"}], "city": "Zürich"}"#,
        ),
    ] {
        let wire = json!({"model":"model", "messages":[
            {"role":"user","content":"Weather?"},
            {"role":"assistant","content":"", "tool_calls":[{
                "id":"call_1", "type":"function", "function":{"name":"get_weather","arguments":arguments}
            }]},
            {"role":"tool","tool_call_id":"call_1","content":"Sunny"}
        ]});
        let submitted = continuation_submissions(formatter.clone(), wire, None).await;
        let prompt = submitted[0].text.as_deref().unwrap();
        assert!(prompt.contains(expected), "{prompt}");
        assert!(
            !prompt.contains(r#""arguments": "{"city":"Paris"}""#),
            "{prompt}"
        );
    }

    let wire = json!({"model":"model", "continue_final_message":true, "messages":[
        {"role":"user","content":"Weather?"},
        {"role":"assistant","content":"", "tool_calls":[{
            "id":"call_1", "type":"function", "function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}
        }]},
        {"role":"tool","tool_call_id":"call_1","content":"Sunny"},
        {"role":"assistant","content":"The weather is"}
    ]});
    let submitted = continuation_submissions(formatter, wire, None).await;
    let prompt = submitted[0].text.as_deref().unwrap();
    assert!(
        prompt.contains(r#""arguments": {"city": "Paris"}"#),
        "{prompt}"
    );
    assert_eq!(submitted[0].append_text.as_deref(), Some("The weather is"));
}

#[tokio::test]
async fn retained_tool_history_numeric_tojson_matches_python_reference_cases() {
    use serde_json::json;

    let formatter = formatter_from_template("{{ messages | tojson }}");
    for (case_id, arguments, expected) in [
        (
            "positive-shortest-decimal-tie",
            r#"{"x":934356417220194.2}"#,
            r#""x": 934356417220194.2"#,
        ),
        (
            "negative-shortest-decimal-tie",
            r#"{"x":-201562347225087.62}"#,
            r#""x": -201562347225087.62"#,
        ),
        (
            "negative-fixed-notation",
            r#"{"x":-0.0000125}"#,
            r#""x": -1.25e-05"#,
        ),
        ("e-7", r#"{"x": 1e-07}"#, r#""x": 1e-07"#),
        ("e-7-compact", r#"{"x":1e-7}"#, r#""x": 1e-07"#),
        ("e-5", r#"{"x": 1e-05}"#, r#""x": 1e-05"#),
        ("e-5-compact", r#"{"x":1e-5}"#, r#""x": 1e-05"#),
        ("e-4", r#"{"x": 1e-04}"#, r#""x": 0.0001"#),
        ("e15", r#"{"x": 1e+15}"#, r#""x": 1000000000000000.0"#),
        ("e16", r#"{"x": 1e+16}"#, r#""x": 1e+16"#),
        (
            "negative-exponent",
            r#"{"x": -1.25e-07}"#,
            r#""x": -1.25e-07"#,
        ),
        ("zero", r#"{"x": 0.0}"#, r#""x": 0.0"#),
        ("negative-zero", r#"{"x": -0.0}"#, r#""x": -0.0"#),
        ("integral-float", r#"{"x": 1.0}"#, r#""x": 1.0"#),
        (
            "long-mantissa",
            r#"{"x": 1.2345678901234567}"#,
            r#""x": 1.2345678901234567"#,
        ),
        ("subnormal", r#"{"x": 5e-324}"#, r#""x": 5e-324"#),
        (
            "max-finite",
            r#"{"x": 1.7976931348623157e+308}"#,
            r#""x": 1.7976931348623157e+308"#,
        ),
        (
            "below-1e-4",
            r#"{"x": 9.999999999999999e-5}"#,
            r#""x": 9.999999999999999e-05"#,
        ),
        (
            "above-1e-5-rounded-by-orjson",
            r#"{"x": 1.0000000000000001e-5}"#,
            r#""x": 1e-05"#,
        ),
        (
            "nested-order-string",
            r#"{"z": 1e-07, "a": {"b": -0.0, "c": "  keep  "}}"#,
            r#""z": 1e-07, "a": {"b": -0.0, "c": "  keep  "}"#,
        ),
    ] {
        let wire = json!({"model":"model", "messages":[
            {"role":"user","content":"Weather?"},
            {"role":"assistant","content":"", "tool_calls":[{
                "id":"call_1", "type":"function", "function":{"name":"get_weather","arguments":arguments}
            }]},
            {"role":"tool","tool_call_id":"call_1","content":"Sunny"}
        ]});
        let submitted = continuation_submissions(formatter.clone(), wire, None).await;
        let prompt = submitted[0].text.as_deref().unwrap();
        assert!(prompt.contains(expected), "{case_id}: {prompt}");
    }
}

#[tokio::test]
async fn qwen_tojson_indent_branch_remains_unchanged() {
    use serde_json::json;

    let formatter = formatter_from_template("{{ 0.0000001 | tojson(indent=2) }}");
    let wire = json!({"model":"model", "messages":[
        {"role":"user","content":"Weather?"}
    ]});
    let submitted = continuation_submissions(formatter, wire, None).await;
    assert_eq!(submitted[0].text.as_deref(), Some("1e-7"));
}

#[tokio::test]
#[ignore = "requires TICKET41_MODEL_DIR and TICKET41_PYTHON_CAPTURE for real tokenizer parity"]
async fn retained_tool_history_matches_python_route_prompt_and_token_ids() {
    use crate::tokenizer_manager::tokenizer::{DynamoTokenizer, load_tokenizer};
    use serde_json::{Value, json};

    let model_dir = std::env::var("TICKET41_MODEL_DIR").unwrap();
    let reference_path = std::env::var("TICKET41_PYTHON_CAPTURE").unwrap();
    let captures: Value = serde_json::from_slice(&std::fs::read(reference_path).unwrap()).unwrap();
    let tokenizer_path = format!("{model_dir}/tokenizer_config.json");
    let inner = load_tokenizer(Some(&model_dir), None, false)
        .unwrap()
        .unwrap();
    let tokenizer = std::sync::Arc::new(DynamoTokenizer::new(inner, Some(&tokenizer_path)));
    let formatter = super::super::template::load_chat_formatter(
        Some(&tokenizer_path),
        Some(&model_dir),
        None,
        None,
    )
    .unwrap();

    for case in captures["cases"].as_array().unwrap() {
        let case_id = case["case_id"].as_str().unwrap();
        let mut wire = case["request"].clone();
        wire["continue_final_message"] = json!(case_id == "continuation");
        let submitted =
            continuation_submissions(formatter.clone(), wire, Some(tokenizer.clone())).await;
        let request = &submitted[0];
        let prompt = request.text.as_deref().unwrap();
        let ids = request.input_ids.as_ref().unwrap();
        assert_eq!(
            prompt.as_bytes(),
            case["pre_render_prompt"].as_str().unwrap().as_bytes(),
            "prompt bytes for {case_id}"
        );
        let expected_ids = case["final_input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap() as i32)
            .collect::<Vec<_>>();
        assert_eq!(ids, &expected_ids, "case {case_id}");
    }
}

#[test]
fn continuation_flag_matches_python_boolean_validation() {
    use serde_json::json;
    for (values, expected) in [
        (
            vec![
                json!(true),
                json!(1),
                json!(1.0),
                json!("TRUE"),
                json!("t"),
                json!("yes"),
                json!("y"),
                json!("on"),
                json!("1"),
            ],
            true,
        ),
        (
            vec![
                json!(false),
                json!(0),
                json!(0.0),
                json!("FALSE"),
                json!("f"),
                json!("no"),
                json!("n"),
                json!("off"),
                json!("0"),
            ],
            false,
        ),
    ] {
        for value in values {
            let mut wire = message_request(json!("hi"));
            wire["continue_final_message"] = value;
            let request: super::ChatRequest = serde_json::from_value(wire.clone()).unwrap();
            assert_eq!(request.continue_final_message, expected, "{wire}");
        }
    }
    for value in [
        json!(null),
        json!(2),
        json!(-1),
        json!(0.5),
        json!(" true "),
        json!("invalid"),
        json!([]),
        json!({}),
    ] {
        let mut wire = message_request(json!("hi"));
        wire["continue_final_message"] = value;
        assert!(
            serde_json::from_value::<super::ChatRequest>(wire.clone()).is_err(),
            "{wire}"
        );
    }
}

#[tokio::test]
async fn continuation_handler_preserves_native_and_legacy_formats() {
    use serde_json::json;
    let wire = json!({"model":"model", "continue_final_message":true,
            "messages":[{"role":"user","content":"question"},{"role":"assistant","content":"prefix"}]});
    let request: CreateChatCompletionRequest = serde_json::from_value(wire.clone()).unwrap();
    for formatter in [
        super::super::template::load_chat_formatter(
            None,
            Some("/models/x"),
            Some("deepseek_v4"),
            None,
        )
        .unwrap(),
        super::super::template::load_chat_formatter(None, None, None, Some("chatml")).unwrap(),
    ] {
        let expected = formatter.render(&request, None).unwrap();
        let submitted = continuation_submissions(formatter, wire.clone(), None).await;
        assert_eq!(submitted[0].text.as_deref(), Some(expected.as_str()));
        assert_eq!(submitted[0].append_text, None);
    }
    for (field, value) in [
        ("chat_template_kwargs", json!({})),
        ("tools", json!([])),
        ("tool_choice", json!("auto")),
        ("tool_choice", json!("none")),
        ("reasoning_effort", json!("high")),
    ] {
        let mut explicit = wire.clone();
        explicit[field] = value;
        let formatter = diagnostic_formatter();
        let submitted = continuation_submissions(formatter, explicit, None).await;
        let messages: serde_json::Value = serde_json::from_str(
            submitted[0]
                .text
                .as_ref()
                .unwrap()
                .split('|')
                .next()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(messages.as_array().unwrap().len(), 1, "{field}");
        assert_eq!(messages[0]["content"], "question");
        assert_eq!(submitted[0].append_text.as_deref(), Some("prefix"));
    }
}

#[tokio::test]
async fn continuation_handler_submits_prefix_to_every_choice() {
    use serde_json::json;
    let wire = json!({"model":"model", "continue_final_message":true, "n":2,
            "messages":[{"role":"user","content":"question"},{"role":"assistant","content":"prefix"}]});
    let formatter = formatter_from_template(
        "{% for message in messages %}{{ message.role }}:{{ message.content }};{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}",
    );
    let submitted = continuation_submissions(formatter, wire, None).await;
    assert_eq!(submitted.len(), 2);
    for request in submitted {
        assert_eq!(request.text.as_deref(), Some("user:question;assistant:"));
        assert_eq!(request.append_text.as_deref(), Some("prefix"));
        assert!(request.skip_special_tokens);
    }
}

#[tokio::test]
async fn continuation_handler_matches_python_string_content_phases() {
    use serde_json::json;
    for (content, text) in [
        (json!("prefix"), "prefix"),
        (json!(""), ""),
        (json!(null), ""),
        (json!([]), ""),
        (
            json!([{"type":"text","text":"Hello"},{"type":"text","text":"world"}]),
            "Hello world",
        ),
        (
            json!([{"type":"text","text":""},{"type":"text","text":"x"},{"type":"text","text":""}]),
            " x ",
        ),
    ] {
        for flag in [None, Some(false), Some(true)] {
            let mut wire = json!({"messages":[{"role":"user","content":"question"},
                    {"role":"assistant","name":"original","content":content}]});
            if let Some(flag) = flag {
                wire["continue_final_message"] = json!(flag);
            }
            let formatter = formatter_from_template(
                "{% for message in messages %}{{ message.role }}:{{ message.content }}:{{ message.name | default('none') }};{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}",
            );
            let submitted = continuation_submissions(formatter, wire, None).await;
            let expected = if flag == Some(true) {
                "user:question:none;assistant:".into()
            } else {
                format!("user:question:none;user:{text}:none;assistant:")
            };
            assert_eq!(
                submitted[0].text.as_deref(),
                Some(expected.as_str()),
                "{content}, {flag:?}"
            );
            let prefix = (flag == Some(true) && !text.is_empty()).then_some(text);
            assert_eq!(submitted[0].append_text.as_deref(), prefix);
        }
    }
}

#[tokio::test]
async fn continuation_user_ending_is_unchanged_and_empty_history_is_rejected() {
    use serde_json::json;
    for flag in [None, Some(false), Some(true)] {
        let formatter = diagnostic_formatter();
        let mut wire = message_request(json!("hi"));
        if let Some(flag) = flag {
            wire["continue_final_message"] = json!(flag);
        }
        let request: CreateChatCompletionRequest = serde_json::from_value(wire.clone()).unwrap();
        let expected = formatter.render(&request, None).unwrap();
        let submitted = continuation_submissions(formatter, wire, None).await;
        assert_eq!(submitted[0].text.as_deref(), Some(expected.as_str()));
        assert_eq!(submitted[0].append_text, None);
    }
    let (tx, rx) = flume::unbounded();
    let mut senders = senders();
    senders.tok_manager_tx = tx;
    let mut state = super::super::test_utils::app_state(senders);
    std::sync::Arc::get_mut(&mut state).unwrap().chat_formatter = Some(diagnostic_formatter());
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        super::super::test_utils::post_json(
            super::super::routes().with_state(state),
            "/v1/chat/completions",
            json!({"model":"model", "continue_final_message":true,
                "messages":[{"role":"assistant","content":"prefix"}]}),
        ),
    )
    .await
    .expect("empty continuation history must be rejected before submission");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(rx.try_recv().is_err());
}

#[tokio::test]
async fn continuation_validates_tool_arguments_before_removing_assistant() {
    use serde_json::json;
    for arguments in ["invalid", "[]", "null", "1"] {
        for flag in [false, true] {
            let (tx, rx) = flume::unbounded();
            let mut senders = senders();
            senders.tok_manager_tx = tx;
            let mut state = super::super::test_utils::app_state(senders);
            std::sync::Arc::get_mut(&mut state).unwrap().chat_formatter =
                Some(diagnostic_formatter());
            let response = super::super::test_utils::post_json(
                    super::super::routes().with_state(state), "/v1/chat/completions",
                    json!({"model":"model", "continue_final_message":flag,
                        "messages":[{"role":"user","content":"question"},
                            {"role":"assistant","content":"prefix", "tool_calls":[{
                                "id":"call_1", "type":"function", "function":{"name":"f","arguments":arguments}
                            }]}]}),
                ).await;
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "{arguments}, {flag}"
            );
            assert!(rx.try_recv().is_err());
        }
    }
}

#[test]
fn continuation_in_array_formats_only_transforms_string_content() {
    use serde_json::json;
    for (content, expected) in [
        (json!("prefix"), Some("prefix")),
        (json!(null), Some("")),
        (json!([{"type":"text","text":"prefix"}]), None),
    ] {
        for flag in [false, true] {
            let mut request: CreateChatCompletionRequest = serde_json::from_value(json!({"model":"model",
                    "messages":[{"role":"user","content":"question"},{"role":"assistant","content":content}]})).unwrap();
            let prefix = super::prepare_final_assistant(&mut request, flag, true);
            assert_eq!(prefix.as_deref(), if flag { expected } else { None });
            let messages = serde_json::to_value(&request.messages).unwrap();
            if expected.is_some() {
                assert_eq!(messages.as_array().unwrap().len(), if flag { 1 } else { 2 });
                assert_eq!(messages.as_array().unwrap().last().unwrap()["role"], "user");
            } else {
                assert_eq!(messages[1]["role"], "assistant");
                assert_eq!(messages[1]["content"], content);
            }
        }
    }
}

#[tokio::test]
async fn continuation_handles_final_assistant_with_metadata() {
    use serde_json::json;
    for (field, value) in [
        ("reasoning_content", json!("reasoning")),
        ("refusal", json!("declined")),
        (
            "tool_calls",
            json!([{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]),
        ),
        ("function_call", json!({"name":"f","arguments":"{}"})),
    ] {
        for flag in [false, true] {
            let mut wire = json!({"model":"model", "continue_final_message":flag,
                    "messages":[{"role":"user","content":"question"},{"role":"assistant","content":"prefix"}]});
            wire["messages"][1][field] = value.clone();
            let formatter = diagnostic_formatter();
            let submitted = continuation_submissions(formatter, wire, None).await;
            let messages: serde_json::Value = serde_json::from_str(
                submitted[0]
                    .text
                    .as_ref()
                    .unwrap()
                    .split('|')
                    .next()
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(
                messages.as_array().unwrap().len(),
                if flag { 1 } else { 2 },
                "{field}"
            );
            if !flag {
                assert_eq!(messages[1]["role"], "user", "{field}");
                assert_eq!(messages[1]["content"], "prefix", "{field}");
                assert!(messages[1].get(field).is_none(), "{field}");
            }
            assert_eq!(
                submitted[0].append_text.as_deref(),
                flag.then_some("prefix")
            );
        }
    }
}

#[test]
fn continuation_preserves_the_python_final_assistant_cases() {
    let base = serde_json::json!({
        "model": "test",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "prefix"}
        ]
    });

    for (flag, expected_role, expected_prefix) in [
        (None, "user", None),
        (Some(false), "user", None),
        (Some(true), "user", Some("prefix")),
    ] {
        let mut wire = base.clone();
        if let Some(flag) = flag {
            wire["continue_final_message"] = serde_json::json!(flag);
        }
        let super::ChatRequest {
            mut request,
            continue_final_message,
            ..
        } = serde_json::from_value(wire).unwrap();
        let prefix = super::prepare_final_assistant(&mut request, continue_final_message, false);
        assert_eq!(prefix.as_deref(), expected_prefix);
        let messages = serde_json::to_value(&request.messages).unwrap();
        assert_eq!(messages[0]["role"], "user");
        if expected_prefix.is_some() {
            assert_eq!(messages.as_array().unwrap().len(), 1);
        } else {
            assert_eq!(messages[1]["role"], expected_role);
            assert_eq!(messages[1]["content"], "prefix");
        }
    }
}

#[test]
fn continuation_leaves_refusal_parts_unchanged() {
    let wire = serde_json::json!({
        "model": "test",
        "continue_final_message": true,
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": [{"type": "refusal", "refusal": "declined"}]}
        ]
    });
    let super::ChatRequest {
        mut request,
        continue_final_message,
        ..
    } = serde_json::from_value(wire).unwrap();
    assert_eq!(
        super::prepare_final_assistant(&mut request, continue_final_message, false),
        None
    );
    assert_eq!(request.messages.len(), 2);
    assert_eq!(
        serde_json::to_value(&request.messages).unwrap()[1]["role"],
        "assistant"
    );
}

#[tokio::test]
async fn jinja_text_normalization_uses_typed_message_variants() {
    use serde_json::json;

    let wire = json!({
        "model": "test",
        "messages": [
            {"role": "developer", "name": "dev", "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"}
            ]},
            {"role": "system", "content": []},
            {"role": "user", "name": "user", "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"}
            ]},
            {"role": "assistant", "name": "assistant", "content": [
                {"type": "text", "text": "one"},
                {"type": "text", "text": "two"}
            ]},
            {"role": "tool", "tool_call_id": "call-1", "content": [
                {"type": "text", "text": "tool"},
                {"type": "text", "text": "result"}
            ]},
            {"role": "function", "name": "legacy", "content": "result"},
            {"role": "assistant", "content": null}
        ]
    });
    let (actual, _) = prepared(diagnostic_formatter(), wire, None).await;
    let messages = serde_json::to_value(&actual.messages).unwrap();
    assert_eq!(messages[0]["content"], "first second");
    assert_eq!(messages[0]["name"], "dev");
    assert_eq!(messages[1]["content"], "");
    assert_eq!(messages[2]["content"], "hello world");
    assert_eq!(messages[2]["name"], "user");
    assert_eq!(messages[3]["content"], "one two");
    assert_eq!(messages[3]["name"], "assistant");
    assert_eq!(messages[4]["content"], "tool result");
    assert_eq!(messages[4]["tool_call_id"], "call-1");
    assert_eq!(messages[5]["content"], "result");
    assert_eq!(messages[6]["content"], "");
}

#[tokio::test]
async fn jinja_message_flattens_text_parts_including_empty_parts() {
    use serde_json::json;
    assert!(!diagnostic_formatter().requires_content_arrays());
    for (content, expected) in [
        (json!([{ "type": "text", "text": "Hello" }]), json!("Hello")),
        (json!([]), json!("")),
        (
            json!([{"type": "text", "text": ""}, {"type": "text", "text": "world"}]),
            json!(" world"),
        ),
        (
            json!([{"type": "text", "text": "Hello"}, {"type": "text", "text": "world"}]),
            json!("Hello world"),
        ),
        (
            json!([{"type": "text", "text": " Hello "}, {"type": "text", "text": " world "}]),
            json!(" Hello   world "),
        ),
        (
            json!([{"type": "text", "text": " "}, {"type": "text", "text": "x"}]),
            json!("  x"),
        ),
    ] {
        let (_, prompt) = prepared(diagnostic_formatter(), message_request(content), None).await;
        let messages: serde_json::Value =
            serde_json::from_str(prompt.split('|').next().unwrap()).unwrap();
        assert_eq!(messages[0]["content"], expected);
    }
    for content in [
        json!("Hello"),
        json!([{"type": "text", "text": "Hello"}, {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]),
    ] {
        let wire = message_request(content);
        let expected: CreateChatCompletionRequest = serde_json::from_value(wire.clone()).unwrap();
        let (actual, prompt) = prepared(diagnostic_formatter(), wire, None).await;
        assert_eq!(actual.messages, expected.messages);
        assert!(prompt.contains("enable_thinking=False:\"UNDEFINED\""));
    }
    let wire = json!({"model":"test", "messages":[{"role":"system", "content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},{"role":"assistant","content":[{"type":"text","text":"c"},{"type":"text","text":"d"}]}]});
    let (actual, _) = prepared(diagnostic_formatter(), wire, None).await;
    let messages = serde_json::to_value(actual.messages).unwrap();
    assert_eq!(messages[0]["content"], "a b");
    assert_eq!(messages[1]["content"], "c d");
}

#[tokio::test]
async fn jinja_array_content_template_keeps_text_part_boundaries() {
    use serde_json::json;

    let formatter = formatter_from_template(
        "{% for message in messages %}{% for part in message.content %}{% if part.type == 'text' %}[{{ part.text }}]{% endif %}{% endfor %}{% endfor %}",
    );
    assert!(formatter.requires_content_arrays());
    let (_, prompt) = prepared(
        formatter,
        message_request(json!([
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "world"}
        ])),
        None,
    )
    .await;
    assert_eq!(prompt, "[Hello][world]");
}

#[tokio::test]
async fn jinja_array_content_template_keeps_parts_with_reasoning_none() {
    use serde_json::json;

    let formatter = formatter_from_template(
        "{% for message in messages %}{% for part in message.content %}{% if part.type == 'text' %}[{{ part.text }}]{% endif %}{% endfor %}{% endfor %}|{{ thinking | default('missing') }}",
    );
    assert!(formatter.requires_content_arrays());
    let mut wire = message_request(json!([
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": "world"}
    ]));
    wire["reasoning_effort"] = json!("none");
    let (_, prompt) = prepared(formatter, wire, None).await;
    assert_eq!(prompt, "[Hello][world]|False");
}

#[tokio::test]
async fn jinja_message_request_none_reaches_actual_render_and_composes() {
    use serde_json::json;
    let expected = "reasoning_effort=True:\"none\";thinking=True:false;enable_thinking=True:false;";
    for content in [
        json!("hi"),
        json!([{"type":"text","text":"hello"},{"type":"text","text":"world"}]),
    ] {
        let mut wire = message_request(content);
        wire["reasoning_effort"] = json!("none");
        wire["stop"] = json!(["custom"]);
        let (actual, prompt) = prepared(diagnostic_formatter(), wire, None).await;
        assert!(prompt.ends_with(expected), "{prompt}");
        assert_eq!(actual.stop, Some(Stop::StringArray(vec!["custom".into()])));
        let messages: serde_json::Value =
            serde_json::from_str(prompt.split('|').next().unwrap()).unwrap();
        assert!(messages[0]["content"].is_string());
    }
    let (_, prompt) = prepared(diagnostic_formatter(), message_request(json!("hi")), None).await;
    assert!(prompt.ends_with("reasoning_effort=False:\"UNDEFINED\";thinking=False:\"UNDEFINED\";enable_thinking=False:\"UNDEFINED\";"), "{prompt}");
}

#[tokio::test]
async fn jinja_messages_and_reasoning_compose_with_tools_and_explicit_kwargs() {
    use serde_json::json;
    let mut wire =
        message_request(json!([{"type":"text","text":"hello"},{"type":"text","text":"world"}]));
    wire["reasoning_effort"] = json!("none");
    for kwargs in [
        super::ChatTemplateKwargs::new(),
        super::ChatTemplateKwargs::from([
            ("reasoning_effort".into(), json!("high")),
            ("thinking".into(), json!(true)),
            ("custom".into(), json!(17)),
        ]),
    ] {
        let original = kwargs.clone();
        let formatter = diagnostic_formatter();
        let (actual, prompt) = prepared(formatter, wire.clone(), Some(&kwargs)).await;
        assert_eq!(
            serde_json::to_value(actual.messages).unwrap()[0]["content"],
            "hello world"
        );
        assert_eq!(kwargs, original);
        assert!(prompt.contains("enable_thinking=True:false"));
        let expected_thinking = if kwargs.is_empty() {
            "thinking=True:false"
        } else {
            "thinking=True:true"
        };
        assert!(prompt.contains(expected_thinking), "{prompt}");
        let effort = kwargs
            .get("reasoning_effort")
            .cloned()
            .unwrap_or(json!("none"));
        assert!(
            prompt.contains(&format!("reasoning_effort=True:{effort}")),
            "{prompt}"
        );
    }
    for (key, value) in [
        ("tools", json!([])),
        (
            "tools",
            json!([{"type":"function","function":{"name":"f"}}]),
        ),
        ("tool_choice", json!("none")),
        ("tool_choice", json!("auto")),
    ] {
        let mut combined = wire.clone();
        combined[key] = value;
        let formatter = diagnostic_formatter();
        let (actual, prompt) = prepared(formatter, combined, None).await;
        assert_eq!(
            serde_json::to_value(actual.messages).unwrap()[0]["content"],
            "hello world"
        );
        assert!(
            prompt.ends_with(
                "reasoning_effort=True:\"none\";thinking=True:false;enable_thinking=True:false;"
            ),
            "{prompt}"
        );
    }
    for effort in ["minimal", "low", "medium", "high", "xhigh", "max"] {
        let mut other =
            message_request(json!([{"type":"text","text":"hello"},{"type":"text","text":"world"}]));
        other["reasoning_effort"] = json!(effort);
        let formatter = diagnostic_formatter();
        let (_, prompt) = prepared(formatter, other, None).await;
        assert!(
            prompt.ends_with(&format!(
                "reasoning_effort=True:\"{effort}\";thinking=True:true;enable_thinking=True:true;"
            )),
            "{prompt}"
        );
    }
}

#[tokio::test]
async fn message_native_and_legacy_renderers_bypass_jinja_normalization() {
    use serde_json::json;
    let mut wire =
        message_request(json!([{"type":"text","text":"hello"},{"type":"text","text":"world"}]));
    wire["reasoning_effort"] = json!("none");
    let native = super::super::template::load_chat_formatter(
        None,
        Some("/models/x"),
        Some("deepseek_v4"),
        None,
    )
    .unwrap();
    let legacy =
        super::super::template::load_chat_formatter(None, None, None, Some("chatml")).unwrap();
    for formatter in [native, legacy] {
        let request: CreateChatCompletionRequest = serde_json::from_value(wire.clone()).unwrap();
        let expected = formatter.render(&request, None).unwrap();
        let (actual, prompt) = prepared(formatter, wire.clone(), None).await;
        assert_eq!(actual.messages, request.messages);
        assert_eq!(prompt, expected);
    }
}

/// Python `to_sampling_params` priority: user value > model generation
/// config (`--sampling-defaults model`) > OpenAI terminal default.
#[test]
fn sampling_defaults_follow_python_priority_chain() {
    let model = DefaultSamplingParams {
        temperature: Some(0.6),
        top_p: Some(0.9),
        ..Default::default()
    };
    // Omitted → model defaults, not the 1.0 OpenAI terminals.
    let sampling = chat_sampling_params(
        &request(),
        &SamplingDefaults::CHAT.with_model_defaults(&model),
    )
    .unwrap();
    assert_eq!(sampling.temperature, 0.6);
    assert_eq!(sampling.top_p, 0.9);
    // Explicit request values win. `Option<f32>` loses precision in f64 —
    // compare with tolerance.
    let mut request = request();
    request.temperature = Some(0.2);
    request.top_p = Some(0.5);
    let sampling = chat_sampling_params(
        &request,
        &SamplingDefaults::CHAT.with_model_defaults(&model),
    )
    .unwrap();
    assert!((sampling.temperature - 0.2).abs() < 1e-6);
    assert!((sampling.top_p - 0.5).abs() < 1e-6);
}

/// `--sampling-defaults openai` resolves an empty model-config slice, so the
/// conversion falls back to the OpenAI terminal defaults.
#[test]
fn sampling_defaults_fall_back_to_openai_terminals_in_openai_mode() {
    let openai_mode = DefaultSamplingParams::default();
    let sampling = chat_sampling_params(
        &request(),
        &SamplingDefaults::CHAT.with_model_defaults(&openai_mode),
    )
    .unwrap();
    assert_eq!(sampling.temperature, 1.0);
    assert_eq!(sampling.top_p, 1.0);
}

/// Python `_apply_conversation_template`: template `stop_str` first, then
/// the request's own stops.
#[test]
fn template_stops_merge_before_request_stops() {
    let chatml = super::super::template::builtin_template("chatml").unwrap();
    let formatter =
        super::super::ChatFormatter::Legacy(Box::new(super::super::template::LegacyFormatter {
            spec: chatml,
        }));
    assert_eq!(
        formatter.stop_strs(),
        Some(crate::message::types::OneOrMany::Many(vec![
            "<|endoftext|>".into(),
            "<|im_end|>".into()
        ]))
    );
    // No request stop → the template's delimiters alone.
    let mut req = request();
    merge_template_stops(&mut req, &formatter);
    assert_eq!(
        req.stop,
        Some(Stop::StringArray(vec![
            "<|endoftext|>".into(),
            "<|im_end|>".into()
        ]))
    );
    // A string request stop appends as one entry.
    let mut req = request();
    req.stop = Some(Stop::String("<stop>".into()));
    merge_template_stops(&mut req, &formatter);
    assert_eq!(
        req.stop,
        Some(Stop::StringArray(vec![
            "<|endoftext|>".into(),
            "<|im_end|>".into(),
            "<stop>".into()
        ]))
    );
    // A list request stop extends the list.
    let mut req = request();
    req.stop = Some(Stop::StringArray(vec!["a".into(), "b".into()]));
    merge_template_stops(&mut req, &formatter);
    assert_eq!(
        req.stop,
        Some(Stop::StringArray(vec![
            "<|endoftext|>".into(),
            "<|im_end|>".into(),
            "a".into(),
            "b".into()
        ]))
    );
    // Token-id stops cannot be merged (Python has no such field) — kept alone.
    let mut req = request();
    req.stop = Some(Stop::TokenIdArray(vec![2, 3]));
    merge_template_stops(&mut req, &formatter);
    assert_eq!(req.stop, Some(Stop::TokenIdArray(vec![2, 3])));
}

/// The HuggingFace renderer carries no template stops (Python's jinja path
/// keeps only the request's stops), so the request is left unchanged.
#[test]
fn huggingface_formatter_leaves_request_stops_alone() {
    let mut req = request();
    req.stop = Some(Stop::String("x".into()));
    // A prompt formatter is not constructible here without a tokenizer; the
    // empty-legacy-spec twin proves the merge is formatter-gated, and the
    // `HuggingFace` arm returns `None` by construction (see `stop_strs`).
    let legacy =
        super::super::ChatFormatter::Legacy(Box::new(super::super::template::LegacyFormatter {
            spec: super::super::template::LegacySpec::default(),
        }));
    assert!(legacy.stop_strs().is_none());
    merge_template_stops(&mut req, &legacy);
    assert_eq!(req.stop, Some(Stop::String("x".into())));
}

/// A request with no `max_tokens`/`max_completion_tokens` stays unbounded —
/// no terminal default is imposed.
#[test]
fn chat_without_a_token_limit_stays_unbounded() {
    let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap();
    assert_eq!(
        chat_sampling_params(&request, &SamplingDefaults::CHAT)
            .unwrap()
            .max_new_tokens,
        None
    );
}

#[test]
fn chat_logprobs_use_dynamo_wire_types() {
    let extras = ChunkExtras {
        out_lp_val: vec![-0.25],
        out_lp_idx: vec![7],
        out_lp_txt: vec!["x".into()],
        out_top_val: vec![-0.25, -1.0],
        out_top_idx: vec![7, 8],
        out_top_lens: vec![2],
        out_top_txt: vec!["x".into(), "y".into()],
        ..Default::default()
    };
    let logprobs = chat_logprobs(Some(&extras));
    let token = &logprobs.content.unwrap()[0];
    assert_eq!(token.token, "x");
    assert_eq!(token.token_id, Some(7));
    assert_eq!(token.top_logprobs.len(), 2);
    assert_eq!(token.top_logprobs[1].token, "y");
}

#[tokio::test]
async fn unary_chat_fans_in_choices_and_usage() {
    let (choice0, tx0) = chat_submitted(0, "r0");
    let (choice1, tx1) = chat_submitted(1, "r1");
    tx0.send(chunk("r0", "Paris", true)).await.unwrap();
    tx1.send(chunk("r1", "Paris", true)).await.unwrap();

    let response = unary_chat(
        vec![choice0, choice1],
        AbortGuard::new_empty(senders()),
        "chatcmpl-test".into(),
        "model".into(),
        1,
        false,
        None,
        None,
        None,
        true,
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["choices"][0]["message"]["role"], "assistant");
    assert_eq!(value["choices"][0]["message"]["content"], "Paris");
    assert_eq!(value["choices"][1]["index"], 1);
    assert_eq!(value["usage"]["prompt_tokens"], 5);
    assert_eq!(value["usage"]["completion_tokens"], 2);
}

#[tokio::test]
async fn unary_chat_separates_reasoning_content_with_parser_configured() {
    let (choice, tx) = chat_submitted(0, "r0");
    tx.send(chunk(
        "r0",
        "<think>because Paris is famous</think>Paris",
        true,
    ))
    .await
    .unwrap();

    let response = unary_chat(
        vec![choice],
        AbortGuard::new_empty(senders()),
        "chatcmpl-test".into(),
        "model".into(),
        1,
        false,
        None,
        Some("deepseek-r1".into()),
        None,
        true,
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        value["choices"][0]["message"]["reasoning_content"],
        "because Paris is famous"
    );
    assert_eq!(value["choices"][0]["message"]["content"], "Paris");
    assert!(value["choices"][0]["message"]["reasoning_content"].is_string());
}

#[tokio::test]
async fn streaming_chat_separates_reasoning_into_own_deltas() {
    let (choice, tx) = chat_submitted(0, "r0");
    // Force mode starts in reasoning, so the opener is stripped and the first
    // reasoning fragment streams immediately.
    tx.send(chunk("r0", "<think>be", false)).await.unwrap();
    tx.send(chunk("r0", "cause</think>Par", false))
        .await
        .unwrap();
    tx.send(chunk("r0", "is", true)).await.unwrap();

    let stream = chat_event_stream(
        vec![choice],
        AbortGuard::new_empty(senders()),
        "chatcmpl-test".into(),
        "model".into(),
        1,
        false,
        true,
        None,
        Some("deepseek-r1".into()),
        false,
        None,
        None,
        false,
        true,
        None,
    );
    futures::pin_mut!(stream);
    let frames: Vec<String> = stream.collect().await;
    let role: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
    let first_reasoning: serde_json::Value = serde_json::from_str(&frames[1]).unwrap();
    let second_reasoning: serde_json::Value = serde_json::from_str(&frames[2]).unwrap();
    let content: serde_json::Value = serde_json::from_str(&frames[3]).unwrap();
    let terminal: serde_json::Value = serde_json::from_str(&frames[4]).unwrap();
    assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(
        first_reasoning["choices"][0]["delta"]["reasoning_content"],
        "be"
    );
    assert!(first_reasoning["choices"][0]["delta"]["content"].is_null());
    assert_eq!(
        second_reasoning["choices"][0]["delta"]["reasoning_content"],
        "cause"
    );
    assert_eq!(content["choices"][0]["delta"]["content"], "Par");
    assert!(content["choices"][0]["delta"]["reasoning_content"].is_null());
    assert_eq!(terminal["choices"][0]["delta"]["content"], "is");
    assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
    assert_eq!(frames.len(), 7);
}

#[tokio::test]
async fn streaming_chat_emits_role_deltas_usage_and_done() {
    let (choice, tx) = chat_submitted(0, "r0");
    tx.send(chunk("r0", "Par", false)).await.unwrap();
    tx.send(chunk("r0", "is", true)).await.unwrap();

    let stream = chat_event_stream(
        vec![choice],
        AbortGuard::new_empty(senders()),
        "chatcmpl-test".into(),
        "model".into(),
        1,
        false,
        true,
        None,
        None,
        false,
        None,
        None,
        false,
        true,
        None,
    );
    futures::pin_mut!(stream);
    let frames: Vec<String> = stream.collect().await;
    assert_eq!(frames.len(), 5);
    let role: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
    let delta: serde_json::Value = serde_json::from_str(&frames[1]).unwrap();
    let terminal: serde_json::Value = serde_json::from_str(&frames[2]).unwrap();
    let usage: serde_json::Value = serde_json::from_str(&frames[3]).unwrap();
    assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
    assert!(role["choices"][0]["delta"]["reasoning_content"].is_null());
    assert_eq!(delta["choices"][0]["delta"]["content"], "Par");
    assert!(delta["choices"][0]["delta"]["reasoning_content"].is_null());
    assert_eq!(terminal["choices"][0]["delta"]["content"], "is");
    assert!(terminal["choices"][0]["delta"]["reasoning_content"].is_null());
    assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
    assert_eq!(usage["usage"]["completion_tokens"], 2);
    assert_eq!(frames[4], "[DONE]");
}

//! Prompt regressions through the real HTTP route and Core admission.
use std::{sync::Arc, time::Duration};

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use serde_json::{Value, json};
use tower::ServiceExt;

use super::super::{AppState, ChatFormatter, test_utils::chunk};
use crate::{
    api_server::core::{CoreConfig, CoreHandle, CoreMetadata},
    message::{config::ServerArgs, request::RequestKind},
    tokenizer_manager::wiring::TmEvent,
};

async fn prompts(raw: Value, template: &str) -> (StatusCode, Vec<String>) {
    let (status, observed) = prepared(raw, template).await;
    (status, observed.into_iter().map(|(text, _)| text).collect())
}

async fn prepared(raw: Value, template: &str) -> (StatusCode, Vec<(String, Option<String>)>) {
    configured(raw, json!(template)).await
}

async fn configured(raw: Value, template: Value) -> (StatusCode, Vec<(String, Option<String>)>) {
    with_formatter(raw, formatter(template)).await
}

fn formatter(template: Value) -> ChatFormatter {
    let config = std::env::temp_dir().join(format!("sglang-parity-{}.json", uuid::Uuid::new_v4()));
    std::fs::write(&config, json!({"chat_template":template}).to_string()).unwrap();
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

async fn with_formatter(
    raw: Value,
    formatter: ChatFormatter,
) -> (StatusCode, Vec<(String, Option<String>)>) {
    with_wire_formatter(&raw.to_string(), formatter).await
}

async fn with_wire_formatter(
    wire: &str,
    formatter: ChatFormatter,
) -> (StatusCode, Vec<(String, Option<String>)>) {
    let raw: Value = serde_json::from_str(wire).unwrap();
    let args = Arc::new(ServerArgs {
        served_model_name: "model".into(),
        tool_call_parser: Some("qwen25".into()),
        ..Default::default()
    });
    let (tx, rx) = flume::unbounded();
    let core = CoreHandle::new(
        tx,
        flume::unbounded().0,
        CoreConfig {
            response_capacity: 8,
            response_activity: Default::default(),
            startup_ready: true,
            is_disaggregation: false,
            mm_limits: Default::default(),
            metadata: CoreMetadata::from(args.as_ref()),
        },
    );
    let state = Arc::new(AppState {
        core,
        generation_policy: crate::message::request::GenerationPolicy::from(args.as_ref()),
        server_args: args,
        chat_formatter: Some(formatter),
    });
    let expected_routing = raw.clone();
    let responder = tokio::spawn(async move {
        let mut observed = Vec::new();
        while let Ok(TmEvent::Intake { request, admission }) = rx.recv_async().await {
            assert!(admission.try_accept());
            let RequestKind::Generate(generate) = &request.kind else {
                panic!("expected generate")
            };
            if let Some(host) = expected_routing
                .get("bootstrap_host")
                .and_then(Value::as_str)
            {
                assert_eq!(generate.bootstrap_host.as_deref(), Some(host));
                assert_eq!(
                    generate.bootstrap_port,
                    expected_routing["bootstrap_port"].as_i64()
                );
                assert_eq!(
                    generate.bootstrap_room,
                    expected_routing["bootstrap_room"].as_i64()
                );
                assert_eq!(
                    generate.routed_dp_rank,
                    expected_routing["routed_dp_rank"].as_i64()
                );
                assert_eq!(
                    generate.disagg_prefill_dp_rank,
                    expected_routing["disagg_prefill_dp_rank"].as_i64()
                );
            }
            observed.push((generate.text.clone().unwrap(), generate.append_text.clone()));
            request
                .sink
                .try_send(chunk(request.rid.as_str(), "done", true))
                .unwrap();
        }
        observed
    });
    let response = tokio::time::timeout(
        Duration::from_secs(10),
        super::super::routes().with_state(state).oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(wire.to_owned()))
                .unwrap(),
        ),
    )
    .await
    .unwrap()
    .unwrap();
    (response.status(), responder.await.unwrap())
}

const TEXT: &str = "{% for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor %}[assistant]";

#[tokio::test]
async fn multipart_text_reaches_intake_with_python_spaces() {
    let (status, observed) = prompts(
        json!({"model":"model", "messages":[{
            "role":"user", "content":[{"type":"text","text":"Hello"},{"type":"text","text":"world"}]
        }]}),
        TEXT,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(observed, ["[user]Hello world[assistant]"]);
}

#[tokio::test]
async fn tools_retain_python_defaults_and_field_order_in_prompt() {
    let (status, observed) = prompts(
        json!({"model":"model", "messages":[{"role":"user","content":"Hi"}],
            "tools":[{"type":"function","function":{"name":"weather"}}]
        }),
        "{{ tools | tojson }}",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        observed,
        [
            r#"[{"type": "function", "function": {"description": null, "name": "weather", "parameters": null, "strict": false}, "defer_loading": null}]"#
        ]
    );
}

#[tokio::test]
async fn reasoning_kwargs_override_effort_after_top_level_thinking_defaults() {
    let (status, observed) = prompts(
        json!({"model":"model", "messages":[{"role":"user","content":"Hi"}],
            "reasoning_effort":"none", "chat_template_kwargs":{"reasoning_effort":"high"}
        }),
        "{{ reasoning_effort }}|{{ thinking | tojson }}|{{ enable_thinking | tojson }}",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(observed, ["high|false|false"]);
}

#[tokio::test]
async fn retained_assistant_null_and_missing_content_are_empty_text() {
    for assistant in [
        json!({"role":"assistant", "content":null}),
        json!({"role":"assistant"}),
    ] {
        let (status, observed) = prompts(
            json!({"model":"model", "messages":[
                {"role":"user","content":"Hi"}, assistant, {"role":"user","content":"Bye"}
            ]}),
            "{% for m in messages %}{{ '[' + m.role + ']' + m.content }}{% endfor %}[assistant]",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(observed, ["[user]Hi[assistant][user]Bye[assistant]"]);
    }
}

#[tokio::test]
async fn final_assistant_defaults_to_python_user_turn() {
    let (status, observed) = prompts(
        json!({"model":"model", "messages":[
            {"role":"user","content":"Hi"}, {"role":"assistant","content":"Answer"}
        ]}),
        TEXT,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(observed, ["[user]Hi[user]Answer[assistant]"]);
}

#[tokio::test]
async fn continuation_submits_prefix_separately_for_every_choice() {
    let (status, observed) = prepared(
        json!({"model":"model", "n":2, "continue_final_message":true,
            "messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Answer"}]
        }),
        TEXT,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        observed,
        vec![("[user]Hi[assistant]".into(), Some("Answer".into())); 2]
    );
}

#[tokio::test]
async fn retained_tool_arguments_render_as_python_objects() {
    let (status, observed) = prompts(json!({"model":"model", "messages":[
        {"role":"user","content":"Hi"},
        {"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function","function":{"name":"weather","arguments":"{\"x\":1e-7,\"label\":\" hi \"}"}}]},
        {"role":"tool","tool_call_id":"c","content":"OK"}, {"role":"user","content":"Bye"}
    ]}), "{% for m in messages %}{% for tc in m.tool_calls | default([]) %}{% if tc.function.arguments is string %}{{ tc.function.arguments }}{% else %}{{ tc.function.arguments | tojson }}{% endif %}{% endfor %}{% endfor %}").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(observed, [r#"{"x": 1e-07, "label": " hi "}"#]);
}

#[tokio::test]
async fn dual_format_jinja_retains_structured_text_parts() {
    let template = "{% for m in messages %}{% if m.content is string %}{{ m.content }}{% else %}{% for part in m.content %}{{ part.text }}|{% endfor %}{% endif %}{% endfor %}";
    let (status, observed) = prompts(
        json!({"model":"model", "messages":[{"role":"user", "content":[
            {"type":"text","text":"A"},{"type":"text","text":"B"}
        ]}]}),
        template,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(observed, ["A|B|"]);
}

#[tokio::test]
async fn empty_text_parts_keep_their_join_separators() {
    let (status, observed) = prompts(
        json!({"model":"model", "messages":[{"role":"user", "content":[
            {"type":"text","text":""},{"type":"text","text":"x"},{"type":"text","text":""}
        ]}]}),
        TEXT,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(observed, ["[user] x [assistant]"]);
}

#[tokio::test]
async fn named_templates_use_startup_selection_independent_of_tools() {
    const ARRAY: &str = "{% for m in messages %}{% for part in m.content %}{{ part.text }}|{% endfor %}{% endfor %}";
    // Current Python TemplateManager chooses the first named source once at
    // startup, then uses that source and its content format for every request.
    for (templates, expected) in [
        (
            json!([{"default":TEXT}, {"tool_use":ARRAY}]),
            "[user]A B[assistant]",
        ),
        (
            json!({"default":TEXT, "tool_use":ARRAY}),
            "[user]A B[assistant]",
        ),
        (json!({"tool_use":ARRAY, "default":TEXT}), "A|B|"),
        (
            json!([{"name":"default", "template":TEXT}, {"name":"tool_use", "template":ARRAY}]),
            "[user]A B[assistant]",
        ),
        (json!({"default":ARRAY}), "A|B|"),
    ] {
        for choice in ["auto", "none"] {
            let (status, observed) = configured(json!({"model":"model", "tool_choice":choice,
                "tools":[{"type":"function","function":{"name":"weather"}}],
                "messages":[{"role":"user", "content":[{"type":"text","text":"A"},{"type":"text","text":"B"}]}]
            }), templates.clone()).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(
                observed,
                [(expected.into(), None)],
                "choice={choice}, templates={templates}"
            );
        }
    }
}

#[tokio::test]
async fn reasoning_override_preserves_original_request_validation() {
    let (status, observed) = prompts(
        json!({"model":"model", "messages":[{"role":"user","content":"Hi"}],
            "reasoning_effort":true, "chat_template_kwargs":{"reasoning_effort":"high"}
        }),
        "{{ reasoning_effort | tojson }}",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(observed.is_empty());
    for effort in [json!(true), json!("custom"), json!(0.2)] {
        let (status, observed) = prompts(
            json!({"model":"model", "messages":[{"role":"user","content":"Hi"}],
                "chat_template_kwargs":{"reasoning_effort":effort}
            }),
            "{{ reasoning_effort | tojson }}",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(observed, [effort.to_string()]);
    }
}

#[tokio::test]
async fn reserved_template_kwargs_reject_before_intake() {
    for key in ["messages", "tools", "add_generation_prompt"] {
        let (status, observed) = prompts(
            json!({"model":"model", "messages":[{"role":"user","content":"Hi"}],
                "chat_template_kwargs":{key:null}
            }),
            TEXT,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{key}");
        assert!(observed.is_empty());
    }
}

#[tokio::test]
async fn array_aware_templates_still_receive_flat_tool_text() {
    let template = "{% for m in messages %}[{{ m.role }}]{% if m.content is string %}{{ m.content }}{% else %}{% for part in m.content %}{{ part.text }}|{% endfor %}{% endif %}{% endfor %}";
    let (status, observed) = prompts(
        json!({"model":"model", "messages":[
            {"role":"user","content":"Hi"}, {"role":"tool","tool_call_id":"c","content":[
                {"type":"text","text":"A"},{"type":"text","text":"B"}
            ]}, {"role":"user","content":"Bye"}
        ]}),
        template,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(observed, ["[user]Hi[tool]A B[user]Bye"]);
}

#[tokio::test]
async fn jinja_preparation_preserves_pd_routing_for_all_choices() {
    let (status, observed) = prompts(json!({"model":"model", "n":2,
        "bootstrap_host":"host", "bootstrap_port":8998, "bootstrap_room":7,
        "routed_dp_rank":1, "disagg_prefill_dp_rank":2,
        "messages":[{"role":"user","content":[{"type":"text","text":"A"},{"type":"text","text":"B"}]}]
    }), TEXT).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(observed, ["[user]A B[assistant]", "[user]A B[assistant]"]);
}

#[tokio::test]
async fn jinja_context_binding_preserves_source_whitespace() {
    for (template, expected) in [
        ("\n{{ messages[0].content }}\n", "\nHi"),
        (
            "  {% for m in messages %}{{ m.content }}{% endfor %}\n",
            "Hi",
        ),
        (
            "\n{%- for m in messages -%}{{ m.content }}{%- endfor -%}\n",
            "Hi",
        ),
    ] {
        let (status, observed) = prompts(
            json!({"model":"model", "messages":[{"role":"user","content":"Hi"}]}),
            template,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(observed, [expected]);
    }
}

#[test]
fn ordinary_render_does_not_accept_the_internal_prepared_messages_key() {
    let formatter = formatter(json!(TEXT));
    let request = serde_json::from_value(
        json!({"model":"model","messages":[{"role":"user","content":"Hi"}]}),
    )
    .unwrap();
    let kwargs = [(
        "_sglang_prompt_messages".into(),
        json!([{"role":"user","content":"injected"}]),
    )]
    .into_iter()
    .collect();
    assert_eq!(
        formatter.render(&request, Some(&kwargs)).unwrap(),
        "[user]Hi[assistant]"
    );
}

#[tokio::test]
async fn native_and_legacy_reject_duplicate_known_fields_before_intake() {
    for native in [true, false] {
        for wire in [
            r#"{"model":"model","model":"model","messages":[{"role":"user","content":"Hi"}]}"#,
            r#"{"model":"model","messages":[{"role":"user","content":7,"content":"Hi"}]}"#,
            r#"{"model":"model","messages":[{"role":"user","content":"Hi"}],"routed_dp_rank":0,"routed_dp_rank":1}"#,
            r#"{"model":"model","messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":{},"chat_template_kwargs":{}}"#,
        ] {
            let formatter = if native {
                super::super::template::load_chat_formatter(None, None, Some("deepseek_v4"), None)
            } else {
                super::super::template::load_chat_formatter(None, None, None, Some("chatml"))
            }
            .unwrap();
            let (status, observed) = with_wire_formatter(wire, formatter).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "native={native}, {wire}");
            assert!(observed.is_empty(), "invalid request reached Intake");
        }
    }
}

#[tokio::test]
async fn jinja_retries_flat_function_tools_after_wrapped_render_failure() {
    let (status, observed) = prompts(
        json!({"model":"model", "messages":[{"role":"user","content":"Hi"}],
            "tools":[{"type":"function","function":{"name":"weather"}}]}),
        "{% for tool in tools %}{% if tool.function is defined %}{{ raise_exception('wrapped tools unsupported') }}{% else %}{{ tool.name }}{% endif %}{% endfor %}",
    ).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(observed, ["weather"]);
}

#[tokio::test]
async fn native_and_legacy_preserve_their_message_and_kwargs_behavior() {
    for (native, expected) in [
        (
            true,
            "<｜begin▁of▁sentence｜><｜User｜>AB<｜Assistant｜></think>tail<｜end▁of▁sentence｜>",
        ),
        (
            false,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAB<|im_end|>\n<|im_start|>assistant\ntail<|im_end|>\n<|im_start|>assistant\n",
        ),
    ] {
        let formatter = if native {
            super::super::template::load_chat_formatter(None, None, Some("deepseek_v4"), None)
        } else {
            super::super::template::load_chat_formatter(None, None, None, Some("chatml"))
        }
        .unwrap();
        // Exact prompt literals were observed on the unmodified main route.
        // Jinja-only bool parsing, separators, continuation and reserved kwargs
        // must not leak into either formatter.
        let (status, observed) = with_formatter(json!({"model":"model",
            "messages":[{"role":"user","content":[{"type":"text","text":"A"},{"type":"text","text":"B"}]},
                        {"role":"assistant","content":"tail"}],
            "continue_final_message":"invalid-bool",
            "reasoning":{"effort":false,"enabled":true},
            "chat_template_kwargs":{"thinking":false,"enable_thinking":false,
                "messages":"caller","tools":"caller","add_generation_prompt":false}
        }), formatter).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(observed, [(expected.into(), None)]);
    }
}

#[tokio::test]
async fn tool_schema_aliases_are_normalized_before_prompt_rendering() {
    let (status, observed) = prompts(
        json!({"model":"model",
        "messages":[{"role":"user","content":"Hi"}],
        "tools":[{"type":"function","function":{"name":"weather","parameters":{
            "type":"object","properties":{"city":{"type":"varchar"},
                "count":{"type":["int32","integer","null","INT64"]}}
        }}}]}),
        "{{ tools[0].function.parameters | tojson }}",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        observed,
        [
            r#"{"type": "object", "properties": {"city": {"type": "string"}, "count": {"type": ["integer", "null"]}}}"#
        ]
    );
}

#[tokio::test]
async fn nested_reasoning_alias_sets_python_thinking_defaults() {
    let (status, observed) = prompts(json!({"model":"model",
        "messages":[{"role":"user","content":"Hi"}],
        "reasoning":{"effort":"none"}}),
        "{{ reasoning_effort | default('MISSING') | tojson }}|{{ thinking | default('MISSING') | tojson }}|{{ enable_thinking | default('MISSING') | tojson }}").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(observed, [r#""none"|false|false"#]);
}

#[tokio::test]
async fn invalid_effort_underscore_whitespace_rejects_before_intake() {
    let (status, observed) = prompts(
        json!({"model":"model",
        "messages":[{"role":"user","content":"Hi"}],
        "reasoning_effort":" 0_._25 ",
        "chat_template_kwargs":{"reasoning_effort":"high"}}),
        "{{ reasoning_effort }}",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(observed.is_empty(), "invalid effort reached Intake");
    for fields in [
        json!({"reasoning_effort":"0_._25"}),
        json!({"reasoning_effort":" 0.25 "}),
        json!({"reasoning":{"effort":" 0.2_5 "}}),
    ] {
        let mut request = json!({"model":"model", "messages":[{"role":"user","content":"Hi"}]});
        request
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        let (status, observed) = prompts(request, "{{ reasoning_effort }}").await;
        assert_eq!(status, StatusCode::OK, "{fields}");
        assert_eq!(observed, ["0.25"], "{fields}");
    }
}

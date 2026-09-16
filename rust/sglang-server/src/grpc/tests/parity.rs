//! Parity checks use the real adapter and frontend with a fake runtime intake.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use serde_json::{Value, json};

use super::*;

fn openai_request(value: Value) -> Request<proto::OpenAiRequest> {
    Request::new(proto::OpenAiRequest {
        json_body: serde_json::to_vec(&value).unwrap(),
        ..Default::default()
    })
}

#[tokio::test]
async fn historical_function_calls_have_the_same_prompt_over_http_and_grpc() {
    use axum::body::Body;
    use dynamo_renderer::{ContextMixins, PromptContextMixin, PromptFormatter};
    use tower::ServiceExt;

    let mut harness = Harness::new(4, false, Duration::from_secs(1));
    let template = serde_json::from_value(json!({
        "chat_template": "{{ messages[1].function_call.name }}:{{ messages[1].function_call.arguments.city }}{{ suffix }}"
    })).unwrap();
    let formatter = PromptFormatter::from_parts(
        template,
        ContextMixins::new(&[PromptContextMixin::OaiChat]),
        true,
    )
    .unwrap();
    Arc::get_mut(&mut harness.service.openai)
        .unwrap()
        .chat_formatter = Some(crate::openai::ChatFormatter::HuggingFace(formatter));
    let body = json!({
        "model": "model",
        "messages": [
            {"role":"user", "content":"Weather in Paris?"},
            {"role":"assistant", "content":null,
             "function_call":{"name":"weather", "arguments":"{\"city\":\"Paris\"}"}},
            {"role":"function", "name":"weather", "content":"Sunny"}
        ]
    });

    // The shared operation keeps HTTP's extra template variables; gRPC's
    // explicit support policy continues to reject that request option below.
    for (grpc, suffix) in [(false, None), (true, None), (false, Some("!"))] {
        let mut body = body.clone();
        if let Some(suffix) = suffix {
            body["chat_template_kwargs"] = json!({"suffix": suffix});
        }
        let rpc = async {
            if grpc {
                let mut stream = harness
                    .service
                    .chat_complete(openai_request(body.clone()))
                    .await
                    .unwrap()
                    .into_inner();
                let response = stream.next().await.unwrap().unwrap();
                assert!(response.finished);
                assert!(stream.next().await.is_none());
                serde_json::from_slice::<Value>(&response.json_chunk).unwrap()
            } else {
                let request = axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap();
                let response = crate::api_server::openai::routes()
                    .with_state(harness.service.openai.clone())
                    .oneshot(request)
                    .await
                    .unwrap();
                assert_eq!(response.status(), axum::http::StatusCode::OK);
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                serde_json::from_slice::<Value>(&bytes).unwrap()
            }
        };
        let runtime = async {
            let intake = harness.next_generation().await;
            assert!(intake.admission.try_accept());
            assert_eq!(
                intake.request.text,
                Some(format!("weather:Paris{}", suffix.unwrap_or_default()))
            );
            assert!(intake.request.skip_special_tokens);
            intake
                .sink
                .try_send(ResponseItem::Done(chunk(&intake.rid, "Sunny", 42, true)))
                .unwrap();
        };
        let (value, ()) =
            tokio::time::timeout(Duration::from_secs(2), async { tokio::join!(rpc, runtime) })
                .await
                .expect("chat must complete over either transport");
        assert_eq!(value["choices"][0]["message"]["content"], "Sunny");
        assert!(harness.abort_rx.try_recv().is_err());
    }

    // Historical message fields do not enable deprecated request-level knobs,
    // and HTTP template kwargs do not widen gRPC's declared support boundary.
    for (key, option) in [
        ("function_call", json!("auto")),
        ("functions", json!([{"name":"weather"}])),
        ("chat_template_kwargs", json!({"suffix": "!"})),
    ] {
        let mut request = body.clone();
        request[key] = option;
        let error = harness
            .service
            .chat_complete(openai_request(request))
            .await
            .err()
            .unwrap();
        assert_eq!(error.code(), Code::Unimplemented);
    }
    assert!(harness.intake_rx.try_recv().is_err());
}

async fn reply_to_generation(harness: &Harness) {
    let intake = harness.next_generation().await;
    assert!(intake.admission.try_accept());
    assert!(intake.request.text.is_some() || intake.request.input_ids.is_some());
    intake
        .sink
        .try_send(ResponseItem::Done(chunk(&intake.rid, "Hello", 42, true)))
        .unwrap();
}

#[tokio::test]
async fn static_metadata_and_model_listing_use_frontend_metadata() {
    let harness = Harness::new(4, false, Duration::from_secs(1));
    let info = harness
        .service
        .get_model_info(Request::new(Default::default()))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.model_path, "/model");
    let json: Value = serde_json::from_str(&info.json_info).unwrap();
    assert_eq!(json["served_model_name"], "model");
    assert_eq!(json["is_generation"], true);
    let models = harness
        .service
        .list_models(Request::new(Default::default()))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(models.models.len(), 1);
    assert_eq!(models.models[0].id, "model");
    assert_eq!(models.models[0].root, "model");
    assert_eq!(models.models[0].max_model_len, Some(4096));
    assert!(models.models[0].parent.is_none());
    assert!(harness.intake_rx.try_recv().is_err());
}

#[tokio::test]
async fn server_info_round_trips_and_does_not_leak_private_scheduler_fields() {
    let harness = Harness::new(4, false, Duration::from_secs(1));
    let rpc = harness
        .service
        .get_server_info(Request::new(Default::default()));
    let runtime = async {
        let TmEvent::Intake { request, admission } = harness.intake_rx.recv_async().await.unwrap()
        else {
            panic!("expected control intake");
        };
        assert!(admission.try_accept());
        assert!(matches!(request.kind, RequestKind::Control(_)));
        let payload = rmp_serde::to_vec_named(&json!({
            "internal_state": {
                "last_gen_throughput": 1.5,
                "api_key": "must-not-leak",
                "admin_api_key": "also-private"
            }
        }))
        .unwrap();
        request
            .sink
            .try_send(ResponseItem::Control(payload.into()))
            .unwrap();
    };
    let (result, ()) = tokio::join!(rpc, runtime);
    let text = result.unwrap().into_inner().json_info;
    let info: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(info["model_path"], "/model");
    assert_eq!(info["max_context_length"], 4096);
    assert_eq!(info["internal_states"][0]["last_gen_throughput"], 1.5);
    assert!(!text.contains("must-not-leak"));
    assert!(!text.contains("also-private"));
    assert!(!text.contains("api_key"));
    assert!(harness.abort_rx.try_recv().is_err());
}

#[tokio::test]
async fn detokenize_uses_the_existing_runtime_operation() {
    let harness = Harness::new(4, false, Duration::from_secs(1));
    let rpc = harness
        .service
        .detokenize(Request::new(proto::DetokenizeRequest {
            tokens: vec![1, 2],
        }));
    let runtime = async {
        let TmEvent::Intake { request, admission } = harness.intake_rx.recv_async().await.unwrap()
        else {
            panic!("expected detokenize intake");
        };
        assert!(admission.try_accept());
        let RequestKind::Detokenize { token_ids } = request.kind else {
            panic!("expected detokenize request");
        };
        assert_eq!(token_ids, vec![1, 2]);
        request
            .sink
            .try_send(ResponseItem::Data(bytes::Bytes::from_static(b"decoded")))
            .unwrap();
    };
    let (result, ()) = tokio::join!(rpc, runtime);
    assert_eq!(result.unwrap().into_inner().text, "decoded");
    assert!(harness.abort_rx.try_recv().is_err());
}

#[tokio::test]
async fn health_uses_readiness_or_the_existing_activity_probe() {
    let mut harness = Harness::new(4, false, Duration::from_secs(1));
    assert!(
        harness
            .service
            .health_check(Request::new(Default::default()))
            .await
            .unwrap()
            .into_inner()
            .healthy
    );
    assert!(harness.intake_rx.try_recv().is_err());

    harness.service.config.health_probe_timeout = Some(Duration::from_secs(1));
    let rpc = harness
        .service
        .health_check(Request::new(Default::default()));
    let runtime = async {
        let intake = harness.next_generation().await;
        assert!(intake.admission.try_accept());
        assert_eq!(intake.request.input_ids, Some(vec![0]));
        harness.activity.fetch_add(1, Ordering::Relaxed);
    };
    let (result, ()) = tokio::join!(rpc, runtime);
    assert!(result.unwrap().into_inner().healthy);
    assert!(harness.abort_rx.try_recv().is_ok());

    harness.service.config.health_probe_timeout = Some(Duration::ZERO);
    assert!(
        !harness
            .service
            .health_check(Request::new(Default::default()))
            .await
            .unwrap()
            .into_inner()
            .healthy
    );
}

#[tokio::test]
async fn unary_chat_and_completion_emit_one_finished_json_chunk() {
    for chat in [false, true] {
        let harness = Harness::new(4, false, Duration::from_secs(1));
        let body = if chat {
            json!({"model": "model", "messages": [{"role": "user", "content": "Hi"}]})
        } else {
            json!({"model": "model", "prompt": [1, 2]})
        };
        let rpc = async {
            if chat {
                harness.service.chat_complete(openai_request(body)).await
            } else {
                harness.service.complete(openai_request(body)).await
            }
        };
        let (result, ()) = tokio::join!(rpc, reply_to_generation(&harness));
        let mut stream = result.unwrap().into_inner();
        let response = stream.next().await.unwrap().unwrap();
        assert!(response.finished);
        let value: Value = serde_json::from_slice(&response.json_chunk).unwrap();
        let field = if chat {
            &value["choices"][0]["message"]["content"]
        } else {
            &value["choices"][0]["text"]
        };
        assert_eq!(field, "Hello");
        assert_eq!(value["usage"]["completion_tokens"], 1);
        assert!(stream.next().await.is_none());
        assert!(harness.abort_rx.try_recv().is_err());
    }
}

#[tokio::test]
async fn streaming_chat_and_completion_send_json_not_sse_and_one_terminal_marker() {
    for chat in [false, true] {
        let harness = Harness::new(4, false, Duration::from_secs(1));
        let body = if chat {
            json!({"model": "model", "messages": [{"role": "user", "content": "Hi"}], "stream": true, "stream_options": {"include_usage": true}})
        } else {
            json!({"model": "model", "prompt": "Hi", "stream": true, "stream_options": {"include_usage": true}})
        };
        let result = if chat {
            harness.service.chat_complete(openai_request(body)).await
        } else {
            harness.service.complete(openai_request(body)).await
        };
        reply_to_generation(&harness).await;
        let mut stream = result.unwrap().into_inner();
        let mut output = String::new();
        let mut finished = 0;
        let mut usage = false;
        while let Some(response) = stream.next().await {
            let response = response.unwrap();
            if response.finished {
                assert!(response.json_chunk.is_empty());
                finished += 1;
                continue;
            }
            assert_eq!(finished, 0);
            let value: Value = serde_json::from_slice(&response.json_chunk).unwrap();
            usage |= value["usage"].is_object();
            let text = if chat {
                &value["choices"][0]["delta"]["content"]
            } else {
                &value["choices"][0]["text"]
            };
            output.push_str(text.as_str().unwrap_or_default());
        }
        assert_eq!(output, "Hello");
        assert!(usage);
        assert_eq!(finished, 1);
        assert!(harness.abort_rx.try_recv().is_err());
    }
}

#[tokio::test]
async fn openai_errors_are_statuses_and_unsupported_features_never_submit() {
    let harness = Harness::new(4, false, Duration::from_secs(1));
    for (body, chat, code) in [
        (
            json!({"model":"model", "prompt":"Hi", "suffix":"!"}),
            false,
            Code::Unimplemented,
        ),
        (
            json!({"model":"model", "prompt":"Hi", "best_of":2}),
            false,
            Code::Unimplemented,
        ),
        (
            json!({"model":"model", "messages":[{"role":"user", "content":[{"type":"image_url","image_url":{"url":"https://unused"}}]}]}),
            true,
            Code::Unimplemented,
        ),
        (
            json!({"model":"model", "messages":[{"role":"user","content":"Hi"}], "stream_options":{"continuous_usage_stats":true}}),
            true,
            Code::Unimplemented,
        ),
        (
            json!({"model":"model", "messages":[{"role":"user","content":"Hi"}], "tools":[{"type":"function","function":{"name":"lookup"}}]}),
            true,
            Code::Unimplemented,
        ),
        (
            json!({"model":"model", "prompt":"Hi", "n":0}),
            false,
            Code::InvalidArgument,
        ),
        (
            json!({"model":"model", "prompt":"Hi", "stream":"yes"}),
            false,
            Code::InvalidArgument,
        ),
        (
            json!({"model":"model", "messages":[]}),
            true,
            Code::InvalidArgument,
        ),
        (json!([]), false, Code::InvalidArgument),
    ] {
        let result = if chat {
            harness
                .service
                .chat_complete(openai_request(body.clone()))
                .await
        } else {
            harness.service.complete(openai_request(body.clone())).await
        };
        assert_eq!(result.err().expect("must reject").code(), code, "{body}");
    }
    let mut request = openai_request(json!({"model":"model","prompt":"Hi"})).into_inner();
    request
        .trace_headers
        .insert("traceparent".into(), "not-supported".into());
    assert_eq!(
        harness
            .service
            .complete(Request::new(request))
            .await
            .err()
            .unwrap()
            .code(),
        Code::Unimplemented
    );
    assert!(harness.intake_rx.try_recv().is_err());
}

#[tokio::test]
async fn tokenizer_dependent_operations_are_explicitly_unsupported_when_disabled() {
    let mut harness = Harness::new(4, false, Duration::from_secs(1));
    let state = Arc::get_mut(&mut harness.service.openai).unwrap();
    Arc::get_mut(&mut state.server_args)
        .unwrap()
        .skip_tokenizer_init = true;
    assert_eq!(
        harness
            .service
            .text_generate(Request::new(proto::TextGenerateRequest {
                text: "Hi".into(),
                ..Default::default()
            }))
            .await
            .err()
            .unwrap()
            .code(),
        Code::Unimplemented
    );
    assert_eq!(
        harness
            .service
            .detokenize(Request::new(proto::DetokenizeRequest { tokens: vec![1] }))
            .await
            .unwrap_err()
            .code(),
        Code::Unimplemented
    );
    for body in [
        json!({"model":"model","prompt":"Hi"}),
        json!({"model":"model","prompt":[1],"echo":true}),
    ] {
        assert_eq!(
            harness
                .service
                .complete(openai_request(body))
                .await
                .err()
                .unwrap()
                .code(),
            Code::Unimplemented
        );
    }
    assert_eq!(
        harness
            .service
            .chat_complete(openai_request(json!({
                "model":"model","messages":[{"role":"user","content":"Hi"}]
            })))
            .await
            .err()
            .unwrap()
            .code(),
        Code::Unimplemented
    );
    assert!(harness.intake_rx.try_recv().is_err());

    // Token-ID completions without echo still work without a tokenizer.
    let rpc = harness
        .service
        .complete(openai_request(json!({"model":"model","prompt":[1]})));
    let (result, ()) = tokio::join!(rpc, reply_to_generation(&harness));
    let mut stream = result.unwrap().into_inner();
    assert!(stream.next().await.unwrap().unwrap().finished);
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn openai_stream_drop_and_timeout_release_all_choices() {
    for timeout in [false, true] {
        let harness = Harness::new(4, false, Duration::from_millis(10));
        let mut stream = harness
            .service
            .complete(openai_request(json!({
                "model": "model", "prompt": [1], "stream": true, "n": 2
            })))
            .await
            .unwrap()
            .into_inner();
        let mut intakes = Vec::new();
        for _ in 0..2 {
            let intake = harness.next_generation().await;
            assert!(intake.admission.try_accept());
            intakes.push(intake);
        }
        if timeout {
            assert_eq!(
                stream.next().await.unwrap().unwrap_err().code(),
                Code::DeadlineExceeded
            );
        } else {
            drop(stream);
        }
        // Timeout must cancel both calls before another poll or client-side drop.
        let aborted: Vec<_> = harness
            .abort_rx
            .try_iter()
            .map(|a| a.rid().clone())
            .collect();
        assert_eq!(aborted.len(), intakes.len());
        assert!(intakes.iter().all(|intake| aborted.contains(&intake.rid)));
    }
}

#[tokio::test]
async fn unary_operation_timeout_cancels_runtime_work() {
    let harness = Harness::new(4, false, Duration::from_millis(10));
    let rpc = harness
        .service
        .complete(openai_request(json!({"model":"model","prompt":[1]})));
    let runtime = async {
        let intake = harness.next_generation().await;
        assert!(intake.admission.try_accept());
        // Keep the response channel open until timeout cancels the call.
        let abort = harness.abort_rx.recv_async().await.unwrap();
        assert_eq!(abort.rid(), &intake.rid);
    };
    let (result, ()) = tokio::join!(rpc, runtime);
    assert_eq!(result.err().unwrap().code(), Code::DeadlineExceeded);
}

#[tokio::test]
async fn missing_chat_template_and_nested_extensions_are_unsupported() {
    let mut harness = Harness::new(4, false, Duration::from_secs(1));
    let state = Arc::get_mut(&mut harness.service.openai).unwrap();
    // A missing parser must not mask rejection of unsupported tool types.
    Arc::get_mut(&mut state.server_args)
        .unwrap()
        .tool_call_parser = Some("qwen25".into());
    for extra in [
        json!({"response_format":{"type":"structural_tag"}}),
        json!({"tools":[{"type":"custom"}]}),
        json!({"messages":[{"role":"assistant","audio":{"id":"audio-id"}}]}),
    ] {
        let mut body = json!({"model":"model","messages":[{"role":"user","content":"Hi"}]});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let result = harness.service.chat_complete(openai_request(body)).await;
        assert_eq!(result.err().unwrap().code(), Code::Unimplemented);
    }
    Arc::get_mut(&mut harness.service.openai)
        .unwrap()
        .chat_formatter = None;
    let result = harness
        .service
        .chat_complete(openai_request(json!({
            "model":"model","messages":[{"role":"user","content":"Hi"}]
        })))
        .await;
    assert_eq!(result.err().unwrap().code(), Code::Unimplemented);
    assert!(harness.intake_rx.try_recv().is_err());
}

#[tokio::test]
async fn streamed_capacity_error_uses_semantic_status_and_cancels_other_choices() {
    let harness = Harness::new(4, false, Duration::from_secs(1));
    let mut stream = harness
        .service
        .complete(openai_request(json!({
            "model":"model","prompt":[1],"stream":true,"n":2
        })))
        .await
        .unwrap()
        .into_inner();
    let first = harness.next_generation().await;
    let second = harness.next_generation().await;
    assert!(first.admission.try_accept());
    assert!(second.admission.try_accept());
    first
        .sink
        .try_send(ResponseItem::Error(crate::utils::error::Error::QueueFull))
        .unwrap();
    assert_eq!(
        stream.next().await.unwrap().unwrap_err().code(),
        Code::ResourceExhausted
    );
    assert_eq!(harness.abort_rx.try_recv().unwrap().rid(), &second.rid);
    assert!(stream.next().await.is_none());
    assert!(harness.abort_rx.try_recv().is_err());
}

#[tokio::test]
async fn unary_scheduler_rejection_uses_semantic_status() {
    let harness = Harness::new(4, false, Duration::from_secs(1));
    let rpc = harness.service.complete(openai_request(json!({
        "model":"model","prompt":[1]
    })));
    let runtime = async {
        let intake = harness.next_generation().await;
        assert!(intake.admission.try_accept());
        let mut rejected = chunk(&intake.rid, "", 42, true);
        rejected.finish_reason = Some(
            serde_json::from_value(json!({
                "type": "abort",
                "message": "upstream unavailable",
                "status_code": 502
            }))
            .unwrap(),
        );
        intake.sink.try_send(ResponseItem::Done(rejected)).unwrap();
    };

    let (result, ()) = tokio::join!(rpc, runtime);
    let error = match result {
        Ok(_) => panic!("scheduler rejection must fail the RPC"),
        Err(error) => error,
    };
    assert_eq!(error.code(), Code::Unavailable);
    assert!(harness.abort_rx.try_recv().is_err());
}

#[tokio::test]
async fn completion_prompt_arrays_submit_each_requested_choice() {
    let harness = Harness::new(4, false, Duration::from_secs(1));
    let rpc = harness.service.complete(openai_request(json!({
        "model":"model","prompt":["first","second"],"n":2
    })));
    let runtime = async {
        for expected in ["first", "first", "second", "second"] {
            let intake = harness.next_generation().await;
            assert!(intake.admission.try_accept());
            assert_eq!(intake.request.text.as_deref(), Some(expected));
            intake
                .sink
                .try_send(ResponseItem::Done(chunk(&intake.rid, "answer", 42, true)))
                .unwrap();
        }
    };
    let (result, ()) = tokio::join!(rpc, runtime);
    let response = result.unwrap().into_inner().next().await.unwrap().unwrap();
    let value: Value = serde_json::from_slice(&response.json_chunk).unwrap();
    assert_eq!(value["choices"].as_array().unwrap().len(), 4);
    assert_eq!(value["usage"]["completion_tokens"], 4);
    assert!(harness.abort_rx.try_recv().is_err());
}

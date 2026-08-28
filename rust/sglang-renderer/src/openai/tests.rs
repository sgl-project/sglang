//! Integration tests for the assembled OpenAI frontend.

mod suite {
    use std::convert::Infallible;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::Duration;

    use axum::{
        Json, Router,
        body::{Body, to_bytes},
        extract::State,
        http::{Request, StatusCode},
        response::sse::{Event, Sse},
        routing::post,
    };
    use tokio::sync::Barrier;
    use tower::ServiceExt;

    use super::super::{
        ChatCompletionRequest, CompletionRequest, HttpGenerateClient, OpenAIHttpFrontend,
        standalone_routes,
    };
    use crate::openai::protocol::{
        lower_chat_request, lower_text_completion_request, lower_token_ids_completion_request,
    };
    use crate::{
        RendererConfig, RendererError, RendererLimits, RendererService, SamplingDefaults,
        TextTokenizer,
    };

    #[test]
    fn chat_lowering_preserves_template_controls_and_metadata() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "rid": "chat-lowering",
            "chat_template_kwargs": {"enable_thinking": false},
            "continue_final_message": true,
            "top_k": 17,
            "min_p": 0.2,
            "min_tokens": 3,
            "stop_regex": "END[0-9]",
            "ignore_eos": true,
            "skip_special_tokens": false,
            "bootstrap_host": "prefill",
            "bootstrap_port": 8998,
            "bootstrap_room": 42
        }))
        .unwrap();

        assert_eq!(request.model, "model");
        assert_eq!(
            request
                .chat_template_kwargs
                .as_ref()
                .and_then(|args| args.get("enable_thinking")),
            Some(&serde_json::Value::Bool(false))
        );
        assert!(request.continue_final_message);
        assert_eq!(request.sampling_overrides.top_k, Some(17));
        assert_eq!(request.sampling_overrides.min_p, Some(0.2));
        assert_eq!(request.sampling_overrides.min_tokens, Some(3));
        assert_eq!(request.sampling_overrides.ignore_eos, Some(true));
        assert_eq!(request.sampling_overrides.skip_special_tokens, Some(false));

        let (response_id, request) = lower_chat_request(&renderer_config(), request).unwrap();

        assert_eq!(response_id, "chat-lowering");
        assert_eq!(request.metadata.bootstrap_host.as_deref(), Some("prefill"));
        assert_eq!(request.metadata.bootstrap_port, Some(8998));
        assert_eq!(request.metadata.bootstrap_room, Some(42));
        assert_eq!(request.sampling_params.top_k, 17);
    }

    #[test]
    fn unsupported_sglang_fields_are_rejected_instead_of_ignored() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "input_ids": [1, 2, 3],
            "task": "domain"
        }))
        .unwrap();

        let error = lower_chat_request(&renderer_config(), request)
            .unwrap_err()
            .to_string();

        assert_eq!(error, "unsupported request fields: input_ids, task");
    }

    #[test]
    fn chat_modalities_keep_the_typed_openai_contract() {
        for modalities in [serde_json::json!("text"), serde_json::json!(["vision"])] {
            let request = serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "hello"}],
                "modalities": modalities
            });
            assert!(serde_json::from_value::<ChatCompletionRequest>(request).is_err());
        }

        let text_request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "modalities": ["text"]
        }))
        .unwrap();
        lower_chat_request(&renderer_config(), text_request).unwrap();
    }

    #[test]
    fn reasoning_inputs_normalize_with_python_precedence() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "high",
            "reasoning": {"effort": "none", "enabled": true},
            "chat_template_kwargs": {"thinking": true}
        }))
        .unwrap();
        let (_, request) = lower_chat_request(&renderer_config(), request).unwrap();
        let args = request.chat_template_args.unwrap();

        assert_eq!(
            serde_json::to_value(request.reasoning_effort).unwrap(),
            serde_json::json!("none")
        );
        assert_eq!(args.get("thinking"), Some(&serde_json::json!(true)));
        assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));

        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "0.5"
        }))
        .unwrap();
        let (_, request) = lower_chat_request(&renderer_config(), request).unwrap();
        assert_eq!(
            serde_json::to_value(request.reasoning_effort).unwrap(),
            serde_json::json!(0.5)
        );
        assert_eq!(
            request
                .chat_template_args
                .as_ref()
                .and_then(|args| args.get("thinking")),
            Some(&serde_json::json!(true))
        );

        for invalid in [serde_json::json!(true), serde_json::json!(1.0)] {
            let request = serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_effort": invalid
            });
            assert!(serde_json::from_value::<ChatCompletionRequest>(request).is_err());
        }
    }

    #[test]
    fn text_completion_lowering_attaches_batched_metadata_in_prompt_major_order() {
        let request: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "prompt": ["one", "two"],
            "n": 2,
            "rid": ["prompt-a", "prompt-b"],
            "cache_salt": ["tenant-a", "tenant-b"],
            "extra_key": ["", "batch"],
            "bootstrap_host": ["prefill-a", "prefill-b"],
            "bootstrap_port": [8998, null],
            "bootstrap_room": [41, 52],
            "priority": 7,
            "routed_dp_rank": 2
        }))
        .unwrap();
        let (response_id, requests) =
            lower_text_completion_request(&renderer_config(), &request).unwrap();

        assert_eq!(response_id, "prompt-a");
        assert_eq!(
            requests
                .iter()
                .flat_map(|request| request.requests.iter())
                .map(|request| request.rid.as_str())
                .collect::<Vec<_>>(),
            ["prompt-a-0", "prompt-a-1", "prompt-b-0", "prompt-b-1"]
        );
        assert_eq!(
            requests[0].requests[0].metadata.cache_salt.as_deref(),
            Some("tenant-a")
        );
        assert_eq!(requests[0].requests[1].metadata.extra_key, None);
        assert_eq!(
            requests[1].requests[0].metadata.extra_key.as_deref(),
            Some("batch")
        );
        assert_eq!(requests[0].requests[0].metadata.bootstrap_port, Some(8998));
        assert_eq!(requests[1].requests[0].metadata.bootstrap_port, None);
        assert_eq!(requests[0].requests[1].metadata.bootstrap_room, Some(41));
        assert_eq!(requests[1].requests[1].metadata.bootstrap_room, Some(52));
        assert_eq!(requests[1].requests[1].metadata.routed_dp_rank, Some(2));
    }

    #[test]
    fn completion_lowering_validates_metadata_lengths_duplicates_and_scalar_rooms() {
        let request: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "prompt": ["one", "two"],
            "rid": ["duplicate", "duplicate"],
            "cache_salt": ["only-one"]
        }))
        .unwrap();
        let error = lower_text_completion_request(&renderer_config(), &request).unwrap_err();
        assert!(error.to_string().contains("duplicate request ID"));

        let request: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "prompt": ["one", "two"],
            "cache_salt": ["only-one"]
        }))
        .unwrap();
        let error = lower_text_completion_request(&renderer_config(), &request).unwrap_err();
        assert!(error.to_string().contains("prompt batch size (2)"));

        let request: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "prompt": ["one", "two"],
            "n": 2,
            "bootstrap_room": 90
        }))
        .unwrap();
        let (_, requests) = lower_text_completion_request(&renderer_config(), &request).unwrap();
        assert_eq!(
            requests
                .iter()
                .flat_map(|request| request.requests.iter())
                .map(|request| request.metadata.bootstrap_room)
                .collect::<Vec<_>>(),
            [Some(90), Some(90), Some(91), Some(91)]
        );
    }

    #[test]
    fn completion_lowering_rejects_zero_max_tokens() {
        let request: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "prompt": "hello",
            "max_tokens": 0
        }))
        .unwrap();

        let error = lower_text_completion_request(&renderer_config(), &request).unwrap_err();

        assert_eq!(error.to_string(), "max_tokens must be positive");
    }

    #[test]
    fn token_id_completion_lowering_attaches_batched_metadata() {
        let request: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "prompt": [[1, 2], [3]],
            "n": 2,
            "rid": ["tokens-a", "tokens-b"],
            "bootstrap_host": ["prefill-a", "prefill-b"],
            "bootstrap_port": [8998, 8999],
            "bootstrap_room": [41, 52]
        }))
        .unwrap();
        let (response_id, requests) =
            lower_token_ids_completion_request(&renderer_config(), &request).unwrap();

        assert_eq!(response_id, "tokens-a");
        assert_eq!(requests[2].rid, "tokens-b-0");
        assert_eq!(requests[2].input_ids, [3]);
        assert_eq!(
            requests[2].metadata.bootstrap_host.as_deref(),
            Some("prefill-b")
        );
        assert_eq!(requests[2].metadata.bootstrap_port, Some(8999));
        assert_eq!(requests[3].metadata.bootstrap_room, Some(52));
    }

    struct WordTokenizer;

    impl TextTokenizer for WordTokenizer {
        fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<i32>, RendererError> {
            Ok(text.split_whitespace().map(|_| 7).collect())
        }
    }

    #[derive(Clone)]
    struct EngineState {
        requests: Arc<Mutex<Vec<serde_json::Value>>>,
    }

    async fn generate(
        State(state): State<EngineState>,
        Json(body): Json<serde_json::Value>,
    ) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
        state.requests.lock().unwrap().push(body);
        let frame = serde_json::json!({
            "output_ids": [104],
            "meta_info": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "finish_reason": {"type": "stop", "matched": null}
            }
        })
        .to_string();
        Sse::new(futures::stream::iter([
            Ok(Event::default().data(frame)),
            Ok(Event::default().data("[DONE]")),
        ]))
    }

    #[derive(Clone)]
    struct ConcurrentEngineState {
        rendezvous: Arc<Barrier>,
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
    }

    async fn concurrent_generate(
        State(state): State<ConcurrentEngineState>,
        Json(body): Json<serde_json::Value>,
    ) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
        let active = state.active.fetch_add(1, Ordering::SeqCst) + 1;
        state.max_active.fetch_max(active, Ordering::SeqCst);
        state.rendezvous.wait().await;

        let choice = body["rid"]
            .as_str()
            .and_then(|rid| rid.rsplit('-').next())
            .and_then(|choice| choice.parse::<u32>().ok())
            .unwrap();
        if choice == 0 {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        state.active.fetch_sub(1, Ordering::SeqCst);

        let frame = serde_json::json!({
            "output_ids": [104],
            "meta_info": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "finish_reason": {"type": "stop", "matched": choice + 10}
            }
        })
        .to_string();
        Sse::new(futures::stream::iter([
            Ok(Event::default().data(frame)),
            Ok(Event::default().data("[DONE]")),
        ]))
    }

    fn renderer_config() -> RendererConfig {
        RendererConfig {
            served_model_name: "model".into(),
            tokenizer_path: ".".into(),
            revision: None,
            model_path: String::new(),
            chat_template: Some("chatml".into()),
            tool_call_parser: None,
            reasoning_parser: None,
            default_chat_template_kwargs: Default::default(),
            stream_response_default_include_usage: false,
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                vocab_size: 128,
                context_len: 128,
                num_reserved_tokens: 0,
                allow_auto_truncate: false,
                enable_return_hidden_states: false,
            },
        }
    }

    fn tiny_tokenizer() -> dynamo_tokenizers::Tokenizer {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../experimental/sgl-router/tests/fixtures/tiny_tokenizer.json");
        dynamo_tokenizers::Tokenizer::from_file_with_options(
            path.to_str().unwrap(),
            dynamo_tokenizers::TokenizerOptions {
                add_special_tokens: false,
            },
        )
        .unwrap()
    }

    async fn post_request(
        app: Router<()>,
        uri: &str,
        body: &serde_json::Value,
    ) -> axum::response::Response {
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn inference_and_render_share_request_preparation() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let engine = tokio::spawn(
            axum::serve(
                listener,
                Router::new()
                    .route("/generate", post(generate))
                    .with_state(EngineState {
                        requests: captured.clone(),
                    }),
            )
            .into_future(),
        );
        let renderer = Arc::new(RendererService::with_tokenizer(
            renderer_config(),
            Arc::new(WordTokenizer),
            2,
            2,
        ));
        let client =
            HttpGenerateClient::new(format!("http://{address}"), tiny_tokenizer()).unwrap();
        let app = standalone_routes(OpenAIHttpFrontend::new(renderer, client));
        let body = serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello world"}],
            "rid": "chatcmpl-parity",
            "max_tokens": 8,
            "temperature": 0.4,
            "top_k": 17,
            "min_p": 0.2,
            "min_tokens": 3,
            "stop_regex": "END[0-9]",
            "ignore_eos": true,
            "skip_special_tokens": false,
            "chat_template_kwargs": {"enable_thinking": false},
            "cache_salt": "tenant-a",
            "extra_key": "interactive",
            "priority": 7,
            "bootstrap_host": "prefill",
            "bootstrap_port": 8998,
            "bootstrap_room": 42,
            "routed_dp_rank": 2,
            "disagg_prefill_dp_rank": 1
        });

        let render_response = post_request(app.clone(), "/v1/chat/completions/render", &body).await;
        assert_eq!(render_response.status(), StatusCode::OK);
        let mut rendered: serde_json::Value = serde_json::from_slice(
            &to_bytes(render_response.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        let inference_response = post_request(app.clone(), "/v1/chat/completions", &body).await;
        assert_eq!(inference_response.status(), StatusCode::OK);
        let engine_request = captured.lock().unwrap().pop().unwrap();
        assert!(engine_request.get("text").is_none());
        assert_eq!(engine_request["bootstrap_host"], "prefill");
        assert_eq!(engine_request["bootstrap_port"], 8998);
        assert_eq!(engine_request["bootstrap_room"], 42);

        rendered["stream"] = serde_json::Value::Bool(true);
        rendered["return_text_in_logprobs"] = serde_json::Value::Bool(false);
        rendered["sampling_params"]["stop"] = serde_json::json!([]);
        rendered["incremental_streaming_output"] = serde_json::Value::Bool(true);
        assert_eq!(engine_request, rendered);

        let batch = serde_json::json!({
            "model": "model",
            "prompt": ["one", "two"],
            "n": 2,
            "rid": ["prompt-a", "prompt-b"],
            "cache_salt": ["tenant-a", "tenant-b"],
            "extra_key": ["interactive", "batch"],
            "bootstrap_host": ["prefill-a", "prefill-b"],
            "bootstrap_port": [8998, null],
            "bootstrap_room": [41, 52]
        });
        let render_response = post_request(app.clone(), "/v1/completions/render", &batch).await;
        assert_eq!(render_response.status(), StatusCode::OK);
        let rendered: serde_json::Value = serde_json::from_slice(
            &to_bytes(render_response.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(rendered[0]["rid"], "prompt-a-0");
        assert_eq!(rendered[1]["rid"], "prompt-a-1");
        assert_eq!(rendered[2]["rid"], "prompt-b-0");
        assert_eq!(rendered[3]["rid"], "prompt-b-1");
        assert_eq!(rendered[3]["cache_salt"], "tenant-b");
        assert_eq!(rendered[3]["bootstrap_room"], 52);

        let inference_response = post_request(app, "/v1/completions", &batch).await;
        assert_eq!(inference_response.status(), StatusCode::OK);
        let mut engine_requests = std::mem::take(&mut *captured.lock().unwrap());
        engine.abort();
        engine_requests.sort_by(|left, right| left["rid"].as_str().cmp(&right["rid"].as_str()));
        assert_eq!(engine_requests.len(), 4);
        assert_eq!(engine_requests[0]["rid"], "prompt-a-0");
        assert_eq!(engine_requests[1]["rid"], "prompt-a-1");
        assert_eq!(engine_requests[2]["rid"], "prompt-b-0");
        assert_eq!(engine_requests[3]["rid"], "prompt-b-1");
        assert_eq!(engine_requests[2]["cache_salt"], "tenant-b");
        assert_eq!(engine_requests[2]["bootstrap_host"], "prefill-b");
        assert_eq!(
            engine_requests[2]["bootstrap_port"],
            serde_json::Value::Null
        );
        assert_eq!(engine_requests[3]["bootstrap_room"], 52);
    }

    #[tokio::test]
    async fn completion_choices_establish_engine_streams_concurrently_in_input_order() {
        let engine_state = ConcurrentEngineState {
            rendezvous: Arc::new(Barrier::new(2)),
            active: Arc::new(AtomicUsize::new(0)),
            max_active: Arc::new(AtomicUsize::new(0)),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let engine = tokio::spawn(
            axum::serve(
                listener,
                Router::new()
                    .route("/generate", post(concurrent_generate))
                    .with_state(engine_state.clone()),
            )
            .into_future(),
        );
        let renderer = Arc::new(RendererService::with_tokenizer(
            renderer_config(),
            Arc::new(WordTokenizer),
            2,
            2,
        ));
        let client =
            HttpGenerateClient::new(format!("http://{address}"), tiny_tokenizer()).unwrap();
        let app = standalone_routes(OpenAIHttpFrontend::new(renderer, client));
        let response = tokio::time::timeout(
            Duration::from_secs(2),
            post_request(
                app,
                "/v1/completions",
                &serde_json::json!({
                    "model": "model",
                    "prompt": "hello",
                    "n": 2
                }),
            ),
        )
        .await
        .expect("both engine requests must be submitted before either responds");
        engine.abort();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(engine_state.max_active.load(Ordering::SeqCst), 2);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 64 * 1024).await.unwrap())
                .unwrap();
        assert_eq!(body["choices"][0]["index"], 0);
        assert_eq!(body["choices"][0]["matched_stop"], 10);
        assert_eq!(body["choices"][1]["index"], 1);
        assert_eq!(body["choices"][1]["matched_stop"], 11);
    }
}

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
        body::{Body, Bytes, to_bytes},
        extract::State,
        http::{HeaderMap, Request, StatusCode},
        response::sse::{Event, Sse},
        response::{IntoResponse, Redirect},
        routing::{get, post},
    };
    use futures::StreamExt;
    use tokio::sync::Barrier;
    use tower::ServiceExt;

    use super::super::{
        DEFAULT_REQUEST_BODY_LIMIT_BYTES, HttpGenerateClient, OpenAIHttpFrontend, hosted_routes,
        render_only_routes, standalone_routes,
    };
    use crate::openai::test_utils::renderer_config;
    use crate::{RendererError, RendererService, TextTokenizer};

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

    #[tokio::test]
    async fn http_framing_preserves_success_and_each_error_phase() {
        async fn scripted_generate(
            Json(body): Json<serde_json::Value>,
        ) -> axum::response::Response {
            let rid = body["rid"].as_str().unwrap();
            let error = serde_json::json!({"error": {"message": "engine refused", "code": 503}});
            if rid.starts_with("submission") {
                return (StatusCode::SERVICE_UNAVAILABLE, Json(error)).into_response();
            }
            let fails = rid.starts_with("midstream");
            let frame = serde_json::json!({
                "output_ids": [104],
                "meta_info": {
                    "prompt_tokens": 1, "completion_tokens": 1,
                    "finish_reason": if fails { serde_json::Value::Null }
                        else { serde_json::json!({"type": "stop", "matched": null}) }
                }
            });
            let mut frames = vec![frame.to_string()];
            if fails {
                frames.push(error.to_string());
            }
            frames.push("[DONE]".into());
            Sse::new(futures::stream::iter(
                frames
                    .into_iter()
                    .map(|data| Ok::<_, Infallible>(Event::default().data(data))),
            ))
            .into_response()
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let engine = tokio::spawn(
            axum::serve(
                listener,
                Router::new().route("/generate", post(scripted_generate)),
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

        for chat in [false, true] {
            let path = if chat {
                "/v1/chat/completions"
            } else {
                "/v1/completions"
            };
            for stream in [false, true] {
                for phase in ["parse", "validation", "submission", "midstream", "success"] {
                    let mut request = serde_json::json!({
                        "model": if phase == "validation" { "missing" } else { "model" },
                        "rid": phase, "stream": stream,
                        "stream_options": {"include_usage": true}
                    });
                    if chat {
                        request["messages"] =
                            serde_json::json!([{"role": "user", "content": "hi"}]);
                    } else {
                        request["prompt"] = serde_json::json!("hi");
                    }
                    let body = if phase == "parse" {
                        "{".into()
                    } else {
                        request.to_string()
                    };
                    let response = post_json(app.clone(), path, body).await;
                    let is_sse = stream && !matches!(phase, "parse" | "validation");
                    let code = match phase {
                        "parse" | "validation" => 400,
                        "submission" | "midstream" => 503,
                        _ => 200,
                    };
                    assert_eq!(
                        response.status().as_u16(),
                        if is_sse { 200 } else { code },
                        "{path} {stream} {phase}"
                    );
                    assert_eq!(
                        response.headers()["content-type"],
                        if is_sse {
                            "text/event-stream"
                        } else {
                            "application/json"
                        }
                    );
                    let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
                    if !is_sse {
                        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                        if phase == "success" {
                            assert_eq!(body["usage"]["completion_tokens"], 1);
                        } else {
                            assert_eq!(body["error"]["code"], code);
                            assert!(body["error"].get("param").unwrap().is_null());
                        }
                        continue;
                    }
                    let body = std::str::from_utf8(&bytes).unwrap();
                    let data: Vec<_> = body
                        .lines()
                        .filter_map(|line| line.strip_prefix("data: "))
                        .collect();
                    assert_eq!(data.last(), Some(&"[DONE]"));
                    assert_eq!(data.iter().filter(|&&frame| frame == "[DONE]").count(), 1);
                    let frames: Vec<serde_json::Value> = data[..data.len() - 1]
                        .iter()
                        .map(|frame| serde_json::from_str(frame).unwrap())
                        .collect();
                    if phase == "submission" {
                        assert_eq!(frames.len(), 1);
                        assert_eq!(frames[0]["error"]["code"], 503);
                    } else {
                        let usage = frames.last().unwrap();
                        assert_eq!(usage["choices"], serde_json::json!([]));
                        assert_eq!(usage["usage"]["completion_tokens"], 1);
                        if phase == "midstream" {
                            assert_eq!(frames[frames.len() - 2]["error"]["code"], 503);
                        } else if chat {
                            assert_eq!(frames[0]["choices"][0]["delta"]["role"], "assistant");
                            assert!(
                                frames[0]["choices"][0]["delta"]
                                    .get("reasoning_content")
                                    .unwrap()
                                    .is_null()
                            );
                            assert!(frames[0].get("usage").unwrap().is_null());
                            assert!(frames[0].get("service_tier").unwrap().is_null());
                        } else {
                            assert!(
                                frames[0]["choices"][0]
                                    .get("matched_stop")
                                    .unwrap()
                                    .is_null()
                            );
                            assert!(frames[0]["choices"][0].get("logprobs").is_none());
                            assert!(frames[0].get("system_fingerprint").is_none());
                        }
                    }
                }
            }
        }
        engine.abort();
    }

    #[tokio::test]
    async fn dropping_http_body_closes_every_upstream_choice() {
        struct DropNotice(tokio::sync::mpsc::UnboundedSender<()>);
        impl Drop for DropNotice {
            fn drop(&mut self) {
                let _ = self.0.send(());
            }
        }
        async fn slow_generate(
            State(notice): State<tokio::sync::mpsc::UnboundedSender<()>>,
        ) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
            let guard = DropNotice(notice);
            Sse::new(async_stream::stream! {
                let _guard = guard;
                yield Ok(Event::default().data(serde_json::json!({
                    "output_ids": [104],
                    "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": null}
                }).to_string()));
                futures::future::pending::<()>().await;
            })
        }
        let (notice, mut dropped) = tokio::sync::mpsc::unbounded_channel();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let engine = tokio::spawn(
            axum::serve(
                listener,
                Router::new()
                    .route("/generate", post(slow_generate))
                    .with_state(notice),
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
        for chat in [false, true] {
            let (path, mut request) = if chat {
                (
                    "/v1/chat/completions",
                    serde_json::json!({"messages": [{"role": "user", "content": "hi"}]}),
                )
            } else {
                ("/v1/completions", serde_json::json!({"prompt": "hi"}))
            };
            request["model"] = serde_json::json!("model");
            request["stream"] = serde_json::json!(true);
            request["n"] = serde_json::json!(2);
            let response = post_request(app.clone(), path, &request).await;
            assert_eq!(response.status(), StatusCode::OK);
            let mut body = response.into_body().into_data_stream();
            assert!(body.next().await.unwrap().is_ok());
            drop(body);
            for _ in 0..2 {
                tokio::time::timeout(Duration::from_secs(2), dropped.recv())
                    .await
                    .expect("HTTP cancellation did not close an engine choice")
                    .unwrap();
            }
        }
        engine.abort();
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
        post_json(app, uri, body.to_string()).await
    }

    async fn post_json(app: Router<()>, uri: &str, body: String) -> axum::response::Response {
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap()
    }

    fn render_only_test_app() -> Router<()> {
        let renderer = Arc::new(RendererService::with_tokenizer(
            renderer_config(),
            Arc::new(WordTokenizer),
            2,
            2,
        ));
        render_only_routes(renderer)
    }

    #[tokio::test]
    async fn render_only_routes_exclude_inference_without_losing_preprocessing() {
        let chat = serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let rendered =
            post_request(render_only_test_app(), "/v1/chat/completions/render", &chat).await;
        assert_eq!(rendered.status(), StatusCode::OK);

        let tokenized = post_request(
            render_only_test_app(),
            "/v1/tokenize",
            &serde_json::json!({"prompt": "hello world"}),
        )
        .await;
        assert_eq!(tokenized.status(), StatusCode::OK);

        let inference = post_request(render_only_test_app(), "/v1/chat/completions", &chat).await;
        assert_eq!(inference.status(), StatusCode::NOT_FOUND);

        let health = render_only_test_app()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn standalone_health_reflects_engine_status_timeout_and_availability() {
        async fn unhealthy(State(hits): State<Arc<AtomicUsize>>) -> StatusCode {
            if hits.fetch_add(1, Ordering::SeqCst) == 0 {
                StatusCode::IM_A_TEAPOT
            } else {
                futures::future::pending().await
            }
        }

        let hits = Arc::new(AtomicUsize::new(0));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let engine = tokio::spawn(
            axum::serve(
                listener,
                Router::new()
                    .route("/health", get(unhealthy))
                    .with_state(hits.clone()),
            )
            .into_future(),
        );
        let renderer = Arc::new(RendererService::with_tokenizer(
            renderer_config(),
            Arc::new(WordTokenizer),
            2,
            2,
        ));
        let client = HttpGenerateClient::new(format!("http://{address}"), tiny_tokenizer())
            .unwrap()
            .with_health_timeout(Duration::from_millis(50));
        let app = standalone_routes(OpenAIHttpFrontend::new(renderer, client));

        let health = app
            .clone()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::IM_A_TEAPOT);
        assert_eq!(hits.load(Ordering::SeqCst), 1);

        let timed_out = app
            .clone()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(timed_out.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(hits.load(Ordering::SeqCst), 2);

        engine.abort();
        let _ = engine.await;
        let unavailable = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(unavailable.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn render_only_routes_accept_bodies_above_axum_default() {
        let body = serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": "x".repeat(2 * 1024 * 1024)
        })
        .to_string();
        assert!(body.len() > 2 * 1024 * 1024);
        assert!(body.len() < DEFAULT_REQUEST_BODY_LIMIT_BYTES);

        let response = post_json(render_only_test_app(), "/v1/chat/completions/render", body).await;

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn render_only_routes_reject_bodies_above_configured_limit() {
        let body = serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": "x".repeat(DEFAULT_REQUEST_BODY_LIMIT_BYTES)
        })
        .to_string();
        assert!(body.len() > DEFAULT_REQUEST_BODY_LIMIT_BYTES);

        let response = post_json(render_only_test_app(), "/v1/chat/completions/render", body).await;

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn hosted_routes_leave_rust_server_routes_authoritative() {
        async fn native(headers: HeaderMap, body: Bytes) -> impl IntoResponse {
            (
                StatusCode::ACCEPTED,
                [("x-rust-server", "native")],
                format!(
                    "{}:{}",
                    headers
                        .get("x-request-marker")
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or_default(),
                    String::from_utf8_lossy(&body)
                ),
            )
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let upstream = tokio::spawn(
            axum::serve(
                listener,
                Router::new()
                    .route(
                        "/health",
                        get(|| async {
                            (
                                StatusCode::IM_A_TEAPOT,
                                [("x-rust-server", "health")],
                                "rust health",
                            )
                        }),
                    )
                    .route("/native", post(native))
                    .route(
                        "/redirect",
                        get(|| async { Redirect::temporary("/native") }),
                    )
                    .route("/generate", post(generate))
                    .fallback(|| async {
                        (
                            StatusCode::NOT_FOUND,
                            [("x-rust-server", "fallback")],
                            "rust missing",
                        )
                    })
                    .with_state(EngineState {
                        requests: Arc::new(Mutex::new(Vec::new())),
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
        let app = hosted_routes(
            OpenAIHttpFrontend::new(renderer, client),
            format!("http://{address}"),
        )
        .unwrap();

        let health = app
            .clone()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::IM_A_TEAPOT);
        assert_eq!(health.headers()["x-rust-server"], "health");
        assert_eq!(
            to_bytes(health.into_body(), 1024).await.unwrap(),
            "rust health"
        );

        let readiness = app
            .clone()
            .oneshot(
                Request::get("/_sglang_renderer/ready")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(readiness.status(), StatusCode::NO_CONTENT);
        assert_eq!(readiness.headers()["x-sglang-renderer"], "ready");

        let native = app
            .clone()
            .oneshot(
                Request::post("/native?room=7")
                    .header("x-request-marker", "forwarded")
                    .body(Body::from("payload"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(native.status(), StatusCode::ACCEPTED);
        assert_eq!(native.headers()["x-rust-server"], "native");
        assert_eq!(
            to_bytes(native.into_body(), 1024).await.unwrap(),
            "forwarded:payload"
        );

        let redirect = app
            .clone()
            .oneshot(Request::get("/redirect").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(redirect.status(), StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(redirect.headers()["location"], "/native");

        let missing = app
            .oneshot(Request::get("/missing").body(Body::empty()).unwrap())
            .await
            .unwrap();
        upstream.abort();
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
        assert_eq!(missing.headers()["x-rust-server"], "fallback");
        assert_eq!(
            to_bytes(missing.into_body(), 1024).await.unwrap(),
            "rust missing"
        );
    }

    #[tokio::test]
    async fn cumulative_engine_frames_preserve_completion_text_logprobs_and_usage() {
        async fn cumulative_generate(
            Json(body): Json<serde_json::Value>,
        ) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
            assert!(body.get("incremental_streaming_output").is_none());
            assert_eq!(body["stream"], true);
            assert_eq!(body["return_logprob"], true);
            assert_eq!(body["return_text_in_logprobs"], false);
            let frames = (1..=2).map(|count| {
                Ok(Event::default().data(serde_json::json!({
                    "output_ids": vec![104; count],
                    "meta_info": {
                        "prompt_tokens": 1,
                        "completion_tokens": count,
                        "output_token_logprobs": vec![serde_json::json!([-0.5, 104, null]); count],
                        "output_top_logprobs": vec![serde_json::json!([[-0.5, 104, null]]); count],
                        "finish_reason": if count == 2 {
                            serde_json::json!({"type": "length", "length": 2})
                        } else { serde_json::Value::Null }
                    }
                }).to_string()))
            });
            Sse::new(futures::stream::iter(
                frames.chain([Ok(Event::default().data("[DONE]"))]),
            ))
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let engine = tokio::spawn(
            axum::serve(
                listener,
                Router::new().route("/generate", post(cumulative_generate)),
            )
            .into_future(),
        );
        let renderer = Arc::new(RendererService::with_tokenizer(
            renderer_config(),
            Arc::new(WordTokenizer),
            2,
            2,
        ));
        let tokenizer = tiny_tokenizer();
        let expected_text = String::from(tokenizer.decode(&[104, 104], true).unwrap());
        let client = HttpGenerateClient::new(format!("http://{address}"), tokenizer).unwrap();
        let app = standalone_routes(OpenAIHttpFrontend::new(renderer, client));

        for stream in [false, true] {
            let response = post_request(
                app.clone(),
                "/v1/completions",
                &serde_json::json!({
                    "model": "model", "prompt": "hello", "max_tokens": 2,
                    "logprobs": 1, "stream": stream,
                    "stream_options": {"include_usage": true}
                }),
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
            let frames: Vec<serde_json::Value> = if stream {
                let body = std::str::from_utf8(&bytes).unwrap();
                assert!(body.ends_with("data: [DONE]\n\n"));
                body.lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter(|data| *data != "[DONE]")
                    .map(|data| serde_json::from_str(data).unwrap())
                    .collect()
            } else {
                vec![serde_json::from_slice(&bytes).unwrap()]
            };
            let choices: Vec<_> = frames
                .iter()
                .flat_map(|frame| frame["choices"].as_array().unwrap())
                .collect();
            let text: String = choices
                .iter()
                .map(|choice| choice["text"].as_str().unwrap())
                .collect();
            let logprobs: Vec<_> = choices
                .iter()
                .flat_map(|choice| choice["logprobs"]["token_logprobs"].as_array().unwrap())
                .collect();
            assert_eq!(text, expected_text);
            assert_eq!(
                logprobs,
                [&serde_json::json!(-0.5), &serde_json::json!(-0.5)]
            );
            assert_eq!(choices.last().unwrap()["finish_reason"], "length");
            assert_eq!(frames.last().unwrap()["usage"]["completion_tokens"], 2);
        }
        engine.abort();
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

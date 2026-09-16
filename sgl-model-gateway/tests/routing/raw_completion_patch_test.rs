//! Regression coverage for the user's raw-completion and model-length patches.

use std::sync::{Arc, Mutex};

use axum::{
    body::{to_bytes, Body, Bytes},
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::Response,
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use smg::{
    config::types::PolicyConfig,
    core::{BasicWorkerBuilder, ModelCard, RuntimeType, WorkerType},
    policies::PolicyRegistry,
    routers::{
        http::router::Router as HttpRouter,
        openai::OpenAIRouter,
        router_manager::{router_ids, RouterManager},
        RouterTrait,
    },
};
use tokio::{net::TcpListener, task::JoinHandle};
use tower::ServiceExt;

struct Upstream {
    url: String,
    requests: Arc<Mutex<Vec<(HeaderMap, Bytes)>>>,
    model_requests: Arc<Mutex<Vec<HeaderMap>>>,
    task: JoinHandle<()>,
}

impl Drop for Upstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl Upstream {
    async fn start(statuses: Vec<StatusCode>, models: Value) -> Self {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = requests.clone();
        let model_requests = Arc::new(Mutex::new(Vec::new()));
        let captured_models = model_requests.clone();
        let app = Router::new()
            .route(
                "/v1/completions",
                post(move |headers: HeaderMap, body: Bytes| {
                    let captured = captured.clone();
                    let statuses = statuses.clone();
                    async move {
                        let index = {
                            let mut calls = captured.lock().unwrap();
                            let index = calls.len();
                            calls.push((headers, body.clone()));
                            index
                        };
                        let status = statuses.get(index).copied().unwrap_or(StatusCode::OK);
                        let payload = serde_json::from_slice::<Value>(&body).unwrap_or(Value::Null);
                        let stream =
                            status.is_success() && payload["stream"].as_bool().unwrap_or(false);
                        let result = if status.is_success() {
                            json!({"id": "cmpl-test", "choices": [{"index": 0, "text": "a",
                            "token_ids": [0, 7], "prompt_token_ids": [1, 23],
                            "logprobs": {"tokens": ["token_id:0"],
                            "top_logprobs": [{"token_id:0": -0.1, "token_id:9": -0.2}]}}]})
                        } else {
                            json!({"error": {"message": "upstream error"}})
                        };
                        let body = if stream {
                            format!("data: {result}\n\ndata: [DONE]\n\n")
                        } else {
                            result.to_string()
                        };
                        Response::builder()
                            .status(status)
                            .header(
                                "content-type",
                                if stream {
                                    "text/event-stream"
                                } else {
                                    "application/json"
                                },
                            )
                            .body(Body::from(body))
                            .unwrap()
                    }
                }),
            )
            .route(
                "/v1/models",
                get(move |headers: HeaderMap| {
                    let models = models.clone();
                    let captured = captured_models.clone();
                    async move {
                        captured.lock().unwrap().push(headers);
                        Json(models)
                    }
                }),
            );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            url,
            requests,
            model_requests,
            task,
        }
    }

    async fn empty(statuses: Vec<StatusCode>) -> Self {
        Self::start(statuses, json!({"data": []})).await
    }
}

async fn app_for(external: bool, upstreams: &[&Upstream], key: Option<&str>) -> Router {
    let mut ctx = crate::common::test_app::create_test_app_context().await;
    Arc::get_mut(&mut ctx).unwrap().policy_registry =
        Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
    for upstream in upstreams {
        let mut builder = BasicWorkerBuilder::new(&upstream.url)
            .worker_type(WorkerType::Regular)
            .runtime_type(if external {
                RuntimeType::External
            } else {
                RuntimeType::Sglang
            })
            .models(vec![ModelCard::new("test-model")]);
        if let Some(key) = key {
            builder = builder.api_key(key);
        }
        ctx.worker_registry.register(Arc::new(builder.build()));
    }
    let router: Arc<dyn RouterTrait> = if external {
        Arc::new(OpenAIRouter::new(&ctx).await.unwrap())
    } else {
        Arc::new(HttpRouter::new(&ctx).await.unwrap())
    };
    let manager = RouterManager::new(ctx.worker_registry.clone());
    manager.register_router(
        if external {
            router_ids::HTTP_OPENAI
        } else {
            router_ids::HTTP_REGULAR
        },
        router,
    );
    crate::common::test_app::create_test_app_with_context(Arc::new(manager), ctx)
}

async fn send(app: &Router, body: &str) -> Response {
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .header("authorization", "Bearer client-key")
                .body(Body::from(body.to_owned()))
                .unwrap(),
        )
        .await
        .unwrap()
}

async fn read_bytes(response: Response) -> Bytes {
    to_bytes(response.into_body(), usize::MAX).await.unwrap()
}

#[tokio::test]
async fn raw_token_requests_and_response_bodies_survive_both_routers() {
    for external in [false, true] {
        let upstream = Upstream::empty(vec![]).await;
        let app = app_for(external, &[&upstream], None).await;
        for prompt in [
            json!("hello"),
            json!(["one", "two"]),
            json!([0, 23, 4294967295u64]),
            json!([[1, 23], [12, 3]]),
        ] {
            for streaming in [false, true] {
                for flag in [Value::Null, json!(false), json!(true)] {
                    let payload = json!({"model": "test-model", "prompt": prompt,
                        "stream": streaming, "return_token_ids": true,
                        "return_tokens_as_token_ids": flag, "logprobs": 2,
                        "custom_extension": {"nested": [1, null]}});
                    // Whitespace must also survive: this is byte-preserving forwarding.
                    let body = format!(" \n{}\n ", serde_json::to_string_pretty(&payload).unwrap());
                    let response = send(&app, &body).await;
                    assert_eq!(response.status(), StatusCode::OK);
                    let bytes = read_bytes(response).await;
                    let text = std::str::from_utf8(&bytes).unwrap();
                    let result: Value = if streaming {
                        assert!(text.ends_with("data: [DONE]\n\n"));
                        serde_json::from_str(
                            text.lines().next().unwrap().strip_prefix("data: ").unwrap(),
                        )
                        .unwrap()
                    } else {
                        serde_json::from_str(text).unwrap()
                    };
                    assert_eq!(result["choices"][0]["token_ids"], json!([0, 7]));
                    assert_eq!(result["choices"][0]["prompt_token_ids"], json!([1, 23]));
                    assert_eq!(
                        result["choices"][0]["logprobs"]["top_logprobs"],
                        json!([{"token_id:0": -0.1, "token_id:9": -0.2}])
                    );
                    assert_eq!(
                        upstream.requests.lock().unwrap().last().unwrap().1.as_ref(),
                        body.as_bytes()
                    );
                }
            }
        }
        assert_eq!(upstream.requests.lock().unwrap().len(), 24);
    }
}

#[tokio::test]
async fn raw_requests_leave_prompt_and_parameter_validation_to_workers() {
    for external in [false, true] {
        let upstream = Upstream::empty(vec![]).await;
        let app = app_for(external, &[&upstream], None).await;
        for body in [
            r#"{"model":"test-model","prompt":[-1],"stream":"custom"}"#,
            r#"{"model":"test-model","prompt":[1,"mixed"],"max_tokens":-1}"#,
            r#"{"model":"test-model","prompt":[4294967296],"return_tokens_as_token_ids":{}}"#,
            r#"{"model":"test-model","custom_extension":true}"#,
            r#"{"model":"test-model","prompt":[0],"custom":1,"custom":2}"#,
        ] {
            assert_eq!(send(&app, body).await.status(), StatusCode::OK);
            assert_eq!(
                upstream.requests.lock().unwrap().last().unwrap().1.as_ref(),
                body.as_bytes()
            );
        }
    }
}

#[tokio::test]
async fn regular_raw_retry_preserves_exact_body_and_stream() {
    for streaming in [false, true] {
        let upstream = Upstream::empty(vec![StatusCode::SERVICE_UNAVAILABLE]).await;
        let app = app_for(false, &[&upstream], None).await;
        let body = format!(
            "{{\"model\":\"test-model\", \"prompt\":[[0,1],[23]], \"stream\":{streaming}}}"
        );
        let response = send(&app, &body).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = read_bytes(response).await;
        if streaming {
            assert!(bytes.ends_with(b"data: [DONE]\n\n"));
        }
        let calls = upstream.requests.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert!(calls
            .iter()
            .all(|(_, bytes)| bytes.as_ref() == body.as_bytes()));
    }
}

#[tokio::test]
async fn raw_worker_errors_preserve_status_and_body() {
    for external in [false, true] {
        for streaming in [false, true] {
            let upstream = Upstream::empty(vec![StatusCode::BAD_REQUEST]).await;
            let app = app_for(external, &[&upstream], None).await;
            let body = json!({"model":"test-model", "prompt":[0], "stream":streaming}).to_string();
            let response = send(&app, &body).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_eq!(
                serde_json::from_slice::<Value>(&read_bytes(response).await).unwrap(),
                json!({"error":{"message":"upstream error"}})
            );
            assert_eq!(upstream.requests.lock().unwrap().len(), 1);
        }
    }
}

#[tokio::test]
async fn regular_raw_traffic_uses_both_workers() {
    let first = Upstream::empty(vec![]).await;
    let second = Upstream::empty(vec![]).await;
    let app = app_for(false, &[&first, &second], None).await;
    for _ in 0..6 {
        let response = send(&app, r#"{"model":"test-model","prompt":[1,23]}"#).await;
        assert_eq!(response.status(), StatusCode::OK);
        read_bytes(response).await;
    }
    assert_eq!(first.requests.lock().unwrap().len(), 3);
    assert_eq!(second.requests.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn regular_raw_worker_key_is_forwarded_without_client_authorization() {
    let upstream = Upstream::empty(vec![]).await;
    let app = app_for(false, &[&upstream], Some("worker-key")).await;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"test-model","prompt":[0]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        upstream.requests.lock().unwrap()[0].0["authorization"],
        "Bearer worker-key"
    );
}

#[tokio::test]
async fn models_refresh_and_manager_delegation_preserve_minimum_length() {
    let first = Upstream::start(
        vec![],
        json!({"data":[
        {"id":"test-model","max_model_len":8192}, {"id":"other","max_model_len":2048},
        {"id":"missing-length"}, {"max_model_len":1}]}),
    )
    .await;
    let second = Upstream::start(
        vec![],
        json!({"data":[
        {"id":"test-model","max_model_len":4096}, {"id":"other"}]}),
    )
    .await;
    for workers in [vec![&first, &second], vec![&second, &first]] {
        let app = app_for(true, &workers, None).await;
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .header("authorization", "Bearer models-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let result: Value = serde_json::from_slice(&read_bytes(response).await).unwrap();
        let models = result["data"].as_array().unwrap();
        assert_eq!(models.len(), 3);
        for (id, length) in [
            ("test-model", json!(4096)),
            ("other", json!(2048)),
            ("missing-length", Value::Null),
        ] {
            assert_eq!(
                models.iter().find(|m| m["id"] == id).unwrap()["max_model_len"],
                length
            );
        }
    }
    for upstream in [&first, &second] {
        assert!(upstream
            .model_requests
            .lock()
            .unwrap()
            .iter()
            .all(|headers| headers["authorization"] == "Bearer models-key"));
    }
}

#[tokio::test]
async fn models_with_registered_lengths_use_minimum_and_preserve_unknown() {
    let first = Upstream::empty(vec![]).await;
    let second = Upstream::empty(vec![]).await;
    let ctx = crate::common::test_app::create_test_app_context().await;
    for (upstream, length) in [(&first, 8192), (&second, 4096)] {
        let mut model = ModelCard::new("test-model");
        model.context_length = Some(length);
        ctx.worker_registry.register(Arc::new(
            BasicWorkerBuilder::new(&upstream.url)
                .runtime_type(RuntimeType::External)
                .worker_type(WorkerType::Regular)
                .models(vec![model, ModelCard::new("unknown-length")])
                .build(),
        ));
    }
    let router = OpenAIRouter::new(&ctx).await.unwrap();
    let manager = RouterManager::new(ctx.worker_registry.clone());
    manager.register_router(router_ids::HTTP_OPENAI, Arc::new(router));
    let app = crate::common::test_app::create_test_app_with_context(Arc::new(manager), ctx);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let result: Value = serde_json::from_slice(&read_bytes(response).await).unwrap();
    let models = result["data"].as_array().unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(
        models.iter().find(|m| m["id"] == "test-model").unwrap()["max_model_len"],
        4096
    );
    assert!(
        models.iter().find(|m| m["id"] == "unknown-length").unwrap()["max_model_len"].is_null()
    );
}

#[tokio::test]
async fn unknown_model_and_no_workers_return_errors() {
    let upstream = Upstream::empty(vec![]).await;
    let app = app_for(true, &[&upstream], None).await;
    assert_eq!(
        send(&app, r#"{"model":"missing","prompt":[0]}"#)
            .await
            .status(),
        StatusCode::NOT_FOUND
    );
    assert!(upstream.requests.lock().unwrap().is_empty());
    let app = app_for(false, &[], None).await;
    assert_eq!(
        send(&app, r#"{"model":"test-model","prompt":[0]}"#)
            .await
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
}

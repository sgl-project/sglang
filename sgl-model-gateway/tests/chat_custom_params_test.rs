//! Exercise the public HTTP extractor and real HTTP dispatch, including RouterManager.

mod common;

use std::{sync::Arc, time::Duration};

use axum::{
    body::{to_bytes, Body},
    http::{HeaderMap, Request, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use serde_json::{json, Value};
use smg::{
    config::PolicyConfig,
    core::{BasicWorkerBuilder, ModelCard, Worker, WorkerType},
    policies::PolicyRegistry,
    protocols::chat::ChatCompletionRequest,
    routers::{
        router_manager::{router_ids, RouterManager},
        RouterFactory, RouterTrait,
    },
};
use tokio::{net::TcpListener, sync::mpsc, task::JoinHandle, time::timeout};
use tower::ServiceExt;

const STREAM_RESPONSE: &str = concat!(
    "data: {\"id\":\"test\",\"object\":\"chat.completion.chunk\",",
    "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
    "data: [DONE]\n\n",
);

struct EchoWorker {
    url: String,
    received: mpsc::UnboundedReceiver<Value>,
    task: JoinHandle<()>,
}

impl Drop for EchoWorker {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl EchoWorker {
    async fn start() -> Self {
        let (tx, received) = mpsc::unbounded_channel();
        let app = Router::new().route("/v1/chat/completions", post(move |Json(body): Json<Value>| {
            let tx = tx.clone();
            async move {
                let stream = body["stream"].as_bool().unwrap_or(false);
                tx.send(body).unwrap();
                if stream {
                    ([("content-type", "text/event-stream")], STREAM_RESPONSE).into_response()
                } else {
                    Json(json!({"id": "test", "object": "chat.completion", "choices": [], "usage": {}}))
                        .into_response()
                }
            }
        }));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            url,
            received,
            task,
        }
    }

    async fn next(&mut self) -> Value {
        timeout(Duration::from_secs(5), self.received.recv())
            .await
            .unwrap()
            .unwrap()
    }
}

fn payload() -> Value {
    json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1000,
        "temperature": 0.0,
        "chat_template_kwargs": {"thinking": true},
        "custom_params": {"thinking_budget": 10, "nested": {"values": [1, true, "text"]}}
    })
}

fn post_request(body: &Value) -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn exercise_custom_params(mode: &str) {
    let mut first = EchoWorker::start().await;
    let mut second = EchoWorker::start().await;
    let mut ctx = common::test_app::create_test_app_context().await;
    let inner = Arc::get_mut(&mut ctx).unwrap();
    inner.router_config.dp_aware = mode == "dp";
    inner.router_config.policy = PolicyConfig::RoundRobin;
    inner.policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
    inner.client = reqwest::Client::builder().no_proxy().build().unwrap();

    let workers = if mode == "pd" {
        vec![
            (
                first.url.clone(),
                WorkerType::Prefill {
                    bootstrap_port: Some(8998),
                },
            ),
            (second.url.clone(), WorkerType::Decode),
        ]
    } else {
        vec![(
            if mode == "dp" {
                format!("{}@2", first.url)
            } else {
                first.url.clone()
            },
            WorkerType::Regular,
        )]
    };
    for (url, kind) in workers {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(kind)
            .models(vec![ModelCard::new("test-model")])
            .build();
        worker.set_healthy(true);
        ctx.worker_registry.register(Arc::new(worker));
    }
    let router = if mode == "pd" {
        RouterFactory::create_pd_router(None, None, &PolicyConfig::RoundRobin, &ctx)
            .await
            .unwrap()
    } else {
        RouterFactory::create_regular_router(&ctx).await.unwrap()
    };
    let manager = RouterManager::new(ctx.worker_registry.clone());
    let router_id = if mode == "pd" {
        router_ids::HTTP_PD
    } else {
        router_ids::HTTP_REGULAR
    };
    manager.register_router(router_id.clone(), Arc::from(router));
    manager.set_default_router(router_id);
    let app = common::test_app::create_test_app_with_context(Arc::new(manager), ctx);

    for (custom_params, stream) in [
        (Some(payload()["custom_params"].clone()), false),
        (Some(json!({"thinking_budget": 0})), true),
        (Some(json!({})), false),
        (Some(Value::Null), false),
        (None, false),
    ] {
        let mut sent = payload();
        sent["stream"] = json!(stream);
        if let Some(params) = custom_params {
            sent["custom_params"] = params;
        } else {
            sent.as_object_mut().unwrap().remove("custom_params");
        }
        let response = timeout(
            Duration::from_secs(10),
            app.clone().oneshot(post_request(&sent)),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "mode={mode}");
        if stream {
            assert_eq!(response.headers()["content-type"], "text/event-stream");
        }
        let response_body = timeout(
            Duration::from_secs(5),
            to_bytes(response.into_body(), 1024 * 1024),
        )
        .await
        .unwrap()
        .unwrap();
        if stream {
            assert_eq!(response_body.as_ref(), STREAM_RESPONSE.as_bytes());
        }
        let forwarded = first.next().await;
        assert_eq!(
            forwarded.get("custom_params"),
            sent.get("custom_params").filter(|v| !v.is_null()),
            "mode={mode}"
        );
        assert_eq!(
            forwarded["chat_template_kwargs"],
            sent["chat_template_kwargs"]
        );
        assert_eq!(forwarded["max_completion_tokens"], 1000);
        assert_eq!(forwarded["stream"], stream);
        if mode == "dp" {
            assert_eq!(forwarded["data_parallel_rank"], 2);
        }
        if mode == "pd" {
            let decoded = second.next().await;
            assert_eq!(decoded.get("custom_params"), forwarded.get("custom_params"));
            assert!(decoded.get("bootstrap_host").is_some());
        }
    }

    for (field, value) in [
        ("temperature", json!(3)),
        ("messages", json!([])),
        ("min_tokens", json!(1001)),
        ("custom_params", json!("invalid")),
        ("custom_params", json!([10])),
    ] {
        let mut invalid = payload();
        invalid[field] = value;
        let response = app.clone().oneshot(post_request(&invalid)).await.unwrap();
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "field={field}, mode={mode}"
        );
    }
    assert!(first.received.try_recv().is_err());
    assert!(second.received.try_recv().is_err());
}

#[tokio::test]
async fn test_custom_params_regular_http() {
    exercise_custom_params("regular").await;
}

#[tokio::test]
async fn test_custom_params_dp_http() {
    exercise_custom_params("dp").await;
}

#[tokio::test]
async fn test_custom_params_pd_http() {
    exercise_custom_params("pd").await;
}

/// A router implementing only the original chat interface, as unsupported transports do.
#[derive(Debug)]
struct LegacyChatRouter;

#[async_trait::async_trait]
impl RouterTrait for LegacyChatRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        Json(serde_json::to_value(body).unwrap()).into_response()
    }

    fn router_type(&self) -> &'static str {
        "legacy-test"
    }
}

#[tokio::test]
async fn test_custom_params_unsupported_router_and_legacy_fallback() {
    let ctx = common::test_app::create_test_app_context().await;
    let app = common::test_app::create_test_app_with_context(Arc::new(LegacyChatRouter), ctx);
    for custom_params in [
        None,
        Some(Value::Null),
        Some(json!({})),
        Some(json!({"thinking_budget": 10})),
    ] {
        let supported = custom_params.as_ref().is_none_or(Value::is_null);
        let mut sent = payload();
        if let Some(params) = custom_params {
            sent["custom_params"] = params;
        } else {
            sent.as_object_mut().unwrap().remove("custom_params");
        }
        let response = app.clone().oneshot(post_request(&sent)).await.unwrap();
        assert_eq!(
            response.status(),
            if supported {
                StatusCode::OK
            } else {
                StatusCode::BAD_REQUEST
            }
        );
        let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        if supported {
            assert_eq!(body["model"], sent["model"]);
            assert_eq!(body["messages"], sent["messages"]);
            assert_eq!(body["max_completion_tokens"], 1000);
            assert_eq!(body["chat_template_kwargs"], sent["chat_template_kwargs"]);
        } else {
            assert_eq!(body["error"]["code"], "unsupported_custom_params");
        }
    }
}

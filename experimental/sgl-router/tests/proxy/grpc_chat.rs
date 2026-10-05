// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Native gRPC `ChatComplete`: routed like `/v1/chat/completions` and sent to
//! the engine's own `ChatComplete`.

use crate::common::cache_aware_fixture;
use crate::common::mock_worker::MockWorker;
use axum::body::Body;
use futures::future::{ready, BoxFuture, Ready};
use futures::stream::{self, BoxStream, StreamExt};
use serde_json::{json, Value};
use sgl_router::config::PolicyKind;
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{EngineProfile, WireProtocol, WorkerRegistry};
use sglang_grpc_types::sglang::runtime::v1::sglang_service_client::SglangServiceClient;
use sglang_grpc_types::sglang::runtime::v1::{OpenAiRequest, OpenAiStreamChunk};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Channel;
use tonic::{Code, Status};
use tower::ServiceExt;

type Reply = BoxStream<'static, Result<OpenAiStreamChunk, Status>>;

/// An engine's `ChatComplete` that records each request and answers with `reply()`.
#[derive(Clone)]
struct MockEngine {
    seen: Arc<Mutex<Vec<OpenAiRequest>>>,
    reply: Arc<dyn Fn() -> Reply + Send + Sync>,
}

impl MockEngine {
    async fn start(reply: impl Fn() -> Reply + Send + Sync + 'static) -> (Self, u16) {
        let engine = Self {
            seen: Default::default(),
            reply: Arc::new(reply),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tonic::transport::Server::builder().add_service(engine.clone());
        tokio::spawn(server.serve_with_incoming(TcpListenerStream::new(listener)));
        (engine, port)
    }

    async fn body(&self) -> Value {
        for _ in 0..200 {
            if let Some(request) = self.seen.lock().unwrap().last() {
                return serde_json::from_slice(&request.json_body).unwrap();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("the engine saw no request");
    }
}

impl tonic::server::NamedService for MockEngine {
    const NAME: &'static str = "sglang.runtime.v1.SglangService";
}

// Answers every method as `ChatComplete`, the only RPC the router forwards.
impl tower::Service<axum::http::Request<tonic::body::Body>> for MockEngine {
    type Response = axum::http::Response<tonic::body::Body>;
    type Error = std::convert::Infallible;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: axum::http::Request<tonic::body::Body>) -> Self::Future {
        let engine = self.clone();
        Box::pin(async move {
            let mut grpc = tonic::server::Grpc::new(tonic_prost::ProstCodec::default());
            Ok(grpc.server_streaming(engine, request).await)
        })
    }
}

impl tonic::server::ServerStreamingService<OpenAiRequest> for MockEngine {
    type Response = OpenAiStreamChunk;
    type ResponseStream = Reply;
    type Future = Ready<Result<tonic::Response<Reply>, Status>>;

    fn call(&mut self, request: tonic::Request<OpenAiRequest>) -> Self::Future {
        self.seen.lock().unwrap().push(request.into_inner());
        ready(Ok(tonic::Response::new((self.reply)())))
    }
}

fn chunk(json: &str, finished: bool, status_code: Option<i32>) -> OpenAiStreamChunk {
    OpenAiStreamChunk {
        json_chunk: json.as_bytes().to_vec(),
        finished,
        status_code,
    }
}

fn replay(chunks: Vec<OpenAiStreamChunk>) -> impl Fn() -> Reply + Send + Sync + 'static {
    move || stream::iter(chunks.clone().into_iter().map(Ok)).boxed()
}

fn chat(stream: bool) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model": "tiny",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
    }))
    .unwrap()
}

/// A round-robin router over `(mode, url, engine gRPC port)` workers.
fn router_ctx(workers: &[(WorkerMode, &str, Option<u16>)]) -> Arc<AppContext> {
    let mut cfg = cache_aware_fixture::config();
    cfg.model.id = "tiny".into();
    cfg.model.policy = PolicyKind::RoundRobin;
    cfg.model.cache_aware = None;
    let registry = WorkerRegistry::default();
    for &(mode, url, grpc_port) in workers {
        let spec = WorkerSpec {
            id: WorkerId(url.into()),
            url: url.into(),
            mode,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: (mode == WorkerMode::Prefill).then_some(8998),
            ..Default::default()
        };
        let profile = EngineProfile {
            protocol: WireProtocol::default(),
            dp_ranks: 1,
            grpc_port,
        };
        registry.add_with_cb(spec, None, profile).unwrap();
    }
    Arc::new(AppContext::new(
        cfg.clone(),
        Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap()),
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        Arc::new(registry),
        Arc::new(build_registry_with_defaults(&cfg).unwrap()),
    ))
}

async fn serve_grpc(ctx: Arc<AppContext>) -> SglangServiceClient<Channel> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let shutdown = std::future::pending();
    tokio::spawn(sgl_router::server::grpc::serve(listener, ctx, shutdown));
    SglangServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap()
}

async fn call(
    client: &mut SglangServiceClient<Channel>,
    body: Vec<u8>,
) -> Result<Vec<Result<OpenAiStreamChunk, Status>>, Status> {
    let request = OpenAiRequest {
        json_body: body,
        trace_headers: Default::default(),
    };
    let stream = client.chat_complete(request).await?.into_inner();
    Ok(stream.collect().await)
}

fn without_rid(mut body: Value) -> Value {
    assert!(body["rid"].is_string(), "the router mints an engine rid");
    body.as_object_mut().unwrap().remove("rid");
    body
}

#[tokio::test]
async fn sends_the_http_body_and_relays_chunks_unchanged() {
    let http = MockWorker::start(vec![]).await;
    let reply = vec![chunk(r#"{"object":"error"}"#, true, Some(400))];
    let (engine, port) = MockEngine::start(replay(reply.clone())).await;
    let ctx = router_ctx(&[(WorkerMode::Plain, &http.url, Some(port))]);

    let request = axum::http::Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(chat(false)))
        .unwrap();
    build_router(Arc::clone(&ctx))
        .oneshot(request)
        .await
        .unwrap();
    let chunks = call(&mut serve_grpc(ctx).await, chat(false)).await.unwrap();

    let chunks: Vec<_> = chunks.into_iter().map(Result::unwrap).collect();
    assert_eq!(chunks, reply);
    assert_eq!(
        without_rid(engine.body().await),
        without_rid(http.captured_json().await)
    );
}

#[tokio::test]
async fn pd_legs_share_one_bootstrap_room() {
    let done = vec![chunk("", true, None)];
    let (prefill, prefill_port) = MockEngine::start(replay(done.clone())).await;
    let decode_reply = vec![chunk(r#"{"id":"d"}"#, false, None), done[0].clone()];
    let (decode, decode_port) = MockEngine::start(replay(decode_reply.clone())).await;
    let ctx = router_ctx(&[
        (
            WorkerMode::Prefill,
            "http://127.0.0.1:1",
            Some(prefill_port),
        ),
        (WorkerMode::Decode, "http://127.0.0.1:2", Some(decode_port)),
    ]);

    let chunks = call(&mut serve_grpc(ctx).await, chat(true)).await.unwrap();

    let chunks: Vec<_> = chunks.into_iter().map(Result::unwrap).collect();
    assert_eq!(chunks, decode_reply);
    let (prefill, decode) = (prefill.body().await, decode.body().await);
    assert!(prefill["bootstrap_room"].is_u64());
    assert_eq!(prefill["bootstrap_room"], decode["bootstrap_room"]);
}

#[tokio::test]
async fn client_cancel_drops_the_engine_call() {
    struct Dropped(tokio::sync::mpsc::UnboundedSender<()>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }
    let (dropped_tx, mut dropped) = tokio::sync::mpsc::unbounded_channel();
    let (_engine, port) = MockEngine::start(move || {
        let guard = Dropped(dropped_tx.clone());
        stream::iter([Ok(chunk("{}", false, None))])
            .chain(stream::pending())
            .map(move |item| {
                let _ = &guard;
                item
            })
            .boxed()
    })
    .await;
    let mut client = serve_grpc(router_ctx(&[(
        WorkerMode::Plain,
        "http://127.0.0.1:1",
        Some(port),
    )]))
    .await;

    let request = OpenAiRequest {
        json_body: chat(true),
        trace_headers: Default::default(),
    };
    let mut stream = client.chat_complete(request).await.unwrap().into_inner();
    assert!(stream.message().await.unwrap().is_some());
    drop(stream);

    tokio::time::timeout(Duration::from_secs(5), dropped.recv())
        .await
        .expect("the engine call outlived the client");
}

#[tokio::test]
async fn errors_arrive_as_grpc_status() {
    let (_engine, port) = MockEngine::start(|| {
        stream::iter([
            Ok(chunk("{}", false, None)),
            Err(Status::internal("engine failed")),
        ])
        .boxed()
    })
    .await;
    let mut client = serve_grpc(router_ctx(&[
        (WorkerMode::Plain, "http://127.0.0.1:1", Some(port)),
        (WorkerMode::Plain, "http://127.0.0.1:2", None),
    ]))
    .await;

    let unknown_model = serde_json::to_vec(&json!({"model": "nope", "messages": []})).unwrap();
    let status = call(&mut client, unknown_model).await.unwrap_err();
    assert_eq!(status.code(), Code::NotFound);
    assert_eq!(
        status.metadata().get("x-router-error-code").unwrap(),
        "model_not_found"
    );

    // Round robin reaches both workers: the engine fails mid-stream, the other has no gRPC port.
    let mut codes = Vec::new();
    for _ in 0..2 {
        codes.push(match call(&mut client, chat(true)).await {
            Ok(chunks) => chunks.last().unwrap().as_ref().unwrap_err().code(),
            Err(status) => status.code(),
        });
    }
    codes.sort_by_key(|code| *code as i32);
    assert_eq!(codes, [Code::Internal, Code::Unavailable]);
}

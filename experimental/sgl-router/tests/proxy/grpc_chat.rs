// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Native gRPC `ChatComplete`: routed like `/v1/chat/completions` and sent to
//! the engine's own `ChatComplete`.

use crate::common::cache_aware_fixture;
use crate::common::mock_worker::MockWorker;
use axum::body::Body;
use axum::http::{Request as HttpRequest, Response as HttpResponse};
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
use sglang_grpc_types::sglang::runtime::v1 as proto;
use sglang_grpc_types::sglang::runtime::v1::sglang_service_client::SglangServiceClient;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::body::Body as GrpcBody;
use tonic::server::{Grpc, NamedService, ServerStreamingService, UnaryService};
use tonic::transport::Channel;
use tonic::{Code, Status};
use tonic_prost::ProstCodec;
use tower::ServiceExt;

type Replies<T> = BoxStream<'static, Result<T, Status>>;
type Handler =
    Arc<dyn Fn(HttpRequest<GrpcBody>) -> BoxFuture<'static, HttpResponse<GrpcBody>> + Send + Sync>;

/// An engine serving the RPCs it is given, by method name.
#[derive(Clone, Default)]
struct MockEngine(HashMap<&'static str, Handler>);

/// Adapts a closure to a tonic service; records nothing itself.
struct Svc<F>(Arc<F>);

impl<Req, Rep, F> ServerStreamingService<Req> for Svc<F>
where
    F: Fn(Req) -> Replies<Rep> + Send + Sync + 'static,
    Rep: Send + 'static,
{
    type Response = Rep;
    type ResponseStream = Replies<Rep>;
    type Future = Ready<Result<tonic::Response<Replies<Rep>>, Status>>;

    fn call(&mut self, request: tonic::Request<Req>) -> Self::Future {
        ready(Ok(tonic::Response::new((self.0)(request.into_inner()))))
    }
}

struct Unary<F>(Arc<F>);

impl<Req, Rep, F> UnaryService<Req> for Unary<F>
where
    F: Fn(Req) -> Rep + Send + Sync + 'static,
{
    type Response = Rep;
    type Future = Ready<Result<tonic::Response<Rep>, Status>>;

    fn call(&mut self, request: tonic::Request<Req>) -> Self::Future {
        ready(Ok(tonic::Response::new((self.0)(request.into_inner()))))
    }
}

impl MockEngine {
    fn stream<Req, Rep>(
        mut self,
        method: &'static str,
        reply: impl Fn(Req) -> Replies<Rep> + Send + Sync + 'static,
    ) -> Self
    where
        Req: prost::Message + Default + Send + 'static,
        Rep: prost::Message + Send + 'static,
    {
        let reply = Arc::new(reply);
        let handler: Handler = Arc::new(move |request| {
            let svc = Svc(Arc::clone(&reply));
            Box::pin(async move {
                let codec = ProstCodec::<Rep, Req>::default();
                Grpc::new(codec).server_streaming(svc, request).await
            })
        });
        self.0.insert(method, handler);
        self
    }

    fn unary<Req, Rep>(
        mut self,
        method: &'static str,
        reply: impl Fn(Req) -> Rep + Send + Sync + 'static,
    ) -> Self
    where
        Req: prost::Message + Default + Send + 'static,
        Rep: prost::Message + Send + 'static,
    {
        let reply = Arc::new(reply);
        let handler: Handler = Arc::new(move |request| {
            let svc = Unary(Arc::clone(&reply));
            Box::pin(async move {
                let codec = ProstCodec::<Rep, Req>::default();
                Grpc::new(codec).unary(svc, request).await
            })
        });
        self.0.insert(method, handler);
        self
    }

    /// Serve on 127.0.0.1 and return the port.
    async fn start(self) -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tonic::transport::Server::builder().add_service(self);
        tokio::spawn(server.serve_with_incoming(TcpListenerStream::new(listener)));
        port
    }
}

impl NamedService for MockEngine {
    const NAME: &'static str = "sglang.runtime.v1.SglangService";
}

impl tower::Service<HttpRequest<GrpcBody>> for MockEngine {
    type Response = HttpResponse<GrpcBody>;
    type Error = std::convert::Infallible;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: HttpRequest<GrpcBody>) -> Self::Future {
        let method = request.uri().path().rsplit('/').next().unwrap_or_default();
        let handler = Arc::clone(&self.0[method]);
        Box::pin(async move { Ok(handler(request).await) })
    }
}

/// Requests an engine saw, polled because a PD prefill is sent in the background.
struct Seen<T>(Arc<Mutex<Vec<T>>>);

impl<T: Clone> Seen<T> {
    fn new() -> Self {
        Self(Default::default())
    }

    fn record(&self) -> impl Fn(T) + Send + Sync + 'static
    where
        T: Send + 'static,
    {
        let seen = Arc::clone(&self.0);
        move |request| seen.lock().unwrap().push(request)
    }

    async fn last(&self) -> T {
        for _ in 0..200 {
            if let Some(request) = self.0.lock().unwrap().last() {
                return request.clone();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("the engine saw no request");
    }
}

fn chunk(json: &str, finished: bool, status_code: Option<i32>) -> proto::OpenAiStreamChunk {
    proto::OpenAiStreamChunk {
        json_chunk: json.as_bytes().to_vec(),
        finished,
        status_code,
    }
}

/// An OpenAI streaming RPC that records each request and replays `chunks`.
fn openai_rpc(
    seen: &Seen<proto::OpenAiRequest>,
    chunks: Vec<proto::OpenAiStreamChunk>,
) -> impl Fn(proto::OpenAiRequest) -> Replies<proto::OpenAiStreamChunk> + Send + Sync + 'static {
    let record = seen.record();
    move |request| {
        record(request);
        stream::iter(chunks.clone().into_iter().map(Ok)).boxed()
    }
}

fn body_of(request: &proto::OpenAiRequest) -> Value {
    serde_json::from_slice(&request.json_body).unwrap()
}

fn chat(stream: bool) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model": "tiny",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
    }))
    .unwrap()
}

fn openai(body: Vec<u8>) -> proto::OpenAiRequest {
    proto::OpenAiRequest {
        json_body: body,
        trace_headers: Default::default(),
    }
}

/// A round-robin router over `(mode, url, engine gRPC port)` workers.
fn router_ctx(workers: &[(WorkerMode, &str, Option<u16>)]) -> Arc<AppContext> {
    router_ctx_with(workers, Proxy::new(Duration::from_secs(5)).unwrap())
}

fn router_ctx_with(workers: &[(WorkerMode, &str, Option<u16>)], proxy: Proxy) -> Arc<AppContext> {
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
        Arc::new(proxy),
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

async fn collect<T>(
    replies: Result<tonic::Response<tonic::Streaming<T>>, Status>,
) -> Result<Vec<T>, Status> {
    replies?
        .into_inner()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect()
}

fn without_rid(mut body: Value) -> Value {
    assert!(body["rid"].is_string(), "the router mints an engine rid");
    body.as_object_mut().unwrap().remove("rid");
    body
}

/// One chunk, then nothing until the call is dropped; `dropped` hears about it.
fn stalled_after_one_chunk(
    dropped: tokio::sync::mpsc::UnboundedSender<()>,
) -> Replies<proto::OpenAiStreamChunk> {
    struct Dropped(tokio::sync::mpsc::UnboundedSender<()>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }
    let guard = Dropped(dropped);
    stream::iter([Ok(chunk("{}", false, None))])
        .chain(stream::pending())
        .map(move |item| {
            let _ = &guard;
            item
        })
        .boxed()
}

#[tokio::test]
async fn sends_the_http_body_and_relays_chunks_unchanged() {
    let http = MockWorker::start(vec![]).await;
    let reply = vec![chunk(r#"{"object":"error"}"#, true, Some(400))];
    let chats = Seen::new();
    let port = MockEngine::default()
        .stream("ChatComplete", openai_rpc(&chats, reply.clone()))
        .start()
        .await;
    let ctx = router_ctx(&[(WorkerMode::Plain, &http.url, Some(port))]);
    let request = HttpRequest::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(chat(false)))
        .unwrap();
    build_router(Arc::clone(&ctx))
        .oneshot(request)
        .await
        .unwrap();
    let mut client = serve_grpc(ctx).await;

    let chunks = collect(client.chat_complete(openai(chat(false))).await).await;
    assert_eq!(chunks.unwrap(), reply);
    assert_eq!(
        without_rid(body_of(&chats.last().await)),
        without_rid(http.captured_json().await)
    );
}

#[tokio::test]
async fn pd_legs_share_one_bootstrap_room() {
    let done = vec![chunk("", true, None)];
    let (prefill_chats, decode_chats) = (Seen::new(), Seen::new());
    let prefill = MockEngine::default()
        .stream("ChatComplete", openai_rpc(&prefill_chats, done.clone()))
        .start()
        .await;
    let decode = MockEngine::default()
        .stream("ChatComplete", openai_rpc(&decode_chats, done))
        .start()
        .await;
    let mut client = serve_grpc(router_ctx(&[
        (WorkerMode::Prefill, "http://127.0.0.1:1", Some(prefill)),
        (WorkerMode::Decode, "http://127.0.0.1:2", Some(decode)),
    ]))
    .await;

    collect(client.chat_complete(openai(chat(true))).await)
        .await
        .unwrap();
    let (p, d) = (
        body_of(&prefill_chats.last().await),
        body_of(&decode_chats.last().await),
    );
    assert!(p["bootstrap_room"].is_u64());
    assert_eq!(p["bootstrap_room"], d["bootstrap_room"]);
}

#[tokio::test]
async fn client_cancel_aborts_the_engine_request() {
    let (dropped_tx, mut dropped) = tokio::sync::mpsc::unbounded_channel();
    let (chats, aborts) = (Seen::new(), Seen::new());
    let (record_chat, record_abort) = (chats.record(), aborts.record());
    let port = MockEngine::default()
        .stream("ChatComplete", move |request| {
            record_chat(request);
            stalled_after_one_chunk(dropped_tx.clone())
        })
        .unary("Abort", move |request: proto::AbortRequest| {
            record_abort(request);
            proto::AbortResponse::default()
        })
        .start()
        .await;
    let mut client = serve_grpc(router_ctx(&[(
        WorkerMode::Plain,
        "http://127.0.0.1:1",
        Some(port),
    )]))
    .await;

    let mut stream = client
        .chat_complete(openai(chat(true)))
        .await
        .unwrap()
        .into_inner();
    assert!(stream.message().await.unwrap().is_some());
    drop(stream);

    tokio::time::timeout(Duration::from_secs(5), dropped.recv())
        .await
        .expect("the engine call outlived the client");
    // The engine's own abort on a dropped call targets another id, so the rid is aborted too.
    let rid = body_of(&chats.last().await)["rid"].clone();
    assert_eq!(aborts.last().await.rid, rid.as_str().unwrap());
}

#[tokio::test]
async fn stream_outcomes_reach_the_breaker_by_status() {
    let calls = Arc::new(Mutex::new(0));
    let counted = Arc::clone(&calls);
    let port = MockEngine::default()
        .stream("ChatComplete", move |_: proto::OpenAiRequest| {
            let mut calls = counted.lock().unwrap();
            *calls += 1;
            match *calls {
                // Backpressure ends three streams; it must not open the breaker.
                1..=3 => stream::iter([
                    Ok(chunk("{}", false, None)),
                    Err(Status::resource_exhausted("queue full")),
                ])
                .boxed(),
                // A rejected request is one finished chunk with its status; faults count.
                _ => stream::iter([Ok(chunk(r#"{"object":"error"}"#, true, Some(500)))]).boxed(),
            }
        })
        .start()
        .await;
    let mut client = serve_grpc(router_ctx(&[(
        WorkerMode::Plain,
        "http://127.0.0.1:1",
        Some(port),
    )]))
    .await;

    for _ in 0..3 {
        let result = collect(client.chat_complete(openai(chat(true))).await).await;
        assert_eq!(result.unwrap_err().code(), Code::ResourceExhausted);
    }
    for _ in 0..3 {
        let chunks = collect(client.chat_complete(openai(chat(true))).await)
            .await
            .unwrap();
        assert_eq!(chunks[0].status_code, Some(500));
    }
    let status = client.chat_complete(openai(chat(true))).await.unwrap_err();
    assert_eq!(
        status.code(),
        Code::Unavailable,
        "three 500s open the breaker"
    );
    assert_eq!(*calls.lock().unwrap(), 6);
}

#[tokio::test]
async fn router_stream_failures_keep_their_error_code() {
    let (dropped_tx, _dropped) = tokio::sync::mpsc::unbounded_channel();
    let port = MockEngine::default()
        .stream("ChatComplete", move |_: proto::OpenAiRequest| {
            stalled_after_one_chunk(dropped_tx.clone())
        })
        .start()
        .await;
    let proxy = Proxy::new(Duration::from_secs(5))
        .unwrap()
        .with_stream_idle_timeout(Duration::from_millis(100));
    let workers = [(WorkerMode::Plain, "http://127.0.0.1:1", Some(port))];
    let mut client = serve_grpc(router_ctx_with(&workers, proxy)).await;

    let result = collect(client.chat_complete(openai(chat(true))).await).await;
    let status = result.unwrap_err();
    assert_eq!(status.code(), Code::DeadlineExceeded);
    assert_eq!(
        status.metadata().get("x-router-error-code").unwrap(),
        "upstream_timeout"
    );
}

#[tokio::test]
async fn errors_arrive_as_grpc_status() {
    let port = MockEngine::default()
        .stream("ChatComplete", |_: proto::OpenAiRequest| {
            stream::iter([
                Ok(chunk("{}", false, None)),
                Err(Status::internal("engine failed")),
            ])
            .boxed()
        })
        .start()
        .await;
    let mut client = serve_grpc(router_ctx(&[
        (WorkerMode::Plain, "http://127.0.0.1:1", Some(port)),
        (WorkerMode::Plain, "http://127.0.0.1:2", None),
    ]))
    .await;

    let unknown_model = json!({"model": "nope", "messages": []}).to_string();
    let status = client
        .chat_complete(openai(unknown_model.into()))
        .await
        .unwrap_err();
    assert_eq!(status.code(), Code::NotFound);
    assert_eq!(
        status.metadata().get("x-router-error-code").unwrap(),
        "model_not_found"
    );

    // Round robin reaches both workers: the engine fails mid-stream, the other has no gRPC port.
    let mut codes = Vec::new();
    for _ in 0..2 {
        let result = collect(client.chat_complete(openai(chat(true))).await).await;
        codes.push(result.unwrap_err().code());
    }
    codes.sort_by_key(|code| *code as i32);
    assert_eq!(codes, [Code::Internal, Code::Unavailable]);
}

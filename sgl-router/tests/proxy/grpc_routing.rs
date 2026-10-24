// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Native gRPC: each RPC is routed like its HTTP route and sent to the
//! engine's own RPC.

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
use sgl_router::policies_reorg::factory::build_resolver;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::{AppContext, ChatRouting};
use sgl_router::state::kv_events::KvEventIndex;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{DecodeGate, EngineProfile, WireProtocol, WorkerRegistry};
use sglang_api_types::runtime::v1 as proto;
use sglang_api_types::runtime::v1::sglang_service_client::SglangServiceClient;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
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
struct RpcEngine<const NATIVE: bool>(HashMap<&'static str, Handler>);

type MockEngine = RpcEngine<false>;

/// Adapts a closure to a tonic service; records nothing itself.
struct Svc<F>(Arc<F>);

impl<Req, Rep, F> ServerStreamingService<Req> for Svc<F>
where
    F: Fn(tonic::Request<Req>) -> Replies<Rep> + Send + Sync + 'static,
    Rep: Send + 'static,
{
    type Response = Rep;
    type ResponseStream = Replies<Rep>;
    type Future = Ready<Result<tonic::Response<Replies<Rep>>, Status>>;

    fn call(&mut self, request: tonic::Request<Req>) -> Self::Future {
        ready(Ok(tonic::Response::new((self.0)(request))))
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

impl<const NATIVE: bool> RpcEngine<NATIVE> {
    fn stream<Req, Rep>(
        self,
        method: &'static str,
        reply: impl Fn(Req) -> Replies<Rep> + Send + Sync + 'static,
    ) -> Self
    where
        Req: prost::Message + Default + Send + 'static,
        Rep: prost::Message + Send + 'static,
    {
        self.stream_with_context(method, move |request| reply(request.into_inner()))
    }

    fn stream_with_context<Req, Rep>(
        mut self,
        method: &'static str,
        reply: impl Fn(tonic::Request<Req>) -> Replies<Rep> + Send + Sync + 'static,
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
    async fn start(mut self) -> u16 {
        if NATIVE && !self.0.contains_key("Abort") {
            self = self.unary("Abort", |_: sglang_api_types::api::v1::AbortRequest| {
                sglang_api_types::api::v1::AbortResponse::default()
            });
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tonic::transport::Server::builder().add_service(self);
        tokio::spawn(server.serve_with_incoming(TcpListenerStream::new(listener)));
        port
    }
}

impl<const NATIVE: bool> NamedService for RpcEngine<NATIVE> {
    const NAME: &'static str = if NATIVE {
        "sglang.api.v1.SglangService"
    } else {
        "sglang.runtime.v1.SglangService"
    };
}

impl<const NATIVE: bool> tower::Service<HttpRequest<GrpcBody>> for RpcEngine<NATIVE> {
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
    router_ctx_with(workers, false, Proxy::new(Duration::from_secs(5)).unwrap())
}

/// [`router_ctx`] on the bucket-first (reorg) selection path when `reorg`.
fn router_ctx_with(
    workers: &[(WorkerMode, &str, Option<u16>)],
    reorg: bool,
    proxy: Proxy,
) -> Arc<AppContext> {
    router_ctx_with_ranks(workers, reorg, proxy, 1)
}

fn router_ctx_with_ranks(
    workers: &[(WorkerMode, &str, Option<u16>)],
    reorg: bool,
    proxy: Proxy,
    dp_ranks: u32,
) -> Arc<AppContext> {
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
            openai: None,
            protocol: WireProtocol::default(),
            dp_ranks,
            grpc_port,
        };
        registry
            .add_with_cb(spec, Some(Default::default()), profile)
            .unwrap();
    }
    if reorg {
        cfg.model.policy = PolicyKind::PowerOfTwo;
    }
    let mut ctx = AppContext::new(
        cfg.clone(),
        Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap()),
        Arc::new(proxy),
        Arc::new(registry),
        Arc::new(build_registry_with_defaults(&cfg).unwrap()),
    );
    if reorg {
        let (resolver, _) = build_resolver(&cfg.model, &KvEventIndex::new(), None).unwrap();
        ctx.chat_routing = ChatRouting::Reorg([(ModelId("tiny".into()), resolver)].into());
    }
    Arc::new(ctx)
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
async fn openai_rpcs_send_the_http_body_and_relay_chunks_unchanged() {
    let http = MockWorker::start(vec![]).await;
    let reply = vec![chunk(r#"{"object":"error"}"#, true, Some(400))];
    let (chats, completions) = (Seen::new(), Seen::new());
    let port = MockEngine::default()
        .stream("ChatComplete", openai_rpc(&chats, reply.clone()))
        .stream("Complete", openai_rpc(&completions, reply.clone()))
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

    let prompt = json!({"model": "tiny", "prompt": "hi", "max_tokens": 4});
    let chunks = collect(client.complete(openai(prompt.to_string().into())).await).await;
    assert_eq!(chunks.unwrap(), reply);
    let sent = without_rid(body_of(&completions.last().await));
    assert_eq!(sent, prompt, "the prompt is forwarded as sent");
}

#[tokio::test]
async fn retries_failures_before_a_response_on_another_worker() {
    for status_code in [None, Some(429), Some(503)] {
        let failed = MockEngine::default()
            .stream("ChatComplete", move |_: proto::OpenAiRequest| {
                let reply = match status_code {
                    Some(code) => Ok(chunk("{}", true, Some(code))),
                    None => Err(Status::unavailable("engine unavailable")),
                };
                stream::iter([reply]).boxed()
            })
            .unary("Abort", |_: proto::AbortRequest| {
                proto::AbortResponse::default()
            })
            .start()
            .await;
        let reply = vec![chunk("{\"ok\":true}", true, Some(200))];
        let healthy = MockEngine::default()
            .stream("ChatComplete", openai_rpc(&Seen::new(), reply.clone()))
            .start()
            .await;
        for max_attempts in [1, 2] {
            for streaming in [false, true] {
                let mut ctx = router_ctx(&[
                    (WorkerMode::Plain, "http://127.0.0.1:1", Some(failed)),
                    (WorkerMode::Plain, "http://127.0.0.1:2", Some(healthy)),
                ]);
                Arc::get_mut(&mut ctx).unwrap().config.proxy.max_attempts =
                    std::num::NonZeroU32::new(max_attempts).unwrap();
                let mut client = serve_grpc(Arc::clone(&ctx)).await;
                let mut successes = 0;
                for _ in 0..2 {
                    let result = collect(client.chat_complete(openai(chat(streaming))).await).await;
                    successes += usize::from(result.is_ok_and(|chunks| chunks == reply));
                }
                assert_eq!(successes, max_attempts as usize);
            }
        }
    }
}

#[tokio::test]
async fn pd_legs_share_one_bootstrap_room() {
    let done = vec![chunk("", true, None)];
    let generated = || {
        let reply = proto::GenerateResponse {
            output_ids: vec![7],
            finished: true,
            ..Default::default()
        };
        stream::iter([Ok(reply)]).boxed()
    };
    let engine = |chats: &Seen<_>, generates: &Seen<proto::GenerateRequest>| {
        let record = generates.record();
        MockEngine::default()
            .stream("ChatComplete", openai_rpc(chats, done.clone()))
            .stream("Generate", move |request| {
                record(request);
                generated()
            })
    };
    let (prefill_chats, prefill_generates) = (Seen::new(), Seen::new());
    let (decode_chats, decode_generates) = (Seen::new(), Seen::new());
    let prefill = engine(&prefill_chats, &prefill_generates).start().await;
    let decode = engine(&decode_chats, &decode_generates).start().await;
    // Legacy policies, then the reorg bucket path.
    for reorg in [false, true] {
        let workers = [
            (WorkerMode::Prefill, "http://127.0.0.1:1", Some(prefill)),
            (WorkerMode::Decode, "http://127.0.0.1:2", Some(decode)),
        ];
        let mut client = serve_grpc(router_ctx_with(
            &workers,
            reorg,
            Proxy::new(Duration::from_secs(5)).unwrap(),
        ))
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

        let request = proto::GenerateRequest {
            input_ids: vec![1, 2, 3],
            stream: Some(true),
            ..Default::default()
        };
        let replies = collect(client.generate(request).await).await.unwrap();
        assert_eq!(
            replies[0].output_ids,
            [7],
            "decode's reply reaches the client"
        );
        let (p, d) = (
            prefill_generates.last().await,
            decode_generates.last().await,
        );
        let room = p.disaggregated_params.as_ref().unwrap();
        assert_eq!(
            (room.bootstrap_host.as_str(), room.bootstrap_port),
            ("127.0.0.1", 8998)
        );
        assert_eq!(p.disaggregated_params, d.disaggregated_params);
        assert!(
            p.rid.is_some() && p.rid == d.rid,
            "both legs carry the router's rid"
        );
    }
}

#[tokio::test]
async fn decode_dispatch_limit_gates_decode_as_over_http() {
    let prefill = MockEngine::default()
        .stream("ChatComplete", |_: proto::OpenAiRequest| {
            stream::once(async {
                tokio::time::sleep(Duration::from_millis(800)).await;
                Ok(chunk("{}", true, None))
            })
            .boxed()
        })
        .start()
        .await;
    let reply = vec![chunk("{}", true, None)];
    let decode = MockEngine::default()
        .stream("ChatComplete", openai_rpc(&Seen::new(), reply))
        .start()
        .await;
    for (streaming, waits) in [(false, true), (true, false)] {
        let mut ctx = router_ctx(&[
            (WorkerMode::Prefill, "http://127.0.0.1:1", Some(prefill)),
            (WorkerMode::Decode, "http://127.0.0.1:2", Some(decode)),
        ]);
        Arc::get_mut(&mut ctx).unwrap().decode_gate = DecodeGate::new(std::num::NonZeroU32::new(1));
        let client = serve_grpc(ctx).await;
        let call = |mut client: SglangServiceClient<Channel>| async move {
            collect(client.chat_complete(openai(chat(streaming))).await)
                .await
                .unwrap()
        };
        let start = Instant::now();
        futures::future::join(call(client.clone()), call(client)).await;
        let waited = start.elapsed() >= Duration::from_millis(700);
        assert_eq!(waited, waits, "streaming={streaming}");
    }
}

#[tokio::test]
async fn typed_unary_rpc_returns_the_engine_reply() {
    let embeds = Seen::new();
    let record = embeds.record();
    let port = MockEngine::default()
        .unary("Embed", move |request: proto::EmbedRequest| {
            record(request);
            proto::EmbedResponse {
                embedding: vec![0.5],
                meta_info: Default::default(),
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

    let request = proto::EmbedRequest {
        input_ids: vec![4, 5],
        ..Default::default()
    };
    let mut request = tonic::Request::new(request);
    let traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
    request
        .metadata_mut()
        .insert("traceparent", traceparent.parse().unwrap());
    let reply = client.embed(request).await.unwrap().into_inner();
    assert_eq!(reply.embedding, [0.5]);
    let sent = embeds.last().await;
    assert_eq!(sent.input_ids, [4, 5]);
    assert!(sent.rid.is_some(), "the router mints an engine rid");
    assert_eq!(
        sent.trace_headers["traceparent"], traceparent,
        "trace context reaches the engine"
    );
}

#[tokio::test]
async fn text_rpcs_reach_the_engine_as_router_tokens() {
    let (generates, embeds) = (Seen::new(), Seen::new());
    let (record_generate, record_embed) = (generates.record(), embeds.record());
    let port = MockEngine::default()
        .stream("Generate", move |request: proto::GenerateRequest| {
            record_generate(request);
            let reply = proto::GenerateResponse {
                text: Some("Paris".into()),
                finished: true,
                ..Default::default()
            };
            stream::iter([Ok(reply)]).boxed()
        })
        .unary("Embed", move |request: proto::EmbedRequest| {
            record_embed(request);
            proto::EmbedResponse {
                embedding: vec![0.5],
                meta_info: Default::default(),
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

    let request = proto::TextGenerateRequest {
        text: "hello world".into(),
        ..Default::default()
    };
    let replies = collect(client.text_generate(request).await).await.unwrap();
    assert_eq!(replies[0].text, "Paris");
    let sent = generates.last().await;
    assert!(!sent.input_ids.is_empty() && sent.return_text == Some(true));

    let request = proto::TextEmbedRequest {
        text: "hello world".into(),
        ..Default::default()
    };
    let reply = client.text_embed(request).await.unwrap().into_inner();
    assert_eq!(reply.embedding, [0.5]);
    assert_eq!(
        embeds.last().await.input_ids,
        sent.input_ids,
        "both tokenize alike"
    );
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
    // Both the request timeout and the stream idle timeout.
    let proxy = Proxy::new(Duration::from_millis(100))
        .unwrap()
        .with_stream_idle_timeout(Duration::from_millis(100));
    let workers = [(WorkerMode::Plain, "http://127.0.0.1:1", Some(port))];
    let mut client = serve_grpc(router_ctx_with(&workers, false, proxy)).await;

    for stream in [true, false] {
        let result = collect(client.chat_complete(openai(chat(stream))).await).await;
        let status = result.unwrap_err();
        assert_eq!(status.code(), Code::DeadlineExceeded, "stream={stream}");
        let code = status.metadata().get("x-router-error-code");
        assert_eq!(code.unwrap(), "upstream_timeout", "stream={stream}");
    }
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

mod native {
    use super::*;
    use sglang_api_types::api::v1 as proto;
    use sglang_api_types::api::v1::sglang_service_client::SglangServiceClient;

    type MockEngine = RpcEngine<true>;

    fn router_ctx(workers: &[(WorkerMode, &str, Option<u16>)]) -> Arc<AppContext> {
        router_ctx_with(workers, false, Proxy::new(Duration::from_secs(5)).unwrap())
    }

    fn router_ctx_with(
        workers: &[(WorkerMode, &str, Option<u16>)],
        reorg: bool,
        proxy: Proxy,
    ) -> Arc<AppContext> {
        router_ctx_with_ranks(workers, reorg, proxy, 2)
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

    fn generate(stream: bool) -> proto::GenerateRequest {
        serde_json::from_value(json!({
            "input_ids": [1, 2, 3], "stream": stream,
            "sampling_params": {"max_new_tokens": 4}
        }))
        .unwrap()
    }

    fn frame(index: Option<u32>) -> proto::GenerateStreamItem {
        proto::GenerateStreamItem {
            item: Some(proto::generate_stream_item::Item::Frame(
                proto::GenerateResponse {
                    text: "four".into(),
                    output_ids: Some(proto::TokenIds {
                        ids: vec![4_000_000_000],
                    }),
                    index,
                    ..Default::default()
                },
            )),
        }
    }

    /// Native batches and per-item errors must survive without the old i32/JSON-map conversions.
    #[tokio::test]
    async fn generate_preserves_native_batches_errors_and_context() {
        let seen = Seen::new();
        let record = seen.record();
        let reply = vec![
            proto::GenerateStreamItem {
                item: Some(proto::generate_stream_item::Item::Error(
                    proto::GenerateStreamError {
                        error: Some(proto::ErrorBody {
                            message: "bad item".into(),
                            code: 400,
                        }),
                        index: Some(0),
                    },
                )),
            },
            frame(Some(1)),
        ];
        let replay = reply.clone();
        let port = MockEngine::default()
            .stream_with_context(
                "Generate",
                move |request: tonic::Request<proto::GenerateRequest>| {
                    let trace = request
                        .metadata()
                        .get("traceparent")
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_owned();
                    let credential = request.metadata().contains_key("authorization");
                    record((request.into_inner(), trace, credential));
                    stream::iter(replay.clone().into_iter().map(Ok)).boxed()
                },
            )
            .start()
            .await;
        let mut client = serve_grpc(router_ctx(&[(
            WorkerMode::Plain,
            "http://127.0.0.1:1",
            Some(port),
        )]))
        .await;
        for streaming in [false, true] {
            let original: proto::GenerateRequest = serde_json::from_value(json!({
                "input_ids": [[4_000_000_000_u64], [2, 3]],
                "stream": streaming,
                "sampling_params": [{"max_new_tokens": 4}, {"max_new_tokens": 5}],
                "cache_salt": ["a", "b"],
                "kv_hints": {"protocol_version": "1", "message_id": "hint", "actions": [{
                    "action_id": "fetch", "action_type": "kv.fetch", "action_version": "1.0",
                    "payload": {"hash": u64::MAX}
                }]}
            }))
            .unwrap();
            let mut request = tonic::Request::new(original.clone());
            let trace = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
            request
                .metadata_mut()
                .insert("traceparent", trace.parse().unwrap());
            request
                .metadata_mut()
                .insert("authorization", "Bearer client-secret".parse().unwrap());
            assert_eq!(
                collect(client.generate(request).await).await.unwrap(),
                reply
            );
            let (sent, sent_trace, credential) = seen.last().await;
            assert_eq!(sent.input_ids, original.input_ids);
            assert_eq!(sent.sampling_params, original.sampling_params);
            assert_eq!(sent.cache_salt, original.cache_salt);
            assert_eq!(sent.kv_hints, original.kv_hints);
            assert_eq!(sent_trace, trace);
            assert!(!credential);
        }
    }

    #[tokio::test]
    async fn generate_pd_legs_share_bootstrap_and_preserve_dp_selection() {
        let (prefills, decodes) = (Seen::new(), Seen::new());
        let engine = |seen: &Seen<proto::GenerateRequest>| {
            let record = seen.record();
            MockEngine::default().stream_with_context(
                "Generate",
                move |request: tonic::Request<proto::GenerateRequest>| {
                    record(request.into_inner());
                    stream::iter([Ok(frame(None))]).boxed()
                },
            )
        };
        let prefill = engine(&prefills).start().await;
        let decode = engine(&decodes).start().await;
        for reorg in [false, true] {
            let workers = [
                (WorkerMode::Prefill, "http://127.0.0.1:1", Some(prefill)),
                (WorkerMode::Decode, "http://127.0.0.1:2", Some(decode)),
            ];
            let mut ctx =
                router_ctx_with(&workers, reorg, Proxy::new(Duration::from_secs(5)).unwrap());
            Arc::get_mut(&mut ctx).unwrap().config.model.dp_aware = true;
            let mut client = serve_grpc(ctx).await;
            assert_eq!(
                collect(client.generate(generate(true)).await)
                    .await
                    .unwrap(),
                vec![frame(None)]
            );
            let p = serde_json::to_value(prefills.last().await).unwrap();
            let d = serde_json::to_value(decodes.last().await).unwrap();
            assert_eq!(p["bootstrap_host"], "127.0.0.1");
            assert_eq!(p["bootstrap_port"], 8998);
            assert!(p["bootstrap_room"].is_u64());
            assert_eq!(p["bootstrap_room"], d["bootstrap_room"]);
            assert!(p["rid"].is_string());
            assert_eq!(p["rid"], d["rid"]);
            assert_eq!(d["disagg_prefill_dp_rank"], p["routed_dp_rank"]);
            assert!(p["routed_dp_rank"].as_u64().unwrap() < 2);
            assert!(d["routed_dp_rank"].as_u64().unwrap() < 2);
        }
    }

    #[tokio::test]
    async fn generate_retries_failures_before_a_response() {
        let failed = MockEngine::default()
            .stream_with_context("Generate", |_: tonic::Request<proto::GenerateRequest>| {
                stream::iter([Err::<proto::GenerateStreamItem, _>(Status::unavailable(
                    "busy",
                ))])
                .boxed()
            })
            .start()
            .await;
        let healthy = MockEngine::default()
            .stream_with_context("Generate", |_: tonic::Request<proto::GenerateRequest>| {
                stream::iter([Ok(frame(None))]).boxed()
            })
            .start()
            .await;
        for attempts in [1, 2] {
            for streaming in [false, true] {
                let mut ctx = router_ctx(&[
                    (WorkerMode::Plain, "http://127.0.0.1:1", Some(failed)),
                    (WorkerMode::Plain, "http://127.0.0.1:2", Some(healthy)),
                ]);
                Arc::get_mut(&mut ctx).unwrap().config.proxy.max_attempts =
                    std::num::NonZeroU32::new(attempts).unwrap();
                let mut client = serve_grpc(ctx).await;
                let mut successes = 0;
                for _ in 0..2 {
                    successes += usize::from(
                        collect(client.generate(generate(streaming)).await)
                            .await
                            .is_ok(),
                    );
                }
                assert_eq!(successes, attempts as usize);
            }
        }
    }

    #[tokio::test]
    async fn generate_disconnect_aborts_through_api_v1() {
        let aborts = Seen::new();
        let record = aborts.record();
        let generates = Seen::new();
        let record_generate = generates.record();
        let (dropped_tx, mut dropped) = tokio::sync::mpsc::unbounded_channel();
        let port = MockEngine::default()
            .stream_with_context(
                "Generate",
                move |request: tonic::Request<proto::GenerateRequest>| {
                    record_generate(request.into_inner());
                    struct Guard(tokio::sync::mpsc::UnboundedSender<()>);
                    impl Drop for Guard {
                        fn drop(&mut self) {
                            let _ = self.0.send(());
                        }
                    }
                    let guard = Guard(dropped_tx.clone());
                    stream::iter([Ok(frame(None))])
                        .chain(stream::pending())
                        .map(move |item| {
                            let _ = &guard;
                            item
                        })
                        .boxed()
                },
            )
            .unary("Abort", move |request: proto::AbortRequest| {
                record(request);
                proto::AbortResponse { success: true }
            })
            .start()
            .await;
        let mut client = serve_grpc(router_ctx(&[(
            WorkerMode::Plain,
            "http://127.0.0.1:1",
            Some(port),
        )]))
        .await;
        let mut output = client.generate(generate(true)).await.unwrap().into_inner();
        assert!(output.message().await.unwrap().is_some());
        drop(output);
        tokio::time::timeout(Duration::from_secs(5), dropped.recv())
            .await
            .unwrap()
            .unwrap();
        let abort = aborts.last().await;
        let sent = serde_json::to_value(generates.last().await).unwrap();
        assert_eq!(sent["rid"], abort.rid);
        assert!(!abort.abort_all);
    }

    #[tokio::test]
    async fn generate_text_reuses_tokenization_and_sampling_overrides() {
        let seen = Seen::new();
        let record = seen.record();
        let port = MockEngine::default()
            .stream_with_context(
                "Generate",
                move |request: tonic::Request<proto::GenerateRequest>| {
                    record(request.into_inner());
                    stream::iter([Ok(frame(None))]).boxed()
                },
            )
            .start()
            .await;
        let mut ctx = router_ctx(&[(WorkerMode::Plain, "http://127.0.0.1:1", Some(port))]);
        let expected = ctx.tokenizers.encode_prompt("tiny", "hello world").unwrap();
        Arc::get_mut(&mut ctx)
            .unwrap()
            .config
            .model
            .sampling_overrides = sgl_router::config::SamplingOverrides {
            params: [(
                sgl_router::config::SamplingField::TopK,
                sgl_router::config::ParamSpec::Exact(20.into()),
            )]
            .into(),
            conflict: sgl_router::config::ConflictPolicy::Allow,
        };
        let mut client = serve_grpc(ctx).await;
        let request = proto::GenerateRequest {
            text: Some(proto::StringOrList {
                value: Some(proto::string_or_list::Value::One("hello world".into())),
            }),
            sampling_params: Some(proto::SamplingParamsOrList {
                value: Some(proto::sampling_params_or_list::Value::One(
                    proto::SamplingParams {
                        temperature: Some(0.5),
                        ..Default::default()
                    },
                )),
            }),
            ..Default::default()
        };
        collect(client.generate(request).await).await.unwrap();
        let sent = serde_json::to_value(seen.last().await).unwrap();
        assert_eq!(sent["input_ids"], json!(expected));
        assert!(sent.get("text").is_none());
        assert_eq!(sent["sampling_params"]["top_k"], 20);
        assert_eq!(sent["sampling_params"]["temperature"], 0.5);
    }

    #[tokio::test]
    async fn generate_timeouts_keep_router_error_codes() {
        let port = MockEngine::default()
            .stream_with_context("Generate", |_: tonic::Request<proto::GenerateRequest>| {
                stream::iter([Ok(frame(None))])
                    .chain(stream::pending())
                    .boxed()
            })
            .start()
            .await;
        let proxy = Proxy::new(Duration::from_millis(100))
            .unwrap()
            .with_stream_idle_timeout(Duration::from_millis(100));
        let mut client = serve_grpc(router_ctx_with(
            &[(WorkerMode::Plain, "http://127.0.0.1:1", Some(port))],
            false,
            proxy,
        ))
        .await;
        for streaming in [true, false] {
            let error = collect(client.generate(generate(streaming)).await)
                .await
                .unwrap_err();
            assert_eq!(error.code(), Code::DeadlineExceeded);
            assert_eq!(
                error.metadata().get("x-router-error-code").unwrap(),
                "upstream_timeout"
            );
        }
    }

    #[tokio::test]
    async fn native_admin_methods_are_explicitly_unimplemented() {
        let mut client = serve_grpc(router_ctx(&[])).await;
        let error = client
            .list_models(proto::ListModelsRequest {})
            .await
            .unwrap_err();
        assert_eq!(error.code(), Code::Unimplemented);
    }
}

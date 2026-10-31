// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Native gRPC: each RPC is routed like its HTTP route and sent to the
//! engine's own RPC.

use crate::common::cache_aware_fixture;
use axum::http::{Request as HttpRequest, Response as HttpResponse};
use futures::future::{ready, BoxFuture, Ready};
use futures::stream::{self, BoxStream, StreamExt};
use serde_json::json;
use sgl_router::config::PolicyKind;
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::policies_reorg::factory::build_resolver;
use sgl_router::proxy::Proxy;
use sgl_router::server::app_context::{AppContext, ChatRouting};
use sgl_router::state::kv_events::KvEventIndex;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{DecodeGate, EngineProfile, WireProtocol, WorkerRegistry};
use sglang_api_types::api::v1 as proto;
use sglang_api_types::api::v1::sglang_service_client::SglangServiceClient;
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

impl MockEngine {
    fn stream<Req, Rep>(
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
    async fn start(self) -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tonic::transport::Server::builder().add_service(self);
        tokio::spawn(server.serve_with_incoming(TcpListenerStream::new(listener)));
        port
    }
}

impl NamedService for MockEngine {
    const NAME: &'static str = "sglang.api.v1.SglangService";
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
        let Some(handler) = self.0.get(method).cloned() else {
            return Box::pin(async {
                Ok(Status::unimplemented("mock RPC is not served").into_http())
            });
        };
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

fn frame(text: &str, finished: bool) -> proto::GenerateStreamItem {
    proto::GenerateStreamItem {
        item: Some(proto::generate_stream_item::Item::Frame(
            proto::GenerateResponse {
                text: text.into(),
                meta_info: Some(proto::GenerateMetaInfo {
                    finish_reason: finished.then(|| proto::FinishReason::stop(None)),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )),
    }
}

fn error(code: u32) -> proto::GenerateStreamItem {
    proto::GenerateStreamItem {
        item: Some(proto::generate_stream_item::Item::Error(
            proto::GenerateStreamError {
                error: Some(proto::ErrorBody {
                    message: "engine failed".into(),
                    code,
                }),
                index: None,
            },
        )),
    }
}

fn generate(stream: bool) -> proto::GenerateRequest {
    proto::GenerateRequest {
        input_ids: Some(proto::TokenIdsOrList {
            value: Some(proto::token_ids_or_list::Value::One(proto::TokenIds {
                ids: vec![1, 2, 3],
            })),
        }),
        stream: Some(stream),
        ..Default::default()
    }
}

fn generate_rpc(
    seen: &Seen<proto::GenerateRequest>,
    items: Vec<proto::GenerateStreamItem>,
) -> impl Fn(tonic::Request<proto::GenerateRequest>) -> Replies<proto::GenerateStreamItem>
       + Send
       + Sync
       + 'static {
    let record = seen.record();
    move |request| {
        record(request.into_inner());
        stream::iter(items.clone().into_iter().map(Ok)).boxed()
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
            dp_ranks: 2,
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

/// One chunk, then nothing until the call is dropped; `dropped` hears about it.
fn stalled_after_one_chunk(
    dropped: tokio::sync::mpsc::UnboundedSender<()>,
) -> Replies<proto::GenerateStreamItem> {
    struct Dropped(tokio::sync::mpsc::UnboundedSender<()>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }
    let guard = Dropped(dropped);
    stream::iter([Ok(frame("hi", false))])
        .chain(stream::pending())
        .map(move |item| {
            let _ = &guard;
            item
        })
        .boxed()
}

#[tokio::test]
async fn generate_preserves_fields_metadata_and_typed_replies() {
    let seen = Seen::new();
    let record = seen.record();
    let replies = vec![frame("Paris", true)];
    let replay = replies.clone();
    let port = MockEngine::default()
        .stream(
            "Generate",
            move |request: tonic::Request<proto::GenerateRequest>| {
                let (metadata, _, body) = request.into_parts();
                record((metadata, body));
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
        let mut body = generate(streaming);
        body.sampling_params = Some(proto::SamplingParamsOrList {
            value: Some(proto::sampling_params_or_list::Value::One(
                proto::SamplingParams {
                    temperature: Some(0.70000000001),
                    max_new_tokens: Some(10),
                    ..Default::default()
                },
            )),
        });
        let mut request = tonic::Request::new(body.clone());
        let traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        request
            .metadata_mut()
            .insert("traceparent", traceparent.parse().unwrap());
        request
            .metadata_mut()
            .insert("authorization", "Bearer secret".parse().unwrap());
        assert_eq!(
            collect(client.generate(request).await).await.unwrap(),
            replies
        );
        let (metadata, sent) = seen.last().await;
        assert_eq!(metadata.get("traceparent").unwrap(), traceparent);
        assert!(metadata.get("authorization").is_none());
        assert!(
            matches!(sent.rid.as_ref().and_then(|r| r.value.as_ref()), Some(proto::string_or_list::Value::One(rid)) if !rid.is_empty())
        );
        body.rid = sent.rid.clone();
        assert_eq!(
            sent, body,
            "only the router's rid is added; absent fields stay absent"
        );
    }
    assert_eq!(
        client
            .health_check(proto::HealthCheckRequest {})
            .await
            .unwrap_err()
            .code(),
        Code::Unimplemented
    );
}

#[tokio::test]
async fn retries_failures_before_a_response_on_another_worker() {
    for status_code in [None, Some(429), Some(503)] {
        let failed = MockEngine::default()
            .stream(
                "Generate",
                move |_: tonic::Request<proto::GenerateRequest>| {
                    let reply = match status_code {
                        Some(code) => Ok(error(code)),
                        None => Err(Status::unavailable("engine unavailable")),
                    };
                    stream::iter([reply]).boxed()
                },
            )
            .unary("Abort", |_: proto::AbortRequest| {
                proto::AbortResponse::default()
            })
            .start()
            .await;
        let reply = vec![frame("ok", true)];
        let healthy = MockEngine::default()
            .stream("Generate", generate_rpc(&Seen::new(), reply.clone()))
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
                let mut client = serve_grpc(ctx).await;
                let mut successes = 0;
                for _ in 0..2 {
                    let result = collect(client.generate(generate(streaming)).await).await;
                    successes += usize::from(result.is_ok_and(|chunks| chunks == reply));
                }
                assert_eq!(successes, max_attempts as usize);
            }
        }
    }
}

#[tokio::test]
async fn pd_legs_share_bootstrap_and_dp_placement() {
    let (prefills, decodes) = (Seen::new(), Seen::new());
    let reply = vec![frame("ok", true)];
    let prefill = MockEngine::default()
        .stream("Generate", generate_rpc(&prefills, reply.clone()))
        .start()
        .await;
    let decode = MockEngine::default()
        .stream("Generate", generate_rpc(&decodes, reply.clone()))
        .start()
        .await;
    for reorg in [false, true] {
        let workers = [
            (WorkerMode::Prefill, "http://127.0.0.1:1", Some(prefill)),
            (WorkerMode::Decode, "http://127.0.0.1:2", Some(decode)),
        ];
        let mut ctx = router_ctx_with(&workers, reorg, Proxy::new(Duration::from_secs(5)).unwrap());
        Arc::get_mut(&mut ctx).unwrap().config.model.dp_aware = true;
        let mut client = serve_grpc(ctx).await;
        let mut request = generate(true);
        request.routed_dp_rank = Some(99);
        assert_eq!(
            collect(client.generate(request).await).await.unwrap(),
            reply
        );
        let (p, d) = (prefills.last().await, decodes.last().await);
        assert!(p.rid.is_some() && p.rid == d.rid);
        assert_eq!(p.bootstrap_host, d.bootstrap_host);
        assert_eq!(p.bootstrap_port, d.bootstrap_port);
        assert_eq!(p.bootstrap_room, d.bootstrap_room);
        let p = serde_json::to_value(p).unwrap();
        let d = serde_json::to_value(d).unwrap();
        assert_eq!(p["bootstrap_host"], "127.0.0.1");
        assert_eq!(p["bootstrap_port"], 8998);
        assert!(p["bootstrap_room"]
            .as_u64()
            .is_some_and(|room| room <= i64::MAX as u64));
        let rank = p["routed_dp_rank"].as_u64().unwrap();
        assert!(rank < 2);
        assert_eq!(p["bootstrap_room"].as_u64().unwrap() % 2, rank);
        assert!(d["routed_dp_rank"].as_u64().is_some_and(|rank| rank < 2));
        assert_eq!(d["disagg_prefill_dp_rank"], rank);
    }
}

#[tokio::test]
async fn decode_dispatch_limit_gates_decode_as_over_http() {
    let prefill = MockEngine::default()
        .stream("Generate", |_: tonic::Request<proto::GenerateRequest>| {
            stream::once(async {
                tokio::time::sleep(Duration::from_millis(800)).await;
                Ok(frame("ok", true))
            })
            .boxed()
        })
        .start()
        .await;
    let decode = MockEngine::default()
        .stream(
            "Generate",
            generate_rpc(&Seen::new(), vec![frame("ok", true)]),
        )
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
            collect(client.generate(generate(streaming)).await)
                .await
                .unwrap()
        };
        let start = Instant::now();
        futures::future::join(call(client.clone()), call(client)).await;
        assert_eq!(
            start.elapsed() >= Duration::from_millis(700),
            waits,
            "streaming={streaming}"
        );
    }
}

#[tokio::test]
async fn text_generation_keeps_forwarding_guards() {
    let seen = Seen::new();
    let reply = vec![frame("Paris", true)];
    let port = MockEngine::default()
        .stream("Generate", generate_rpc(&seen, reply.clone()))
        .start()
        .await;
    for (disable_ids, text_logprobs, media) in [
        (false, false, false),
        (true, false, false),
        (false, true, false),
        (false, false, true),
    ] {
        let mut ctx = router_ctx(&[(WorkerMode::Plain, "http://127.0.0.1:1", Some(port))]);
        Arc::get_mut(&mut ctx)
            .unwrap()
            .config
            .model
            .disable_input_ids_forwarding = disable_ids;
        let expected_ids = ctx.tokenizers.encode_prompt("tiny", "hello world").unwrap();
        let mut client = serve_grpc(ctx).await;
        let mut body = json!({"text": "hello world", "return_text_in_logprobs": text_logprobs});
        if media {
            body["image_data"] = json!("https://example.com/image.png");
        }
        let request: proto::GenerateRequest = serde_json::from_value(body).unwrap();
        assert_eq!(
            collect(client.generate(request.clone()).await)
                .await
                .unwrap(),
            reply
        );
        let sent = seen.last().await;
        if disable_ids || text_logprobs || media {
            assert_eq!(sent.text, request.text);
            assert_eq!(sent.image_data, request.image_data);
            assert!(sent.input_ids.is_none());
        } else {
            assert!(sent.text.is_none());
            assert_eq!(
                sent.input_ids,
                Some(proto::TokenIdsOrList {
                    value: Some(proto::token_ids_or_list::Value::One(proto::TokenIds {
                        ids: expected_ids.into_iter().map(i64::from).collect(),
                    })),
                })
            );
        }
    }
}

#[tokio::test]
async fn unsupported_batches_and_unrepresentable_token_ids_are_rejected() {
    let mut client = serve_grpc(router_ctx(&[(
        WorkerMode::Plain,
        "http://127.0.0.1:1",
        None,
    )]))
    .await;
    for body in [
        json!({"text": ["hi"]}),
        json!({"input_ids": [[1]]}),
        json!({"input_ids": [1], "rid": ["id"]}),
        json!({"input_ids": [1], "sampling_params": [{}]}),
        json!({"input_ids": [-1]}),
        json!({"input_ids": [4294967296i64]}),
        json!({"input_ids": [1], "text": "hi"}),
    ] {
        let request: proto::GenerateRequest = serde_json::from_value(body.clone()).unwrap();
        let status = client.generate(request).await.unwrap_err();
        assert_eq!(status.code(), Code::InvalidArgument, "{body}");
        assert!(status.metadata().get("x-router-error-code").is_some());
    }
}

#[tokio::test]
async fn client_cancel_aborts_the_engine_request() {
    let (dropped_tx, mut dropped) = tokio::sync::mpsc::unbounded_channel();
    let (generates, aborts) = (Seen::new(), Seen::new());
    let (record_generate, record_abort) = (generates.record(), aborts.record());
    let port = MockEngine::default()
        .stream(
            "Generate",
            move |request: tonic::Request<proto::GenerateRequest>| {
                record_generate(request.into_inner());
                stalled_after_one_chunk(dropped_tx.clone())
            },
        )
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
    let mut stream = client.generate(generate(true)).await.unwrap().into_inner();
    assert!(stream.message().await.unwrap().is_some());
    drop(stream);
    tokio::time::timeout(Duration::from_secs(5), dropped.recv())
        .await
        .expect("the engine call outlived the client");
    let rid = generates.last().await.rid.unwrap().value.unwrap();
    assert_eq!(
        rid,
        proto::string_or_list::Value::One(aborts.last().await.rid)
    );
}

#[tokio::test]
async fn stream_outcomes_reach_the_breaker_by_status() {
    for late in [false, true] {
        let calls = Arc::new(Mutex::new(0));
        let counted = Arc::clone(&calls);
        let port = MockEngine::default()
            .stream(
                "Generate",
                move |_: tonic::Request<proto::GenerateRequest>| {
                    let mut calls = counted.lock().unwrap();
                    *calls += 1;
                    match *calls {
                        1..=3 => stream::iter([
                            Ok(frame("hi", false)),
                            Err(Status::resource_exhausted("queue full")),
                        ])
                        .boxed(),
                        _ => stream::iter(
                            late.then(|| Ok(frame("hi", false)))
                                .into_iter()
                                .chain([Ok(error(500))]),
                        )
                        .boxed(),
                    }
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
        for _ in 0..3 {
            assert_eq!(
                collect(client.generate(generate(true)).await)
                    .await
                    .unwrap_err()
                    .code(),
                Code::ResourceExhausted
            );
        }
        for _ in 0..3 {
            let items = collect(client.generate(generate(true)).await)
                .await
                .unwrap();
            assert_eq!(items.last(), Some(&error(500)));
        }
        assert_eq!(
            client.generate(generate(true)).await.unwrap_err().code(),
            Code::Unavailable
        );
        assert_eq!(*calls.lock().unwrap(), 6);
    }
}

#[tokio::test]
async fn router_stream_failures_keep_their_error_code() {
    let (dropped_tx, _dropped) = tokio::sync::mpsc::unbounded_channel();
    let port = MockEngine::default()
        .stream(
            "Generate",
            move |_: tonic::Request<proto::GenerateRequest>| {
                stalled_after_one_chunk(dropped_tx.clone())
            },
        )
        .unary("Abort", |_: proto::AbortRequest| {
            proto::AbortResponse::default()
        })
        .start()
        .await;
    let proxy = Proxy::new(Duration::from_millis(100))
        .unwrap()
        .with_stream_idle_timeout(Duration::from_millis(100));
    let workers = [(WorkerMode::Plain, "http://127.0.0.1:1", Some(port))];
    let mut client = serve_grpc(router_ctx_with(&workers, false, proxy)).await;
    for streaming in [true, false] {
        let result = collect(client.generate(generate(streaming)).await).await;
        let status = result.unwrap_err();
        assert_eq!(status.code(), Code::DeadlineExceeded, "stream={streaming}");
        assert_eq!(
            status.metadata().get("x-router-error-code").unwrap(),
            "upstream_timeout"
        );
    }
}

#[tokio::test]
async fn errors_arrive_as_grpc_status() {
    let port = MockEngine::default()
        .stream("Generate", |_: tonic::Request<proto::GenerateRequest>| {
            stream::iter([
                Ok(frame("hi", false)),
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
    let mut codes = Vec::new();
    for _ in 0..2 {
        codes.push(
            collect(client.generate(generate(true)).await)
                .await
                .unwrap_err()
                .code(),
        );
    }
    codes.sort_by_key(|code| *code as i32);
    assert_eq!(codes, [Code::Internal, Code::Unavailable]);
}

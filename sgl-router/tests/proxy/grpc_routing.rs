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
use sgl_router::workers::{EngineProfile, WireProtocol, WorkerRegistry};
use sglang_api_types::api::v1 as proto;
use sglang_api_types::api::v1::sglang_service_client::SglangServiceClient;
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
    async fn start(mut self) -> u16 {
        if !self.0.contains_key("Abort") {
            self = self.unary("Abort", |_: proto::AbortRequest| {
                proto::AbortResponse::default()
            });
        }
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
        let handler = Arc::clone(&self.0[method]);
        Box::pin(async move { Ok(handler(request).await) })
    }
}

/// Requests an engine saw, including a PD prefill dispatched in the background.
struct Seen<T>(Arc<Mutex<Vec<T>>>, Arc<tokio::sync::Notify>);

impl<T: Clone> Seen<T> {
    fn new() -> Self {
        Self(Default::default(), Default::default())
    }

    fn record(&self) -> impl Fn(T) + Send + Sync + 'static
    where
        T: Send + 'static,
    {
        let seen = Arc::clone(&self.0);
        let changed = Arc::clone(&self.1);
        move |request| {
            seen.lock().unwrap().push(request);
            changed.notify_one();
        }
    }

    async fn last(&self) -> T {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let changed = self.1.notified();
                if let Some(request) = self.0.lock().unwrap().last().cloned() {
                    return request;
                }
                changed.await;
            }
        })
        .await
        .expect("the engine saw no request")
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
        .stream(
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
        MockEngine::default().stream(
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
        let mut ctx = router_ctx_with(&workers, reorg, Proxy::new(Duration::from_secs(5)).unwrap());
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
async fn generate_retries_only_before_output() {
    let failed = MockEngine::default()
        .stream("Generate", |_: tonic::Request<proto::GenerateRequest>| {
            stream::iter([Err::<proto::GenerateStreamItem, _>(Status::unavailable(
                "busy",
            ))])
            .boxed()
        })
        .start()
        .await;
    let healthy = MockEngine::default()
        .stream("Generate", |_: tonic::Request<proto::GenerateRequest>| {
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
        .stream(
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
        .stream(
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
        .sampling_overrides = sgl_router::config::parse_sampling_overrides(
        r#"{"top_k": 20}"#,
        sgl_router::config::ConflictPolicy::Allow,
    )
    .unwrap();
    let mut client = serve_grpc(ctx).await;
    let request = serde_json::from_value::<proto::GenerateRequest>(json!({
        "text": "hello world", "sampling_params": {"temperature": 0.5}
    }))
    .unwrap();
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
        .stream("Generate", |_: tonic::Request<proto::GenerateRequest>| {
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

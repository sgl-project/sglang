//! The gRPC transport (tonic), serving `sglang.api.v1.SglangService` over the
//! same `api_server::core` the HTTP transport mounts — one core, two wire formats.
//! Only this module knows tonic; endpoint logic stays in `api_server::core`.
//!
//! Runs on the api runtime next to the HTTP server (its own listener,
//! `--rust-grpc-port`), dark unless the port is configured. Dropping the
//! serve future (runtime shutdown) drops in-flight streams, so `AbortGuard`
//! fires exactly as it does on an HTTP disconnect.

use std::pin::Pin;
use std::sync::Arc;

use sglang_api_types::api::v1 as genapi;
use sglang_api_types::api::v1::sglang_service_server::{SglangService, SglangServiceServer};
use tonic::{Request, Response, Status};

use crate::api_server::core::error::ApiError;
use crate::api_server::core::generate::{
    GeneratePlan, PbFrameShaper, generation_event_stream_with,
};
use crate::api_server::core::health::{HealthStatus, health_probe};
use crate::api_server::core::state::CoreState;
use crate::api_server::core::{control, generate};
use crate::api_server::layers::access_log;
use crate::api_server::layers::auth;
use crate::message::convert;
use crate::utils::environ;

/// The one [`ApiError`] → gRPC status mapping. The exact HTTP code (including
/// the non-IANA 499 and scheduler abort statuses) rides the
/// `x-sglang-http-code` metadata so no client loses it to the coarser
/// canonical codes.
fn status(e: &ApiError) -> Status {
    let code = match e.http_code {
        400 => tonic::Code::InvalidArgument,
        401 => tonic::Code::Unauthenticated,
        404 => tonic::Code::NotFound,
        429 => tonic::Code::ResourceExhausted,
        499 => tonic::Code::Cancelled,
        503 => tonic::Code::Unavailable,
        _ => tonic::Code::Internal,
    };
    let mut status = Status::new(code, e.message.clone());
    if let Ok(value) = e.http_code.to_string().parse() {
        status.metadata_mut().insert("x-sglang-http-code", value);
    }
    status
}

struct GrpcApi {
    state: Arc<CoreState>,
    /// Frozen at server build, like the HTTP router's health knob.
    health_timeout: std::time::Duration,
}

#[tonic::async_trait]
impl SglangService for GrpcApi {
    type GenerateStream =
        Pin<Box<dyn futures::Stream<Item = Result<genapi::GenerateStreamItem, Status>> + Send>>;

    async fn generate(
        &self,
        request: Request<genapi::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        use futures::StreamExt;
        let mut request = request.into_inner();
        // Protobuf has no JSON-null: unset defaulted fields (incl.
        // max_new_tokens) mean their schema defaults on this wire.
        request.apply_absent_defaults();
        let body = convert::generate_body(request);
        // Pre-submit failures end the RPC with a status; per-item failures
        // after this point ride the stream as typed error items.
        let plan = generate::generate_start(&self.state, body)
            .await
            .map_err(|e| status(&e))?;
        let GeneratePlan {
            receivers,
            guard,
            is_batch,
            incremental,
        } = plan;
        let stream =
            generation_event_stream_with(receivers, guard, incremental, is_batch, PbFrameShaper)
                .map(Ok);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn health_check(
        &self,
        _request: Request<genapi::HealthCheckRequest>,
    ) -> Result<Response<genapi::HealthCheckResponse>, Status> {
        match health_probe(&self.state, self.health_timeout).await {
            Ok(HealthStatus::Alive) => {
                Ok(Response::new(genapi::HealthCheckResponse { healthy: true }))
            }
            // The HTTP handler answers a stalled pipeline with 503; the RPC's
            // analogue is UNAVAILABLE.
            Ok(HealthStatus::Stalled) => Err(Status::unavailable("pipeline stalled")),
            Err(e) => Err(status(&e)),
        }
    }

    async fn get_model_info(
        &self,
        _request: Request<genapi::GetModelInfoRequest>,
    ) -> Result<Response<genapi::GetModelInfoResponse>, Status> {
        // The same blob `/get_model_info` shapes, through the generated type.
        let value = control::model_info_value(&self.state.server_args);
        let info: genapi::GetModelInfoResponse = serde_json::from_value(value)
            .map_err(|e| Status::internal(format!("bad model_info shape: {e}")))?;
        Ok(Response::new(info))
    }

    async fn get_server_info(
        &self,
        _request: Request<genapi::GetServerInfoRequest>,
    ) -> Result<Response<genapi::GetServerInfoResponse>, Status> {
        let bytes = control::server_info_json(&self.state)
            .await
            .map_err(|e| status(&e))?;
        let info: genapi::GetServerInfoResponse = serde_json::from_slice(&bytes)
            .map_err(|e| Status::internal(format!("bad server_info shape: {e}")))?;
        Ok(Response::new(info))
    }
}

/// Serve the gRPC transport on the pre-bound listener until shutdown. Mirrors
/// `api_server::http::app::serve`: selecting away drops in-flight RPC futures and
/// streams, and every armed `AbortGuard` fires.
pub async fn serve(
    listener: std::net::TcpListener,
    state: Arc<CoreState>,
    shutdown: flume::Receiver<()>,
) {
    let health_timeout =
        std::time::Duration::from_secs(environ::env_u64("SGLANG_HEALTH_CHECK_TIMEOUT", 20));
    let api = GrpcApi {
        state: state.clone(),
        health_timeout,
    };
    let listener = match tokio::net::TcpListener::from_std(listener) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(error = %e, "failed to adopt pre-bound gRPC listener");
            return;
        }
    };
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
    // The same shared layers as the HTTP stack: first .layer added = outermost
    // (access log wraps auth, so a rejected RPC is still logged).
    let access_log = state
        .server_args
        .http_access_log_enabled()
        .then_some(access_log::AccessLogLayer);
    let auth = state.api_key.as_deref().map(auth::ApiKeyAuthLayer::new);
    let serve = tonic::transport::Server::builder()
        .layer(tower::util::option_layer(access_log))
        .layer(tower::util::option_layer(auth))
        .add_service(SglangServiceServer::new(api))
        .serve_with_incoming(incoming);
    tokio::select! {
        r = serve => {
            if let Err(e) = r {
                tracing::error!(error = %e, "grpc serve exited");
            }
        }
        _ = shutdown.recv_async() => {
            tracing::info!("shutdown: stopping gRPC accepts, aborting in-flight RPCs");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::core::openai::test_utils::senders;
    use crate::message::response::{ChunkEvent, ResponseItem, ResponseSink};
    use crate::tokenizer_manager::wiring::{Senders, TmEvent};
    use sglang_api_types::api::v1::sglang_service_client::SglangServiceClient;

    fn test_state_with_key(senders: Senders, api_key: Option<&str>) -> Arc<CoreState> {
        Arc::new(CoreState {
            senders,
            response_buf: 8,
            api_key: api_key.map(str::to_owned),
            server_args: Arc::new(crate::message::config::ServerArgs {
                served_model_name: "model".into(),
                model_path: "/m".into(),
                ..Default::default()
            }),
            chat_formatter: None,
            response_activity: Default::default(),
        })
    }

    fn test_state(senders: Senders) -> Arc<CoreState> {
        Arc::new(CoreState {
            senders,
            response_buf: 8,
            api_key: None,
            server_args: Arc::new(crate::message::config::ServerArgs {
                served_model_name: "model".into(),
                model_path: "/m".into(),
                ..Default::default()
            }),
            chat_formatter: None,
            response_activity: Default::default(),
        })
    }

    /// A live Senders set whose TM inbox is answered by `respond` for every
    /// intake (the scripted scheduler).
    fn scripted_senders(
        respond: impl Fn(crate::message::request::Request) + Send + 'static,
    ) -> Senders {
        let (tm_tx, tm_rx) = flume::unbounded::<TmEvent>();
        let (abort_tx, _abort_rx_keepalive) = flume::unbounded();
        let (tok_tx, _tok_rx_keepalive) = flume::unbounded();
        std::thread::spawn(move || {
            while let Ok(TmEvent::Intake(request)) = tm_rx.recv() {
                respond(request);
            }
        });
        // The keepalive receivers leak with the thread; fine for a test.
        std::mem::forget(_abort_rx_keepalive);
        std::mem::forget(_tok_rx_keepalive);
        Senders {
            tok_manager_tx: tm_tx,
            abort_tx,
            tokenizer_tx: tok_tx,
            detokenizer_tx: vec![],
        }
    }

    async fn serve_on_ephemeral(state: Arc<CoreState>) -> (String, flume::Sender<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = flume::unbounded::<()>();
        tokio::spawn(serve(listener, state, shutdown_rx));
        (format!("http://{addr}"), shutdown_tx)
    }

    /// End to end over a real socket: Generate streams typed frames built by
    /// the shared state machine, terminated by stream close (no sentinel).
    #[tokio::test(flavor = "multi_thread")]
    async fn generate_streams_typed_frames() {
        let senders = scripted_senders(|request| {
            let crate::message::request::Request { rid, sink, .. } = request;
            let ResponseSink::Local(tx) = sink;
            let frame = |text: &str, done: bool| {
                let ev = ChunkEvent {
                    rid: rid.clone(),
                    text: text.into(),
                    token_ids: vec![1],
                    prompt_tokens: 3,
                    completion_tokens: 1,
                    finish_reason: done.then(|| {
                        serde_json::from_value(serde_json::json!({"type": "stop"})).unwrap()
                    }),
                    ..Default::default()
                };
                if done {
                    ResponseItem::Done(ev)
                } else {
                    ResponseItem::Frame(ev)
                }
            };
            tx.try_send(frame("Hel", false)).unwrap();
            tx.try_send(frame("lo", true)).unwrap();
        });
        let (url, _shutdown) = serve_on_ephemeral(test_state(senders)).await;
        let mut client = SglangServiceClient::connect(url).await.unwrap();
        let request = genapi::GenerateRequest {
            text: Some(genapi::StringOrList {
                value: Some(genapi::string_or_list::Value::One("hi".into())),
            }),
            stream: Some(true),
            ..Default::default()
        };
        let mut stream = client.generate(request).await.unwrap().into_inner();
        let mut texts = Vec::new();
        let mut finish = None;
        while let Some(item) = stream.message().await.unwrap() {
            match item.item.unwrap() {
                genapi::generate_stream_item::Item::Frame(f) => {
                    texts.push(f.text.clone());
                    if let Some(reason) = f.meta_info.and_then(|m| m.finish_reason) {
                        finish = reason.kind_name().map(|k| k.into_owned());
                    }
                }
                genapi::generate_stream_item::Item::Error(e) => panic!("unexpected error: {e:?}"),
            }
        }
        // Cumulative streaming (the default): the terminal frame carries the
        // full text.
        assert_eq!(texts.last().map(String::as_str), Some("Hello"));
        assert_eq!(finish.as_deref(), Some("stop"));
    }

    /// A submit failure (TM inbox closed) ends the RPC with UNAVAILABLE and
    /// the exact HTTP code in metadata.
    #[tokio::test(flavor = "multi_thread")]
    async fn submit_failure_maps_to_unavailable() {
        let (url, _shutdown) = serve_on_ephemeral(test_state(senders())).await;
        let mut client = SglangServiceClient::connect(url).await.unwrap();
        let err = client
            .generate(genapi::GenerateRequest {
                text: Some(genapi::StringOrList {
                    value: Some(genapi::string_or_list::Value::One("hi".into())),
                }),
                ..Default::default()
            })
            .await
            .expect_err("closed inbox must fail the RPC");
        assert_eq!(err.code(), tonic::Code::Unavailable);
        assert_eq!(
            err.metadata().get("x-sglang-http-code").unwrap(),
            &"503".parse::<tonic::metadata::MetadataValue<_>>().unwrap()
        );
    }

    /// GetModelInfo answers from ServerArgs through the generated type.
    #[tokio::test(flavor = "multi_thread")]
    async fn model_info_round_trips() {
        let (url, _shutdown) = serve_on_ephemeral(test_state(senders())).await;
        let mut client = SglangServiceClient::connect(url).await.unwrap();
        let info = client
            .get_model_info(genapi::GetModelInfoRequest {})
            .await
            .unwrap()
            .into_inner();
        assert_eq!(info.served_model_name, "model");
        assert_eq!(info.model_path, "/m");
        assert!(info.is_generation);
    }

    /// The shared auth layer on the gRPC stack: an unauthenticated RPC gets
    /// UNAUTHENTICATED; HealthCheck is exempt (the k8s-probe rule); the right
    /// bearer passes.
    #[tokio::test(flavor = "multi_thread")]
    async fn auth_gates_rpcs_but_not_health() {
        let (url, _shutdown) =
            serve_on_ephemeral(test_state_with_key(senders(), Some("sk-1"))).await;
        let mut client = SglangServiceClient::connect(url).await.unwrap();

        let denied = client
            .get_model_info(genapi::GetModelInfoRequest {})
            .await
            .expect_err("no bearer must be rejected");
        assert_eq!(denied.code(), tonic::Code::Unauthenticated);

        // HealthCheck reaches the handler (the closed test inbox then fails
        // the probe submit with 503 -> UNAVAILABLE, proving auth let it in).
        let probe = client
            .health_check(genapi::HealthCheckRequest {})
            .await
            .expect_err("closed inbox fails the probe AFTER auth");
        assert_eq!(probe.code(), tonic::Code::Unavailable);

        let mut authed = tonic::Request::new(genapi::GetModelInfoRequest {});
        authed.metadata_mut().insert(
            "authorization",
            "Bearer sk-1".parse().expect("valid metadata"),
        );
        let info = client.get_model_info(authed).await.unwrap().into_inner();
        assert_eq!(info.served_model_name, "model");
    }
}

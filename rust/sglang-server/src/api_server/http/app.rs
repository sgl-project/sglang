//! The hand-built route table and hyper accept loop. Every endpoint is one
//! match arm in [`HttpApi::respond`]; [`serve`] runs the layered service on
//! the pre-bound listener until shutdown. The shared handler state is built
//! once in `utils::runtime` (so the gRPC transport mounts the same instance)
//! and arrives as `Arc<CoreState>`.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use http::{Method, StatusCode};

use super::disaggregation::bootstrap as pd_bootstrap;
use super::response::{HttpResponse, method_not_allowed, status_response};
use super::{common, native_api, openai};
use crate::api_server::layers::Peer;
use crate::api_server::layers::access_log::AccessLogLayer;
use crate::api_server::layers::auth::ApiKeyAuthLayer;
use crate::utils::environ;

/// The HTTP transport's name for the shared [`CoreState`].
pub(super) use crate::api_server::core::state::CoreState as AppState;

/// The route table, as one cloneable tower `Service` over any request body.
/// Both env knobs are resolved ONCE at construction (server startup) —
/// changing them on a live process needs a restart.
#[derive(Clone)]
pub(crate) struct HttpApi {
    state: Arc<AppState>,
    bootstrap: Option<Arc<pd_bootstrap::Registry>>,
    health_timeout: Duration,
    /// `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` (default true, mirroring
    /// Python): whether `/health` runs the deep generate probe or is a plain
    /// 200 (routing the request already proves the frontend is up).
    health_probes_generation: bool,
}

impl HttpApi {
    pub(crate) fn new(
        state: Arc<AppState>,
        bootstrap: Option<Arc<pd_bootstrap::Registry>>,
    ) -> Self {
        HttpApi {
            state,
            bootstrap,
            health_timeout: Duration::from_secs(environ::env_u64(
                "SGLANG_HEALTH_CHECK_TIMEOUT",
                20,
            )),
            health_probes_generation: environ::env_bool(
                "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION",
                true,
            ),
        }
    }

    async fn respond<B>(self, req: http::Request<B>) -> HttpResponse
    where
        B: http_body::Body,
    {
        let state = self.state;
        let method = req.method().clone();
        let path = req.uri().path().to_owned();
        // GET routes serve HEAD too (hyper omits the body on the wire), as
        // axum's `get` did.
        let get = |m: &Method| *m == Method::GET || *m == Method::HEAD;
        match path.as_str() {
            "/generate" if method == Method::POST => native_api::generate(state, req).await,
            "/generate" => method_not_allowed("POST"),
            "/health" if get(&method) => {
                if self.health_probes_generation {
                    native_api::health_generate(state, self.health_timeout).await
                } else {
                    status_response(StatusCode::OK)
                }
            }
            "/health" => method_not_allowed("GET,HEAD"),
            "/health_generate" if get(&method) => {
                native_api::health_generate(state, self.health_timeout).await
            }
            "/health_generate" => method_not_allowed("GET,HEAD"),
            "/server_info" if get(&method) => common::server_info(state).await,
            "/server_info" => method_not_allowed("GET,HEAD"),
            "/get_model_info" | "/model_info" if get(&method) => common::model_info(state).await,
            "/get_model_info" | "/model_info" => method_not_allowed("GET,HEAD"),
            "/v1/models" if get(&method) => openai::models::available_models(state).await,
            "/v1/models" => method_not_allowed("GET,HEAD"),
            "/v1/completions" if method == Method::POST => {
                openai::completions::completions(state, req).await
            }
            "/v1/completions" => method_not_allowed("POST"),
            "/v1/chat/completions" if method == Method::POST => {
                openai::chat::chat_completions(state, req).await
            }
            "/v1/chat/completions" => method_not_allowed("POST"),
            _ => {
                // `/v1/models/{model}`: exactly one more segment,
                // percent-decoded (axum `Path<String>` parity).
                if let Some(model) = path
                    .strip_prefix("/v1/models/")
                    .filter(|rest| !rest.is_empty() && !rest.contains('/'))
                {
                    if !get(&method) {
                        return method_not_allowed("GET,HEAD");
                    }
                    let model = percent_encoding::percent_decode_str(model).decode_utf8_lossy();
                    return openai::models::retrieve_model(state, &model).await;
                }
                if let Some(registry) = &self.bootstrap
                    && let Some(response) = pd_bootstrap::dispatch(registry, req).await
                {
                    return response;
                }
                status_response(StatusCode::NOT_FOUND)
            }
        }
    }
}

impl<B> tower::Service<http::Request<B>> for HttpApi
where
    B: http_body::Body + Send + 'static,
    B::Data: Send,
    B::Error: Send,
{
    type Response = HttpResponse;
    type Error = Infallible;
    type Future = std::pin::Pin<Box<dyn Future<Output = Result<HttpResponse, Infallible>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Infallible>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: http::Request<B>) -> Self::Future {
        let api = self.clone();
        Box::pin(async move { Ok(api.respond(req).await) })
    }
}

pub async fn serve(
    listener: std::net::TcpListener,
    state: Arc<AppState>,
    // The runtime's shutdown signal, shared with every worker stage: it fires
    // (disconnects) when `Runtime::request_shutdown` drops the sender, at
    // which point `serve` stops accepting and its in-flight handlers are
    // aborted with the api runtime.
    shutdown: flume::Receiver<()>,
) {
    let server_args = state.server_args.clone();

    // Prefill-only KV bootstrap registry, resolved after the main table and
    // before the 404 fallback. Python gates its routes with the same auth
    // middleware, so it sits inside the layers below too.
    let bootstrap = server_args.enable_pd_bootstrap().then(|| {
        let (registry, sweeper) = pd_bootstrap::registry_and_sweeper();
        tokio::spawn(sweeper); // cancelled with the runtime on shutdown
        tracing::info!("PD KV bootstrap registry mounted on the api listener");
        registry
    });

    let auth = tower::util::option_layer(state.api_key.as_deref().map(ApiKeyAuthLayer::new));
    let access_log = tower::util::option_layer(
        server_args
            .http_access_log_enabled()
            .then_some(AccessLogLayer),
    );
    // The shared tower layers, both stacks alike: auth inside, access log
    // outermost (a 401 is still logged). No body limit, matching the Python
    // server (`read_json` buffers uncapped).
    let service = tower::ServiceBuilder::new()
        .layer(access_log)
        .layer(auth)
        .service(HttpApi::new(state, bootstrap));

    // The listener was already bound synchronously in `runtime::start` (so a
    // port conflict fails startup); adopt it into the tokio reactor here.
    let listener = match tokio::net::TcpListener::from_std(listener) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(error = %e, "failed to adopt pre-bound listener");
            return;
        }
    };

    // Per-connection tasks live in the JoinSet: dropping it on shutdown
    // aborts in-flight connections, which drops their handler futures and
    // fires the AbortGuards — the drop semantics the axum select had.
    let mut connections = tokio::task::JoinSet::new();
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (stream, peer) = match accepted {
                    Ok(pair) => pair,
                    Err(e) => {
                        tracing::debug!(error = %e, "accept failed");
                        continue;
                    }
                };
                let service = service.clone();
                connections.spawn(async move {
                    let io = hyper_util::rt::TokioIo::new(stream);
                    // Stamp the connection's peer on each request — the
                    // access-log layer reads it back as `Peer`.
                    let svc = hyper_util::service::TowerToHyperService::new(tower::service_fn(
                        move |mut req: http::Request<hyper::body::Incoming>| {
                            req.extensions_mut().insert(Peer(peer));
                            let service = service.clone();
                            async move { tower::ServiceExt::oneshot(service, req).await }
                        },
                    ));
                    let result = hyper_util::server::conn::auto::Builder::new(
                        hyper_util::rt::TokioExecutor::new(),
                    )
                    .serve_connection(io, svc)
                    .await;
                    if let Err(e) = result {
                        tracing::debug!(error = %e, "connection closed with error");
                    }
                });
            }
            // Reap finished connection tasks so the set doesn't grow.
            Some(_) = connections.join_next(), if !connections.is_empty() => {}
            _ = shutdown.recv_async() => {
                tracing::info!("shutdown: stopping accepts, aborting in-flight handlers");
                return;
            }
        }
    }
}

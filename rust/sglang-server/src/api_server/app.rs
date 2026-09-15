//! Router assembly and the shared handler state: every endpoint module
//! registers its routes here, and [`serve`] runs the assembled app on the
//! pre-bound listener until shutdown.

use std::sync::Arc;

use axum::Router;

use super::disaggregation::bootstrap as pd_bootstrap;
use super::transport::{Http2Settings, serve_listener};
use super::{common, log, native_api, openai};
use crate::message::config::ServerArgs;
use crate::message::request::{GenerateRequest, MmData};
use crate::tokenizer_manager::from_scheduler::ActivityCounter;
use crate::tokenizer_manager::wiring::Senders;

pub struct Listeners {
    pub api: std::net::TcpListener,
    pub bootstrap: Option<std::net::TcpListener>,
}

/// Model packages may add native routes and middleware at startup. Request
/// handling remains on the Rust HTTP runtime and never calls into Python.
pub trait HttpExtension: std::fmt::Debug + Send + Sync {
    fn apply(&self, router: Router) -> Router;

    /// Model-owned measurements join the worker registry at startup. Scrapes
    /// are handled by the same collector as the frontend's native metrics.
    fn register_metrics(
        &self,
        _registry: &prometheus::Registry,
        _labels: &std::collections::BTreeMap<String, String>,
    ) -> Result<(), String> {
        Ok(())
    }

    /// One actual remote-media fetch, independent of generation completion.
    fn observe_media_fetch(
        &self,
        _modality: &str,
        _succeeded: bool,
        _elapsed: std::time::Duration,
    ) {
    }

    /// Preserve model-specific chat fields before the standard OpenAI adapter
    /// projects the response. Runs on the HTTP runtime's blocking pool.
    fn render_chat(&self, _request: &serde_json::Value) -> Result<Option<ChatInput>, String> {
        Ok(None)
    }

    /// Model-owned fields emitted once, on each native request's first result.
    fn initial_response_metadata(&self) -> Option<serde_json::Map<String, serde_json::Value>> {
        None
    }

    /// A fresh reducer for each request. It observes scheduler deltas on the
    /// detokenizer shard before HTTP buffering or stream coalescing.
    fn new_response_processor(&self) -> Option<Box<dyn ResponseProcessor>> {
        None
    }

    /// Resolve model-specific request metadata before media I/O or scheduler
    /// admission. Dropping the HTTP task cancels this future with it.
    fn prepare_request<'a>(&'a self, _request: &'a mut GenerateRequest) -> RequestPreparation<'a> {
        Box::pin(async { Ok(()) })
    }
}

pub trait ResponseProcessor: std::fmt::Debug + Send {
    /// Called once after tokenization and sampling normalization, before output.
    fn prepare(&mut self, _sampling_params: &crate::SamplingParams) {}

    fn process(
        &mut self,
        output: &mut crate::message::response::OutputMetadata,
        finished: bool,
    ) -> Result<(), String>;
}

pub type RequestPreparation<'a> = futures::future::BoxFuture<'a, Result<(), String>>;

#[derive(Clone, Debug)]
pub struct ChatInput {
    pub text: String,
    pub mm: Option<Box<MmData>>,
}

pub struct AuxiliaryRoutes {
    pub startup_ready: Arc<std::sync::atomic::AtomicBool>,
    pub metrics: Router,
    pub loads: Router,
    pub extension: Option<Arc<dyn HttpExtension>>,
    pub(crate) frontend_metrics: Option<Arc<crate::metrics::FrontendMetrics>>,
}

/// Shared handler state: submission handles, immutable server configuration,
/// and the API-owned chat formatter.
///
/// axum clones the router state into **every** request, so it is mounted as
/// `Arc<AppState>` — one refcount bump per request instead of cloning each
/// `flume::Sender` and the chat formatter. Deliberately not `Clone`, so it
/// can only be shared through that `Arc`.
pub(super) struct AppState {
    pub(super) senders: Senders,
    pub(super) response_buf: usize,
    pub(super) server_args: Arc<ServerArgs>,
    pub(super) chat_formatter: Option<openai::ChatFormatter>,
    pub(super) http_extension: Option<Arc<dyn HttpExtension>>,
    /// Response heartbeat (bumped per drained ring frame).
    pub(super) response_activity: ActivityCounter,
    pub(super) startup_ready: Arc<std::sync::atomic::AtomicBool>,
    pub(super) frontend_metrics: Option<Arc<crate::metrics::FrontendMetrics>>,
}

pub async fn serve(
    listeners: Listeners,
    senders: Senders,
    response_buf: usize,
    server_args: Arc<ServerArgs>,
    response_activity: ActivityCounter,
    auxiliary: AuxiliaryRoutes,
    // The runtime's shutdown signal, shared with every worker stage: it fires
    // (disconnects) when `Runtime::request_shutdown` drops the sender, at
    // which point `serve` stops accepting and its in-flight handlers are
    // aborted with the api runtime.
    shutdown: flume::Receiver<()>,
) {
    let chat_formatter = openai::load_chat_support(&server_args);
    let state = Arc::new(AppState {
        senders,
        response_buf,
        server_args: server_args.clone(),
        chat_formatter,
        http_extension: auxiliary.extension.clone(),
        response_activity,
        startup_ready: auxiliary.startup_ready,
        frontend_metrics: auxiliary.frontend_metrics.clone(),
    });
    // Each endpoint module registers its own routes and merges here.
    let router = Router::new()
        .merge(common::routes())
        .merge(native_api::routes())
        .merge(openai::routes());

    // TODO(auth): no API-key boundary yet. Python gates every route (except
    // /health*, /metrics*, OPTIONS) via `add_api_key_middleware`; until ported,
    // a configured `api_key` does NOT protect these routes.
    //
    // No body limit, matching the Python server.
    let mut app = router
        .layer(axum::extract::DefaultBodyLimit::disable())
        .with_state(state)
        .merge(auxiliary.metrics)
        .merge(auxiliary.loads);

    // Prefill-only KV bootstrap registry. Merged AFTER `with_state` — its
    // router carries its own Arc<Registry> state, so it cannot merge into the
    // Router<Arc<AppState>> above — and before `log::apply`, so bootstrap traffic
    // shows in the access log.
    let mut bootstrap_app = None;
    if server_args.enable_pd_bootstrap() {
        let (routes, sweeper) = pd_bootstrap::router_and_sweeper();
        tokio::spawn(sweeper); // cancelled with the runtime on shutdown
        app = app.merge(routes.clone());
        bootstrap_app = Some(log::apply(
            routes.route("/health", axum::routing::get(|| async { "OK" })),
            &server_args,
        ));
        tracing::info!(
            port = server_args.disaggregation_bootstrap_port,
            "PD KV bootstrap registry ready"
        );
    }

    if let Some(extension) = auxiliary.extension {
        app = extension.apply(app);
    }

    // Apply logging and access log middleware.
    let app = super::decompression::apply(app, server_args.enable_request_decompression);
    let app = match auxiliary
        .frontend_metrics
        .filter(|_| server_args.dp_size == 1)
    {
        Some(metrics) => crate::metrics::apply_http(app, metrics),
        None => app,
    };
    let app = log::apply(app, &server_args);

    let bootstrap = async {
        if let Some(listener) = listeners.bootstrap {
            let routes = bootstrap_app
                .ok_or_else(|| std::io::Error::other("bootstrap listener has no registry"))?;
            serve_listener(listener, routes, None).await
        } else {
            std::future::pending::<std::io::Result<()>>().await
        }
    };
    tokio::select! {
        r = serve_listener(listeners.api, app, server_args.enable_http2.then_some(Http2Settings {
            max_concurrent_streams: server_args.http2_max_concurrent_streams,
            initial_connection_window_size: server_args.http2_initial_connection_window_size,
        })) => {
            if let Err(e) = r {
                tracing::error!(error = %e, "axum serve exited");
            }
        }
        r = bootstrap => {
            if let Err(e) = r {
                tracing::error!(error = %e, "PD bootstrap listener exited");
            }
        }
        _ = shutdown.recv_async() => {
            tracing::info!("shutdown: stopping accepts, aborting in-flight handlers");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::config::{RuntimeConfig, RustServerServerArgs};
    use std::sync::atomic::Ordering;

    #[test]
    fn parent_acknowledgement_controls_readiness_even_when_warmup_is_skipped() {
        for wait_for_parent_warmup in [false, true] {
            for skip_server_warmup in [false, true] {
                let runtime = crate::utils::runtime::start(RuntimeConfig {
                    rust_server_args: RustServerServerArgs {
                        http_addr: "127.0.0.1:0".parse().unwrap(),
                        http_api_worker_num: 1,
                        ..Default::default()
                    },
                    server_args: Arc::new(ServerArgs {
                        skip_tokenizer_init: true,
                        wait_for_parent_warmup,
                        skip_server_warmup,
                        ..Default::default()
                    }),
                })
                .unwrap();
                assert_eq!(
                    runtime.startup_ready.load(Ordering::Acquire),
                    !wait_for_parent_warmup,
                );
                runtime.startup_ready.store(true, Ordering::Release);
                assert!(runtime.startup_ready.load(Ordering::Acquire));
                runtime.request_shutdown();
            }
        }
    }
}

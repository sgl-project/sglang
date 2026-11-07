//! Router assembly and the shared handler state: every endpoint module
//! registers its routes here, and [`serve`] runs the assembled app on the
//! pre-bound listener until a drain ends or the runtime shuts down.

use std::sync::Arc;
use std::time::Duration;

use axum::Router;

use super::generate_hook::GenerateHook;
use super::{auth, hicache, info, loads, metrics, native_api, openai, pre_generate};
use crate::HttpLayer;
use crate::api_server::core::{DrainEnd, FLUSH_INTERVAL};
use crate::api_server::disaggregation::bootstrap as pd_bootstrap;
use crate::api_server::handlers::{CoreHandle, HiCacheManager};
use crate::api_server::log;
use crate::message::config::ServerArgs;
use crate::message::request::GenerationPolicy;
use crate::utils::environ;

/// HTTP adapter state: the shared core capability, immutable server
/// configuration needed for HTTP request preparation, and the chat formatter.
///
/// axum clones the router state into **every** request, so it is mounted as
/// `Arc<AppState>` — one refcount bump per request instead of cloning the
/// core handle and chat formatter. Deliberately not `Clone`, so it
/// can only be shared through that `Arc`.
pub(super) struct AppState {
    pub(super) core: CoreHandle,
    pub(super) server_args: Arc<ServerArgs>,
    pub(super) generation_policy: GenerationPolicy,
    pub(super) chat_formatter: Option<openai::ChatFormatter>,
    pub(super) generate_hook: Option<Arc<dyn GenerateHook>>,
    /// The processor's chat layer, tried before the Dynamo-typed chat handler.
    pub(super) openai_chat: Option<openai::OpenAiChat>,
}

/// Every listener warms its own rank, as Python's `_execute_server_warmup`
/// warms the server, and health stays 503 until the warmup settles. A failed
/// or timed-out warmup takes the server down as Python's does: the error goes
/// to `startup_failure`, which `Server.wait_startup_failure` hands to the
/// launching process. Python's PD exception holds here too: a warmup the
/// scheduler answers with an error (Python: a non-200 status) leaves the
/// server up, and the health probe decides whether it serves.
async fn startup_warmup(
    core: CoreHandle,
    server_args: Arc<ServerArgs>,
    timeout: Duration,
    startup_failure: flume::Sender<String>,
) {
    if core.is_ready() {
        return;
    }
    let error =
        match tokio::time::timeout(timeout, core.warm_up(server_args.skip_tokenizer_init)).await {
            Ok(Ok(())) => {
                core.mark_ready();
                tracing::info!("The server is fired up and ready to roll!");
                return;
            }
            Ok(Err(e)) if server_args.is_disaggregation() => {
                tracing::info!(
                    "Disaggregation warmup failed (mode={:?}), error: {e}",
                    server_args.disaggregation_mode
                );
                core.mark_ready();
                return;
            }
            Ok(Err(e)) => e.to_string(),
            Err(_) => format!("timed out after {timeout:?}"),
        };
    tracing::error!("Initialization failed. warmup error: {error}");
    let _ = startup_failure.try_send(error);
}

/// Python's warmup request timeout: `SGLANG_WARMUP_TIMEOUT` seconds when
/// positive, else 1800 s in PD mode and 600 s otherwise.
fn warmup_timeout(configured_secs: f64, is_disaggregation: bool) -> Duration {
    match configured_secs {
        secs if secs > 0.0 && secs.is_finite() => Duration::from_secs_f64(secs),
        _ if is_disaggregation => Duration::from_secs(1800),
        _ => Duration::from_secs(600),
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn serve(
    listener: std::net::TcpListener,
    core: CoreHandle,
    server_args: Arc<ServerArgs>,
    // Whether this listener serves the node's `/metrics`.
    metrics_listener: bool,
    http_layer: Option<HttpLayer>,
    generate_hook: Option<Arc<dyn GenerateHook>>,
    // The runtime's shutdown signal, shared with every worker stage: it fires
    // (disconnects) when `Runtime::request_shutdown` drops the sender, at
    // which point `serve` stops accepting and its in-flight handlers are
    // aborted with the api runtime. A graceful drain
    // (`CoreHandle::begin_drain`) instead returns once its requests finished.
    shutdown: flume::Receiver<()>,
    // Where a failed startup warmup reports (see `startup_warmup`).
    startup_failure: flume::Sender<String>,
    hicache_manager: Arc<HiCacheManager>,
    tokenizer: Option<Arc<sglang_processor::DynamoTokenizer>>,
) {
    let chat_formatter = openai::load_chat_support(&server_args);
    let drain = core.clone();
    // Legacy conversation templates add stop strings `lower_chat` never sees;
    // they stay on the typed handler.
    let openai_chat = chat_formatter
        .as_ref()
        .filter(|formatter| !matches!(formatter, openai::ChatFormatter::Legacy(_)))
        .zip(tokenizer)
        .map(|(formatter, tokenizer)| openai::OpenAiChat::new(&server_args, formatter, tokenizer));
    let generation_policy = GenerationPolicy::from(server_args.as_ref());
    let state = Arc::new(AppState {
        core,
        server_args: server_args.clone(),
        generation_policy,
        chat_formatter,
        generate_hook,
        openai_chat,
    });
    // Cancelled with the api runtime on shutdown.
    tokio::spawn(startup_warmup(
        state.core.clone(),
        server_args.clone(),
        warmup_timeout(
            environ::env_f64("SGLANG_WARMUP_TIMEOUT", -1.0),
            server_args.is_disaggregation(),
        ),
        startup_failure,
    ));

    // Each endpoint module registers its own routes and merges here.
    let mut router = Router::new()
        .merge(info::routes())
        .merge(loads::routes())
        .merge(native_api::routes())
        .merge(openai::routes());
    if server_args.metrics.is_some() && metrics_listener {
        router = router.merge(metrics::routes());
    }
    // Keep this frontend's metrics files current for the directory's other
    // readers (the node's metrics listener, when this is a DP rank's offset
    // listener); stops with the runtime.
    if let Some(frontend) = state.core.metrics().cloned() {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(FLUSH_INTERVAL);
            loop {
                interval.tick().await;
                let frontend = frontend.clone();
                let _ = tokio::task::spawn_blocking(move || frontend.flush_or_warn()).await;
            }
        });
    }

    // No body limit, matching the Python server.
    let core = state.core.clone();
    let mut app = router
        .with_state(state)
        .merge(hicache::routes(hicache_manager))
        .layer(axum::extract::DefaultBodyLimit::disable());
    if let Some(http_layer) = http_layer {
        app = pre_generate::apply(app, &http_layer, core);
    }
    // Outside the extension layer, so its routes are keyed too. The bootstrap
    // registry below stays open, as Python serves it on its own server.
    app = auth::apply(app, &server_args.api_keys);

    // Prefill-only KV bootstrap registry. Merged AFTER `with_state` — its
    // router carries its own Arc<Registry> state, so it cannot merge into the
    // Router<Arc<AppState>> above — and before `log::apply`, so bootstrap traffic
    // shows in the access log.
    if server_args.enable_pd_bootstrap() {
        let (routes, sweeper) = pd_bootstrap::router_and_sweeper();
        tokio::spawn(sweeper); // cancelled with the runtime on shutdown
        app = app.merge(routes);
        tracing::info!("PD KV bootstrap registry mounted on the api listener");
    }
    app = auth::guard_fallback(app, &server_args.api_keys);

    // Apply logging and access log middleware.
    let app = log::apply(app, &server_args);

    // The listener was already bound synchronously in `runtime::start` (so a port
    // conflict fails startup); adopt it into the tokio reactor here.
    let listener = match tokio::net::TcpListener::from_std(listener) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(error = %e, "failed to adopt pre-bound listener");
            return;
        }
    };
    // `with_connect_info` exposes the peer address to the access-log middleware.
    // Once a drain ends, stop accepting, let the requests still on open
    // connections finish (the last bytes of the drained responses, or one
    // that raced the close), close idle keep-alive connections, and return.
    let graceful = drain.clone();
    let serve = axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        if graceful.drain_ended().await == DrainEnd::Drained {
            tracing::info!("drain: no request left in flight; closing the HTTP listener");
        }
    });
    tokio::select! {
        r = serve => {
            if let Err(e) = r {
                tracing::error!(error = %e, "axum serve exited");
            }
        }
        // Dropping `serve` closes the listener without waiting; the requests
        // still in flight end with the api runtime.
        () = drain.drain_close_deadline() => {
            if drain.drain_force_reason().is_none() {
                tracing::warn!(
                    "drain: connections still open past the graceful-close timeout after \
                     the requests drained; closing them"
                );
            }
        }
        _ = shutdown.recv_async() => {
            tracing::info!("shutdown: stopping accepts, aborting in-flight handlers");
        }
    }
    if let Some(reason) = drain.drain_force_reason() {
        // TODO(crash-dump): dump the in-flight requests here, as Python's
        // `dump_requests_before_crash` does on the health-check path, once
        // the Rust server supports `--crash-dump-folder`.
        tracing::error!("{reason}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::handlers::{CoreConfig, CoreMetadata};

    /// `SGLANG_WARMUP_TIMEOUT` is a float, as Python's `EnvFloat`.
    #[test]
    fn warmup_timeout_reads_python_float_seconds() {
        let configured = environ::parse_f64("3600.0").unwrap();
        assert_eq!(warmup_timeout(configured, false), Duration::from_secs(3600));
        assert_eq!(warmup_timeout(0.5, true), Duration::from_millis(500));
        assert_eq!(warmup_timeout(-1.0, false), Duration::from_secs(600));
        assert_eq!(warmup_timeout(-1.0, true), Duration::from_secs(1800));
    }

    /// A warmup the scheduler never answers times out and is reported, as
    /// Python's request timeout fails its warmup; the server stays unready.
    #[tokio::test]
    async fn timed_out_startup_warmup_is_reported() {
        // Held, so the request is queued but never answered.
        let (intake_tx, _intake_rx) = flume::unbounded();
        let core = CoreHandle::new(
            intake_tx,
            flume::unbounded().0,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: false,
                is_disaggregation: false,
                mm_limits: Default::default(),
                multimodal: false,
                metadata: CoreMetadata::default(),
                metrics: None,
                request_log: None,
                peer_endpoints: Vec::new(),
            },
        );
        let (failure_tx, failure_rx) = flume::bounded(1);
        startup_warmup(
            core.clone(),
            Arc::new(ServerArgs::default()),
            Duration::from_millis(10),
            failure_tx,
        )
        .await;
        assert_eq!(failure_rx.try_recv().unwrap(), "timed out after 10ms");
        assert!(!core.is_ready());
    }
}

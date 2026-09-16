//! Runtime bootstrap: wires channels, pins CPU-bound pools, starts the tokio
//! API server, and returns a handle the Python boundary uses for
//! `recv_requests` and `push_decode_result_batch`.
//!
//! Thread layout:
//!   * API transports — one tokio multi-thread runtime (I/O bound), pinned core set A
//!   * Tokenizer      — N pinned OS threads (CPU bound), core set B
//!   * Detokenizer    — M pinned OS threads (CPU bound), core set C
//!   * To_scheduler   — 1 thread driving the FSM
//!   * From_scheduler — 1 thread draining the scheduler → detok shards
//!   * MM workers     — K unpinned OS threads, spawned late via
//!     [`Runtime::start_mm_workers`] (multimodal models only)
//!
//! Keeping CPU-bound tokenize/detokenize off the async executor avoids stalling
//! the HTTP/gRPC worker threads.

use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::message::config::RuntimeConfig;
use crate::message::detok::DetokMsg;

use super::threads::{join_all_with_timeout, plan_cores, spawn_pool};
use crate::tokenizer_manager::channel::{
    FromSchedulerRx, FromSchedulerTx, ToSchedulerRx, ToSchedulerTx, from_scheduler, to_scheduler,
};
use crate::tokenizer_manager::wiring::{Senders, TmEvent};
use crate::utils::sock::bind_tcp_listener;
use crate::{
    api_server, tokenizer_manager, tokenizer_manager::detokenizer, tokenizer_manager::tokenizer,
};

/// A pipeline stage that owns its channel handles + config and runs a blocking
/// loop until its inbox closes.
pub trait Runnable: Send + 'static {
    fn run(self);
}

/// Live runtime. Held by the pyo3 bridge; the Python boundary reads the `to_scheduler_rx` channel,
/// and write to `from_scheduler_tx` channel. `request_shutdown` (also run on `Drop`) stops every stage.
pub struct Runtime {
    pub to_scheduler_rx: ToSchedulerRx,
    pub from_scheduler_tx: FromSchedulerTx,
    /// Wiring for the late-spawned MM pool ([`Runtime::start_mm_workers`]).
    mm_wiring: crate::multi_modality::worker::MmWiring,
    /// Worker join handles, joined by `request_shutdown` / `Drop`.
    threads: Mutex<Vec<JoinHandle<()>>>,
    /// The single shutdown sender.
    shutdown_tx: Mutex<Option<flume::Sender<()>>>,
}

/// Deadline for joining worker threads on shutdown. Past it we abandon the join
/// so process teardown can't deadlock on a worker that somehow failed to exit.
const SHUTDOWN_JOIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

impl Runtime {
    /// Build the family context from `spec` and spawn `workers`
    /// `mm-worker-{i}` threads into the shutdown join set — late, once Python
    /// has built the mm spec (`Server.start_mm_workers`).
    ///
    /// Deliberately unpinned: the threads inherit the launch thread's affinity,
    /// already narrowed by `RustServer.launch` to the server cores, so bursty
    /// MM preprocessing floats over that whole set (rather than owning cores
    /// that idle between bursts) and never preempts the scheduler's reserved
    /// cores.
    pub fn start_mm_workers(
        &self,
        spec: crate::message::config::MmSpec,
        workers: usize,
    ) -> Result<(), String> {
        let ctx = Arc::new(crate::multi_modality::worker::MmContext::new(spec)?);
        self.spawn_mm_pool(workers, ctx);
        Ok(())
    }

    /// Start the shared worker pool with a processor supplied by an external
    /// model package. `feature_shm` is that package's `_use_feature_shm`
    /// answer: place feature tensors in POSIX shm for the TP broadcast.
    pub fn start_mm_workers_with_processor(
        &self,
        processor: Arc<dyn crate::multi_modality::worker::MmProcessor>,
        workers: usize,
        feature_shm: bool,
    ) {
        let ctx = Arc::new(crate::multi_modality::worker::MmContext::with_processor(
            processor,
            feature_shm,
        ));
        self.spawn_mm_pool(workers, ctx);
    }

    fn spawn_mm_pool(&self, workers: usize, ctx: Arc<crate::multi_modality::worker::MmContext>) {
        let mut threads = self.threads.lock().unwrap();
        spawn_pool("mm-worker", None, workers.max(1), &mut threads, |_| {
            crate::multi_modality::worker::MmWorker::new(
                self.mm_wiring.mm_rx.clone(),
                self.mm_wiring.tm_tx.clone(),
                ctx.clone(),
            )
        });
    }

    /// Stop the runtime and join every worker thread (with a bounded wait).
    pub fn request_shutdown(&self) {
        drop(self.shutdown_tx.lock().unwrap().take());
        // Idempotent: a `Drop` after an explicit shutdown finds nothing to join.
        let handles = std::mem::take(&mut *self.threads.lock().unwrap());
        if !join_all_with_timeout(handles, SHUTDOWN_JOIN_TIMEOUT) {
            tracing::warn!(
                "shutdown: workers did not exit within {SHUTDOWN_JOIN_TIMEOUT:?}; abandoning join"
            );
        }
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.request_shutdown();
    }
}

/// Boot the whole frontend. Returns once threads are spawned (non-blocking).
/// `Err` on a startup misconfiguration (e.g. no tokenizer for a non-skip server).
pub fn start(cfg: RuntimeConfig) -> Result<Runtime, String> {
    let (shutdown_tx, shutdown_rx) = flume::unbounded::<()>();
    let mut threads = Vec::new();
    let plan = plan_cores(&cfg);

    // --- rings (Rust ↔ Python) ---
    let (to_scheduler_tx, to_scheduler_rx): (ToSchedulerTx, ToSchedulerRx) =
        to_scheduler(cfg.rust_server_args.to_scheduler_cap);
    let (from_scheduler_tx, from_scheduler_rx): (FromSchedulerTx, FromSchedulerRx) =
        from_scheduler(cfg.rust_server_args.from_scheduler_cap);

    // --- inter-stage channels ---
    let (tok_manager_tx, tok_manager_rx) =
        flume::bounded::<TmEvent>(cfg.rust_server_args.stage_channel_cap);
    let (tokenizer_tx, tokenizer_rx) =
        flume::bounded::<crate::message::request::Request>(cfg.rust_server_args.stage_channel_cap);
    // Encoding → MM worker pool. Bounded like the other stage edges so a slow
    // pool back-pressures instead of buffering unboundedly.
    let (mm_worker_tx, mm_worker_rx) =
        flume::bounded::<crate::message::request::Request>(cfg.rust_server_args.stage_channel_cap);
    let detokenizer_worker_num = cfg.server_args.detokenizer_worker_num;
    let mut detokenizer_tx = Vec::with_capacity(detokenizer_worker_num);
    let mut detokenizer_rx = Vec::with_capacity(detokenizer_worker_num);
    for _ in 0..detokenizer_worker_num {
        let (tx, rx) = flume::bounded::<DetokMsg>(cfg.rust_server_args.stage_channel_cap);
        detokenizer_tx.push(tx);
        detokenizer_rx.push(rx);
    }

    // Aborts get their own UNBOUNDED lane: on the bounded inbox they are dropped
    // exactly under the overload that makes them necessary (see `Senders::abort`).
    let (abort_tx, abort_rx) = flume::unbounded::<crate::tokenizer_manager::wiring::AbortSource>();
    let senders = Senders {
        tok_manager_tx: tok_manager_tx.clone(),
        abort_tx: abort_tx.clone(),
        tokenizer_tx,
        detokenizer_tx,
    };

    // `skip_tokenizer_init`: clients send token ids and receive token ids — no
    // tokenizer is loaded, and the server emits raw `output_ids` (no decode).
    let skip_tokenizer_init = cfg.server_args.skip_tokenizer_init;

    // The same instance is shared by the tokenizer pool (encode) and the detok
    // shards (decode); `None` only under `skip_tokenizer_init`.
    let dyn_tokenizer = tokenizer::load_tokenizer(
        // Empty only in standalone (test) configs (the Python handoff always
        // resolves it); empty → no tokenizer, allowed only under
        // `skip_tokenizer_init`.
        (!cfg.server_args.tokenizer_path.is_empty()).then_some(&*cfg.server_args.tokenizer_path),
        cfg.server_args.revision.as_deref(),
        skip_tokenizer_init,
    )?;
    // The `TextTokenizer` view of it, for the tokenizer pool. The MM workers
    // never tokenize: a multimodal text prompt passes through the pool first.
    let text_tokenizer: Option<Arc<dyn tokenizer::TextTokenizer>> = dyn_tokenizer
        .as_ref()
        .map(|t| Arc::new(tokenizer::DynamoTokenizer::new(t.clone())) as _);

    // The potentially slow/fallible tokenizer load above happens before the
    // ports become visible. Own both sockets before starting any worker or
    // transport thread, though, so startup remains all-or-nothing: if either
    // configured port is unavailable, both local listeners are dropped and no
    // partial runtime needs cleanup.
    let http_addr = cfg.rust_server_args.http_addr;
    let http_listener = bind_tcp_listener(http_addr)
        .map_err(|e| format!("binding HTTP listener on {http_addr} failed: {e}"))?;
    let grpc_listener = match cfg.rust_server_args.grpc_addr {
        Some(addr) => Some(
            bind_tcp_listener(addr)
                .map_err(|e| format!("binding gRPC listener on {addr} failed: {e}"))?,
        ),
        None => None,
    };

    // --- Detokenizer shards (pinned, CPU bound) ---
    {
        // Default: a real tokenizer decodes to text. `None` (→ `Skip`, raw
        // `output_ids`) only happens under `skip_tokenizer_init` —
        // `load_tokenizer` rejects a non-skip server with no tokenizer.
        let backend = match &dyn_tokenizer {
            Some(t) => detokenizer::DetokenizerBackend::Dynamo(t.clone()),
            None => detokenizer::DetokenizerBackend::Skip,
        };
        let detok_cores = plan.as_ref().map(|p| p.detok.clone());
        // Each shard owns its receiver outright (one consumer per shard), so the
        // owned `detok_rx` Vec is moved out element-by-element via the iterator.
        let count = detokenizer_rx.len();
        let mut detokenizer_rxs = detokenizer_rx.into_iter();
        spawn_pool("detokenizer", detok_cores, count, &mut threads, |i| {
            detokenizer::DetokenizerWorker::new(
                i,
                detokenizer_rxs.next().unwrap(),
                backend.clone(),
                abort_tx.clone(),
            )
        });
    }

    // --- Tokenizer pool (pinned, CPU bound) ---
    // Only spawned when a real tokenizer is loaded; under `skip_tokenizer_init`
    // there is none and request never routes to the pool, so we skip it.
    if let Some(tokenizer) = &text_tokenizer {
        // Reuse the single loaded tokenizer (shared with the detok shards).
        let tokenizer = tokenizer.clone();
        let tok_cores = plan.as_ref().map(|p| p.tok.clone());
        // Workers share the MPMC inbox (`tok_rx`) and the read-only backend, so
        // each gets a cheap clone of both.
        spawn_pool(
            "tokenizer",
            tok_cores,
            cfg.server_args.tokenizer_worker_num,
            &mut threads,
            |_i| {
                tokenizer::TokenizerWorker::new(
                    tokenizer_rx.clone(),
                    tok_manager_tx.clone(),
                    tokenizer.clone(),
                )
            },
        );
    }

    // Response heartbeat: bumped per drained frame, watched by `/health_generate`.
    let response_activity: tokenizer_manager::from_scheduler::ActivityCounter =
        Arc::new(std::sync::atomic::AtomicU64::new(0));

    // --- Response dispatcher: drains from_scheduler channel → routes chunks to shards ---
    {
        // First TM core; from_scheduler is the hotter router (every output token). One
        // worker today via `spawn_pool`, so sharding by `Rid::shard` later (see
        // `TM_CORES`) is just a larger count + per-shard receivers.
        let cores = plan
            .as_ref()
            .and_then(|p| p.tm.first().copied())
            .map(|c| vec![c]);
        let mut from_scheduler_rx = Some(from_scheduler_rx); // moved into the single worker
        let activity = response_activity.clone();
        let shutdown_rx = shutdown_rx.clone();
        spawn_pool("from-scheduler", cores, 1, &mut threads, |_| {
            tokenizer_manager::from_scheduler::Dispatcher::new(
                from_scheduler_rx.take().unwrap(),
                senders.clone(),
                activity.clone(),
                shutdown_rx.clone(),
            )
        });
    }

    // --- TokenizerManager to_scheduler loop ---
    {
        // Second TM core when present, else share the first (1-core / API-set
        // fallback) — still off the CPU-bound pool cores either way.
        let cores = plan
            .as_ref()
            .and_then(|p| p.tm.get(1).or_else(|| p.tm.first()).copied())
            .map(|c| vec![c]);
        let limits = tokenizer_manager::to_scheduler::Limits::from(&*cfg.server_args);
        let mm = tokenizer_manager::to_scheduler::MmDispatch {
            enabled: cfg.server_args.model_is_multimodal(),
            tx: mm_worker_tx,
        };
        let mut parts = Some((tok_manager_rx, to_scheduler_tx)); // moved into the single worker
        let shutdown_rx = shutdown_rx.clone();
        spawn_pool("to-scheduler", cores, 1, &mut threads, |_| {
            let (tok_manager_rx, to_scheduler_tx) = parts.take().unwrap();
            tokenizer_manager::to_scheduler::Intake::new(
                tok_manager_rx,
                abort_rx.clone(),
                senders.clone(),
                to_scheduler_tx,
                limits.clone(),
                mm.clone(),
                shutdown_rx.clone(),
            )
        });
    }

    // One transport-neutral entrance to the runtime. Each configured listener
    // receives a clone; no listener owns or reconstructs scheduler wiring.
    let frontend = crate::frontend::FrontendHandle::new(
        senders.tok_manager_tx.clone(),
        senders.abort_tx.clone(),
        crate::frontend::FrontendConfig {
            response_capacity: cfg.rust_server_args.stage_channel_cap,
            response_activity: response_activity.clone(),
            startup_ready: cfg.server_args.skip_server_warmup,
            is_disaggregation: cfg.server_args.is_disaggregation(),
            mm_limits: cfg.server_args.limit_mm_data_per_request.clone(),
            metadata: crate::frontend::FrontendMetadata::from(cfg.server_args.as_ref()),
        },
    );

    // --- HTTP + optional gRPC adapters (one tokio runtime, I/O bound) ---
    {
        let cfg = cfg.clone();
        let api_cores = plan.as_ref().map(|p| p.api.clone());
        let shutdown_rx = shutdown_rx.clone();
        let handle = std::thread::Builder::new()
            .name("api-runtime".into())
            .spawn(move || {
                let mut builder = tokio::runtime::Builder::new_multi_thread();
                builder
                    .worker_threads(cfg.rust_server_args.http_api_worker_num)
                    .enable_all();
                if let Some(cores) = api_cores {
                    let next = std::sync::atomic::AtomicUsize::new(0);
                    builder.on_thread_start(move || {
                        let idx = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if let Some(c) = cores.get(idx % cores.len()) {
                            core_affinity::set_for_current(*c);
                        }
                    });
                }
                let rt = builder.build().expect("build api runtime");
                rt.block_on(async move {
                    let state = Arc::new(crate::openai::OpenAiState::new(
                        frontend,
                        cfg.server_args.clone(),
                    ));
                    let http =
                        api_server::app::serve(http_listener, state.clone(), shutdown_rx.clone());
                    if let Some(listener) = grpc_listener {
                        let service = crate::grpc::GrpcService::new(state);
                        tokio::join!(http, crate::grpc::serve(listener, service, shutdown_rx));
                    } else {
                        http.await;
                    }
                })
            })
            .expect("spawn api runtime");
        threads.push(handle);
    }

    Ok(Runtime {
        to_scheduler_rx,
        from_scheduler_tx,
        mm_wiring: crate::multi_modality::worker::MmWiring {
            mm_rx: mm_worker_rx,
            tm_tx: tok_manager_tx,
        },
        threads: Mutex::new(threads),
        shutdown_tx: Mutex::new(Some(shutdown_tx)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::config::{RuntimeConfig, RustServerServerArgs, ServerArgs};
    use crate::message::response::{BatchHeader, frame_decode_batch_cols};
    use sglang_grpc_types::sglang::runtime::v1 as proto;
    use sglang_grpc_types::sglang::runtime::v1::sglang_service_client::SglangServiceClient;

    fn free_loopback_addrs() -> (std::net::SocketAddr, std::net::SocketAddr) {
        // Hold both probes at once so the OS cannot return the same ephemeral
        // port twice. Release them together immediately before runtime startup.
        let first = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let second = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addrs = (first.local_addr().unwrap(), second.local_addr().unwrap());
        drop((first, second));
        addrs
    }

    fn test_config(
        http_addr: std::net::SocketAddr,
        grpc_addr: Option<std::net::SocketAddr>,
    ) -> RuntimeConfig {
        RuntimeConfig {
            rust_server_args: RustServerServerArgs {
                http_addr,
                grpc_addr,
                http_api_worker_num: 1,
                ..Default::default()
            },
            server_args: Arc::new(test_server_args()),
        }
    }

    async fn connect_grpc(
        addr: std::net::SocketAddr,
    ) -> SglangServiceClient<tonic::transport::Channel> {
        let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{addr}"))
            .unwrap()
            .connect_timeout(std::time::Duration::from_secs(2));
        SglangServiceClient::new(endpoint.connect().await.expect("connect gRPC client"))
    }

    fn scheduler_rid(header: &[u8]) -> String {
        let value = rmpv::decode::read_value(&mut std::io::Cursor::new(header)).unwrap();
        value
            .as_array()
            .and_then(|fields| fields.get(1))
            .and_then(rmpv::Value::as_str)
            .expect("TokenizedGenerateReqInput rid")
            .to_owned()
    }

    fn terminal_token_frame(rid: String, output_ids: &[i64]) -> bytes::Bytes {
        let header = BatchHeader {
            rids: vec![rid],
            finish_reasons: vec![Some(
                serde_json::from_value(serde_json::json!({"type": "stop"})).unwrap(),
            )],
            prompt_tokens: vec![3],
            tok_lens: vec![output_ids.len() as u32],
            ..Default::default()
        };
        let header = rmp_serde::to_vec(&header).unwrap();
        let data = output_ids
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect::<Vec<_>>();
        frame_decode_batch_cols(&header, &[&data])
    }

    /// Minimal boot config: no tokenizer load, complete `model_config` (from
    /// `Default`), unified role.
    fn test_server_args() -> ServerArgs {
        ServerArgs {
            skip_tokenizer_init: true,
            ..Default::default()
        }
    }

    /// Regression: `request_shutdown` must actually stop the API server — it joins
    /// the api thread once the listener closes, so the port stops accepting.
    /// (Previously it set an unread flag and the port kept accepting.)
    #[test]
    fn request_shutdown_closes_listener() {
        // Pick a free port: bind :0, read the assigned addr, release it.
        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = probe.local_addr().unwrap();
        drop(probe);

        // `skip_tokenizer_init` → no tokenizer/detok model load; minimal boot.
        let server_args = test_server_args();
        let cfg = RuntimeConfig {
            rust_server_args: RustServerServerArgs {
                http_addr: addr,
                http_api_worker_num: 1,
                ..Default::default()
            },
            server_args: Arc::new(server_args),
        };
        // Bind is synchronous in `start`, so the port is already accepting.
        let rt = start(cfg).expect("start runtime");
        assert!(
            std::net::TcpStream::connect(addr).is_ok(),
            "server not listening on {addr} after start returned",
        );

        // Joins the api thread; the listener is closed by the time it returns.
        rt.request_shutdown();

        assert!(
            std::net::TcpStream::connect(addr).is_err(),
            "port still accepting connections after shutdown",
        );
    }

    /// Regression: shutdown must return promptly even with an in-flight
    /// `/generate`.
    #[test]
    fn shutdown_returns_with_in_flight_request() {
        use std::io::Write;
        use std::time::{Duration, Instant};

        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = probe.local_addr().unwrap();
        drop(probe);

        let server_args = test_server_args();
        let cfg = RuntimeConfig {
            rust_server_args: RustServerServerArgs {
                http_addr: addr,
                http_api_worker_num: 1,
                ..Default::default()
            },
            server_args: Arc::new(server_args),
        };
        let rt = start(cfg).expect("start runtime");

        // Fire a request that will block (already-tokenized → valid → pushed to the
        // ring, then the handler awaits decode frames that never arrive).
        let mut conn = std::net::TcpStream::connect(addr).expect("connect");
        let body = r#"{"input_ids":[1,2,3],"stream":false,"sampling_params":{"max_new_tokens":8}}"#;
        let req = format!(
            "POST /generate HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        conn.write_all(req.as_bytes()).unwrap();
        conn.flush().unwrap();
        std::thread::sleep(Duration::from_millis(300)); // reach the blocked state

        let t = Instant::now();
        rt.request_shutdown();
        let elapsed = t.elapsed();
        assert!(
            elapsed < Duration::from_secs(3),
            "shutdown took {elapsed:?} with an in-flight request (deadlock?)",
        );
        drop(conn);
    }

    /// Regression: a >2MB body must reach the JSON layer and fail on its
    /// *content* (unknown field → 4xx), never on size (413).
    #[test]
    fn accepts_multi_megabyte_generate_body() {
        use std::io::{Read, Write};

        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = probe.local_addr().unwrap();
        drop(probe);

        let server_args = test_server_args();
        let cfg = RuntimeConfig {
            rust_server_args: RustServerServerArgs {
                http_addr: addr,
                http_api_worker_num: 1,
                ..Default::default()
            },
            server_args: Arc::new(server_args),
        };
        let rt = start(cfg).expect("start runtime");

        // ~3MB of input_ids plus a `text`, which is mutually exclusive with them:
        // the body parses in full and is then rejected by `into_requests` with a
        // 400, proving it got past any size limit (a 413 would fire before
        // parsing). The rejection must come from OUR validation, not from serde —
        // an unknown field used to serve here, but unknown fields are now ignored
        // to match Python, so such a body would be accepted, dispatched to a ring
        // nobody drains in this test, and hang the connection.
        let ids = "1,".repeat(1_500_000);
        let body = format!(
            r#"{{"input_ids":[{}1],"text":"x","sampling_params":{{"max_new_tokens":1}}}}"#,
            ids
        );
        assert!(body.len() > 2 * 1024 * 1024, "test body must exceed 2MB");

        let mut conn = std::net::TcpStream::connect(addr).expect("connect");
        let req = format!(
            "POST /generate HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        conn.write_all(req.as_bytes()).unwrap();
        conn.flush().unwrap();

        let mut response = String::new();
        conn.read_to_string(&mut response).unwrap();
        let status_line = response.lines().next().unwrap_or("");
        let code: u16 = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|c| c.parse().ok())
            .unwrap_or(0);
        // A 400 from the mutually-exclusive-inputs check proves the body was read
        // and parsed in full; 413 would mean it was rejected on size beforehand.
        assert!(
            (400..500).contains(&code) && code != 413,
            "expected a JSON-layer 4xx (not 413), got: {status_line}"
        );

        rt.request_shutdown();
    }

    /// Regression: a port conflict must fail `start` (so the scheduler doesn't
    /// advertise ready), not return an `Ok` runtime whose listener never binds.
    #[test]
    fn start_fails_on_port_conflict() {
        // Hold the port so the runtime's bind conflicts (EADDRINUSE).
        let hog = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = hog.local_addr().unwrap();

        let server_args = test_server_args();
        let cfg = RuntimeConfig {
            rust_server_args: RustServerServerArgs {
                http_addr: addr,
                http_api_worker_num: 1,
                ..Default::default()
            },
            server_args: Arc::new(server_args),
        };
        let err = match start(cfg) {
            Ok(_) => panic!("bind conflict must fail startup, got Ok"),
            Err(e) => e,
        };
        assert!(err.contains("bind"), "error should mention bind: {err}");
    }

    /// A configured gRPC bind is part of startup, not a best-effort side task:
    /// failure returns synchronously and releases the HTTP listener acquired
    /// immediately before it.
    #[test]
    fn grpc_port_conflict_fails_startup_without_leaking_http_listener() {
        let grpc_hog = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let grpc_addr = grpc_hog.local_addr().unwrap();
        // Choose HTTP while the gRPC port remains occupied, guaranteeing that
        // the two addresses differ.
        let http_probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let http_addr = http_probe.local_addr().unwrap();
        drop(http_probe);

        let error = match start(test_config(http_addr, Some(grpc_addr))) {
            Ok(_) => panic!("gRPC bind conflict must fail startup"),
            Err(error) => error,
        };
        assert!(error.contains("gRPC listener"), "unexpected error: {error}");
        assert!(error.contains(&grpc_addr.to_string()));

        let rebound = std::net::TcpListener::bind(http_addr)
            .expect("failed startup must release its pre-bound HTTP listener");
        drop(rebound);
    }

    /// End to end over the configured socket: the generated runtime.v1 client
    /// reaches the existing scheduler ring, and an existing scheduler frame is
    /// translated back into a protobuf response. Shutdown then closes both
    /// transport listeners.
    #[test]
    fn configured_grpc_serves_generation_and_closes_with_http() {
        use std::time::Duration;

        let (http_addr, grpc_addr) = free_loopback_addrs();
        let rt = start(test_config(http_addr, Some(grpc_addr))).expect("start runtime");
        assert!(std::net::TcpStream::connect(http_addr).is_ok());

        let client_runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let mut stream = client_runtime.block_on(async {
            let mut client = connect_grpc(grpc_addr).await;
            let models = client
                .list_models(proto::ListModelsRequest {})
                .await
                .unwrap()
                .into_inner();
            assert_eq!(models.models.len(), 1);
            let info = client
                .get_model_info(proto::GetModelInfoRequest {})
                .await
                .unwrap()
                .into_inner();
            let info: serde_json::Value = serde_json::from_str(&info.json_info).unwrap();
            assert_eq!(info["served_model_name"], models.models[0].id);
            tokio::time::timeout(
                Duration::from_secs(5),
                client.generate(proto::GenerateRequest {
                    input_ids: vec![1, 2, 3],
                    stream: Some(true),
                    rid: Some("network-generation".into()),
                    ..Default::default()
                }),
            )
            .await
            .expect("timed out starting Generate RPC")
            .expect("Generate RPC")
            .into_inner()
        });

        assert!(
            rt.to_scheduler_rx.wait(Duration::from_secs(2)),
            "gRPC request did not reach the scheduler ring"
        );
        let requests = rt.to_scheduler_rx.drain(1);
        assert_eq!(requests.len(), 1);
        let input = requests[0]
            .buffers
            .iter()
            .find(|buffer| buffer.name == "input_ids")
            .expect("input token buffer");
        assert_eq!(input.shape, vec![3]);
        let rid = scheduler_rid(&requests[0].header);
        assert!(
            rt.from_scheduler_tx
                .push(terminal_token_frame(rid, &[9, 10]))
        );

        let response = client_runtime
            .block_on(async {
                tokio::time::timeout(Duration::from_secs(2), stream.message()).await
            })
            .expect("timed out waiting for gRPC response")
            .expect("gRPC stream error")
            .expect("gRPC stream ended without a response");
        assert_eq!(response.output_ids, vec![9, 10]);
        assert!(response.finished);
        assert_eq!(response.meta_info["id"], r#""network-generation""#);

        rt.request_shutdown();
        assert!(std::net::TcpStream::connect(http_addr).is_err());
        assert!(std::net::TcpStream::connect(grpc_addr).is_err());
    }

    /// The existing runtime.v1 listener accepts bodies above Tonic's 4 MiB
    /// default. Keep that transport policy when mounting the same protocol on
    /// the Rust frontend; reaching adapter validation proves the body decoded.
    #[test]
    fn grpc_listener_preserves_existing_large_message_limit() {
        use tonic::Code;

        let (http_addr, grpc_addr) = free_loopback_addrs();
        let rt = start(test_config(http_addr, Some(grpc_addr))).expect("start runtime");
        let client_runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let error = client_runtime.block_on(async {
            let client = connect_grpc(grpc_addr).await;
            let mut client = client.max_encoding_message_size(64 * 1024 * 1024);
            tokio::time::timeout(
                std::time::Duration::from_secs(10),
                client.generate(proto::GenerateRequest {
                    input_ids: vec![1],
                    // This field is intentionally unsupported. An
                    // UNIMPLEMENTED status proves the >4 MiB protobuf reached
                    // the adapter instead of Tonic rejecting it on size.
                    routing_key: Some("x".repeat(5 * 1024 * 1024)),
                    ..Default::default()
                }),
            )
            .await
            .expect("timed out sending large Generate RPC")
            .expect_err("unsupported routing_key must fail")
        });
        assert_eq!(error.code(), Code::Unimplemented);

        rt.request_shutdown();
        assert!(std::net::TcpStream::connect(grpc_addr).is_err());
    }

    /// Tonic's unbounded graceful-drain API waits for open HTTP/2 connections.
    /// Keep a generated response stream live to ensure the frontend's shared
    /// shutdown remains bounded and cancels it instead.
    #[test]
    fn grpc_shutdown_returns_promptly_with_in_flight_generation() {
        use std::time::{Duration, Instant};

        let (http_addr, grpc_addr) = free_loopback_addrs();
        let rt = start(test_config(http_addr, Some(grpc_addr))).expect("start runtime");
        let client_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let (_client, _stream) = client_runtime.block_on(async {
            let mut client = connect_grpc(grpc_addr).await;
            let stream = tokio::time::timeout(
                Duration::from_secs(5),
                client.generate(proto::GenerateRequest {
                    input_ids: vec![1, 2, 3],
                    stream: Some(true),
                    rid: Some("in-flight".into()),
                    ..Default::default()
                }),
            )
            .await
            .expect("timed out starting Generate RPC")
            .expect("Generate RPC")
            .into_inner();
            (client, stream)
        });
        assert!(rt.to_scheduler_rx.wait(Duration::from_secs(2)));

        let started = Instant::now();
        rt.request_shutdown();
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "shutdown waited for an open gRPC stream: {:?}",
            started.elapsed()
        );
        assert!(std::net::TcpStream::connect(http_addr).is_err());
        assert!(std::net::TcpStream::connect(grpc_addr).is_err());
    }
}

//! Runtime bootstrap: wires channels, pins the frontend threads, starts the
//! tokio API server, and returns a handle the Python boundary uses for
//! `recv_requests` and `push_decode_result_batch`.
//!
//! Thread layout (every pinned thread shares one CPU set, see
//! [`frontend_cpu_set`]):
//!   * API transports — one tokio multi-thread runtime (I/O bound), pinned
//!   * Tokenizer      — N pinned OS threads (CPU bound)
//!   * Detokenizer    — M pinned OS threads (CPU bound)
//!   * To_scheduler   — 1 pinned thread driving the FSM
//!   * From_scheduler — 1 pinned thread draining the scheduler → detok shards
//!   * MM workers     — K unpinned OS threads, spawned late via
//!     [`Runtime::start_mm_workers`] (multimodal models only)
//!
//! Keeping CPU-bound tokenize/detokenize off the async executor avoids stalling
//! the HTTP/gRPC worker threads.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::HttpLayer;
use crate::message::config::RuntimeConfig;
use crate::message::detok::DetokMsg;

use super::threads::{frontend_cpu_set, join_all_with_timeout, spawn_pool};
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
    /// The shared core, which runs the MM pool processor's media hooks.
    core: crate::api_server::handlers::CoreHandle,
    /// Worker join handles, joined by `request_shutdown` / `Drop`.
    threads: Mutex<Vec<JoinHandle<()>>>,
    /// The single shutdown sender.
    shutdown_tx: Mutex<Option<flume::Sender<()>>>,
    /// Disconnects with `shutdown_tx`, ending a [`Runtime::wait_startup_failure`].
    shutdown_rx: flume::Receiver<()>,
    /// The error of a failed startup warmup, sent at most once.
    startup_failure_rx: flume::Receiver<String>,
    /// Set once the API listeners have closed and the drain's end is no
    /// longer held (see [`Runtime::hold_drain_end`]).
    listeners_closed: Arc<AtomicBool>,
    /// Set once the API listeners have stopped serving and closed, held or not.
    own_listeners_closed: Arc<AtomicBool>,
    drain_end_gate: Arc<DrainEndGate>,
    /// Where to report a drain's listeners closing (see
    /// [`Runtime::drain_signal_fd`]).
    drain_notify: Arc<std::sync::OnceLock<String>>,
}

/// [`Runtime::drain_state`] bit: a drain has begun.
pub const DRAIN_DRAINING: u8 = 1;
/// [`Runtime::drain_state`] bit: the drain was begun forced
/// (`SGLANG_FORCE_SHUTDOWN`); a failed health probe does not set it.
pub const DRAIN_FORCE_REQUESTED: u8 = 2;
/// [`Runtime::drain_state`] bit: this runtime's own listeners have closed,
/// whether or not the drain's end is held.
pub const DRAIN_CLOSED: u8 = 4;

/// Holds a drain's end past the listeners' close until released (see
/// [`Runtime::hold_drain_end`]).
#[derive(Default)]
struct DrainEndGate {
    held: AtomicBool,
    released: AtomicBool,
    changed: tokio::sync::Notify,
}

impl DrainEndGate {
    fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
        self.changed.notify_waiters();
    }

    async fn released(&self) {
        loop {
            // Register before checking, so a release between the check and
            // the await still wakes this waiter.
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.released.load(Ordering::SeqCst) {
                return;
            }
            notified.await;
        }
    }
}

/// Deadline for joining worker threads on shutdown. Past it we abandon the join
/// so process teardown can't deadlock on a worker that somehow failed to exit.
const SHUTDOWN_JOIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

impl Runtime {
    /// The crash dump's requests (`None` without `--crash-dump-folder`).
    pub fn crash_dump_snapshot(&self) -> Option<Vec<u8>> {
        self.core.crash_dump_snapshot()
    }

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
        let ctx = Arc::new(
            crate::multi_modality::worker::MmContext::new(spec)?
                .with_limits(self.mm_wiring.limits.clone()),
        );
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
        let ctx = Arc::new(
            crate::multi_modality::worker::MmContext::with_processor(processor, feature_shm)
                .with_limits(self.mm_wiring.limits.clone()),
        );
        self.spawn_mm_pool(workers, ctx);
    }

    fn spawn_mm_pool(&self, workers: usize, ctx: Arc<crate::multi_modality::worker::MmContext>) {
        if !self.core.set_mm_processor(ctx.processor.clone()) {
            tracing::warn!("MM workers started again; admission keeps the first processor");
        }
        let mut threads = self.threads.lock().unwrap();
        spawn_pool("mm-worker", None, workers.max(1), &mut threads, |_| {
            crate::multi_modality::worker::MmWorker::new(
                self.mm_wiring.mm_rx.clone(),
                self.mm_wiring.tm_tx.clone(),
                ctx.clone(),
            )
        });
    }

    /// Block until the startup warmup fails, returning its error, or until
    /// the runtime shuts down (`None`, whether the warmup succeeded, was
    /// skipped, or never finished). The launching process takes the server
    /// down on a failure, as Python's server does.
    pub fn wait_startup_failure(&self) -> Option<String> {
        flume::Selector::new()
            .recv(&self.startup_failure_rx, |failure| failure.ok())
            .recv(&self.shutdown_rx, |_| None)
            .wait()
    }

    /// Begin the graceful drain (see `CoreHandle::begin_drain`): health
    /// answers 503, requests are still admitted, and the listeners close once
    /// none is left in flight. `force` closes them without waiting.
    pub fn begin_drain(&self, force: bool) {
        self.core.begin_drain(force);
    }

    /// Python's `--warmups` finished (see [`CoreHandle::finish_custom_warmups`]).
    pub fn finish_custom_warmups(&self) {
        self.core.finish_custom_warmups();
    }

    /// Whether a drain has begun and every listener has since closed, its
    /// end released if held: the scheduler may stop. The other stages keep
    /// running until [`Runtime::request_shutdown`].
    pub fn drain_complete(&self) -> bool {
        self.core.is_draining() && self.listeners_closed.load(Ordering::SeqCst)
    }

    /// The drain's progress as `DRAIN_*` bits, for the caller to share with
    /// the runtimes of the other attention-DP ranks.
    pub fn drain_state(&self) -> u8 {
        let mut state = 0;
        if self.core.is_draining() {
            state |= DRAIN_DRAINING;
        }
        if self.core.is_drain_forced() {
            state |= DRAIN_FORCE_REQUESTED;
        }
        if self.own_listeners_closed.load(Ordering::SeqCst) {
            state |= DRAIN_CLOSED;
        }
        state
    }

    /// Hold the end of every later graceful drain until
    /// [`Runtime::release_drain_end`]: the listeners still close once their
    /// requests drained, but [`Runtime::drain_complete`], the scheduler
    /// wake-up and the drain-ended datagram wait for the release. For DP rank
    /// 0, whose drain end stops the schedulers of every DP rank, so that the
    /// other DP ranks' listeners finish their requests first. A forced drain
    /// (or a failed health probe), even one forced mid-hold, does not wait,
    /// as Python's does not. While held, connections left open past the
    /// graceful-close timeout stay open until the release. Must be called
    /// before any drain begins.
    pub fn hold_drain_end(&self) -> Result<(), String> {
        if self.core.is_draining() {
            return Err("a drain has already begun".into());
        }
        self.drain_end_gate.held.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Release a held drain end (see [`Runtime::hold_drain_end`]); latched,
    /// so a release before the listeners closed ends the drain as they close.
    /// Ignored, returning `false`, unless a drain has begun, so it can never
    /// pre-release a later drain.
    pub fn release_drain_end(&self) -> bool {
        if !self.core.is_draining() {
            return false;
        }
        self.drain_end_gate.release();
        true
    }

    /// A pipe whose write end the caller installs as its signal wakeup fd
    /// (Python's `signal.set_wakeup_fd`, which writes each caught signal's
    /// number as one byte): `signum` begins the drain from a thread of this
    /// runtime, so a scheduler thread blocked in a GPU call or collective
    /// cannot delay it. Returns the write end, which stays open for the life
    /// of the process; the watcher thread is detached, since it may block
    /// until then.
    ///
    /// When the drain's listeners have closed, a datagram goes to the
    /// abstract Unix socket `notify`, if given: the launching process's
    /// positive signal that the drain ended, even while the scheduler thread
    /// is wedged.
    pub fn drain_signal_fd(
        &self,
        signum: u8,
        force: bool,
        notify: Option<String>,
    ) -> std::io::Result<std::os::fd::OwnedFd> {
        use std::io::Read;

        if let Some(notify) = notify {
            let _ = self.drain_notify.set(notify);
        }
        let (read, write) = super::platform::pipe_cloexec()?;
        // Python requires a non-blocking wakeup fd.
        rustix::fs::fcntl_setfl(&write, rustix::fs::OFlags::NONBLOCK)?;
        let core = self.core.clone();
        std::thread::Builder::new()
            .name("drain-signal".into())
            .spawn(move || {
                let mut read = std::fs::File::from(read);
                let mut buf = [0u8; 64];
                loop {
                    match read.read(&mut buf) {
                        Ok(0) => return, // every write end closed
                        Ok(n) if buf[..n].contains(&signum) => core.begin_drain(force),
                        Ok(_) => {}
                        // Python's signal handlers are installed without
                        // SA_RESTART, so any caught signal can interrupt this.
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                        Err(e) => {
                            tracing::error!(error = %e, "drain signal watcher stopped");
                            return;
                        }
                    }
                }
            })?;
        Ok(write)
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
        crate::api_server::core::flush_request_log();
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.request_shutdown();
    }
}

/// Send the drain-ended datagram to the Unix socket `name`; best effort,
/// since the launching process falls back to scheduler exits.
fn notify_drained(name: &str) {
    if let Err(error) = super::platform::send_notify_datagram(name, b"drained") {
        tracing::warn!(%error, name, "could not report the drain's end");
    }
}

/// Boot the whole frontend, `http_layer` wrapping the HTTP app (see
/// [`HttpLayer`]). Returns once threads are spawned (non-blocking).
/// `Err` on a startup misconfiguration (e.g. no tokenizer for a non-skip server).
pub fn start(cfg: RuntimeConfig, http_layer: Option<HttpLayer>) -> Result<Runtime, String> {
    crate::message::info::InternalState::check_registered_keys(&cfg.server_info_state_keys)?;
    let (shutdown_tx, shutdown_rx) = flume::unbounded::<()>();
    let (startup_failure_tx, startup_failure_rx) = flume::bounded::<String>(1);
    let mut threads = Vec::new();
    let cpus = frontend_cpu_set(&cfg);
    // Validated before anything is spawned, so a bad bucket list fails startup
    // cleanly.
    let metrics = match &cfg.server_args.metrics {
        Some(config) => Some(Arc::new(crate::api_server::core::FrontendMetrics::new(
            config,
            cfg.server_args.disaggregation_mode,
        )?)),
        None => None,
    };

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

    // The same instance is shared by the tokenizer pool (encode), the OpenAI chat
    // layer and the detok shards (decode); `None` only under `skip_tokenizer_init`.
    let tokenizer = tokenizer::load_tokenizer(
        // Empty only in standalone (test) configs (the Python handoff always
        // resolves it); empty → no tokenizer, allowed only under
        // `skip_tokenizer_init`.
        (!cfg.server_args.tokenizer_path.is_empty()).then_some(&*cfg.server_args.tokenizer_path),
        cfg.server_args.revision.as_deref(),
        skip_tokenizer_init,
    )?
    .map(Arc::new);
    let dyn_tokenizer = tokenizer.as_ref().map(|t| t.tokenizer().clone());

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
        // Each shard owns its receiver outright (one consumer per shard), so the
        // owned `detok_rx` Vec is moved out element-by-element via the iterator.
        let count = detokenizer_rx.len();
        let mut detokenizer_rxs = detokenizer_rx.into_iter();
        spawn_pool("detokenizer", cpus, count, &mut threads, |i| {
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
    if let Some(tokenizer) = &tokenizer {
        // Reuse the single loaded tokenizer (shared with the detok shards).
        let tokenizer = tokenizer.clone();
        // Workers share the MPMC inbox (`tok_rx`) and the read-only backend, so
        // each gets a cheap clone of both.
        spawn_pool(
            "tokenizer",
            cpus,
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
        // The hotter router (every output token). One worker today via
        // `spawn_pool`, so sharding by `Rid::shard` later is just a larger
        // count + per-shard receivers.
        let mut from_scheduler_rx = Some(from_scheduler_rx); // moved into the single worker
        let activity = response_activity.clone();
        let shutdown_rx = shutdown_rx.clone();
        spawn_pool("from-scheduler", cpus, 1, &mut threads, |_| {
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
        // TODO(tm-scaling): this thread and `from-scheduler` are single-consumer
        // serialization points, each with its own ceiling. to-scheduler runs
        // validate + `normalize_sampling_params` for *every* request before
        // fanning out to the (pooled) tokenizer workers, so a high
        // request-arrival / short-request workload is bounded by this one
        // thread's per-request cost (kept O(fields), see
        // `sampling::normalize_sampling_params`). Sharding it by rid — like the
        // tokenizer/detok pools — lifts that ceiling.
        let limits = tokenizer_manager::to_scheduler::Limits::from(&*cfg.server_args);
        let mm = tokenizer_manager::to_scheduler::MmDispatch {
            enabled: cfg.server_args.model_is_multimodal(),
            tx: mm_worker_tx,
        };
        let mut parts = Some((tok_manager_rx, to_scheduler_tx)); // moved into the single worker
        let shutdown_rx = shutdown_rx.clone();
        spawn_pool("to-scheduler", cpus, 1, &mut threads, |_| {
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
    let core = crate::api_server::handlers::CoreHandle::new(
        senders.tok_manager_tx.clone(),
        senders.abort_tx.clone(),
        crate::api_server::handlers::CoreConfig {
            response_capacity: cfg.rust_server_args.stage_channel_cap,
            response_activity: response_activity.clone(),
            startup_ready: cfg.server_args.skip_server_warmup,
            is_disaggregation: cfg.server_args.is_disaggregation(),
            mm_limits: cfg.server_args.limit_mm_data_per_request.clone(),
            multimodal: cfg.server_args.model_is_multimodal(),
            peer_endpoints: crate::api_server::core::peer_endpoints(
                &cfg.server_args.deployment_info,
                cfg.server_args.deployment_info.dp_rank.unwrap_or(0),
                &cfg.server_args.api_keys,
            )?,
            metadata: crate::api_server::handlers::CoreMetadata::from(cfg.server_args.as_ref())
                .with_server_info_state_keys(cfg.server_info_state_keys.clone()),
            metrics,
            request_log: crate::api_server::core::RequestLog::from_server_args(&cfg.server_args)?,
            generate_validator: cfg.generate_validator.clone(),
            abort_hook: cfg.abort_hook.clone(),
        },
    );
    core.set_graceful_close_timeout(cfg.rust_server_args.graceful_close_timeout);
    if cfg.server_args.custom_warmups_pending {
        core.hold_for_custom_warmups();
    }

    // --- HTTP + optional gRPC adapters (one tokio runtime, I/O bound) ---
    let listeners_closed = Arc::new(AtomicBool::new(false));
    let own_listeners_closed = Arc::new(AtomicBool::new(false));
    let drain_end_gate = Arc::new(DrainEndGate::default());
    let drain_notify = Arc::new(std::sync::OnceLock::<String>::new());
    {
        let cfg = cfg.clone();
        let shutdown_rx = shutdown_rx.clone();
        let shutdown_for_gate = shutdown_rx.clone();
        let core = core.clone();
        let listeners_closed = listeners_closed.clone();
        let own_listeners_closed = own_listeners_closed.clone();
        let drain_end_gate = drain_end_gate.clone();
        let drain_notify = drain_notify.clone();
        let waker = to_scheduler_rx.waker();
        let core_for_close = core.clone();
        let hicache_manager = Arc::new(api_server::handlers::HiCacheManager::new(
            core.clone(),
            &cfg.server_args,
        ));
        let grpc_server = grpc_listener.map(|listener| {
            let service = api_server::grpc::service::GrpcService::new(
                core.clone(),
                cfg.server_args.clone(),
                hicache_manager.clone(),
            );
            (listener, service)
        });
        let handle = std::thread::Builder::new()
            .name("api-runtime".into())
            .spawn(move || {
                // This thread runs `block_on` (the listeners' accept loops);
                // the runtime's worker and blocking threads pin on start.
                super::platform::pin_current(cpus.as_ref());
                let mut builder = tokio::runtime::Builder::new_multi_thread();
                builder
                    .worker_threads(cfg.rust_server_args.http_api_worker_num)
                    .enable_all()
                    .on_thread_start(move || super::platform::pin_current(cpus.as_ref()));
                let rt = builder.build().expect("build api runtime");
                rt.block_on(async move {
                    let http = api_server::http::app::serve(
                        http_listener,
                        core.clone(),
                        cfg.server_args.clone(),
                        cfg.rust_server_args.metrics_listener,
                        http_layer,
                        cfg.generate_hook.clone(),
                        shutdown_rx.clone(),
                        startup_failure_tx,
                        hicache_manager,
                        tokenizer.clone(),
                    );
                    if let Some((listener, service)) = grpc_server {
                        tokio::join!(
                            http,
                            api_server::grpc::app::serve(listener, service, core, shutdown_rx)
                        );
                    } else {
                        http.await;
                    }
                });
                own_listeners_closed.store(true, Ordering::SeqCst);
                if drain_end_gate.held.load(Ordering::SeqCst)
                    && core_for_close.is_draining()
                    && core_for_close.drain_force_reason().is_none()
                {
                    tracing::info!(
                        "drain: listeners closed; holding the drain's end until released"
                    );
                    rt.block_on(async {
                        tokio::select! {
                            () = drain_end_gate.released() => {}
                            () = core_for_close.drain_forced() => {}
                            _ = shutdown_for_gate.recv_async() => {}
                        }
                    });
                }
                // Every listener is closed: report it before dropping the
                // runtime, whose drop waits for blocking tasks (media fetches)
                // and ends the connections still open.
                listeners_closed.store(true, Ordering::SeqCst);
                if core_for_close.is_draining() {
                    // The scheduler stops at once rather than at its next idle
                    // wake-up, and the launching process learns the drain ended.
                    let _ = waker.send(());
                    if let Some(name) = drain_notify.get() {
                        notify_drained(name);
                    }
                }
                drop(rt);
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
            limits: tokenizer_manager::to_scheduler::Limits::from(&*cfg.server_args),
        },
        core,
        threads: Mutex::new(threads),
        shutdown_tx: Mutex::new(Some(shutdown_tx)),
        shutdown_rx,
        startup_failure_rx,
        listeners_closed,
        own_listeners_closed,
        drain_end_gate,
        drain_notify,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::config::{RuntimeConfig, RustServerServerArgs, ServerArgs};
    use crate::message::response::{BatchHeader, frame_decode_batch_cols};
    use sglang_api_types::api::v1 as proto;
    use sglang_api_types::api::v1::sglang_service_client::SglangServiceClient;

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
            generate_hook: None,
            generate_validator: None,
            abort_hook: None,
            server_info_state_keys: Vec::new(),
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
        token_frame(rid, output_ids, true)
    }

    /// One scheduler output frame for `rid`, final when `finished`.
    fn token_frame(rid: String, output_ids: &[i64], finished: bool) -> bytes::Bytes {
        let header = BatchHeader {
            rids: vec![rid],
            finish_reasons: vec![
                finished
                    .then(|| serde_json::from_value(serde_json::json!({"type": "stop"})).unwrap()),
            ],
            prompt_tokens: vec![3],
            tok_lens: vec![output_ids.len() as u32],
            cached_tokens: vec![0],
            reasoning_tokens: vec![0],
            ..Default::default()
        };
        let header = rmp_serde::to_vec(&header).unwrap();
        let data = output_ids
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect::<Vec<_>>();
        frame_decode_batch_cols(&header, &[&data])
    }

    /// Take the post-startup `FreezeGCReq`, the first message on the scheduler
    /// ring when the startup warmup is skipped, so a test sees its own
    /// requests next.
    fn take_startup_freeze_gc(rt: &Runtime) {
        assert!(
            rt.to_scheduler_rx.wait(std::time::Duration::from_secs(5)),
            "no FreezeGCReq after startup"
        );
        let first = rt.to_scheduler_rx.drain(1);
        let header = rmpv::decode::read_value(&mut &first[0].header[..]).unwrap();
        assert_eq!(header[0].as_str(), Some("FreezeGCReq"));
    }

    /// Minimal boot config: no tokenizer load, complete `model_config` (from
    /// `Default`), unified role.
    fn test_server_args() -> ServerArgs {
        ServerArgs {
            skip_tokenizer_init: true,
            // No scheduler answers here, so a startup warmup would only
            // occupy the scheduler ring that the tests inspect.
            skip_server_warmup: true,
            ..Default::default()
        }
    }

    /// Write an HTTP/1.1 request on a fresh connection (closed after the
    /// response), for the caller to read the response from.
    fn send_http(
        addr: std::net::SocketAddr,
        method: &str,
        path: &str,
        body: &str,
    ) -> std::net::TcpStream {
        use std::io::Write;
        let mut conn = std::net::TcpStream::connect(addr).expect("connect");
        conn.set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .unwrap();
        write!(
            conn,
            "{method} {path} HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        conn
    }

    fn read_response(mut conn: std::net::TcpStream) -> String {
        use std::io::Read;
        let mut response = String::new();
        conn.read_to_string(&mut response).expect("read response");
        response
    }

    /// Wait for the next request on the scheduler ring and return its rid.
    fn next_scheduler_rid(rt: &Runtime) -> String {
        assert!(
            rt.to_scheduler_rx.wait(std::time::Duration::from_secs(5)),
            "no request reached the scheduler ring"
        );
        let requests = rt.to_scheduler_rx.drain(1);
        assert_eq!(requests.len(), 1);
        scheduler_rid(&requests[0].header)
    }

    fn wait_drain_complete(rt: &Runtime) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !rt.drain_complete() {
            assert!(
                std::time::Instant::now() < deadline,
                "drain did not complete"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// The signal wakeup fd begins the drain from a runtime thread: only the
    /// drain signal's byte counts, as a wakeup fd carries every caught signal.
    /// Once the listeners have closed, the scheduler's parked wait ends and
    /// the launching process gets its datagram.
    #[cfg(target_os = "linux")]
    #[test]
    fn drain_signal_fd_begins_the_drain_and_reports_its_end() {
        use std::io::Write;

        let name = format!("sglang-drain-test-{}", std::process::id());
        let launcher = crate::utils::platform::bind_notify_socket(&name).unwrap();
        launcher
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let (addr, _) = free_loopback_addrs();
        let rt = start(test_config(addr, None), None).expect("start runtime");
        take_startup_freeze_gc(&rt);
        let mut wakeup = std::fs::File::from(rt.drain_signal_fd(15, false, Some(name)).unwrap());
        wakeup.write_all(&[2]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(!rt.core.is_draining(), "another signal began the drain");

        let parked = std::time::Instant::now();
        wakeup.write_all(&[15]).unwrap();
        assert!(rt.to_scheduler_rx.wait(std::time::Duration::from_secs(5)));
        assert!(
            parked.elapsed() < std::time::Duration::from_secs(4),
            "the scheduler's wait was not woken"
        );
        assert!(rt.drain_complete());
        let mut buf = [0u8; 16];
        let n = launcher.recv(&mut buf).expect("no drain-ended datagram");
        assert_eq!(&buf[..n], b"drained");
        rt.request_shutdown();
    }

    fn wait_drain_state(rt: &Runtime, state: u8) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while rt.drain_state() != state {
            assert!(
                std::time::Instant::now() < deadline,
                "drain state {} never became {state}",
                rt.drain_state()
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// A held drain end: the listener closes once its requests drained, but
    /// the drain completes, and the launching process hears of it, only on
    /// the release. A release before any drain began is ignored, and the
    /// hold cannot be taken once a drain began.
    #[cfg(target_os = "linux")]
    #[test]
    fn held_drain_end_waits_for_the_release() {
        let name = format!("sglang-drain-hold-test-{}", std::process::id());
        let launcher = crate::utils::platform::bind_notify_socket(&name).unwrap();
        launcher.set_nonblocking(true).unwrap();
        let (addr, _) = free_loopback_addrs();
        let rt = start(test_config(addr, None), None).expect("start runtime");
        drop(rt.drain_signal_fd(15, false, Some(name)).unwrap());
        rt.hold_drain_end().unwrap();
        assert!(!rt.release_drain_end(), "released with no drain under way");
        assert_eq!(rt.drain_state(), 0);

        rt.begin_drain(false);
        wait_drain_state(&rt, DRAIN_DRAINING | DRAIN_CLOSED);
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(std::net::TcpStream::connect(addr).is_err());
        assert!(!rt.drain_complete(), "the early release was latched");
        let mut buf = [0u8; 16];
        assert!(launcher.recv(&mut buf).is_err(), "drain end reported early");
        assert!(rt.hold_drain_end().is_err());

        assert!(rt.release_drain_end());
        wait_drain_complete(&rt);
        launcher.set_nonblocking(false).unwrap();
        launcher
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let n = launcher.recv(&mut buf).expect("no drain-ended datagram");
        assert_eq!(&buf[..n], b"drained");
        rt.request_shutdown();
    }

    /// A release that arrives while the listener is still finishing a
    /// request is latched: the drain completes as the listener closes.
    #[test]
    fn drain_end_release_is_latched() {
        let (addr, _) = free_loopback_addrs();
        let rt = start(test_config(addr, None), None).expect("start runtime");
        take_startup_freeze_gc(&rt);
        rt.hold_drain_end().unwrap();
        let streaming = send_http(
            addr,
            "POST",
            "/generate",
            r#"{"input_ids":[1,2,3],"stream":true,"sampling_params":{"max_new_tokens":8}}"#,
        );
        let rid = next_scheduler_rid(&rt);
        rt.begin_drain(false);
        assert!(rt.release_drain_end());
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert_eq!(rt.drain_state(), DRAIN_DRAINING);
        assert!(rt.from_scheduler_tx.push(terminal_token_frame(rid, &[8])));
        assert!(read_response(streaming).contains("data: [DONE]"));
        wait_drain_complete(&rt);
        rt.request_shutdown();
    }

    /// A forced drain does not wait for the release, whether forced from the
    /// start or mid-hold, and a shutdown mid-hold still stops the runtime.
    #[test]
    fn forced_drain_or_shutdown_ends_a_held_drain() {
        use std::time::{Duration, Instant};

        for case in ["forced", "forced mid-hold", "shutdown mid-hold"] {
            let (addr, _) = free_loopback_addrs();
            let rt = start(test_config(addr, None), None).expect("start runtime");
            rt.hold_drain_end().unwrap();
            if case == "forced" {
                rt.begin_drain(true);
                wait_drain_complete(&rt);
                assert_eq!(
                    rt.drain_state(),
                    DRAIN_DRAINING | DRAIN_FORCE_REQUESTED | DRAIN_CLOSED
                );
                rt.request_shutdown();
                continue;
            }
            rt.begin_drain(false);
            wait_drain_state(&rt, DRAIN_DRAINING | DRAIN_CLOSED);
            if case == "forced mid-hold" {
                rt.begin_drain(true);
                wait_drain_complete(&rt);
                rt.request_shutdown();
            } else {
                let started = Instant::now();
                rt.request_shutdown();
                assert!(
                    started.elapsed() < Duration::from_secs(3),
                    "the held drain kept the runtime from stopping"
                );
            }
        }
    }

    /// Answer the startup warmup request with a scheduler error.
    fn fail_warmup(rt: &Runtime) {
        assert!(
            rt.to_scheduler_rx.wait(std::time::Duration::from_secs(5)),
            "the startup warmup did not reach the scheduler ring"
        );
        let requests = rt.to_scheduler_rx.drain(1);
        let rid = scheduler_rid(&requests[0].header);
        assert!(
            rt.from_scheduler_tx
                .push(crate::message::response::frame_error(&rid, "warmup boom"))
        );
    }

    /// A startup warmup the scheduler fails reaches `wait_startup_failure`,
    /// which the launching process turns into Python's teardown; the health
    /// endpoint never reports the server ready.
    #[test]
    fn failed_startup_warmup_is_reported() {
        let (http_addr, _) = free_loopback_addrs();
        let mut cfg = test_config(http_addr, None);
        cfg.server_args = Arc::new(ServerArgs {
            skip_server_warmup: false,
            ..test_server_args()
        });
        let rt = start(cfg, None).expect("start runtime");
        fail_warmup(&rt);
        let failure = std::thread::scope(|scope| {
            let waiter = scope.spawn(|| rt.wait_startup_failure());
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while !waiter.is_finished() && std::time::Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            // Unblocks the wait if nothing was reported.
            rt.request_shutdown();
            waiter.join().unwrap()
        });
        let failure = failure.expect("a reported failure");
        assert!(failure.contains("warmup boom"), "{failure}");
        assert!(!rt.core.is_ready());
    }

    /// In PD mode a warmup the scheduler answers with an error leaves the
    /// server up for the health probe to judge, as Python's does for a
    /// non-200 warmup; nothing is reported, and shutdown ends the wait.
    #[test]
    fn failed_pd_startup_warmup_leaves_health_to_the_probe() {
        let (http_addr, _) = free_loopback_addrs();
        let mut cfg = test_config(http_addr, None);
        cfg.server_args = Arc::new(ServerArgs {
            skip_server_warmup: false,
            disaggregation_mode: crate::message::config::DisaggregationMode::Decode,
            ..test_server_args()
        });
        let rt = start(cfg, None).expect("start runtime");
        fail_warmup(&rt);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !rt.core.is_ready() {
            assert!(std::time::Instant::now() < deadline, "warmup never settled");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        rt.request_shutdown();
        assert_eq!(rt.wait_startup_failure(), None);
    }

    /// With `custom_warmups_pending`, the server is not serving and freezes no
    /// garbage collector until Python reports its `--warmups` finished, as
    /// Python freezes only after them, with or without a startup warmup.
    #[test]
    fn custom_warmups_hold_health_and_the_gc_freeze() {
        let (http_addr, _) = free_loopback_addrs();
        let mut cfg = test_config(http_addr, None);
        cfg.server_args = Arc::new(ServerArgs {
            custom_warmups_pending: true,
            ..test_server_args()
        });
        let rt = start(cfg, None).expect("start runtime");
        assert!(rt.core.is_ready(), "the startup warmup is skipped");
        assert!(!rt.core.is_serving());
        assert!(
            !rt.to_scheduler_rx
                .wait(std::time::Duration::from_millis(200)),
            "froze GC before the custom warmups finished"
        );
        rt.finish_custom_warmups();
        assert!(rt.core.is_serving());
        take_startup_freeze_gc(&rt);
        rt.request_shutdown();
    }

    /// Every abort by rid reaches the package's abort hook, from each entry:
    /// `/abort_request`, an abort another DP rank's listener relays
    /// (`/internal/peer`), and the gRPC `Abort`. An `abort_all` or an empty
    /// rid names no request, and does not.
    #[test]
    fn every_abort_by_rid_reaches_the_abort_hook() {
        use crate::message::ids::Rid;
        use crate::message::io_struct::{AbortReq, ControlRequest};
        use std::io::Write;

        #[derive(Debug, Default)]
        struct Named(Mutex<Vec<String>>);
        impl crate::AbortHook for Named {
            fn named_abort(&self, rid: &str) {
                self.0.lock().unwrap().push(rid.to_owned());
            }
        }

        let (http_addr, grpc_addr) = free_loopback_addrs();
        let named = Arc::new(Named::default());
        let cfg = RuntimeConfig {
            abort_hook: Some(named.clone()),
            ..test_config(http_addr, Some(grpc_addr))
        };
        let rt = start(cfg, None).expect("start runtime");
        for body in [
            r#"{"rid":"http"}"#,
            r#"{"rid":""}"#,
            r#"{"rid":"all","abort_all":true}"#,
        ] {
            let response = read_response(send_http(http_addr, "POST", "/abort_request", body));
            assert!(response.starts_with("HTTP/1.1 200"), "{body}: {response}");
        }
        for abort in [
            AbortReq::new("relayed".into(), false),
            AbortReq::new(Rid::new().to_string(), true),
        ] {
            let body = ControlRequest::AbortReq(abort).encode().unwrap();
            let mut conn = std::net::TcpStream::connect(http_addr).unwrap();
            write!(
                conn,
                "POST /internal/peer HTTP/1.1\r\nHost: t\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            conn.write_all(&body).unwrap();
            // The reply body is msgpack.
            let mut response = Vec::new();
            std::io::Read::read_to_end(&mut conn, &mut response).unwrap();
            assert!(response.starts_with(b"HTTP/1.1 200"));
        }
        let client_runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        client_runtime.block_on(async {
            let mut client = connect_grpc(grpc_addr).await;
            for (rid, abort_all) in [("grpc", false), ("", true)] {
                let request = proto::AbortRequest {
                    rid: rid.into(),
                    abort_all,
                };
                client.abort(request).await.expect("Abort RPC");
            }
        });
        assert_eq!(*named.0.lock().unwrap(), ["http", "relayed", "grpc"]);
        rt.request_shutdown();
    }
}

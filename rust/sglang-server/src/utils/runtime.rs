//! Runtime bootstrap: wires channels, pins CPU-bound pools, starts the tokio
//! API server, and returns a handle the Python boundary uses for
//! `recv_requests` and `push_decode_result_batch`.
//!
//! Thread layout:
//!   * API server     — tokio multi-thread runtime (I/O bound), pinned core set A
//!   * Tokenizer      — N pinned OS threads (CPU bound), core set B
//!   * Detokenizer    — M pinned OS threads (CPU bound), core set C
//!   * To_scheduler   — 1 thread driving the FSM
//!   * From_scheduler — 1 thread draining the scheduler → detok shards
//!   * MM workers     — K unpinned OS threads, spawned late via
//!     [`Runtime::start_mm_workers`] (multimodal models only)
//!
//! Keeping CPU-bound tokenize/detokenize off the async executor avoids stalling
//! axum's worker threads.

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
    pub http_addr: std::net::SocketAddr,
    pub startup_ready: Arc<std::sync::atomic::AtomicBool>,
    pub load_snapshots: Arc<api_server::loads::LoadSnapshotStore>,
    pub to_scheduler_rx: ToSchedulerRx,
    pub from_scheduler_tx: FromSchedulerTx,
    /// MM results parked between a worker's `MmEncoded` and the scheduler drain
    /// (`Server.take_mm_result`).
    pub mm_results: crate::multi_modality::result_store::MmResultStore,
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
        let ctx = Arc::new(crate::multi_modality::worker::MmContext::new(
            spec,
            self.mm_wiring.tokenizer.clone(),
            self.mm_results.clone(),
        )?);
        self.spawn_mm_pool(workers, ctx);
        Ok(())
    }

    pub fn start_mm_workers_with_processor(
        &self,
        processor: Arc<dyn crate::multi_modality::worker::MmProcessor>,
        workers: usize,
    ) {
        let ctx = Arc::new(crate::multi_modality::worker::MmContext::with_processor(
            processor,
            self.mm_wiring.tokenizer.clone(),
            self.mm_results.clone(),
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
    // Reserve both listeners before creating worker threads. The bootstrap
    // address remains the one advertised to decode engines and PD routers.
    let http_addr = cfg.rust_server_args.http_addr;
    let listener = bind_tcp_listener(http_addr)
        .map_err(|e| format!("binding API listener on {http_addr} failed: {e}"))?;
    let bound_http_addr = listener.local_addr().map_err(|error| error.to_string())?;
    let bootstrap_listener = cfg
        .server_args
        .disaggregation_bootstrap_port
        .filter(|port| {
            cfg.server_args.enable_pd_bootstrap()
                && cfg.server_args.dp_rank.unwrap_or(0) == 0
                && *port != http_addr.port()
                && !(cfg.server_args.dp_size > 1 && *port == cfg.server_args.port)
        })
        .map(|port| {
            let addr = std::net::SocketAddr::new(http_addr.ip(), port);
            bind_tcp_listener(addr)
                .map_err(|error| format!("binding PD bootstrap listener on {addr} failed: {error}"))
        })
        .transpose()?;
    // The shared Python registry is one engine source. DP worker listeners
    // must not export additional copies of its counters and sums.
    let frontend_metrics = cfg
        .server_args
        .enable_metrics
        .then(|| crate::metrics::FrontendMetrics::new(&cfg.server_args))
        .transpose()?;
    if let (Some(extension), Some(metrics)) =
        (&cfg.rust_server_args.http_extension, &frontend_metrics)
    {
        extension.register_metrics(&metrics.registry, &metrics.config.labels)?;
    }
    let metrics_router = api_server::metrics::router(
        &cfg.server_args,
        cfg.server_args.dp_rank.unwrap_or(0) == 0,
        frontend_metrics.clone(),
    )?;
    let load_snapshots = Arc::new(api_server::loads::LoadSnapshotStore::default());
    let loads_router = api_server::loads::router(load_snapshots.clone(), &cfg.server_args);
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
    let (mm_worker_tx, mm_worker_rx) = flume::bounded::<crate::message::request::MmRequest>(
        cfg.rust_server_args.stage_channel_cap,
    );
    let detokenizer_worker_num = cfg.server_args.detokenizer_worker_num;
    let mut detokenizer_tx = Vec::with_capacity(detokenizer_worker_num);
    let mut detokenizer_rx = Vec::with_capacity(detokenizer_worker_num);
    for _ in 0..detokenizer_worker_num {
        let (tx, rx) = flume::bounded::<DetokMsg>(cfg.rust_server_args.stage_channel_cap);
        detokenizer_tx.push(tx);
        detokenizer_rx.push(rx);
    }

    // Terminal notifications cannot wait behind a saturated work inbox.
    let (lifecycle_tx, lifecycle_rx) =
        flume::unbounded::<crate::tokenizer_manager::wiring::LifecycleEvent>();
    let senders = Senders {
        tok_manager_tx: tok_manager_tx.clone(),
        lifecycle_tx: lifecycle_tx.clone(),
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
    // The `TextTokenizer` view of it, shared by the tokenizer pool and the MM
    // worker path (which encodes the placeholder-expanded prompt itself).
    let text_tokenizer: Option<Arc<dyn tokenizer::TextTokenizer>> =
        dyn_tokenizer.as_ref().map(|t| Arc::new(t.clone()) as _);

    // Shared: MM workers park, the Python drain pops.
    let mm_results: crate::multi_modality::result_store::MmResultStore = Default::default();

    // --- Detokenizer shards (pinned, CPU bound) ---
    {
        // Default: a real tokenizer decodes to text. `None` (→ `Skip`, raw
        // `output_ids`) only happens under `skip_tokenizer_init` —
        // `load_tokenizer` rejects a non-skip server with no tokenizer.
        let backend = match &dyn_tokenizer {
            Some(t) => detokenizer::DetokenizerBackend::Dynamo {
                tokenizer: t.decoder(),
                vocab_size: cfg.server_args.tokenizer_vocab_size,
            },
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
                lifecycle_tx.clone(),
            )
            .with_http_extension(cfg.rust_server_args.http_extension.clone())
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
    let startup_ready = Arc::new(std::sync::atomic::AtomicBool::new(
        !cfg.server_args.wait_for_parent_warmup,
    ));
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
            results: mm_results.clone(),
        };
        let mut parts = Some((tok_manager_rx, to_scheduler_tx)); // moved into the single worker
        let shutdown_rx = shutdown_rx.clone();
        spawn_pool("to-scheduler", cores, 1, &mut threads, |_| {
            let (tok_manager_rx, to_scheduler_tx) = parts.take().unwrap();
            tokenizer_manager::to_scheduler::Intake::new(
                tok_manager_rx,
                lifecycle_rx.clone(),
                senders.clone(),
                to_scheduler_tx,
                limits.clone(),
                mm.clone(),
                shutdown_rx.clone(),
                frontend_metrics.clone(),
            )
        });
    }

    // --- API server (tokio, I/O bound) ---
    {
        let cfg = cfg.clone();
        let api_cores = plan.as_ref().map(|p| p.api.clone());
        let senders = senders.clone();
        let response_activity = response_activity.clone();
        let startup_ready = startup_ready.clone();
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
                rt.block_on(api_server::app::serve(
                    api_server::app::Listeners {
                        api: listener,
                        bootstrap: bootstrap_listener,
                    },
                    senders,
                    cfg.rust_server_args.stage_channel_cap,
                    cfg.server_args.clone(),
                    // Response heartbeat watched by `/health_generate`.
                    response_activity,
                    api_server::app::AuxiliaryRoutes {
                        startup_ready,
                        metrics: metrics_router,
                        loads: loads_router,
                        extension: cfg.rust_server_args.http_extension.clone(),
                        frontend_metrics,
                    },
                    shutdown_rx,
                ))
            })
            .expect("spawn api runtime");
        threads.push(handle);
    }

    Ok(Runtime {
        http_addr: bound_http_addr,
        startup_ready,
        load_snapshots,
        to_scheduler_rx,
        from_scheduler_tx,
        mm_results,
        mm_wiring: crate::multi_modality::worker::MmWiring {
            mm_rx: mm_worker_rx,
            tm_tx: tok_manager_tx,
            tokenizer: text_tokenizer,
        },
        threads: Mutex::new(threads),
        shutdown_tx: Mutex::new(Some(shutdown_tx)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::config::{RuntimeConfig, RustServerServerArgs, ServerArgs};

    /// Minimal boot config: no tokenizer load, complete `model_config` (from
    /// `Default`), unified role.
    fn test_server_args() -> ServerArgs {
        ServerArgs {
            skip_tokenizer_init: true,
            ..Default::default()
        }
    }

    #[test]
    fn dp_controls_wait_for_distinct_ranks_and_aggregate_in_rank_order() {
        use crate::message::response::frame_control_result_part;
        use std::time::Duration;

        let mut args = test_server_args();
        args.dp_size = 2;
        args.dp_rank = Some(0);
        let rt = start(RuntimeConfig {
            rust_server_args: RustServerServerArgs {
                http_addr: "127.0.0.1:0".parse().unwrap(),
                http_api_worker_num: 1,
                ..Default::default()
            },
            server_args: Arc::new(args),
        })
        .unwrap();
        for endpoint in ["flush_cache", "server_info", "flush_cache"] {
            let url = format!("http://{}/{endpoint}", rt.http_addr);
            let client = std::thread::spawn(move || {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(async move {
                        let response = reqwest::Client::builder()
                            .no_proxy()
                            .timeout(Duration::from_secs(5))
                            .build()
                            .unwrap()
                            .get(url)
                            .send()
                            .await
                            .unwrap();
                        (response.status().as_u16(), response.text().await.unwrap())
                    })
            });
            assert!(rt.to_scheduler_rx.wait(Duration::from_secs(5)));
            let requests = rt.to_scheduler_rx.drain(1);
            let request: rmpv::Value = rmp_serde::from_slice(&requests.headers[0]).unwrap();
            let rid = request.as_array().unwrap()[1].as_str().unwrap();
            let payload = |rank| {
                rmp_serde::to_vec(&serde_json::json!({
                    "success": rank == 1, "message": "Busy on rank 0",
                    "internal_state": {"last_gen_throughput":rank, "api_key":"secret"}
                }))
                .unwrap()
            };
            for _ in 0..2 {
                assert!(
                    rt.from_scheduler_tx
                        .push(frame_control_result_part(rid, 1, &payload(1)))
                );
            }
            std::thread::sleep(Duration::from_millis(20));
            assert!(
                !client.is_finished(),
                "duplicate rank must not complete a global control"
            );
            assert!(
                rt.from_scheduler_tx
                    .push(frame_control_result_part(rid, 0, &payload(0)))
            );
            let (status, body) = client.join().unwrap();
            if endpoint == "flush_cache" {
                assert_eq!(status, 400);
                assert_eq!(body, "Busy on rank 0");
            } else {
                assert_eq!(status, 200);
                assert!(!body.contains("secret"));
                let body: serde_json::Value = serde_json::from_str(&body).unwrap();
                assert_eq!(body["internal_states"].as_array().unwrap().len(), 2);
                assert_eq!(body["internal_states"][0]["last_gen_throughput"], 0);
                assert_eq!(body["internal_states"][1]["last_gen_throughput"], 1);
            }
            // Late replies must not complete a future control request.
            assert!(
                rt.from_scheduler_tx
                    .push(frame_control_result_part(rid, 0, &payload(0)))
            );
        }
        rt.request_shutdown();
    }

    #[test]
    fn scheduler_abort_reaches_http_with_partial_output_and_original_status() {
        use crate::message::response::{frame_abort_result, frame_decode_batch_cols};
        use std::time::Duration;

        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = probe.local_addr().unwrap();
        drop(probe);
        let rt = start(RuntimeConfig {
            rust_server_args: RustServerServerArgs {
                http_addr: addr,
                http_api_worker_num: 1,
                ..Default::default()
            },
            server_args: Arc::new(ServerArgs {
                enable_metrics: true,
                metrics_socket: Some("/unused/collector.sock".into()),
                ..test_server_args()
            }),
        })
        .unwrap();

        for stream in [false, true] {
            for status in [None, Some(503)] {
                let client = std::thread::spawn(move || {
                    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap()
                        .block_on(async move {
                            let response = reqwest::Client::builder().no_proxy().timeout(Duration::from_secs(10)).build().unwrap()
                                .post(format!("http://{addr}/generate"))
                                .json(&serde_json::json!({"input_ids": [1,2,3], "stream": stream, "return_logprob": true,
                                    "sampling_params": {"max_new_tokens": 8}}))
                                .send().await.unwrap();
                            (response.status(), response.text().await.unwrap())
                        })
                });
                assert!(rt.to_scheduler_rx.wait(Duration::from_secs(10)));
                let requests = rt.to_scheduler_rx.drain(1);
                let request: rmpv::Value = rmp_serde::from_slice(&requests.headers[0]).unwrap();
                let rid = request.as_array().unwrap()[1].as_str().unwrap();
                let mut header = vec![
                    serde_json::json!([rid]),
                    serde_json::json!([null]),
                    serde_json::json!([3]),
                    serde_json::json!([2]),
                ];
                header.push(serde_json::json!([2])); // output-token logprobs
                header.extend((0..11).map(|_| serde_json::json!([])));
                header.push(
                    serde_json::json!({"reasoning_tokens":[0], "cached_tokens":[1],
                    "weight_version":"old", "weight_versions":[[["old",0,2]]]}),
                );
                let ids: Vec<_> = [11i32, 12].into_iter().flat_map(i32::to_le_bytes).collect();
                let logprobs: Vec<_> = [-0.25f32, -0.5]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect();
                assert!(rt.from_scheduler_tx.push(frame_decode_batch_cols(
                    &rmp_serde::to_vec(&header).unwrap(),
                    &[&ids, &logprobs, &ids]
                )));
                let payload = rmp_serde::to_vec(&serde_json::json!({
                    "finished_reason": {"type":"abort", "message":"queue rejected", "status_code":status},
                    "weight_version":"new", "weight_versions":[["old",0,1],["new",1,2]]
                })).unwrap();
                // A late duplicate must not resurrect or complete another request.
                for _ in 0..2 {
                    assert!(rt.from_scheduler_tx.push(frame_abort_result(rid, &payload)));
                }
                let (http_status, body) = client.join().unwrap();
                assert_eq!(
                    http_status.as_u16(),
                    if stream { 200 } else { status.unwrap_or(200) }
                );
                if status.is_some() {
                    assert!(body.contains("queue rejected"), "{body}");
                    assert!(body.contains("503"), "{body}");
                } else {
                    let output: serde_json::Value = if stream {
                        let frames: Vec<_> = body
                            .lines()
                            .filter_map(|line| line.strip_prefix("data: "))
                            .collect();
                        assert_eq!(frames.last(), Some(&"[DONE]"));
                        serde_json::from_str(frames[frames.len() - 2]).unwrap()
                    } else {
                        serde_json::from_str(&body).unwrap()
                    };
                    assert_eq!(output["output_ids"], serde_json::json!([11, 12]));
                    assert_eq!(output["meta_info"]["completion_tokens"], 2);
                    assert_eq!(output["meta_info"]["prompt_tokens"], 3);
                    assert_eq!(output["meta_info"]["cached_tokens"], 1);
                    assert_eq!(output["meta_info"]["finish_reason"]["type"], "abort");
                    assert_eq!(
                        output["meta_info"]["output_token_logprobs"],
                        serde_json::json!([[-0.25, 11, null], [-0.5, 12, null]])
                    );
                    assert_eq!(output["meta_info"]["weight_version"], "new");
                    assert_eq!(
                        output["meta_info"]["weight_versions"],
                        serde_json::json!([
                            {"version":"old", "start":0, "end":1}, {"version":"new", "start":1, "end":2}
                        ])
                    );
                }
            }
        }
        let text = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                reqwest::Client::builder()
                    .no_proxy()
                    .build()
                    .unwrap()
                    .get(format!("http://{addr}/metrics/native"))
                    .send()
                    .await
                    .unwrap()
                    .text()
                    .await
                    .unwrap()
            });
        assert!(
            text.contains("sglang:http_requests_total{endpoint=\"/generate\",method=\"POST\"} 4"),
            "{text}"
        );
        assert_eq!(
            text.lines()
                .filter(
                    |line| line.starts_with("sglang:time_to_first_token_seconds_count")
                        && line.ends_with(" 2")
                )
                .count(),
            2
        );
        assert!(
            !text.contains("sglang:num_requests_total"),
            "AbortReq must not fabricate completion metrics"
        );
        assert!(
            !text.contains("sglang:num_aborted_requests_total"),
            "scheduler aborts are not client abort operations"
        );
        rt.request_shutdown();
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
}

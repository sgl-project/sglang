//! Runtime bootstrap: wires channels, pins CPU-bound pools, starts the tokio
//! API server, and returns a handle the Python boundary uses for
//! `recv_requests` (ingress drain) and `push_chunk` (egress push).
//!
//! Thread layout:
//!   * API server   — tokio multi-thread runtime (I/O bound), pinned core set A
//!   * Tokenizer    — N pinned OS threads (CPU bound), core set B
//!   * Detokenizer  — M pinned OS threads / shards (CPU bound), core set C
//!   * TM ingress   — 1 thread driving the ingress FSM
//!   * TM egress    — 1 thread draining the egress ring → detok shards
//!
//! Keeping CPU-bound tokenize/detokenize off the async executor avoids stalling
//! axum's worker threads.

use std::net::SocketAddr;
use std::sync::Arc;
use std::thread::JoinHandle;

pub mod channels;
pub mod ring;

use crate::ids::RequestIdGen;
use crate::runtime::channels::{DetokMsg, Senders, TmEvent};
use crate::runtime::ring::{
    EgressConsumer, EgressProducer, IngressConsumer, IngressProducer, egress_ring, ingress_ring,
};
use crate::{api_server, detokenizer, tokenizer, tokenizer_manager};

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub bind: SocketAddr,
    pub api_worker_num: usize,
    pub tokenizer_worker_num: usize,
    pub detokenizer_worker_num: usize,
    pub ingress_ring_cap: usize,
    pub egress_ring_cap: usize,
    pub channel_cap: usize,
    /// If true, pin pools to distinct CPU cores via `core_affinity`.
    pub pin_cores: bool,
    /// Explicit CPU core ids the pools may pin to (e.g. this rank's NUMA-local
    /// cores minus the scheduler's reserved launch cores). `None` → use every
    /// core this process is allowed on (`sched_getaffinity`).
    pub cores: Option<Vec<usize>>,
    /// Path to a `tokenizer.json` (or a model dir containing one, a tiktoken
    /// model file, or an HF Hub repo id resolved via the local cache). `None` →
    /// byte-stub tokenizer (tests / no real model).
    pub tokenizer_path: Option<String>,
    /// HF revision used only when `tokenizer_path` is a repo id. `None` → main.
    pub revision: Option<String>,
    /// Static server metadata (server_args + model_config) for config endpoints.
    /// `Arc` so cloning the config (and, downstream, each `AppState`) is cheap;
    /// `ServerArgs` itself is immutable after construction.
    pub server_args: Arc<ServerArgs>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:30000".parse().unwrap(),
            api_worker_num: 2,
            tokenizer_worker_num: 2,
            detokenizer_worker_num: 2,
            ingress_ring_cap: 8192,
            egress_ring_cap: 8192,
            channel_cap: 8192,
            pin_cores: true,
            cores: None,
            tokenizer_path: None,
            revision: None,
            server_args: Arc::new(ServerArgs::default()),
        }
    }
}

/// Static server metadata for config endpoints (`/v1/models`, …) — the JSON
/// blob the Python scheduler dumps from `server_args` + `model_config` at
/// startup, parsed once and read by key. Immutable after construction; the
/// per-request sharing into each `AppState` is done via an external `Arc`
/// (see `api_server::AppState`), so the struct itself just owns its data. There
/// is no exposure concern: these threads run inside the scheduler process, and
/// each endpoint chooses what to return.
#[derive(Default, Debug)]
pub struct ServerArgs {
    data: serde_json::Value,
}

impl ServerArgs {
    /// Parse the JSON blob; errors on malformed JSON.
    pub fn from_json(s: &str) -> Result<Self, String> {
        let data: serde_json::Value = serde_json::from_str(s).map_err(|e| e.to_string())?;
        Ok(Self { data })
    }

    /// Fail fast at startup if a field an endpoint depends on can't be resolved.
    pub fn validate_mandatory(&self) -> Result<(), String> {
        if self.served_model_name().is_empty() {
            return Err("no 'served_model_name' or 'model_path' in server_args".into());
        }
        if self.context_len().is_none() {
            return Err("no resolvable context length (model_config.context_len)".into());
        }
        Ok(())
    }

    /// `served_model_name`, falling back to `model_path` (the dump mirrors
    /// server_args, where `served_model_name` is `None` unless the user set it).
    pub fn served_model_name(&self) -> &str {
        self.str_field("served_model_name")
            .filter(|s| !s.is_empty())
            .or_else(|| self.str_field("model_path"))
            .unwrap_or("")
    }

    /// `max_model_len` for `/v1/models`: the resolved `model_config.context_len`
    /// (model_config is attached to server_args before the dump), falling back
    /// to the `context_length` user override.
    pub fn context_len(&self) -> Option<u64> {
        self.data
            .get("model_config")
            .and_then(|m| m.get("context_len"))
            .and_then(|v| v.as_u64())
            .or_else(|| self.data.get("context_length").and_then(|v| v.as_u64()))
    }

    /// Bind address `host:port` from the dumped server_args (both must be
    /// present). `host` is expected to be an IP — it's parsed as a `SocketAddr`.
    pub fn bind(&self) -> Option<String> {
        let host = self.str_field("host")?;
        let port = self.usize_field("port")?;
        Some(format!("{host}:{port}"))
    }

    /// Tokenizer source: explicit `tokenizer_path`, falling back to `model_path`
    /// (a model dir / HF repo id the Rust backend resolves). `None` → byte stub.
    /// Mirrors the Python `server_args.tokenizer_path or server_args.model_path`.
    pub fn tokenizer_path(&self) -> Option<String> {
        self.str_field("tokenizer_path")
            .filter(|s| !s.is_empty())
            .or_else(|| self.str_field("model_path"))
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    }

    /// HF `revision`, used only when `tokenizer_path` is a repo id. `None` → main.
    pub fn revision(&self) -> Option<String> {
        self.str_field("revision")
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    }

    /// `tokenizer_worker_num` — pinned tokenizer threads (server_args default 1).
    pub fn tokenizer_worker_num(&self) -> usize {
        self.usize_field("tokenizer_worker_num").unwrap_or(1)
    }

    /// `detokenizer_worker_num` — pinned detok shards (server_args default 1).
    pub fn detokenizer_worker_num(&self) -> usize {
        self.usize_field("detokenizer_worker_num").unwrap_or(1)
    }

    /// `skip_tokenizer_init`: when set the server neither tokenizes input nor
    /// detokenizes output — clients send token ids and receive token ids back.
    /// Drives the detok backend (`SkipDetok`, raw `output_ids` frames) and the
    /// ingress branch (already-tokenized only). Read from the dumped blob so it
    /// stays a single source of truth with the rest of `server_args`.
    pub fn skip_tokenizer_init(&self) -> bool {
        self.data
            .get("skip_tokenizer_init")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    fn str_field(&self, key: &str) -> Option<&str> {
        self.data.get(key).and_then(|v| v.as_str())
    }

    /// A positive integer field (zero/absent → `None`, so callers default it).
    fn usize_field(&self, key: &str) -> Option<usize> {
        self.data
            .get(key)
            .and_then(|v| v.as_u64())
            .filter(|&n| n > 0)
            .map(|n| n as usize)
    }
}

/// Live runtime. Held by the pyo3 bridge; the Python boundary reads
/// `ingress` and `egress`. Dropping joins all threads via `shutdown`.
pub struct Runtime {
    pub ingress: IngressConsumer,
    pub egress: EgressProducer,
    /// Join handles kept alive for the lifetime of the runtime; threads are
    /// detached daemons that exit when their channels close on shutdown.
    #[allow(dead_code)]
    threads: Vec<JoinHandle<()>>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl Runtime {
    pub fn request_shutdown(&self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Cores reserved for the two TokenizerManager router threads (`tm-ingress`,
/// `tm-egress`) — light, latency-sensitive channel routers, so one core each.
///
/// TODO(tm-scaling): if a single egress dispatcher becomes a bottleneck at high
/// aggregate token rates, the FSM dispatch can be sharded by `RequestId` (each
/// shard draining its own ring / channel) — at which point this needs to grow
/// to one core per ingress/egress shard rather than a fixed 2.
const TM_CORES: usize = 2;

/// Partition the machine's cores into four disjoint sets: the I/O-bound API
/// pool, the CPU-bound tokenizer and detokenizer pools, and the two TM router
/// threads. Falls back to no pinning if affinity isn't available or there aren't
/// enough cores for the (CPU-bound) pools.
struct CorePlan {
    api: Vec<core_affinity::CoreId>,
    tok: Vec<core_affinity::CoreId>,
    detok: Vec<core_affinity::CoreId>,
    tm: Vec<core_affinity::CoreId>,
}

fn plan_cores(cfg: &RuntimeConfig) -> Option<CorePlan> {
    if !cfg.pin_cores {
        return None;
    }
    // Prefer an explicit core list (NUMA-local cores minus the scheduler's
    // reserved launch cores); otherwise use every core this process is allowed
    // on. When NUMA binding is active, `get_core_ids` already reflects the
    // NUMA-local set via `sched_getaffinity`.
    let cores: Vec<core_affinity::CoreId> = match &cfg.cores {
        Some(ids) if !ids.is_empty() => {
            ids.iter().map(|&id| core_affinity::CoreId { id }).collect()
        }
        _ => core_affinity::get_core_ids()?,
    };
    if cores.len() < cfg.api_worker_num + cfg.tokenizer_worker_num + cfg.detokenizer_worker_num {
        tracing::warn!(
            available = cores.len(),
            "not enough cores to pin all pools; running unpinned"
        );
        return None;
    }
    let mut it = cores.into_iter();
    let api: Vec<core_affinity::CoreId> = it.by_ref().take(cfg.api_worker_num).collect();
    let tok = it.by_ref().take(cfg.tokenizer_worker_num).collect();
    let detok = it.by_ref().take(cfg.detokenizer_worker_num).collect();
    // The two TM router threads get up to `TM_CORES` leftover cores; when none
    // are spare they fall back to the API set so they never float onto the
    // CPU-bound tokenizer/detok cores.
    let mut tm: Vec<core_affinity::CoreId> = it.by_ref().take(TM_CORES).collect();
    if tm.is_empty() {
        tm = api.clone();
    }
    Some(CorePlan {
        api,
        tok,
        detok,
        tm,
    })
}

/// Pin the calling thread to `core` if one was assigned (no-op otherwise).
fn pin_current(core: Option<core_affinity::CoreId>) {
    if let Some(c) = core {
        core_affinity::set_for_current(c);
    }
}

/// Spawn `n` pinned OS threads running `f(worker_index)`.
fn spawn_pinned_pool<F>(
    name: &str,
    n: usize,
    cores: Option<Vec<core_affinity::CoreId>>,
    threads: &mut Vec<JoinHandle<()>>,
    f: F,
) where
    F: Fn(usize) + Send + Sync + 'static,
{
    let f = Arc::new(f);
    for i in 0..n {
        let f = f.clone();
        let core = cores.as_ref().and_then(|c| c.get(i).copied());
        let tname = format!("{name}-{i}");
        let handle = std::thread::Builder::new()
            .name(tname)
            .spawn(move || {
                if let Some(c) = core {
                    core_affinity::set_for_current(c);
                }
                f(i);
            })
            .expect("spawn pinned thread");
        threads.push(handle);
    }
}

/// Boot the whole frontend. Returns once threads are spawned (non-blocking),
/// so the Python caller regains control of the GIL immediately.
pub fn start(cfg: RuntimeConfig) -> Runtime {
    let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut threads = Vec::new();
    let plan = plan_cores(&cfg);

    // --- rings (Rust ↔ Python) ---
    let (ingress_tx, ingress_rx): (IngressProducer, IngressConsumer) =
        ingress_ring(cfg.ingress_ring_cap);
    let (egress_tx, egress_rx): (EgressProducer, EgressConsumer) = egress_ring(cfg.egress_ring_cap);

    // --- inter-stage channels ---
    let (tm_tx, tm_rx) = flume::bounded::<TmEvent>(cfg.channel_cap);
    let (tok_tx, tok_rx) = flume::bounded::<crate::message::Request>(cfg.channel_cap);
    let mut detok_tx = Vec::with_capacity(cfg.detokenizer_worker_num);
    let mut detok_rx = Vec::with_capacity(cfg.detokenizer_worker_num);
    for _ in 0..cfg.detokenizer_worker_num {
        let (tx, rx) = flume::bounded::<DetokMsg>(cfg.channel_cap);
        detok_tx.push(tx);
        detok_rx.push(rx);
    }

    let senders = Senders {
        tm: tm_tx.clone(),
        tok: tok_tx,
        detok: detok_tx,
    };
    let id_gen = Arc::new(RequestIdGen::default());

    // `skip_tokenizer_init`: clients send token ids and receive token ids — no
    // tokenizer is loaded, and the egress emits raw `output_ids` (no decode).
    let skip_tokenizer_init = cfg.server_args.skip_tokenizer_init();

    // Load the tokenizer once; the same Arc-backed instance is shared by both
    // the tokenizer pool (encode) and the detokenizer shards (decode_stream).
    // Skipped entirely in `skip_tokenizer_init` mode.
    let dyn_tokenizer: Option<dynamo_tokenizers::Tokenizer> = if skip_tokenizer_init {
        tracing::info!("skip_tokenizer_init: token ids in and out; no tokenizer/detokenizer");
        None
    } else {
        cfg.tokenizer_path.as_deref().and_then(|path| {
            match tokenizer::load_tokenizer(path, cfg.revision.as_deref()) {
                Ok(t) => {
                    tracing::info!(%path, "loaded tokenizer");
                    Some(t)
                }
                Err(e) => {
                    tracing::error!(%path, error = %e, "tokenizer load failed; using byte stub");
                    None
                }
            }
        })
    };
    if !skip_tokenizer_init && dyn_tokenizer.is_none() && cfg.tokenizer_path.is_none() {
        tracing::warn!("no tokenizer_path configured; using byte stub");
    }

    // --- Detokenizer shards (pinned, CPU bound) ---
    {
        let backend = if skip_tokenizer_init {
            detokenizer::DetokBackend::SkipDetok
        } else {
            match &dyn_tokenizer {
                Some(t) => detokenizer::DetokBackend::Dynamo(t.clone()),
                None => detokenizer::DetokBackend::Stub,
            }
        };
        let detok_cores = plan.as_ref().map(|p| p.detok.clone());
        let rxs = Arc::new(std::sync::Mutex::new(detok_rx)); // drained once at spawn
        spawn_pinned_pool("detok", cfg.detokenizer_worker_num, detok_cores, &mut threads, {
            move |i| {
                let rx = rxs.lock().unwrap()[i].clone();
                detokenizer::run_shard(i, rx, backend.clone());
            }
        });
    }

    // --- Egress dispatcher: drains egress ring → routes chunks to shards ---
    {
        let senders = senders.clone();
        // First TM core: egress is the hotter router (every output token).
        let core = plan.as_ref().and_then(|p| p.tm.first().copied());
        let handle = std::thread::Builder::new()
            .name("tm-egress".into())
            .spawn(move || {
                pin_current(core);
                tokenizer_manager::run_egress(egress_rx, senders)
            })
            .expect("spawn tm egress");
        threads.push(handle);
    }

    // --- Tokenizer pool (pinned, CPU bound) ---
    {
        // Reuse the single loaded tokenizer (shared with the detok shards).
        let tokenizer: Arc<dyn tokenizer::TextTokenizer> = match &dyn_tokenizer {
            Some(t) => Arc::new(tokenizer::DynamoTokenizer::new(t.clone())),
            None => Arc::new(tokenizer::StubTokenizer),
        };

        let tok_cores = plan.as_ref().map(|p| p.tok.clone());
        let tm_tx = tm_tx.clone();
        let tok_rx = tok_rx.clone();
        spawn_pinned_pool(
            "tokenizer",
            cfg.tokenizer_worker_num,
            tok_cores,
            &mut threads,
            move |_i| tokenizer::run_worker(tok_rx.clone(), tm_tx.clone(), tokenizer.clone()),
        );
    }

    // --- TokenizerManager ingress loop ---
    {
        let senders = senders.clone();
        let ingress_tx = ingress_tx.clone();
        // Second TM core when present, else share the first (1-core / API-set
        // fallback) — still off the CPU-bound pool cores either way.
        let core = plan
            .as_ref()
            .and_then(|p| p.tm.get(1).or_else(|| p.tm.first()).copied());
        let handle = std::thread::Builder::new()
            .name("tm-ingress".into())
            .spawn(move || {
                pin_current(core);
                tokenizer_manager::run_ingress(tm_rx, senders, ingress_tx, skip_tokenizer_init)
            })
            .expect("spawn tm ingress");
        threads.push(handle);
    }

    // --- API server (tokio, I/O bound) ---
    {
        let cfg = cfg.clone();
        let api_cores = plan.as_ref().map(|p| p.api.clone());
        let senders = senders.clone();
        let id_gen = id_gen.clone();
        let handle = std::thread::Builder::new()
            .name("api-runtime".into())
            .spawn(move || {
                let mut builder = tokio::runtime::Builder::new_multi_thread();
                builder.worker_threads(cfg.api_worker_num).enable_all();
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
                rt.block_on(api_server::serve(
                    cfg.bind,
                    senders,
                    id_gen,
                    cfg.channel_cap,
                    cfg.server_args.clone(),
                ));
            })
            .expect("spawn api runtime");
        threads.push(handle);
    }

    Runtime {
        ingress: ingress_rx,
        egress: egress_tx,
        threads,
        shutdown,
    }
}

//! Runtime bootstrap: wires channels, pins CPU-bound pools, starts the tokio
//! API server, and returns a handle the Python boundary uses for
//! `recv_requests` (ingress drain) and `push_batch` (egress push).
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
pub mod threads;

use crate::ids::RequestIdGen;
use crate::runtime::channels::{DetokMsg, Senders, TmEvent};
use crate::runtime::ring::{
    EgressConsumer, EgressProducer, IngressConsumer, IngressProducer, egress_ring, ingress_ring,
};
use crate::runtime::threads::{plan_cores, spawn_pool};
use crate::{api_server, detokenizer, tokenizer, tokenizer_manager};

// Re-export so stages keep importing `crate::runtime::Runnable`.
pub use threads::Runnable;

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
    /// model file, or an HF Hub repo id resolved via the local cache). `None`
    /// requires `skip_tokenizer_init`; otherwise startup is a hard error.
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

    /// The model path (HF repo id / local dir) reported by `/get_model_info`.
    /// The SGLang lang backend (`RuntimeEndpoint`) uses it for chat-template
    /// detection. Falls back to the served name if `model_path` is absent.
    pub fn model_path(&self) -> &str {
        self.str_field("model_path")
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| self.served_model_name())
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
    pub fn bind(&self) -> String {
        let host = self.str_field("host").unwrap_or("localhost");
        let port = self.usize_field("port").unwrap_or(30000);
        format!("{host}:{port}")
    }

    /// Tokenizer source: explicit `tokenizer_path`, falling back to `model_path`
    /// (a model dir / HF repo id the Rust backend resolves). `None` → no tokenizer.
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

    /// Pinned API threads for the embedded HTTP api-server (server_args
    /// default 4).
    pub fn api_worker_num(&self) -> usize {
        let default_worker_num = 4
            .max(self.tokenizer_worker_num())
            .max(self.detokenizer_worker_num());
        self.usize_field("api_worker_num")
            .unwrap_or(default_worker_num)
    }

    /// `skip_tokenizer_init`: when set the server neither tokenizes input nor
    /// detokenizes output — clients send token ids and receive token ids back.
    /// Drives the detok backend (`Skip`, raw `output_ids` frames) and the
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

/// Boot the whole frontend. Returns once threads are spawned (non-blocking),
/// so the Python caller regains control of the GIL immediately. `Err` on a
/// startup misconfiguration (e.g. no tokenizer for a non-skip server).
pub fn start(cfg: RuntimeConfig) -> Result<Runtime, String> {
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

    // The same instance is shared by the tokenizer pool (encode) and the detok
    // shards (decode); `None` only under `skip_tokenizer_init`.
    let dyn_tokenizer = tokenizer::load_tokenizer(
        cfg.tokenizer_path.as_deref(),
        cfg.revision.as_deref(),
        skip_tokenizer_init,
    )?;

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
        let count = detok_rx.len();
        let mut rxs = detok_rx.into_iter();
        spawn_pool("detokenizer", detok_cores, count, &mut threads, |i| {
            detokenizer::DetokenizerWorker::new(i, rxs.next().unwrap(), backend.clone())
        });
    }

    // --- Tokenizer pool (pinned, CPU bound) ---
    // Only spawned when a real tokenizer is loaded; under `skip_tokenizer_init`
    // there is none and ingress never routes to the pool, so we skip it.
    if let Some(t) = &dyn_tokenizer {
        // Reuse the single loaded tokenizer (shared with the detok shards).
        let tokenizer: Arc<dyn tokenizer::TextTokenizer> =
            Arc::new(tokenizer::DynamoTokenizer::new(t.clone()));
        let tok_cores = plan.as_ref().map(|p| p.tok.clone());
        // Workers share the MPMC inbox (`tok_rx`) and the read-only backend, so
        // each gets a cheap clone of both.
        spawn_pool(
            "tokenizer",
            tok_cores,
            cfg.tokenizer_worker_num,
            &mut threads,
            |_i| tokenizer::TokenizerWorker::new(tok_rx.clone(), tm_tx.clone(), tokenizer.clone()),
        );
    }

    // --- Egress dispatcher: drains egress ring → routes chunks to shards ---
    {
        // First TM core; egress is the hotter router (every output token). One
        // worker today via `spawn_pool`, so sharding by `RequestId` later (see
        // `TM_CORES`) is just a larger count + per-shard receivers.
        let cores = plan
            .as_ref()
            .and_then(|p| p.tm.first().copied())
            .map(|c| vec![c]);
        let mut egress_rx = Some(egress_rx); // moved into the single worker
        spawn_pool("tm-egress", cores, 1, &mut threads, |_| {
            tokenizer_manager::Egress::new(egress_rx.take().unwrap(), senders.clone())
        });
    }

    // --- TokenizerManager ingress loop ---
    {
        // Second TM core when present, else share the first (1-core / API-set
        // fallback) — still off the CPU-bound pool cores either way.
        let cores = plan
            .as_ref()
            .and_then(|p| p.tm.get(1).or_else(|| p.tm.first()).copied())
            .map(|c| vec![c]);
        let mut parts = Some((tm_rx, ingress_tx)); // moved into the single worker
        spawn_pool("tm-ingress", cores, 1, &mut threads, |_| {
            let (tm_rx, ingress_tx) = parts.take().unwrap();
            tokenizer_manager::Ingress::new(tm_rx, senders.clone(), ingress_tx, skip_tokenizer_init)
        });
    }

    // --- API server (tokio, I/O bound) ---
    {
        let cfg = cfg.clone();
        let api_cores = plan.as_ref().map(|p| p.api.clone());
        let senders = senders.clone();
        let id_gen = id_gen.clone();
        // Shared (Arc-backed) with the detok shards; used to decode logprob token
        // text at frame time.
        let api_tokenizer = dyn_tokenizer.clone();
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
                    // Shared with detok/tokenizer pool — decodes logprob token
                    // text when `return_text_in_logprobs` is set.
                    cfg.server_args.clone(),
                    api_tokenizer,
                ))
            })
            .expect("spawn api runtime");
        threads.push(handle);
    }

    Ok(Runtime {
        ingress: ingress_rx,
        egress: egress_tx,
        threads,
        shutdown,
    })
}

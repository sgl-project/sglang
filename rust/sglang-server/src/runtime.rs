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
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

pub mod channels;
pub mod ring;
pub mod threads;

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
    pub http_addr: SocketAddr,
    pub api_worker_num: usize,
    pub tokenizer_worker_num: usize,
    pub detokenizer_worker_num: usize,
    pub ingress_ring_cap: usize,
    pub egress_ring_cap: usize,
    pub channel_cap: usize,
    /// CPU core ids the pools pin to (e.g. this rank's NUMA-local cores minus
    /// the scheduler's reserved launch cores). `None` → run unpinned.
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
            http_addr: "127.0.0.1:30000".parse().unwrap(),
            api_worker_num: 2,
            tokenizer_worker_num: 2,
            detokenizer_worker_num: 2,
            ingress_ring_cap: 8192,
            egress_ring_cap: 8192,
            channel_cap: 8192,
            cores: None,
            tokenizer_path: None,
            revision: None,
            server_args: Arc::new(
                ServerArgs::from_json("{}").expect("empty server_args blob parses"),
            ),
        }
    }
}

/// The scheduler's startup blob (`RustServer._build_server_args`) parsed once into
/// typed fields: values are post-`__post_init__`, unknown keys (e.g. `api_key`) are dropped.
#[derive(Debug, serde::Deserialize)]
pub struct ServerArgs {
    /// HF repo id / local dir of the model, reported by `/get_model_info`.
    #[serde(default)]
    pub model_path: String,
    /// Model name reported by `/v1/models` and `/server_info`.
    #[serde(default)]
    pub served_model_name: String,
    /// Tokenizer source (model dir / `tokenizer.json` / HF repo id). Empty only
    /// in minimal standalone blobs — then boot requires `skip_tokenizer_init`.
    #[serde(default)]
    pub tokenizer_path: String,
    /// HF revision, used only when `tokenizer_path` is a repo id. `None` → main.
    #[serde(default)]
    pub revision: Option<String>,
    /// HTTP bind address (see [`Self::bind`]).
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    /// Log levels driving the access log — uvicorn runs at
    /// `log_level_http or log_level` (see [`Self::http_access_log_enabled`]).
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default)]
    pub log_level_http: Option<String>,
    /// Pinned tokenizer threads / detok shards (Python asserts both ≥ 1).
    #[serde(default = "default_worker_num")]
    pub tokenizer_worker_num: usize,
    #[serde(default = "default_worker_num")]
    pub detokenizer_worker_num: usize,
    /// Token-ids-in / token-ids-out mode: no tokenizer load, raw `output_ids`
    /// frames (drives the `Skip` detok backend and the ingress branch).
    #[serde(default)]
    pub skip_tokenizer_init: bool,
    /// Streamed `/generate` frames carry per-step deltas instead of cumulative
    /// text. Matches the Python `TokenizerManager`.
    #[serde(default)]
    pub incremental_streaming_output: bool,
    /// The resolved Python `ModelConfig`, attached to the blob at dump time.
    #[serde(default)]
    pub model_config: ModelConfig,
    /// Launch-time stamps (not `server_args` fields): sglang package version
    /// and the scheduler-derived KV token capacity, reported by `/server_info`.
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub max_total_num_tokens: Option<u64>,
}

/// The slice of the resolved Python `ModelConfig` the rust server reads.
#[derive(Debug, Default, serde::Deserialize)]
pub struct ModelConfig {
    /// Resolved context length (`max_model_len` in `/v1/models`); mandatory at
    /// boot ([`ServerArgs::validate_mandatory`]).
    #[serde(default)]
    pub context_len: Option<u64>,
    /// Bounds client-supplied token ids — ingress 400s out-of-vocab ids before
    /// they crash the scheduler's embedding lookup. `None` → unvalidated.
    #[serde(default)]
    pub vocab_size: Option<u64>,
}

fn default_host() -> String {
    "127.0.0.1".into()
}
fn default_port() -> u16 {
    30000
}
fn default_log_level() -> String {
    "info".into()
}
fn default_worker_num() -> usize {
    1
}

impl ServerArgs {
    /// Parse the blob; errors on malformed JSON or a wrongly-typed field.
    pub fn from_json(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| e.to_string())
    }

    /// Fail fast at startup if a field an endpoint depends on is missing.
    pub fn validate_mandatory(&self) -> Result<(), String> {
        if self.served_model_name.is_empty() {
            return Err("no 'served_model_name' in server_args".into());
        }
        if self.model_config.context_len.is_none() {
            return Err("no resolvable context length (model_config.context_len)".into());
        }
        Ok(())
    }

    /// Bind address `host:port`. `host` is expected to be an IP — the result is
    /// parsed as a `SocketAddr`.
    pub fn bind(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Whether the HTTP access log is emitted, mirroring the Python server:
    /// uvicorn runs at `log_level_http or log_level` and prints access lines
    /// only at info/debug. `--log-level-http warning` turns them off.
    pub fn http_access_log_enabled(&self) -> bool {
        let level = self
            .log_level_http
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or(&self.log_level);
        matches!(
            level.to_ascii_lowercase().as_str(),
            "trace" | "debug" | "info"
        )
    }

    /// Pinned API threads for the embedded HTTP api-server. Python `server_args`
    /// has no such field — this is derived: enough to cover the widest pool.
    pub fn api_worker_num(&self) -> usize {
        4.max(self.tokenizer_worker_num)
            .max(self.detokenizer_worker_num)
    }
}

/// Live runtime. Held by the pyo3 bridge; the Python boundary reads `ingress`
/// and `egress`. `request_shutdown` (also run on `Drop`) stops every stage.
pub struct Runtime {
    pub ingress: IngressConsumer,
    pub egress: EgressProducer,
    /// Worker join handles, joined by `request_shutdown` / `Drop`.
    threads: Mutex<Vec<JoinHandle<()>>>,
    /// The single shutdown sender.
    shutdown_tx: Mutex<Option<flume::Sender<()>>>,
}

/// Deadline for joining worker threads on shutdown. Past it we abandon the join
/// so process teardown can't deadlock on a worker that somehow failed to exit.
const SHUTDOWN_JOIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

impl Runtime {
    /// Stop the runtime and join every worker thread (with a bounded wait).
    ///
    /// Dropping `shutdown_tx` wakes the tm-ingress/tm-egress selectors (which
    /// otherwise never see their inbox close — one self-holds a `tm` sender, the
    /// other's inbox is the Python-fed egress ring). Those exit and drop their
    /// `Senders` clones; the api thread's `serve` returns non-gracefully, so its
    /// `block_on` unwinds and the api tokio runtime is dropped — cancelling
    /// in-flight handlers, whose `AbortGuard`s release the remaining clones. With
    /// every clone gone the tok/detok channels close and those workers exit.
    ///
    /// In-flight requests are **aborted**, not drained — this is the hard-stop
    /// path (also run on `Drop`). Clients of aborted requests retry.
    pub fn request_shutdown(&self) {
        drop(self.shutdown_tx.lock().unwrap().take());
        let handles: Vec<JoinHandle<()>> = self.threads.lock().unwrap().drain(..).collect();
        if handles.is_empty() {
            return; // Idempotent: a `Drop` after an explicit shutdown has nothing to join.
        }
        // Join off-thread and wait with a deadline: a stuck worker can't wedge exit.
        let (done_tx, done_rx) = flume::bounded::<()>(1);
        std::thread::spawn(move || {
            for h in handles {
                let _ = h.join();
            }
            let _ = done_tx.send(());
        });
        if done_rx.recv_timeout(SHUTDOWN_JOIN_TIMEOUT).is_err() {
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

/// Boot the whole frontend. Returns once threads are spawned (non-blocking),
/// so the Python caller regains control of the GIL immediately. `Err` on a
/// startup misconfiguration (e.g. no tokenizer for a non-skip server).
pub fn start(cfg: RuntimeConfig) -> Result<Runtime, String> {
    // Bind the API server port before spawning any thread, so an unavailable
    // port (EADDRINUSE) is a hard startup error.
    let listener = std::net::TcpListener::bind(cfg.http_addr)
        .map_err(|e| format!("bind {} failed: {e}", cfg.http_addr))?;
    listener
        .set_nonblocking(true)
        .map_err(|e| format!("listener set_nonblocking failed: {e}"))?;

    let (shutdown_tx, shutdown_rx) = flume::unbounded::<()>();
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

    // `skip_tokenizer_init`: clients send token ids and receive token ids — no
    // tokenizer is loaded, and the egress emits raw `output_ids` (no decode).
    let skip_tokenizer_init = cfg.server_args.skip_tokenizer_init;

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
            detokenizer::DetokenizerWorker::new(
                i,
                rxs.next().unwrap(),
                backend.clone(),
                tm_tx.clone(),
            )
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

    // Egress heartbeat: bumped per drained frame, watched by `/health_generate`.
    let egress_activity: tokenizer_manager::ActivityCounter =
        Arc::new(std::sync::atomic::AtomicU64::new(0));

    // --- Egress dispatcher: drains egress ring → routes chunks to shards ---
    {
        // First TM core; egress is the hotter router (every output token). One
        // worker today via `spawn_pool`, so sharding by `RidHash` later (see
        // `TM_CORES`) is just a larger count + per-shard receivers.
        let cores = plan
            .as_ref()
            .and_then(|p| p.tm.first().copied())
            .map(|c| vec![c]);
        let mut egress_rx = Some(egress_rx); // moved into the single worker
        let activity = egress_activity.clone();
        let shutdown_rx = shutdown_rx.clone();
        spawn_pool("tm-egress", cores, 1, &mut threads, |_| {
            tokenizer_manager::Egress::new(
                egress_rx.take().unwrap(),
                senders.clone(),
                activity.clone(),
                shutdown_rx.clone(),
            )
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
        let shutdown_rx = shutdown_rx.clone();
        spawn_pool("tm-ingress", cores, 1, &mut threads, |_| {
            let (tm_rx, ingress_tx) = parts.take().unwrap();
            tokenizer_manager::Ingress::new(
                tm_rx,
                senders.clone(),
                ingress_tx,
                skip_tokenizer_init,
                cfg.server_args.model_config.vocab_size,
                shutdown_rx.clone(),
            )
        });
    }

    // --- API server (tokio, I/O bound) ---
    {
        let cfg = cfg.clone();
        let api_cores = plan.as_ref().map(|p| p.api.clone());
        let senders = senders.clone();
        let api_activity = egress_activity.clone();
        let shutdown_rx = shutdown_rx.clone();
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
                    listener,
                    senders,
                    cfg.channel_cap,
                    cfg.server_args.clone(),
                    // Egress heartbeat watched by `/health_generate`.
                    api_activity,
                    shutdown_rx,
                ))
            })
            .expect("spawn api runtime");
        threads.push(handle);
    }

    Ok(Runtime {
        ingress: ingress_rx,
        egress: egress_tx,
        threads: Mutex::new(threads),
        shutdown_tx: Mutex::new(Some(shutdown_tx)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let server_args = ServerArgs::from_json(r#"{"skip_tokenizer_init": true}"#).unwrap();
        let cfg = RuntimeConfig {
            http_addr: addr,
            api_worker_num: 1,
            tokenizer_worker_num: 1,
            detokenizer_worker_num: 1,
            server_args: Arc::new(server_args),
            ..Default::default()
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

    /// Regression: shutdown must return promptly even with an in-flight `/generate`.
    /// No scheduler drains the ingress ring or feeds the egress ring here, so the
    /// handler parks on its egress channel forever. Graceful shutdown would wait
    /// for it (deadlock → only the 5s bounded-join fallback returns); the
    /// non-graceful path cancels the handler via the api runtime drop, whose
    /// `AbortGuard` releases the last `Senders` clone so the workers exit.
    #[test]
    fn shutdown_returns_with_in_flight_request() {
        use std::io::Write;
        use std::time::{Duration, Instant};

        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = probe.local_addr().unwrap();
        drop(probe);

        let server_args = ServerArgs::from_json(r#"{"skip_tokenizer_init": true}"#).unwrap();
        let cfg = RuntimeConfig {
            http_addr: addr,
            api_worker_num: 1,
            tokenizer_worker_num: 1,
            detokenizer_worker_num: 1,
            server_args: Arc::new(server_args),
            ..Default::default()
        };
        let rt = start(cfg).expect("start runtime");

        // Fire a request that will block (already-tokenized → valid → pushed to the
        // ring, then the handler awaits egress frames that never arrive).
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

        let server_args = ServerArgs::from_json(r#"{"skip_tokenizer_init": true}"#).unwrap();
        let cfg = RuntimeConfig {
            http_addr: addr,
            api_worker_num: 1,
            tokenizer_worker_num: 1,
            detokenizer_worker_num: 1,
            server_args: Arc::new(server_args),
            ..Default::default()
        };
        let rt = start(cfg).expect("start runtime");

        // ~3MB of input_ids + an unknown field: deny_unknown_fields fails it
        // fast at the JSON layer with a 400 — proving the body got past any
        // size limit (a 413 would fire before parsing).
        let ids = "1,".repeat(1_500_000);
        let body = format!(
            r#"{{"input_ids":[{}1],"bogus":1,"sampling_params":{{"max_new_tokens":1}}}}"#,
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
        // 422 (axum Json unknown-field rejection) proves the body was parsed;
        // 413 would mean it was rejected on size before parsing.
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

        let server_args = ServerArgs::from_json(r#"{"skip_tokenizer_init": true}"#).unwrap();
        let cfg = RuntimeConfig {
            http_addr: addr,
            api_worker_num: 1,
            tokenizer_worker_num: 1,
            detokenizer_worker_num: 1,
            server_args: Arc::new(server_args),
            ..Default::default()
        };
        let err = match start(cfg) {
            Ok(_) => panic!("bind conflict must fail startup, got Ok"),
            Err(e) => e,
        };
        assert!(err.contains("bind"), "error should mention bind: {err}");
    }
}

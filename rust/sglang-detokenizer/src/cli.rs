// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Async DetokenizerManager pipeline.
//!
//! Architecture:
//!
//!   zmq_recv_task ──[raw_tx]──► async event loop ──[out_tx]──► zmq_send_task
//!   (tokio task,               • Phase 1: prepare               (tokio task,
//!    AsyncFd on ZMQ fd)        • Phase 2: tokenize               AsyncFd on
//!                              • Phase 3: finalize               ZMQ fd)
//!
//! Both ZMQ sockets are driven by tokio's epoll reactor via AsyncFd — no
//! dedicated OS threads.  When the ZMQ notification fd becomes readable/
//! writable, the async task wakes up and calls recv_multipart / send_multipart
//! with DONTWAIT, keeping tokio worker threads unblocked.

use crate::detokenizer::{
    build_batch_str_output, execute_decode, extract_batch_input, finalize_decode,
    msgpack_decode, msgpack_encode, prepare_decode_work, read_type_tag, Config,
    LimitedCapacityDict,
};
use crate::DetokenizerConfig;
use log::{error, info, warn};
use rmpv::Value;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::io::unix::AsyncFd;
use tokio::sync::mpsc;

pub const MSGPACK_MAGIC: &[u8] = b"0xSG02";
pub const PICKLE_MAGIC: &[u8] = b"0xSG01";

const DEFAULT_WORKER_THREADS: usize = 8;
const DEFAULT_RAYON_THREADS: usize = 16;

// ─────────────────────────────── CLI args ───────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Args {
    pub detokenizer_ipc_name: String,
    pub tokenizer_ipc_name: String,
    pub tokenizer_path: String,
    pub skip_tokenizer_init: bool,
    pub max_states: usize,
    pub disable_tokenizer_batch_decode: bool,
    pub tool_call_parser: String,
}

impl From<DetokenizerConfig> for Args {
    fn from(args: DetokenizerConfig) -> Self {
        Args {
            detokenizer_ipc_name: args.detokenizer_ipc_name,
            tokenizer_ipc_name: args.tokenizer_ipc_name,
            tokenizer_path: args.tokenizer_path,
            skip_tokenizer_init: args.skip_tokenizer_init,
            max_states: args.max_states,
            disable_tokenizer_batch_decode: args.disable_tokenizer_batch_decode,
            tool_call_parser: args.tool_call_parser,
        }
    }
}

// ────────────────────────────── tokenizer ───────────────────────────────────

fn load_tokenizer(path: &str) -> Option<Tokenizer> {
    if path.is_empty() {
        return None;
    }

    let json_path = if Path::new(path).is_dir() {
        format!("{}/tokenizer.json", path)
    } else {
        path.to_string()
    };

    if Path::new(&json_path).exists() {
        info!("Loading tokenizer from local file: {json_path}");
        match Tokenizer::from_file(&json_path) {
            Ok(t) => {
                info!("Tokenizer loaded successfully.");
                return Some(t);
            }
            Err(e) => warn!("Failed to load tokenizer from file ({e}), trying from_pretrained…"),
        }
    }

    info!("Loading tokenizer from HuggingFace Hub: {path}");
    match Tokenizer::from_pretrained(path, None) {
        Ok(t) => {
            info!("Tokenizer loaded from HuggingFace Hub.");
            Some(t)
        }
        Err(e) => {
            error!("Failed to load tokenizer: {e}");
            None
        }
    }
}

// ──────────────────────── AsyncFd wrapper for ZMQ ───────────────────────────

/// Newtype that lets us register ZMQ's notification fd with tokio's epoll reactor.
///
/// ZMQ exposes one fd per socket (`socket.get_fd()`).  It is NOT the raw data
/// socket — it is a signalling fd that becomes readable when the ZMQ socket has
/// pending POLLIN events, and writable when it can accept POLLOUT.  The actual
/// data transfer still uses `recv_multipart(DONTWAIT)` / `send_multipart(DONTWAIT)`.
struct ZmqNotifyFd(std::os::unix::io::RawFd);

impl AsRawFd for ZmqNotifyFd {
    fn as_raw_fd(&self) -> std::os::unix::io::RawFd {
        self.0
    }
}

// ──────────────────────── ZMQ async recv task ───────────────────────────────

/// Receives multipart messages from a ZMQ PULL socket and forwards them to
/// `tx` without ever blocking a tokio worker thread.
///
/// How it works:
///   1. `AsyncFd::readable().await` suspends the task until ZMQ's notification
///      fd signals that at least one message is ready (via epoll POLLIN).
///   2. `guard.clear_ready()` re-arms the watch *before* draining, so any
///      message that arrives during the drain loop will fire a new wakeup.
///   3. We drain all buffered messages in a tight DONTWAIT loop; EAGAIN means
///      the queue is empty and we go back to awaiting.
///
/// Drop order: `async_fd` is declared after `sock`, so it is dropped first —
/// ensuring the fd is unregistered from epoll before the socket is closed.
async fn zmq_recv_task(sock: zmq::Socket, tx: mpsc::Sender<Vec<Vec<u8>>>) {
    let raw_fd = sock.get_fd().expect("zmq::Socket::get_fd failed");
    let async_fd = AsyncFd::new(ZmqNotifyFd(raw_fd)).expect("AsyncFd::new failed");

    'outer: loop {
        let mut guard = match async_fd.readable().await {
            Ok(g) => g,
            Err(e) => {
                error!("ZMQ recv poll error: {e}");
                break;
            }
        };
        // Clear before draining to avoid missing edge notifications.
        guard.clear_ready();

        loop {
            match sock.recv_multipart(zmq::DONTWAIT) {
                Ok(parts) => {
                    if tx.send(parts).await.is_err() {
                        break 'outer; // event loop shut down
                    }
                }
                Err(zmq::Error::EAGAIN) => break, // queue empty, back to epoll
                Err(e) => {
                    error!("ZMQ recv error: {e}");
                    break 'outer;
                }
            }
        }
    }
}

// ──────────────────────── ZMQ async send task ───────────────────────────────

/// Sends msgpack frames to a ZMQ PUSH socket without blocking a tokio worker.
///
/// With `sndhwm(0)` (unlimited) `send_multipart(DONTWAIT)` almost never
/// returns EAGAIN, but when it does (e.g. peer not yet connected) we wait on
/// `async_fd.writable()` instead of spinning.
async fn zmq_send_task(sock: zmq::Socket, mut rx: mpsc::Receiver<Vec<u8>>) {
    let raw_fd = sock.get_fd().expect("zmq::Socket::get_fd failed");
    let async_fd = AsyncFd::new(ZmqNotifyFd(raw_fd)).expect("AsyncFd::new failed");

    while let Some(data) = rx.recv().await {
        loop {
            match sock.send_multipart(&[MSGPACK_MAGIC, data.as_slice()], zmq::DONTWAIT) {
                Ok(()) => break,
                Err(zmq::Error::EAGAIN) => {
                    // Socket not ready to send; wait for POLLOUT.
                    if let Ok(mut guard) = async_fd.writable().await {
                        guard.clear_ready();
                    }
                }
                Err(e) => {
                    error!("ZMQ send error: {e}");
                    return;
                }
            }
        }
    }
}

// ────────────────────── ZMQ multipart validation ────────────────────────────

fn validate_zmq_parts(parts: Vec<Vec<u8>>) -> Result<Vec<u8>, String> {
    if parts.len() == 1 {
        return Err("Received single-frame (pickle) message. Set SGLANG_IPC_USE_MSGPACK=1.".into());
    }
    let magic = &parts[0];
    if magic == PICKLE_MAGIC {
        return Err("Received pickle message. Set SGLANG_IPC_USE_MSGPACK=1.".into());
    }
    if magic != MSGPACK_MAGIC {
        return Err(format!("Unknown IPC magic: {magic:?}"));
    }
    Ok(parts[1].clone())
}

// ─────────────────────────── async event loop ───────────────────────────────

async fn run(args: Args) {
    let tokenizer: Option<Arc<Tokenizer>> = if args.skip_tokenizer_init {
        info!("skip_tokenizer_init=true; tokenizer not loaded.");
        None
    } else {
        load_tokenizer(&args.tokenizer_path).map(Arc::new)
    };

    let config = Arc::new(Config {
        disable_batch_decode: args.disable_tokenizer_batch_decode,
        is_gpt_oss: args.tool_call_parser == "gpt-oss",
        max_states: args.max_states,
    });

    let (raw_tx, mut raw_rx) = mpsc::channel::<Vec<Vec<u8>>>(256);
    let (out_tx, out_rx) = mpsc::channel::<Vec<u8>>(256);

    // ── ZMQ PULL socket → async recv task ───────────────────────────────────
    let zmq_ctx_recv = zmq::Context::new();
    let recv_sock = zmq_ctx_recv
        .socket(zmq::PULL)
        .expect("Failed to create PULL socket");
    recv_sock.set_rcvhwm(0).ok();
    recv_sock
        .bind(&args.detokenizer_ipc_name)
        .unwrap_or_else(|e| panic!("Failed to bind PULL to {}: {e}", args.detokenizer_ipc_name));
    info!("PULL socket bound to: {}", args.detokenizer_ipc_name);

    tokio::spawn(zmq_recv_task(recv_sock, raw_tx));

    // ── ZMQ PUSH socket → async send task ───────────────────────────────────
    let zmq_ctx_send = zmq::Context::new();
    let send_sock = zmq_ctx_send
        .socket(zmq::PUSH)
        .expect("Failed to create PUSH socket");
    send_sock.set_sndhwm(0).ok();
    send_sock
        .connect(&args.tokenizer_ipc_name)
        .unwrap_or_else(|e| panic!("Failed to connect PUSH to {}: {e}", args.tokenizer_ipc_name));
    info!("PUSH socket connected to: {}", args.tokenizer_ipc_name);

    tokio::spawn(zmq_send_task(send_sock, out_rx));

    // ── Main async event loop ────────────────────────────────────────────────
    info!("DetokenizerManager event loop started.");

    let mut state = LimitedCapacityDict::new(config.max_states);

    while let Some(parts) = raw_rx.recv().await {
        let data = match validate_zmq_parts(parts) {
            Ok(d) => d,
            Err(e) => {
                error!("{e}");
                continue;
            }
        };

        let val = match msgpack_decode(&data) {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to decode msgpack: {e}");
                continue;
            }
        };

        let map = match val {
            Value::Map(m) => m,
            _ => {
                error!("Expected msgpack map, got: {:?}", val);
                continue;
            }
        };

        let msg_type = match read_type_tag(&map) {
            Some(t) => t,
            None => {
                error!("Message missing 'type' field");
                continue;
            }
        };

        let encoded: Vec<u8> = match msg_type.as_str() {
            "BatchTokenIDOutput" => {
                process_batch(map, &mut state, tokenizer.as_ref(), &config).await
            }
            "BatchEmbeddingOutput" => data,
            "FreezeGCReq" => {
                info!("FreezeGCReq received – no-op in Rust detokenizer.");
                continue;
            }
            other => {
                warn!("Unknown message type: {other}");
                continue;
            }
        };

        if out_tx.send(encoded).await.is_err() {
            break;
        }
    }
}

/// Process one `BatchTokenIDOutput` message through the three-phase pipeline.
///
/// Phases 1 and 3 borrow `state` mutably but contain no `.await`, so the
/// borrow never spans a yield point and the enclosing future stays `Send`.
async fn process_batch(
    map: Vec<(Value, Value)>,
    state: &mut LimitedCapacityDict,
    tokenizer: Option<&Arc<Tokenizer>>,
    config: &Arc<Config>,
) -> Vec<u8> {
    let fields = extract_batch_input(&map);
    let bs = fields.rids.len();

    let output_strs: Vec<String> = if bs == 0 || tokenizer.is_none() {
        vec![String::new(); bs]
    } else {
        let tok = Arc::clone(tokenizer.unwrap());
        let cfg = config.clone();

        // Phase 1: update state, collect token-ID slices (no .await)
        let work = prepare_decode_work(
            &fields.rids,
            fields.finished_reasons,
            &fields.decoded_texts,
            &fields.new_decode_ids,
            &fields.read_offsets,
            fields.skip_special_tokens,
            fields.spaces_between_special_tokens,
            fields.no_stop_trim,
            state,
            &cfg,
        );

        // Phase 2: CPU-bound tokenizer decode (parallelized with rayon, no .await)
        let results = execute_decode(&work, &tok, &cfg);

        // Phase 3: compute incremental strings, update state (no .await)
        finalize_decode(&work, &results, state)
    };

    let out_val = build_batch_str_output(map, output_strs);
    msgpack_encode(&out_val)
}

// ─────────────────────────────────── main ───────────────────────────────────

pub fn start(args: Args) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Configure rayon's global thread pool before any parallel work is done.
    rayon::ThreadPoolBuilder::new()
        .num_threads(DEFAULT_RAYON_THREADS)
        .thread_name(|i| format!("sglang-detok-rayon-{i}"))
        .build_global()
        .ok();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(DEFAULT_WORKER_THREADS)
        .enable_all()
        .thread_name("sglang-detok-tokio")
        .build()
        .expect("Failed to build tokio runtime");

    rt.block_on(run(args));
}

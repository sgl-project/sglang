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

use crate::detokenizer::{
    build_batch_str_output, execute_decode, extract_batch_input, finalize_decode,
    msgpack_decode, msgpack_encode, prepare_decode_work, read_type_tag, Config,
    LimitedCapacityDict,
};
use crate::ipc::{validate_zmq_parts, zmq_recv_task, zmq_send_task};
use crate::DetokenizerConfig;
use log::{error, info, warn};
use rmpv::Value;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

const DEFAULT_WORKER_THREADS: usize = 8;

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

pub fn load_tokenizer(path: &str) -> Option<Tokenizer> {
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

        let results = execute_decode(&work, &tok, &cfg);
        finalize_decode(&work, &results, state)
    };

    let out_val = build_batch_str_output(map, output_strs);
    msgpack_encode(&out_val)
}

// ─────── in-process variant (no ZMQ PUSH; output goes to mpsc channel) ─────

/// Run the detokenizer pipeline sending decoded output via an in-process channel.
///
/// Identical to the ZMQ-based `run()` except the outgoing side: instead of
/// forwarding `BatchStrOutput` bytes over a ZMQ PUSH socket, they are sent
/// directly to `out_tx`.  Used by the engine when all Rust components share a
/// single tokio runtime so the inter-component ZMQ hop can be eliminated.
pub async fn run_in_process(args: Args, out_tx: mpsc::UnboundedSender<Vec<u8>>) {
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

    // ZMQ PULL ← scheduler (cross-process; ZMQ stays here)
    let zmq_ctx_recv = zmq::Context::new();
    let recv_sock = zmq_ctx_recv
        .socket(zmq::PULL)
        .expect("Failed to create PULL socket");
    recv_sock.set_rcvhwm(0).ok();
    recv_sock
        .bind(&args.detokenizer_ipc_name)
        .unwrap_or_else(|e| panic!("Failed to bind PULL to {}: {e}", args.detokenizer_ipc_name));
    info!("Detokenizer PULL bound to: {}", args.detokenizer_ipc_name);

    tokio::spawn(zmq_recv_task(recv_sock, raw_tx));

    info!("Detokenizer in-process event loop started.");
    let mut state = LimitedCapacityDict::new(config.max_states);

    while let Some(parts) = raw_rx.recv().await {
        let data = match validate_zmq_parts(parts) {
            Ok(d) => d,
            Err(e) => { error!("{e}"); continue; }
        };
        let val = match msgpack_decode(&data) {
            Ok(v) => v,
            Err(e) => { error!("Detokenizer: msgpack decode: {e}"); continue; }
        };
        let map = match val {
            Value::Map(m) => m,
            _ => { error!("Detokenizer: expected map"); continue; }
        };
        let msg_type = match read_type_tag(&map) {
            Some(t) => t,
            None => { error!("Detokenizer: missing 'type' field"); continue; }
        };

        let encoded = match msg_type.as_str() {
            "BatchTokenIDOutput" => {
                process_batch(map, &mut state, tokenizer.as_ref(), &config).await
            }
            "BatchEmbeddingOutput" => data,
            "FreezeGCReq" => { info!("FreezeGCReq – no-op."); continue; }
            other => { warn!("Detokenizer: unknown message type: {other}"); continue; }
        };

        if out_tx.send(encoded).is_err() {
            break;
        }
    }
}

// ─────────────────────────────────── main ───────────────────────────────────

pub fn start(args: Args) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(DEFAULT_WORKER_THREADS)
        .enable_all()
        .thread_name("sglang-detok")
        .build()
        .expect("Failed to build tokio runtime");

    rt.block_on(run(args));
}

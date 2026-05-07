// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Rust implementation of DetokenizerManager.
//!
//! Receives `BatchTokenIDOutput` (and `BatchEmbeddingOutput` / `FreezeGCReq`)
//! from the scheduler over a ZMQ PULL socket, detokenizes token-id batches
//! into strings, and forwards the resulting `BatchStrOutput` to the tokenizer
//! worker over a ZMQ PUSH socket.
//!
//! Only the msgpack IPC path is supported (`SGLANG_IPC_USE_MSGPACK=1`).

use crate::detokenizer::{
    msgpack_decode, msgpack_encode, read_type_tag, transform_batch_token_id, Config,
    LimitedCapacityDict,
};
use crate::ipc::IpcChannels;
use crate::DetokenizerConfig;
use log::{error, info, warn};
use rmpv::Value;
use std::path::Path;
use tokenizers::Tokenizer;

// ─────────────────────────────── CLI args ───────────────────────────────────
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Args {
    /// ZMQ endpoint to PULL batches from the scheduler (bind).
    detokenizer_ipc_name: String,

    /// ZMQ endpoint to PUSH results to the tokenizer worker (connect).
    tokenizer_ipc_name: String,

    /// HuggingFace model id or local path that contains tokenizer.json.
    tokenizer_path: String,

    /// Do not load a tokenizer; output empty strings (used with skip_tokenizer_init).
    skip_tokenizer_init: bool,

    /// Maximum number of in-flight decode states before oldest are evicted.
    max_states: usize,

    /// Decode each sequence individually instead of using batch_decode.
    disable_tokenizer_batch_decode: bool,

    /// Tool-call parser mode. Set to "gpt-oss" to enable special stop-token handling.
    tool_call_parser: String,
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

    // If path is a local directory, look for tokenizer.json inside it.
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

    // Fall back to downloading from the HuggingFace Hub.
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

// ─────────────────────────────── event loop ─────────────────────────────────

fn event_loop(ipc: &IpcChannels, tokenizer: Option<&Tokenizer>, config: &Config, state: &mut LimitedCapacityDict) {
    info!("DetokenizerManager event loop started.");
    loop {
        let data = ipc.recv();

        let val = match msgpack_decode(&data) {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to decode msgpack message: {e}");
                continue;
            }
        };

        let map = match &val {
            Value::Map(m) => m,
            _ => {
                error!("Expected a msgpack map, got: {:?}", val);
                continue;
            }
        };

        let msg_type = match read_type_tag(map) {
            Some(t) => t,
            None => {
                error!("Message missing 'type' field");
                continue;
            }
        };

        match msg_type.as_str() {
            "BatchTokenIDOutput" => {
                let out = transform_batch_token_id(val, state, tokenizer, config);
                let encoded = msgpack_encode(&out);
                ipc.send_msgpack(&encoded);
            }

            "BatchEmbeddingOutput" => {
                // Embedding models need no detokenization; forward as-is.
                ipc.send_msgpack(&data);
            }

            "FreezeGCReq" => {
                // In Python this triggers gc.freeze(). Nothing to do in Rust.
                info!("FreezeGCReq received – no-op in Rust detokenizer.");
            }

            other => {
                warn!("Unknown message type: {other}");
            }
        }
    }
}

// ─────────────────────────────────── main ───────────────────────────────────

pub fn start(args: Args) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let tokenizer: Option<Tokenizer> = if args.skip_tokenizer_init {
        info!("skip_tokenizer_init=true; tokenizer not loaded.");
        None
    } else {
        load_tokenizer(&args.tokenizer_path)
    };

    let config = Config {
        disable_batch_decode: args.disable_tokenizer_batch_decode,
        is_gpt_oss: args.tool_call_parser == "gpt-oss",
        max_states: args.max_states,
    };

    let mut state = LimitedCapacityDict::new(config.max_states);

    info!(
        "Binding PULL socket to: {}",
        args.detokenizer_ipc_name
    );
    info!(
        "Connecting PUSH socket to: {}",
        args.tokenizer_ipc_name
    );

    let ctx = zmq::Context::new();
    let ipc = IpcChannels::new(&ctx, &args.detokenizer_ipc_name, &args.tokenizer_ipc_name);

    event_loop(&ipc, tokenizer.as_ref(), &config, &mut state);
}

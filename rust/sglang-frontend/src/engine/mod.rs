// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Rust engine entry point — single-process, multi-threaded.
//!
//! All three Rust components run as tokio tasks on one shared runtime:
//!
//!   HTTP client
//!       │ HTTP (axum)
//!       ▼
//!   ┌─────────────────────────────────────────────────────┐
//!   │  Single tokio runtime                               │
//!   │                                                     │
//!   │  [axum handlers] ──mpsc──► [TokenizerManager]       │
//!   │                                 │           ▲        │
//!   │                                 │ ZMQ PUSH  │ mpsc   │
//!   │                                 ▼           │        │
//!   │                          [Python Scheduler] │        │
//!   │                                 │ ZMQ PUSH  │        │
//!   │                                 ▼           │        │
//!   │                          [Detokenizer task] ┘        │
//!   │                          (ZMQ PULL from scheduler)   │
//!   └─────────────────────────────────────────────────────┘
//!
//! Cross-component communication within the process uses `tokio::sync::mpsc`
//! channels — no ZMQ sockets between Rust components.
//!
//! ZMQ is only used at the two process boundaries:
//!   • TokenizerManager → Python Scheduler  (PUSH, crosses process)
//!   • Python Scheduler → Detokenizer       (PULL, crosses process)

use crate::detokenizer::runner::{self as det_runner, Args as DetokenizerArgs};
use crate::http_server;
use crate::tokenizer::{TmConfig, TokenizerManager};
use crate::{DetokenizerConfig, EngineConfig};
use log::info;
use std::sync::Arc;
use tokio::sync::mpsc;

pub fn start(config: EngineConfig) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(config.worker_threads.unwrap_or(8) as usize)
        .enable_all()
        .thread_name("sglang-engine")
        .build()
        .expect("Failed to build tokio runtime");

    rt.block_on(async move {
        // ── In-process channel: detokenizer → TokenizerManager ──────────────
        // Replaces the ZMQ PUSH/PULL pair that would otherwise cross a thread
        // boundary.  The sender is given to the detokenizer task; the receiver
        // is consumed by the TokenizerManager dispatch loop.
        let (det_tx, det_rx) = mpsc::unbounded_channel::<Vec<u8>>();

        // ── Detokenizer task ─────────────────────────────────────────────────
        // Runs on this runtime; receives BatchTokenIDOutput from the Python
        // scheduler via ZMQ PULL, decodes tokens, and sends BatchStrOutput
        // bytes into `det_tx`.
        let det_args = DetokenizerArgs::from(DetokenizerConfig {
            detokenizer_ipc_name: config.detokenizer_ipc_name.clone(),
            tokenizer_ipc_name: String::new(), // unused in in-process mode
            tokenizer_path: config.tokenizer_path.clone(),
            skip_tokenizer_init: config.skip_tokenizer_init,
            max_states: config.max_states,
            disable_tokenizer_batch_decode: config.disable_tokenizer_batch_decode,
            tool_call_parser: config.tool_call_parser.clone(),
        });
        tokio::spawn(det_runner::run_in_process(det_args, det_tx));

        // ── TokenizerManager ─────────────────────────────────────────────────
        // Tokenizes incoming HTTP requests, sends TokenizedGenerateReqInput to
        // the scheduler via ZMQ PUSH, and routes BatchStrOutput chunks received
        // from the detokenizer (via `det_rx`) back to waiting HTTP handlers.
        let tm_cfg = TmConfig {
            // The scheduler sends direct messages (e.g. abort responses) to the
            // TM via this ZMQ address, so the TM must bind PULL here even in the
            // in-process engine. Also used as http_worker_ipc in outgoing requests.
            tokenizer_ipc_name: config.tokenizer_ipc_name.clone(),
            scheduler_ipc_name: config.scheduler_ipc_name.clone(),
            tokenizer_path: config.tokenizer_path.clone(),
            skip_tokenizer_init: config.skip_tokenizer_init,
            model_name: config.model_name.clone(),
        };
        let tm = TokenizerManager::start_in_process(tm_cfg, det_rx);

        // ── HTTP server ───────────────────────────────────────────────────────
        let state = Arc::new(http_server::AppState {
            tm,
            model_name: config.model_name.clone(),
        });
        let app = http_server::build_router(state);
        let addr = format!("{}:{}", config.host, config.port);
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .unwrap_or_else(|e| panic!("Failed to bind HTTP on {addr}: {e}"));
        info!("Engine HTTP server listening on {addr}");
        axum::serve(listener, app)
            .await
            .expect("Engine HTTP server error");
    });
}

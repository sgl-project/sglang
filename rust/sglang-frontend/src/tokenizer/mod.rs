// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Rust TokenizerManager — HTTP-facing counterpart of the Python TokenizerManager.
//!
//! Responsibilities:
//!   • Tokenize incoming text (HuggingFace tokenizers crate).
//!   • Apply chat templates via minijinja (loaded from tokenizer_config.json).
//!   • Send `TokenizedGenerateReqInput` to the scheduler via ZMQ PUSH.
//!   • Receive `BatchStrOutput` from the detokenizer via ZMQ PULL, routing
//!     each incremental chunk to the correct HTTP handler by request ID.

use crate::detokenizer::{map_get, msgpack_decode, read_type_tag};
use crate::ipc::wire::{build_abort_req, build_generate_req, SamplingParams};
use crate::ipc::{validate_zmq_parts, zmq_recv_task, zmq_send_task};
use dashmap::DashMap;
use log::{error, info, warn};
use minijinja::{Environment, Value as JValue};
use rmpv::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use uuid::Uuid;

// ─────────────────────────────── config ─────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TmConfig {
    /// ZMQ PULL socket address (we bind here; detokenizer connects and pushes results).
    pub tokenizer_ipc_name: String,
    /// ZMQ PUSH socket address (scheduler binds; we connect and push requests).
    pub scheduler_ipc_name: String,
    /// Path to HuggingFace tokenizer directory or tokenizer.json.
    pub tokenizer_path: String,
    /// Skip loading the tokenizer (use raw token IDs only).
    pub skip_tokenizer_init: bool,
    /// Model name reported in API responses.
    pub model_name: String,
}

// ─────────────────────── per-request response chunk ─────────────────────────

/// A single streaming unit dispatched to an in-flight HTTP request.
#[derive(Debug)]
pub enum ResponseChunk {
    /// Incremental decoded text (request still in progress).
    Delta { text: String },
    /// Request completed. `finish_reason` is e.g. "stop" or "length".
    Done { text: String, finish_reason: String },
    /// Hard error from the scheduler / detokenizer.
    Error(String),
}

// ─────────────────────────── TokenizerManager ───────────────────────────────

pub struct TokenizerManager {
    pub config: Arc<TmConfig>,
    /// Map from RID to the sender half of the per-request response channel.
    pending: Arc<DashMap<String, mpsc::UnboundedSender<ResponseChunk>>>,
    /// Send pre-encoded msgpack bytes to the ZMQ PUSH → scheduler task.
    sched_tx: mpsc::Sender<Vec<u8>>,
    pub tokenizer: Option<Arc<Tokenizer>>,
    /// Jinja2 chat template loaded from tokenizer_config.json, if present.
    chat_template: Option<String>,
    /// EOS token string loaded from tokenizer_config.json.
    eos_token: Option<String>,
    /// BOS token string loaded from tokenizer_config.json.
    bos_token: Option<String>,
}

impl TokenizerManager {
    /// Start the tokenizer manager: load tokenizer, spin up ZMQ tasks, return handle.
    pub fn start(config: TmConfig) -> Arc<Self> {
        let cfg = Arc::new(config.clone());
        let pending: Arc<DashMap<String, mpsc::UnboundedSender<ResponseChunk>>> =
            Arc::new(DashMap::new());

        // ── Load tokenizer and chat template ────────────────────────────────
        let (tokenizer, chat_template, eos_token, bos_token) =
            load_tokenizer_and_template(&config.tokenizer_path, config.skip_tokenizer_init);

        // ── ZMQ PUSH → scheduler ────────────────────────────────────────────
        let (sched_tx, sched_rx) = mpsc::channel::<Vec<u8>>(1024);

        // TM binds PUSH; scheduler connects PULL (mirrors Python TM bind=True).
        let zmq_ctx_send = zmq::Context::new();
        let send_sock = zmq_ctx_send
            .socket(zmq::PUSH)
            .expect("Failed to create scheduler PUSH socket");
        send_sock.set_sndhwm(0).ok();
        send_sock
            .bind(&config.scheduler_ipc_name)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to bind PUSH at scheduler {}: {e}",
                    config.scheduler_ipc_name
                )
            });
        info!("Scheduler PUSH socket bound at: {}", config.scheduler_ipc_name);

        tokio::spawn(zmq_send_task(send_sock, sched_rx));

        // TM binds PULL; scheduler connects PUSH (mirrors Python TM bind=True).
        let (raw_tx, raw_rx) = mpsc::channel::<Vec<Vec<u8>>>(512);

        let zmq_ctx_recv = zmq::Context::new();
        let recv_sock = zmq_ctx_recv
            .socket(zmq::PULL)
            .expect("Failed to create tokenizer PULL socket");
        recv_sock.set_rcvhwm(0).ok();
        recv_sock
            .bind(&config.tokenizer_ipc_name)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to bind PULL at {}: {e}",
                    config.tokenizer_ipc_name
                )
            });
        info!("Tokenizer PULL socket bound to: {}", config.tokenizer_ipc_name);

        tokio::spawn(zmq_recv_task(recv_sock, raw_tx));

        let tm = Arc::new(TokenizerManager {
            config: cfg,
            pending: pending.clone(),
            sched_tx,
            tokenizer,
            chat_template,
            eos_token,
            bos_token,
        });

        // Dispatch task: decode BatchStrOutput and route to pending requests.
        tokio::spawn(dispatch_loop(raw_rx, pending));

        tm
    }

    /// Start the TokenizerManager for the in-process engine.
    ///
    /// Two receive paths run concurrently:
    ///   1. ZMQ PULL bound at `tokenizer_ipc_name` — for direct scheduler → TM
    ///      messages (e.g. abort responses).  Required because the scheduler
    ///      routes replies back using the address stored in `http_worker_ipc`.
    ///   2. In-process `mpsc` channel (`det_rx`) — carries `BatchStrOutput` from
    ///      the Rust detokenizer without an extra ZMQ hop.
    /// Both paths share the same `pending` map and call the same dispatch helpers.
    pub fn start_in_process(config: TmConfig, det_rx: mpsc::UnboundedReceiver<Vec<u8>>) -> Arc<Self> {
        let cfg = Arc::new(config.clone());
        let pending: Arc<DashMap<String, mpsc::UnboundedSender<ResponseChunk>>> =
            Arc::new(DashMap::new());

        let (tokenizer, chat_template, eos_token, bos_token) =
            load_tokenizer_and_template(&config.tokenizer_path, config.skip_tokenizer_init);

        // ── ZMQ PUSH → scheduler (TM binds; scheduler connects) ────────────
        let (sched_tx, sched_rx) = mpsc::channel::<Vec<u8>>(1024);

        let zmq_ctx_send = zmq::Context::new();
        let send_sock = zmq_ctx_send
            .socket(zmq::PUSH)
            .expect("Failed to create scheduler PUSH socket");
        send_sock.set_sndhwm(0).ok();
        send_sock
            .bind(&config.scheduler_ipc_name)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to bind PUSH at scheduler {}: {e}",
                    config.scheduler_ipc_name
                )
            });
        info!("Scheduler PUSH socket bound at: {}", config.scheduler_ipc_name);
        tokio::spawn(zmq_send_task(send_sock, sched_rx));

        // ── ZMQ PULL ← scheduler direct messages (TM binds; scheduler connects) ─
        let (raw_tx, raw_rx) = mpsc::channel::<Vec<Vec<u8>>>(512);

        let zmq_ctx_recv = zmq::Context::new();
        let recv_sock = zmq_ctx_recv
            .socket(zmq::PULL)
            .expect("Failed to create tokenizer PULL socket");
        recv_sock.set_rcvhwm(0).ok();
        recv_sock
            .bind(&config.tokenizer_ipc_name)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to bind PULL at {}: {e}",
                    config.tokenizer_ipc_name
                )
            });
        info!("Tokenizer PULL socket bound to: {}", config.tokenizer_ipc_name);
        tokio::spawn(zmq_recv_task(recv_sock, raw_tx));

        let tm = Arc::new(TokenizerManager {
            config: cfg,
            pending: pending.clone(),
            sched_tx,
            tokenizer,
            chat_template,
            eos_token,
            bos_token,
        });

        // Dispatch ZMQ messages (scheduler direct) and channel messages (detokenizer).
        tokio::spawn(dispatch_loop(raw_rx, pending.clone()));
        tokio::spawn(dispatch_loop_channel(det_rx, pending));
        tm
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Tokenize `text` using the loaded tokenizer.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, String> {
        let tok = self.tokenizer.as_ref().ok_or("Tokenizer not loaded")?;
        tok.encode(text, false)
            .map(|enc| enc.get_ids().to_vec())
            .map_err(|e| e.to_string())
    }

    /// Apply the model's chat template (Jinja2) to a list of messages.
    /// Falls back to a simple role-prefixed concatenation if no template loaded.
    pub fn apply_chat_template(
        &self,
        messages: &[(&str, &str)], // (role, content)
        add_generation_prompt: bool,
    ) -> String {
        if let Some(template) = &self.chat_template {
            match apply_jinja_template(
                template,
                messages,
                add_generation_prompt,
                self.eos_token.as_deref(),
                self.bos_token.as_deref(),
            ) {
                Ok(s) => return s,
                Err(e) => warn!("Chat template render failed ({e}), using fallback"),
            }
        }
        // Fallback: simple concatenation
        simple_format_messages(messages, add_generation_prompt)
    }

    /// Submit a generation request to the scheduler.
    ///
    /// Returns `(rid, receiver)`. The receiver yields `ResponseChunk`s as
    /// the detokenizer streams incremental output.
    pub async fn submit(
        &self,
        input_ids: Vec<u32>,
        mut sp: SamplingParams,
        stream: bool,
    ) -> (String, mpsc::UnboundedReceiver<ResponseChunk>) {
        sp.normalize(self.tokenizer.as_deref());
        let rid = Uuid::new_v4().to_string();
        let (tx, rx) = mpsc::unbounded_channel();
        self.pending.insert(rid.clone(), tx);

        let payload = build_generate_req(
            &rid,
            &input_ids,
            &sp,
            stream,
            &self.config.tokenizer_ipc_name,
        );

        if self.sched_tx.send(payload).await.is_err() {
            error!("Scheduler channel closed; dropping request {rid}");
            self.pending.remove(&rid);
        }

        (rid, rx)
    }

    /// Send an abort request for a pending RID.
    pub async fn abort(&self, rid: &str) {
        self.pending.remove(rid);
        let payload = build_abort_req(rid);
        self.sched_tx.send(payload).await.ok();
    }
}

// ─────────────────────── dispatch loop (recv side) ──────────────────────────

/// Reads `BatchStrOutput` messages from the ZMQ recv channel and routes each
/// `(rid, text, finished_reason)` tuple to the correct pending request sender.
async fn dispatch_loop(
    mut raw_rx: mpsc::Receiver<Vec<Vec<u8>>>,
    pending: Arc<DashMap<String, mpsc::UnboundedSender<ResponseChunk>>>,
) {
    while let Some(parts) = raw_rx.recv().await {
        let data = match validate_zmq_parts(parts) {
            Ok(d) => d,
            Err(e) => {
                error!("TM recv: {e}");
                continue;
            }
        };
        let val = match msgpack_decode(&data) {
            Ok(v) => v,
            Err(e) => {
                error!("TM msgpack decode error: {e}");
                continue;
            }
        };
        let map = match val {
            Value::Map(m) => m,
            _ => {
                error!("TM: expected map, got {:?}", val);
                continue;
            }
        };

        let msg_type = read_type_tag(&map).unwrap_or_default();
        match msg_type.as_str() {
            "BatchStrOutput" => dispatch_batch_str_output(&map, &pending),
            "BatchEmbeddingOutput" => {
                dispatch_batch_embedding_output(&map, &pending);
            }
            other => {
                warn!("TM: unknown message type '{other}'");
            }
        }
    }
}

/// Like `dispatch_loop` but reads raw msgpack bytes from an in-process channel
/// instead of ZMQ multipart frames — no magic-byte stripping needed.
async fn dispatch_loop_channel(
    mut rx: mpsc::UnboundedReceiver<Vec<u8>>,
    pending: Arc<DashMap<String, mpsc::UnboundedSender<ResponseChunk>>>,
) {
    while let Some(data) = rx.recv().await {
        let val = match msgpack_decode(&data) {
            Ok(v) => v,
            Err(e) => { error!("TM msgpack decode error: {e}"); continue; }
        };
        let map = match val {
            Value::Map(m) => m,
            _ => { error!("TM: expected map, got {:?}", val); continue; }
        };
        let msg_type = read_type_tag(&map).unwrap_or_default();
        match msg_type.as_str() {
            "BatchStrOutput" => dispatch_batch_str_output(&map, &pending),
            "BatchEmbeddingOutput" => dispatch_batch_embedding_output(&map, &pending),
            other => warn!("TM: unknown message type '{other}'"),
        }
    }
}

fn dispatch_batch_str_output(
    map: &[(Value, Value)],
    pending: &DashMap<String, mpsc::UnboundedSender<ResponseChunk>>,
) {
    let rids = extract_str_vec(map, "rids");
    let output_strs = extract_str_vec(map, "output_strs");
    let finished_reasons = extract_finished_reasons(map);

    let mut done_rids: Vec<String> = Vec::new();

    for (i, rid) in rids.iter().enumerate() {
        let text = output_strs.get(i).cloned().unwrap_or_default();
        let finished = finished_reasons.get(i).and_then(|x| x.as_ref());

        let chunk = match finished {
            Some(fr) => {
                let reason = extract_finish_reason_str(fr);
                done_rids.push(rid.clone());
                ResponseChunk::Done { text, finish_reason: reason }
            }
            None => ResponseChunk::Delta { text },
        };

        if let Some(entry) = pending.get(rid) {
            entry.send(chunk).ok();
        }
    }

    for rid in done_rids {
        pending.remove(&rid);
    }
}

fn dispatch_batch_embedding_output(
    map: &[(Value, Value)],
    pending: &DashMap<String, mpsc::UnboundedSender<ResponseChunk>>,
) {
    let rids = extract_str_vec(map, "rids");
    for rid in &rids {
        if let Some(entry) = pending.get(rid) {
            entry.send(ResponseChunk::Done {
                text: String::new(),
                finish_reason: "stop".to_string(),
            }).ok();
        }
        pending.remove(rid);
    }
}

// ──────────────────────────── map field readers ──────────────────────────────

fn extract_str_vec(map: &[(Value, Value)], key: &str) -> Vec<String> {
    map_get(map, key)
        .and_then(|v| match v {
            Value::Array(arr) => Some(
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect(),
            ),
            Value::Nil => Some(vec![]),
            _ => None,
        })
        .unwrap_or_default()
}

fn extract_finished_reasons(map: &[(Value, Value)]) -> Vec<Option<Value>> {
    map_get(map, "finished_reasons")
        .and_then(|v| match v {
            Value::Array(arr) => Some(
                arr.iter()
                    .map(|x| match x {
                        Value::Nil => None,
                        other => Some(other.clone()),
                    })
                    .collect(),
            ),
            _ => None,
        })
        .unwrap_or_default()
}

fn extract_finish_reason_str(fr: &Value) -> String {
    // Scheduler encodes finish reason as {"type": "...", "matched": ...}
    match fr {
        Value::Map(m) => {
            let type_str = map_get(m, "type")
                .and_then(|v| v.as_str())
                .unwrap_or("stop");
            match type_str {
                "length" => "length".to_string(),
                _ => "stop".to_string(),
            }
        }
        Value::String(s) => s.as_str().unwrap_or("stop").to_string(),
        _ => "stop".to_string(),
    }
}

// ─────────────────── tokenizer + chat template loading ──────────────────────

fn load_tokenizer_and_template(
    path: &str,
    skip: bool,
) -> (Option<Arc<Tokenizer>>, Option<String>, Option<String>, Option<String>) {
    let (chat_template, eos_token, bos_token) = load_template_config(path);

    if skip || path.is_empty() {
        return (None, chat_template, eos_token, bos_token);
    }

    let json_path = if Path::new(path).is_dir() {
        format!("{}/tokenizer.json", path)
    } else {
        path.to_string()
    };

    let tok = if Path::new(&json_path).exists() {
        info!("Loading tokenizer from: {json_path}");
        match Tokenizer::from_file(&json_path) {
            Ok(t) => {
                info!("Tokenizer loaded.");
                Some(Arc::new(t))
            }
            Err(e) => {
                warn!("Failed loading tokenizer from file ({e})");
                try_load_from_pretrained(path)
            }
        }
    } else {
        try_load_from_pretrained(path)
    };

    (tok, chat_template, eos_token, bos_token)
}

fn try_load_from_pretrained(path: &str) -> Option<Arc<Tokenizer>> {
    info!("Loading tokenizer from HuggingFace Hub: {path}");
    match Tokenizer::from_pretrained(path, None) {
        Ok(t) => {
            info!("Tokenizer loaded from Hub.");
            Some(Arc::new(t))
        }
        Err(e) => {
            error!("Failed to load tokenizer: {e}");
            None
        }
    }
}

fn load_template_config(path: &str) -> (Option<String>, Option<String>, Option<String>) {
    let config_path = if Path::new(path).is_dir() {
        format!("{}/tokenizer_config.json", path)
    } else if let Some(dir) = Path::new(path).parent() {
        format!("{}/tokenizer_config.json", dir.display())
    } else {
        return (None, None, None);
    };

    let content = match std::fs::read_to_string(&config_path) {
        Ok(s) => s,
        Err(_) => return (None, None, None),
    };

    let json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            warn!("Failed to parse tokenizer_config.json: {e}");
            return (None, None, None);
        }
    };

    let chat_template = json
        .get("chat_template")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let eos_token = json
        .get("eos_token")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let bos_token = json
        .get("bos_token")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    (chat_template, eos_token, bos_token)
}

// ────────────────────────── chat template rendering ─────────────────────────

fn apply_jinja_template(
    template: &str,
    messages: &[(&str, &str)],
    add_generation_prompt: bool,
    eos_token: Option<&str>,
    bos_token: Option<&str>,
) -> Result<String, minijinja::Error> {
    let mut env = Environment::new();

    // Register common custom functions used in HuggingFace chat templates.
    env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
        Err(minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            msg,
        ))
    });

    env.add_template("chat", template)?;
    let tmpl = env.get_template("chat")?;

    // Build messages list as minijinja values.
    let msgs: Vec<JValue> = messages
        .iter()
        .map(|(role, content)| {
            let mut m = HashMap::new();
            m.insert("role", *role);
            m.insert("content", *content);
            JValue::from_serialize(&m)
        })
        .collect();

    let ctx = minijinja::context! {
        messages => msgs,
        add_generation_prompt => add_generation_prompt,
        eos_token => eos_token.unwrap_or(""),
        bos_token => bos_token.unwrap_or(""),
    };

    tmpl.render(ctx)
}

fn simple_format_messages(messages: &[(&str, &str)], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        match *role {
            "system" => {
                out.push_str(content);
                out.push('\n');
            }
            "user" => {
                out.push_str("User: ");
                out.push_str(content);
                out.push('\n');
            }
            "assistant" => {
                out.push_str("Assistant: ");
                out.push_str(content);
                out.push('\n');
            }
            _ => {
                out.push_str(content);
                out.push('\n');
            }
        }
    }
    if add_generation_prompt {
        out.push_str("Assistant: ");
    }
    out
}

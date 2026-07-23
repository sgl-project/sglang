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
#![allow(dead_code)] // TODO: remove when the consumer PR lands

pub mod channels;
pub mod ring;

/// A pipeline stage that owns its channel handles + config and runs a blocking
/// loop until its inbox closes. Lets the runtime spawn stages uniformly via
/// [`spawn_stage`] / [`spawn_pool`] instead of free `run_*` functions with
/// positional handles. Implemented by every CPU-bound worker and TM router.
pub trait Runnable: Send + 'static {
    fn run(self);
}

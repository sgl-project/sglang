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

mod config;
mod runnable;

// Re-export so stages keep importing `crate::runtime::Runnable`.

//! TokenizerManager — owns the request lifecycle across two isolated threads:
//!
//!   * [`ingress`] — drives the ingress FSM (Received → Validating →
//!     Normalizing → {Tokenizing | PreSendValidating}) and pushes tokenized requests to the
//!     scheduler ring.
//!   * [`egress`] — drains the scheduler-output ring and routes each chunk to
//!     the owning detokenizer shard.
//!
//! The two run on separate pinned threads with no shared state, connected to
//! the rest of the pipeline only through `flume` channels: [`wiring::TmEvent`] into
//! the ingress loop, [`wiring::Senders`] fanning out to the pools.

pub mod channel;
pub mod detokenizer;
pub mod egress;
pub mod ingress;
pub mod tokenizer;
pub mod wiring;

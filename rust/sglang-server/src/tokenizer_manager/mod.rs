//! TokenizerManager — owns the request lifecycle across two isolated threads:
//!
//!   * [`ingress`] — drives the ingress FSM (Received → Validating →
//!     {Tokenizing | Queued}) and pushes tokenized requests to the scheduler
//!     ring.
//!   * [`egress`] — drains the scheduler-output ring and routes each chunk to
//!     the owning detokenizer shard.
//!
//! The two run on separate pinned threads with no shared state, connected to
//! the rest of the pipeline only through `runtime::channels`.

mod egress;
mod ingress;

pub use egress::Egress;
pub use ingress::Ingress;

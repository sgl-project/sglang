//! Configuration and multimodal handles for scheduler intake.

use crate::message::config::ServerArgs;
use crate::message::request::MmRequest;

/// The intake side of the MM path.
#[derive(Clone)]
pub struct Mm {
    /// Whether the model is multimodal. When false, mm fields are silently
    /// ignored, as the Python `TokenizerManager` does with `mm_processor is
    /// None`.
    pub enabled: bool,
    /// → MM worker pool (spawned via `Server.start_mm_workers`).
    pub tx: flume::Sender<MmRequest>,
    /// Results sidecar. Purged here when a late result arrives for a request
    /// that is no longer parked; otherwise it would leak, since only the
    /// scheduler drain pops entries.
    pub sidecar: crate::multi_modality::sidecar::Sidecar,
}

/// Resolved once at boot from the scheduler's `server_args`.
#[derive(Clone, Debug)]
pub struct Limits {
    /// Token-ids-in mode: a generate request must arrive already tokenized.
    pub skip_tokenizer_init: bool,
    /// `model_config.vocab_size`; bounds client-supplied token ids. A required
    /// field of the `ServerArgs` schema, so intake can check unconditionally.
    pub vocab_size: u64,
    /// `model_config.context_len`, the ceiling for input + `max_new_tokens`.
    pub context_len: u64,
    /// Output slots reserved on top of the input (eagle draft tokens).
    pub num_reserved_tokens: u64,
    /// Clamp `max_new_tokens` to what fits instead of rejecting the request.
    pub allow_auto_truncate: bool,
    /// Whether the server can produce hidden states at all.
    pub enable_return_hidden_states: bool,
}

impl From<&ServerArgs> for Limits {
    fn from(sa: &ServerArgs) -> Self {
        Self {
            skip_tokenizer_init: sa.skip_tokenizer_init,
            vocab_size: sa.model_config.vocab_size,
            context_len: sa.model_config.context_len,
            num_reserved_tokens: sa.num_reserved_tokens,
            allow_auto_truncate: sa.allow_auto_truncate,
            enable_return_hidden_states: sa.enable_return_hidden_states,
        }
    }
}

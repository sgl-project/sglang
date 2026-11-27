//! Configuration and multimodal handles for scheduler intake.

use crate::message::config::{HiddenStatesMode, ServerArgs};
use crate::message::request::Request;

/// Dispatches multimodal requests onto the MM worker channel.
#[derive(Clone)]
pub struct MmDispatch {
    /// Whether the model is multimodal. When false, mm fields are silently
    /// ignored, as the Python `TokenizerManager` does with `mm_processor is
    /// None`.
    pub enabled: bool,
    /// -> MM worker pool (spawned via `Server.start_mm_workers`); carries the
    /// whole request, like the tokenizer pool's edge.
    pub tx: flume::Sender<Request>,
}

/// Resolved once at boot from the scheduler's `server_args`.
#[derive(Clone, Debug)]
pub struct Limits {
    /// Token-ids-in mode: a generate request must arrive already tokenized.
    pub skip_tokenizer_init: bool,
    /// `model_config.vocab_size`; bounds client-supplied token ids. A required
    /// field of the `ServerArgs` schema, so intake can check unconditionally.
    pub vocab_size: u64,
    /// The scheduler's `max_req_len`: input, reserved slots and `max_new_tokens` together.
    pub max_req_len: u64,
    /// The scheduler's `max_req_input_len`, inclusive; reserved slots are already out of it.
    pub max_req_input_len: u64,
    /// Output slots reserved on top of the input (eagle draft tokens).
    pub num_reserved_tokens: u64,
    /// Truncate a text input and clamp `max_new_tokens` to what fits instead
    /// of rejecting the request.
    pub allow_auto_truncate: bool,
    /// The most a request's `return_hidden_states` may ask for.
    pub return_hidden_states_mode: HiddenStatesMode,
}

impl From<&ServerArgs> for Limits {
    fn from(sa: &ServerArgs) -> Self {
        Self {
            skip_tokenizer_init: sa.skip_tokenizer_init,
            vocab_size: sa.model_config.vocab_size,
            max_req_len: sa.max_req_len,
            max_req_input_len: sa.max_req_input_len,
            num_reserved_tokens: sa.num_reserved_tokens,
            allow_auto_truncate: sa.allow_auto_truncate,
            return_hidden_states_mode: sa.hidden_states_mode(),
        }
    }
}

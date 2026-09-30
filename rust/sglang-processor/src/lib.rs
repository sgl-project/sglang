//! Shared SGLang wrappers over Dynamo's frontend crates.
//!
//! The library loads and applies model chat templates, tokenizes rendered
//! prompts, derives tool-call constraints, and parses generated output into
//! chat events, using `dynamo-renderer`, `dynamo-tokenizers`, and
//! `dynamo-parsers`. Hosts own request types, sampling, validation,
//! transport, and runtime.

mod error;
mod postprocessing;
mod preprocessing;

pub use error::ProcessorError;
pub use postprocessing::{
    ChatEvent, ChatFinishReason, ChatResponseProcessor, ChatToolCallDelta, DecodedChatEvent,
};
pub use preprocessing::{
    ChatConfig, ChatPreprocessor, ChatRequest, ReasoningEffort, RenderedChat, ToolConstraint,
};
pub use preprocessing::{
    DynamoTokenizer, TextTokenizer, load_tokenizer, resolve_chat_template_file, resolve_model_file,
    resolve_tokenizer_file,
};

// Dynamo crates whose types appear in this crate's public API. Hosts use these
// re-exports so their Dynamo versions always match the processor's.
pub use dynamo_protocols;
pub use dynamo_renderer;
pub use dynamo_tokenizers;

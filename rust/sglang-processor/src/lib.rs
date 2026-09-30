//! Shared SGLang wrappers over Dynamo's frontend crates.
//!
//! The library loads and applies model chat templates, tokenizes rendered
//! prompts, and parses generated output into chat events, using
//! `dynamo-renderer`, `dynamo-tokenizers`, and `dynamo-parsers`. Hosts own
//! request types, sampling, validation, transport, and runtime.

mod error;
mod parsers;
mod postprocessing;
mod preprocessing;

pub use error::ProcessorError;
pub use parsers::dynamo_parser_name;
pub use postprocessing::{
    ChatEvent, ChatFinishReason, ChatResponseProcessor, ChatToolCallDelta, DecodedChatEvent,
};
pub use preprocessing::{
    ChatFormatter, ChatTemplateConfig, DeepSeekV4Profile, OneOrMany, TemplateError,
    ThinkingTemplates, load_chat_formatter, load_chat_support,
};
pub use preprocessing::{
    DynamoTokenizer, TextTokenizer, load_tokenizer, resolve_chat_template_file, resolve_model_file,
    resolve_tokenizer_file,
};
#[cfg(feature = "test-support")]
pub use preprocessing::{test_hugging_face_formatter, test_hugging_face_formatter_from_config};

// Dynamo crates whose types appear in this crate's public API. Hosts use these
// re-exports so their Dynamo versions always match the processor's.
pub use dynamo_protocols;
pub use dynamo_renderer;
pub use dynamo_tokenizers;

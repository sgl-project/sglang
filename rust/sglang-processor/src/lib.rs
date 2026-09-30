//! Shared SGLang wrappers over Dynamo's frontend crates.
//!
//! The library resolves model files, loads and applies chat templates,
//! tokenizes rendered prompts, and parses generated output into chat events using
//! `dynamo-renderer`, `dynamo-tokenizers`, and `dynamo-parsers`. Hosts own
//! request types, sampling, validation, transport, and runtime.

mod error;
mod model_files;
mod output;
mod template;
mod tokenizer;
mod tool_parser;

pub use error::ProcessorError;
pub use model_files::{resolve_model_file, resolve_tokenizer_file};
pub use output::{
    ChatEvent, ChatFinishReason, ChatResponseProcessor, ChatToolCallDelta, DecodedChatEvent,
};
pub use template::{
    ChatFormatter, ChatFormatterOptions, DeepSeekV4Profile, OneOrMany, TemplateError,
    ThinkingTemplates, load_chat_formatter, select_chat_formatter,
};
pub use tokenizer::{DynamoTokenizer, TextTokenizer, load_tokenizer};
pub use tool_parser::dynamo_tool_parser_name;

// Dynamo crates whose types appear in this crate's public API. Hosts use these
// re-exports so their Dynamo versions always match the processor's.
pub use dynamo_protocols;
pub use dynamo_renderer;
pub use dynamo_tokenizers;

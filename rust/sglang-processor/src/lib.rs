//! Shared SGLang wrappers over Dynamo's frontend crates.
//!
//! The library resolves model files, loads and applies chat templates,
//! tokenizes rendered prompts, and parses generated output into chat events using
//! `dynamo-renderer`, `dynamo-tokenizers`, and `dynamo-parsers`. Hosts own
//! request types, sampling, validation, transport, and runtime.

mod error;
mod model_files;
#[cfg(feature = "openai")]
pub mod openai;
#[cfg(feature = "parser")]
mod parser;
#[cfg(feature = "render")]
mod render;
#[cfg(any(feature = "parser", feature = "openai"))]
mod think;
#[cfg(feature = "tokenizer")]
mod tokenizer;

pub use error::ProcessorError;
pub use model_files::{resolve_model_file, resolve_tokenizer_file};
#[cfg(feature = "parser")]
pub use parser::{
    ChatEvent, ChatFinishReason, ChatResponseProcessor, ChatToolCallDelta, DecodedChatEvent,
    ReasoningStreamSplitter, ToolConstraint, chat_tool_definitions, dynamo_tool_choice,
    dynamo_tool_parser_name, split_reasoning, tool_constraint,
};
#[cfg(feature = "render")]
pub use render::{
    ChatFormatter, ChatFormatterOptions, DeepSeekV4Profile, OneOrMany, TemplateError,
    ThinkingTemplates, load_chat_formatter, requested_effort, requested_thinking,
    select_chat_formatter,
};
#[cfg(any(feature = "parser", feature = "openai"))]
pub use think::ReasoningOptions;
#[cfg(feature = "tokenizer")]
pub use tokenizer::{DynamoTokenizer, TextTokenizer, load_tokenizer};

// Dynamo crates whose types appear in this crate's public API. Hosts use these
// re-exports so their Dynamo versions always match the processor's.
#[cfg(any(feature = "parser", feature = "render"))]
pub use dynamo_protocols;
#[cfg(feature = "render")]
pub use dynamo_renderer;
#[cfg(feature = "tokenizer")]
pub use dynamo_tokenizers;

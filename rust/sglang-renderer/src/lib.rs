//! Engine-free OpenAI request rendering for SGLang.
//!
//! This crate owns request lowering, chat templating, tokenization and the
//! prepared token-in contract. It deliberately has no HTTP, gRPC, PyO3,
//! scheduler or GPU runtime dependency.

pub mod config;
pub mod error;
pub mod openai;
pub mod output;
pub mod request;
pub mod sampling;
pub mod service;
pub mod template;
pub mod tokenizer;
pub mod types;

mod regex;

pub use config::{RendererConfig, RendererLimits, SamplingDefaults};
pub use error::{RendererError, RendererErrorKind};
pub use output::{
    ChatEvent, ChatFinishReason, ChatOutputError, ChatOutputInput, ChatOutputItem,
    ChatOutputProcessor, ChatToolCall, ChatToolCallDelta, DecodedChatEvent, ParsedChatOutput,
};
pub use request::{PreparedGenerateRequest, PreparedSamplingParams, TextCompletionRequest};
pub use sampling::SamplingParams;
pub use service::{LoweredChat, RendererService, RequestLowerer, TokenizationBackend};
pub use template::ChatFormatter;
pub use tokenizer::{
    DynamoTokenizer, TextTokenizer, check_completion_token_budget, check_total_tokens,
    load_tokenizer, prepare_direct_request, resolve_model_file, tokenize_text_completion,
    validate_completion_fields, validate_request,
};
pub use types::{OneOrMany, OneOrManyItem, TokenIds};

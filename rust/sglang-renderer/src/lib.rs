//! Engine-free OpenAI request rendering for SGLang.
//!
//! This crate owns OpenAI protocol lowering, chat templating, tokenization,
//! response interpretation, and the prepared token-in contract. Its optional
//! HTTP frontend sends prepared token requests to SGLang's `/generate`
//! endpoint, so the crate has no PyO3, scheduler, or GPU runtime dependency.

pub mod chat;
pub mod config;
pub mod error;
pub mod generation;
#[cfg(feature = "http")]
pub mod http;
pub mod output;
pub mod protocol;
pub mod request;
pub mod sampling;
pub mod service;
pub mod template;
pub mod tokenizer;
pub mod types;

mod regex;

pub use chat::{ChatPreprocessor, ChatRequest, LoweredChat};
pub use config::{RendererConfig, RendererLimits, SamplingDefaults};
pub use error::{RendererError, RendererErrorKind};
pub use generation::{
    FrontendError, GenerationEvent, GenerationFinishReason, GenerationOutput,
    GenerationOutputExtras, GenerationStream, MatchedStop,
};
pub use output::{
    ChatEvent, ChatFinishReason, ChatResponseError, ChatResponseInput, ChatResponseItem,
    ChatResponseProcessor, ChatToolCall, ChatToolCallDelta, DecodedChatEvent, ParsedChatChoice,
};
pub use request::{
    GenerationOptions, PreparedGenerateRequest, PreparedSamplingParams, TextPrompt, TextRequest,
    TokenIdsRequest,
};
pub use sampling::SamplingParams;
pub use service::{OpenAIRequestLowerer, RendererService, TokenizationBackend};
pub use template::ChatFormatter;
pub use tokenizer::{
    DynamoTokenizer, NoTokenizer, PooledTokenizer, TextTokenizer, check_completion_token_budget,
    check_total_tokens, load_tokenizer, prepare_direct_request, resolve_chat_template_file,
    resolve_model_file, resolve_tokenizer_file, tokenize_text_prompt, tokenize_text_request,
    validate_completion_fields, validate_text_request,
};
pub use types::{OneOrMany, OneOrManyItem, TokenIds};

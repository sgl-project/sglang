//! Engine-free OpenAI request rendering for SGLang.
//!
//! This crate owns OpenAI protocol lowering, chat templating, tokenization and the
//! prepared token-in contract. It deliberately has no HTTP, gRPC, PyO3,
//! scheduler or GPU runtime dependency.

pub mod config;
pub mod error;
#[cfg(feature = "http")]
pub mod http;
pub mod inference;
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
pub use inference::{
    FrontendError, GenerationEvent, GenerationFinishReason, GenerationOutput,
    GenerationOutputExtras, GenerationSubmission, InferenceBackend, InferenceSession, MatchedStop,
};
pub use openai::LoweredChat;
pub use output::{
    ChatEvent, ChatFinishReason, ChatResponseError, ChatResponseInput, ChatResponseItem,
    ChatResponseProcessor, ChatToolCall, ChatToolCallDelta, DecodedChatEvent, ParsedChatChoice,
};
pub use request::{
    GenerationInput, GenerationOptions, PreparedGenerateRequest, PreparedSamplingParams,
    TextRequest, TokenIdsRequest,
};
pub use sampling::SamplingParams;
pub use service::{OpenAIRequestLowerer, RendererService, TokenizationBackend};
pub use template::ChatFormatter;
pub use tokenizer::{
    DynamoTokenizer, TextTokenizer, check_completion_token_budget, check_total_tokens,
    load_tokenizer, prepare_direct_request, resolve_model_file, tokenize_text_prompt,
    tokenize_text_request, validate_completion_fields, validate_generation_input,
};
pub use types::{OneOrMany, OneOrManyItem, TokenIds};

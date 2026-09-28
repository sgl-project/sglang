//! Reusable SGLang rendering, tokenization, and output parsing.
//! Protocol handling lives in sglang-frontend; applications own their runtimes.

mod config;
mod error;
mod postprocessing;
mod preprocessing;
mod types;

pub use config::{RendererConfig, RendererLimits, SamplingDefaults};
pub use error::{
    RendererError, RendererErrorKind, ResponseError, ResponseErrorKind, UpstreamErrorCode,
};
pub use postprocessing::{
    ChatEvent, ChatFinishReason, ChatResponseProcessor, ChatToolCallDelta, DecodedChatEvent,
};
pub(crate) use preprocessing::ChatFormatter;
pub use preprocessing::{
    ChatPreprocessor, ChatRequest, DynamoTokenizer, LoweredChat, PreparedChat, ReasoningEffort,
    RendererService, SamplingParams, TextTokenizer, load_tokenizer,
};
pub use preprocessing::{
    GenerateRequest, GenerateRequestMetadata, GenerateSamplingParams, GenerationOptions,
    TextRequest, TokenIdsRequest,
};
pub use types::{OneOrMany, TokenIds};

pub use preprocessing::{
    GenerateRequestIdentity, TextRequestGroup, resolve_model_file, resolve_tokenizer_file,
};

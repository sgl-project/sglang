//! Reusable request preprocessing for SGLang over the Dynamo frontend crates.
//!
//! The library renders normalized chat requests, lowers textual completions,
//! tokenizes prompts, and produces the token-in contract consumed by SGLang.
//! It also parses generated output into chat events. Hosts own their protocol
//! handling, transport, and runtime.

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
pub(crate) use preprocessing::{ChatPreprocessor, LoweredChat};
pub use preprocessing::{
    ChatRequest, DynamoTokenizer, PreparedChat, ReasoningEffort, RendererService, SamplingParams,
    TextTokenizer, load_tokenizer,
};
pub use preprocessing::{
    GenerateRequest, GenerateRequestMetadata, GenerateSamplingParams, GenerationOptions,
    TextRequest, TokenIdsRequest,
};
pub use preprocessing::{
    GenerateRequestIdentity, TextRequestGroup, resolve_model_file, resolve_tokenizer_file,
};
pub use types::{OneOrMany, TokenIds};

// Dynamo crates whose types appear in this crate's public API. Hosts use these
// re-exports so their Dynamo versions always match the processor's.
pub use dynamo_protocols;
pub use dynamo_renderer;
pub use dynamo_tokenizers;

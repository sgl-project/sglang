//! Request processing from protocol-neutral inputs to token-only generation requests.

mod chat;
mod regex;
mod request;
mod sampling;
mod service;
mod template;
mod tokenizer;

pub(crate) use chat::{ChatPreprocessor, LoweredChat, dynamo_parser_name};
pub use chat::{ChatRequest, ReasoningEffort};
pub use request::{
    GenerateRequest, GenerateRequestMetadata, GenerateSamplingParams, GenerationOptions,
    TextRequest, TokenIdsRequest,
};
pub(crate) use request::{GenerateRequestIdentity, TextRequestGroup};
pub use sampling::SamplingParams;
#[cfg(feature = "http")]
pub(crate) use sampling::SamplingParamsOverrides;
pub use service::{PreparedChat, RendererService};
pub(crate) use template::ChatFormatter;
#[cfg(all(test, feature = "http"))]
pub(crate) fn load_test_chat_formatter(name: &str) -> ChatFormatter {
    template::load_chat_formatter(None, None, Some(name)).unwrap()
}
pub use tokenizer::{DynamoTokenizer, TextTokenizer, load_tokenizer};
#[cfg(feature = "http")]
pub(crate) use tokenizer::{resolve_model_file, resolve_tokenizer_file};

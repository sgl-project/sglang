//! Request processing from protocol-neutral inputs to token-only generation requests.

mod chat;
mod regex;
mod request;
mod sampling;
mod service;
mod tokenizer;

pub(crate) use chat::{ChatPreprocessor, LoweredChat};
pub use chat::{ChatRequest, ReasoningEffort};
pub use request::{
    GenerateRequest, GenerateRequestMetadata, GenerateSamplingParams, GenerationOptions,
    TextRequest, TokenIdsRequest,
};
pub(crate) use request::{GenerateRequestIdentity, TextRequestGroup};
pub use sampling::SamplingParams;
pub use service::{PreparedChat, RendererService};
pub(crate) use sglang_processor::ChatFormatter;
#[cfg(test)]
pub(crate) fn load_test_chat_formatter(name: &str) -> ChatFormatter {
    sglang_processor::load_chat_formatter(None, None, None, Some(name)).unwrap()
}
pub use sglang_processor::{DynamoTokenizer, TextTokenizer, load_tokenizer};
pub(crate) use sglang_processor::{resolve_model_file, resolve_tokenizer_file};

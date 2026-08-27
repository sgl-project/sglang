//! Engine-free OpenAI request rendering for SGLang.
//!
//! This crate owns request lowering, chat templating, tokenization and the
//! prepared token-in contract. It deliberately has no HTTP, gRPC, PyO3,
//! scheduler or GPU runtime dependency.

pub mod config;
pub mod error;
pub mod request;
pub mod sampling;
pub mod tokenizer;
pub mod types;

mod regex;

pub use config::{RendererConfig, RendererLimits, SamplingDefaults};
pub use error::{RendererError, RendererErrorKind};
pub use request::{PreparedGenerateRequest, PreparedSamplingParams, RendererRequest};
pub use sampling::SamplingParams;
pub use tokenizer::{DynamoTokenizer, TextTokenizer, load_tokenizer, resolve_model_file};
pub use types::{OneOrMany, TokenIds};

//! Shared SGLang request DTOs and their validation.
//!
//! Leaf crate: plain data plus the `SamplingParams` normalize/verify pipeline.
//! No service behavior, no transport, no tokenizer — both `sglang-renderer` and
//! `sglang-server` depend on this crate instead of on each other for types.

mod error;
mod regex;
mod sampling;
mod types;

pub use error::ValidationError;
pub use regex::RegexPattern;
pub use sampling::{CustomParamValue, JsonScalar, SamplingParams};
pub use types::{OneOrMany, OneOrManyItem, TokenIds};

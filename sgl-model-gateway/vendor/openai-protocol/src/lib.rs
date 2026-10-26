// Protocol definitions and validation for various LLM APIs
// This module provides a structured approach to handling different API protocols

/// Default model identifier used when no model is specified.
///
/// This constant should be used instead of hardcoded "unknown" strings
/// throughout the codebase for consistency.
pub const UNKNOWN_MODEL_ID: &str = "unknown";

pub mod builders;
pub mod chat;
pub mod classify;
pub mod common;
pub mod completion;
pub mod embedding;
pub mod event_types;
pub mod generate;
pub mod messages;
pub mod parser;
pub mod rerank;
pub mod responses;
pub mod sampling_params;
pub mod tokenize;
pub mod validated;
pub mod worker_spec;

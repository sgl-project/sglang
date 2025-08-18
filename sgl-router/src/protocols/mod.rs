// Protocol definitions and validation for various LLM APIs
// This module provides a structured approach to handling different API protocols

pub mod common;
pub mod generate;
pub mod openai;

// Re-export common types
pub use common::{default_true, GenerationRequest, LoRAPath, StringOrArray};

// Re-export generate API types
pub use generate::{GenerateParameters, GenerateRequest, InputIds, SamplingParams};

// Re-export OpenAI types at the protocols level for backward compatibility
pub use openai::*;

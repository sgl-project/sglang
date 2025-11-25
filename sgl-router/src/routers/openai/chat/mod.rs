//! Chat pipeline module
//!
//! Lightweight pipeline for /v1/chat/completions endpoint.
//! Pure proxy with minimal overhead (4 stages).

pub mod context;
pub mod pipeline;
pub mod stages;

// Re-export main types
pub use context::{
    ChatDependencies, ChatProcessingState, ChatRequestContext, ChatRequestInput,
    DiscoveryOutput, PayloadOutput, ValidationOutput,
};
pub use pipeline::ChatPipeline;

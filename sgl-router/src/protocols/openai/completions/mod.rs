// Completions API module (v1/completions)

pub mod request;
pub mod response;

// Re-export main types for convenience
pub use request::CompletionRequest;
pub use response::{
    CompletionChoice, CompletionResponse, CompletionStreamChoice, CompletionStreamResponse,
};

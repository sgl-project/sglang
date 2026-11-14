//! Builders for Chat Completion API response types

pub mod response;
pub mod stream_response;

pub use response::ChatCompletionResponseBuilder;
pub use stream_response::ChatCompletionStreamResponseBuilder;

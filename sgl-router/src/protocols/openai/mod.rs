// OpenAI protocol module
// This module contains all OpenAI API-compatible types and future validation logic

pub mod chat;
pub mod common;
pub mod completions;
pub mod errors;

// Re-export all types for backward compatibility
pub use chat::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse,
    ChatMessage, ChatMessageDelta, ChatStreamChoice, ContentPart, Function, FunctionCall,
    FunctionCallDelta, FunctionCallResponse, FunctionChoice, ImageUrl, JsonSchemaFormat,
    ResponseFormat, Tool, ToolCall, ToolCallDelta, ToolChoice, UserMessageContent,
};
pub use common::{
    ChatLogProbs, ChatLogProbsContent, CompletionTokensDetails, LogProbs, StreamOptions,
    TopLogProb, Usage,
};
pub use completions::{
    CompletionChoice, CompletionRequest, CompletionResponse, CompletionStreamChoice,
    CompletionStreamResponse,
};
pub use errors::{ErrorDetail, ErrorResponse};

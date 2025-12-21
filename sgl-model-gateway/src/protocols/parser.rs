use serde::Deserialize;

use crate::protocols::common::Tool;

/// Request to parse function calls from model output text
#[derive(Deserialize)]
pub struct ParseFunctionCallRequest {
    /// The text to parse for function calls
    pub text: String,
    /// The parser type/name to use for parsing (e.g., "json", "pythonic")
    pub tool_call_parser: String,
    /// The list of available tools that the model can call
    pub tools: Vec<Tool>,
}

/// Request to separate reasoning from normal text in model output
#[derive(Deserialize)]
pub struct SeparateReasoningRequest {
    /// The text to parse for reasoning content
    pub text: String,
    /// The parser type/name to use for reasoning detection (e.g., "step3", "deepseek_r1")
    pub reasoning_parser: String,
}

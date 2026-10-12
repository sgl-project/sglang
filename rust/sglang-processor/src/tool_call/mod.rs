//! SGLang's tool-call detectors ported from Python (`function_call/`), for the
//! parser names where Dynamo's parsers split output differently. Free of
//! Dynamo, like `think`, so `openai` needs no `parser` feature.

mod models;

pub(crate) use self::models::tool_detector;

/// One parsed call: `ToolCallItem`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolCallItem {
    pub tool_index: i64,
    pub name: String,
    pub parameters: String,
}

/// `(normal_text, calls)`: `StreamingParseResult`.
pub(crate) type Parsed = (String, Vec<ToolCallItem>);

/// SGLang's `BaseFormatDetector`, as `FunctionCallParser` drives it.
pub(crate) trait ToolDetector: Send + Sync {
    fn has_tool_call(&self, text: &str) -> bool;
    /// `FunctionCallParser.parse_non_stream`.
    fn parse_non_stream(&self, text: &str) -> Parsed;
    /// `parse_stream_chunk`, plus `parse_stream_end` when `flush`.
    fn parse_stream(&mut self, text: &str, flush: bool) -> Parsed;
}

/// Python's `call_` id: 24 hex digits of a UUID4.
pub(crate) fn tool_call_id() -> String {
    format!("call_{}", &uuid::Uuid::new_v4().simple().to_string()[..24])
}

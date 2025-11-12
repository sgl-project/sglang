// Qwen3 specific reasoning parser.
// This parser starts with in_reasoning=false, requiring an explicit
// start token to enter reasoning mode.

use crate::reasoning_parser::{
    parsers::BaseReasoningParser,
    traits::{ParseError, ParserConfig, ParserResult, ReasoningParser},
};

/// Qwen3 reasoning parser.
///
/// This parser requires explicit <think> tokens to enter reasoning mode
/// (in_reasoning=false initially).
pub struct Qwen3Parser {
    base: BaseReasoningParser,
}

impl Qwen3Parser {
    /// Create a new Qwen3 parser.
    pub fn new() -> Self {
        let config = ParserConfig {
            think_start_token: "<think>".to_string(),
            think_end_token: "</think>".to_string(),
            stream_reasoning: true,
            max_buffer_size: 65536,
            initial_in_reasoning: false, // Requires explicit start token
        };

        Self {
            base: BaseReasoningParser::new(config).with_model_type("qwen3".to_string()),
        }
    }
}

impl Default for Qwen3Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for Qwen3Parser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        self.base.detect_and_parse_reasoning(text)
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        self.base.parse_reasoning_streaming_incremental(text)
    }

    fn reset(&mut self) {
        self.base.reset()
    }

    fn model_type(&self) -> &str {
        self.base.model_type()
    }

    fn is_in_reasoning(&self) -> bool {
        self.base.is_in_reasoning()
    }
}

/// QwenThinking parser - variant that assumes reasoning from start.
///
/// This is for qwen*thinking models that behave like DeepSeek-R1.
pub struct QwenThinkingParser {
    base: BaseReasoningParser,
}

impl QwenThinkingParser {
    /// Create a new QwenThinking parser.
    pub fn new() -> Self {
        let config = ParserConfig {
            think_start_token: "<think>".to_string(),
            think_end_token: "</think>".to_string(),
            stream_reasoning: true,
            max_buffer_size: 65536,
            initial_in_reasoning: true, // Assumes reasoning from start
        };

        Self {
            base: BaseReasoningParser::new(config).with_model_type("qwen_thinking".to_string()),
        }
    }
}

impl Default for QwenThinkingParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for QwenThinkingParser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        self.base.detect_and_parse_reasoning(text)
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        self.base.parse_reasoning_streaming_incremental(text)
    }

    fn reset(&mut self) {
        self.base.reset()
    }

    fn model_type(&self) -> &str {
        self.base.model_type()
    }

    fn is_in_reasoning(&self) -> bool {
        self.base.is_in_reasoning()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_initial_state() {
        let mut parser = Qwen3Parser::new();

        // Should NOT treat text as reasoning without start token
        let result = parser
            .detect_and_parse_reasoning("This is normal content")
            .unwrap();
        assert_eq!(result.normal_text, "This is normal content");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn test_qwen3_with_tokens() {
        let mut parser = Qwen3Parser::new();

        // Should extract reasoning with proper tokens
        let result = parser
            .detect_and_parse_reasoning("<think>reasoning</think>answer")
            .unwrap();
        assert_eq!(result.normal_text, "answer");
        assert_eq!(result.reasoning_text, "reasoning");
    }

    #[test]
    fn test_qwen_thinking_initial_state() {
        let mut parser = QwenThinkingParser::new();

        // Should treat text as reasoning even without start token
        let result = parser
            .detect_and_parse_reasoning("This is reasoning content")
            .unwrap();
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "This is reasoning content");
    }

    #[test]
    fn test_qwen3_streaming() {
        let mut parser = Qwen3Parser::new();

        // First chunk - normal text (no start token yet)
        let result1 = parser
            .parse_reasoning_streaming_incremental("normal text ")
            .unwrap();
        assert_eq!(result1.normal_text, "normal text ");
        assert_eq!(result1.reasoning_text, "");

        // Second chunk - enters reasoning
        let result2 = parser
            .parse_reasoning_streaming_incremental("<think>reasoning")
            .unwrap();
        assert_eq!(result2.normal_text, "");
        assert_eq!(result2.reasoning_text, "reasoning");
    }

    #[test]
    fn test_model_types() {
        let qwen3 = Qwen3Parser::new();
        assert_eq!(qwen3.model_type(), "qwen3");

        let qwen_thinking = QwenThinkingParser::new();
        assert_eq!(qwen_thinking.model_type(), "qwen_thinking");
    }
}

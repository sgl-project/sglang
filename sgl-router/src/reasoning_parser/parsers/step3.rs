// Step3 specific reasoning parser.
// Uses the same format as DeepSeek-R1 but has its own implementation for debugging.

use crate::reasoning_parser::{
    parsers::BaseReasoningParser,
    traits::{ParseError, ParserConfig, ParserResult, ReasoningParser},
};

/// Step3 reasoning parser.
///
/// This parser uses the same format as DeepSeek-R1 (<think>...</think>) but has
/// its own implementation for better debugging and potential future customization.
pub struct Step3Parser {
    base: BaseReasoningParser,
}

impl Step3Parser {
    /// Create a new Step3 parser.
    pub fn new() -> Self {
        let config = ParserConfig {
            think_start_token: "<think>".to_string(),
            think_end_token: "</think>".to_string(),
            stream_reasoning: true,
            max_buffer_size: 65536,
            initial_in_reasoning: true, // Assumes reasoning from start like DeepSeek-R1
        };

        Self {
            base: BaseReasoningParser::new(config).with_model_type("step3".to_string()),
        }
    }
}

impl Default for Step3Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for Step3Parser {
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
    fn test_step3_initial_state() {
        let mut parser = Step3Parser::new();

        // Should treat text as reasoning even without start token
        let result = parser
            .detect_and_parse_reasoning("This is reasoning content")
            .unwrap();
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "This is reasoning content");
    }

    #[test]
    fn test_step3_with_end_token() {
        let mut parser = Step3Parser::new();

        // Should handle text with end token
        let result = parser
            .detect_and_parse_reasoning("reasoning content</think>answer")
            .unwrap();
        assert_eq!(result.normal_text, "answer");
        assert_eq!(result.reasoning_text, "reasoning content");
    }

    #[test]
    fn test_step3_with_both_tokens() {
        let mut parser = Step3Parser::new();

        // Should handle both start and end tokens
        let result = parser
            .detect_and_parse_reasoning("<think>reasoning content</think>answer")
            .unwrap();
        assert_eq!(result.normal_text, "answer");
        assert_eq!(result.reasoning_text, "reasoning content");
    }

    #[test]
    fn test_step3_streaming() {
        let mut parser = Step3Parser::new();

        // First chunk - treated as reasoning (initial_in_reasoning=true)
        let result1 = parser
            .parse_reasoning_streaming_incremental("reasoning text ")
            .unwrap();
        assert_eq!(result1.normal_text, "");
        assert_eq!(result1.reasoning_text, "reasoning text ");

        // Second chunk - continues reasoning until end token
        let result2 = parser
            .parse_reasoning_streaming_incremental("more reasoning</think>answer")
            .unwrap();
        assert_eq!(result2.normal_text, "answer");
        assert_eq!(result2.reasoning_text, "more reasoning");
    }

    #[test]
    fn test_model_type() {
        let parser = Step3Parser::new();
        assert_eq!(parser.model_type(), "step3");
    }
}

// DeepSeek-R1 specific reasoning parser.
// This parser starts with in_reasoning=true, assuming all text is reasoning
// until an end token is encountered.

use crate::reasoning_parser::{
    parsers::BaseReasoningParser,
    traits::{ParseError, ParserConfig, ParserResult, ReasoningParser},
};

/// DeepSeek-R1 reasoning parser.
///
/// This parser assumes reasoning from the start of text (in_reasoning=true)
/// and uses <think> and </think> tokens.
pub struct DeepSeekR1Parser {
    base: BaseReasoningParser,
}

impl DeepSeekR1Parser {
    /// Create a new DeepSeek-R1 parser.
    pub fn new() -> Self {
        let config = ParserConfig {
            think_start_token: "<think>".to_string(),
            think_end_token: "</think>".to_string(),
            stream_reasoning: true,
            max_buffer_size: 65536,
            initial_in_reasoning: true, // Always starts with reasoning
        };

        Self {
            base: BaseReasoningParser::new(config).with_model_type("deepseek_r1".to_string()),
        }
    }
}

impl Default for DeepSeekR1Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for DeepSeekR1Parser {
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
    fn test_deepseek_r1_initial_state() {
        let mut parser = DeepSeekR1Parser::new();

        // Should treat text as reasoning even without start token
        let result = parser
            .detect_and_parse_reasoning("This is reasoning content")
            .unwrap();
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "This is reasoning content");
    }

    #[test]
    fn test_deepseek_r1_with_end_token() {
        let mut parser = DeepSeekR1Parser::new();

        // Should extract reasoning until end token
        let result = parser
            .detect_and_parse_reasoning("reasoning content</think>normal content")
            .unwrap();
        assert_eq!(result.normal_text, "normal content");
        assert_eq!(result.reasoning_text, "reasoning content");
    }

    #[test]
    fn test_deepseek_r1_streaming() {
        let mut parser = DeepSeekR1Parser::new();

        // First chunk - all reasoning
        let result1 = parser
            .parse_reasoning_streaming_incremental("thinking about")
            .unwrap();
        assert_eq!(result1.reasoning_text, "thinking about");
        assert_eq!(result1.normal_text, "");

        // Second chunk - ends reasoning
        let result2 = parser
            .parse_reasoning_streaming_incremental(" the problem</think>answer")
            .unwrap();
        assert_eq!(result2.reasoning_text, "the problem"); // Text is trimmed
        assert_eq!(result2.normal_text, "answer");
    }

    #[test]
    fn test_model_type() {
        let parser = DeepSeekR1Parser::new();
        assert_eq!(parser.model_type(), "deepseek_r1");
    }
}

// Kimi specific reasoning parser.
// This parser uses Unicode tokens and starts with in_reasoning=false.

use crate::reasoning_parser::{
    parsers::BaseReasoningParser,
    traits::{ParseError, ParserConfig, ParserResult, ReasoningParser},
};

/// Kimi reasoning parser.
///
/// This parser uses Unicode tokens (◁think▷ and ◁/think▷) and requires
/// explicit start tokens to enter reasoning mode.
pub struct KimiParser {
    base: BaseReasoningParser,
}

impl KimiParser {
    /// Create a new Kimi parser.
    pub fn new() -> Self {
        let config = ParserConfig {
            think_start_token: "◁think▷".to_string(),
            think_end_token: "◁/think▷".to_string(),
            stream_reasoning: true,
            max_buffer_size: 65536,
            initial_in_reasoning: false, // Requires explicit start token
        };

        Self {
            base: BaseReasoningParser::new(config).with_model_type("kimi".to_string()),
        }
    }
}

impl Default for KimiParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for KimiParser {
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
    fn test_kimi_initial_state() {
        let mut parser = KimiParser::new();

        // Should NOT treat text as reasoning without start token
        let result = parser
            .detect_and_parse_reasoning("This is normal content")
            .unwrap();
        assert_eq!(result.normal_text, "This is normal content");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn test_kimi_with_unicode_tokens() {
        let mut parser = KimiParser::new();

        // Should extract reasoning with Unicode tokens
        let result = parser
            .detect_and_parse_reasoning("◁think▷reasoning content◁/think▷answer")
            .unwrap();
        assert_eq!(result.normal_text, "answer");
        assert_eq!(result.reasoning_text, "reasoning content");
    }

    #[test]
    fn test_kimi_partial_unicode() {
        let mut parser = KimiParser::new();

        let result1 = parser
            .parse_reasoning_streaming_incremental("◁thi")
            .unwrap();
        assert_eq!(result1.normal_text, "");
        assert_eq!(result1.reasoning_text, "");

        // Complete the token
        let result2 = parser
            .parse_reasoning_streaming_incremental("nk▷reasoning")
            .unwrap();
        assert_eq!(result2.normal_text, "");
        assert_eq!(result2.reasoning_text, "reasoning");
    }

    #[test]
    fn test_kimi_streaming() {
        let mut parser = KimiParser::new();

        // Normal text first
        let result1 = parser
            .parse_reasoning_streaming_incremental("normal ")
            .unwrap();
        assert_eq!(result1.normal_text, "normal ");
        assert_eq!(result1.reasoning_text, "");

        // Enter reasoning with Unicode token
        let result2 = parser
            .parse_reasoning_streaming_incremental("◁think▷thinking")
            .unwrap();
        assert_eq!(result2.normal_text, "");
        assert_eq!(result2.reasoning_text, "thinking");

        // Exit reasoning
        let result3 = parser
            .parse_reasoning_streaming_incremental("◁/think▷answer")
            .unwrap();
        assert_eq!(result3.normal_text, "answer");
        assert_eq!(result3.reasoning_text, ""); // Already returned in stream mode
    }

    #[test]
    fn test_model_type() {
        let parser = KimiParser::new();
        assert_eq!(parser.model_type(), "kimi");
    }
}

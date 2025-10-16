// Base implementation of reasoning parser that handles common logic
// for detecting and extracting reasoning blocks from text.

use crate::reasoning_parser::traits::{ParseError, ParserConfig, ParserResult, ReasoningParser};

/// Base reasoning parser implementation.
///
/// This parser handles the common logic for detecting reasoning blocks
/// delimited by start and end tokens (e.g., <think> and </think>).
#[derive(Debug, Clone)]
pub struct BaseReasoningParser {
    config: ParserConfig,
    in_reasoning: bool,
    buffer: String,
    stripped_think_start: bool,
    model_type: String,
}

impl BaseReasoningParser {
    /// Create a new BaseReasoningParser with the given configuration.
    pub fn new(config: ParserConfig) -> Self {
        let in_reasoning = config.initial_in_reasoning;
        Self {
            config,
            in_reasoning,
            buffer: String::new(),
            stripped_think_start: false,
            model_type: "base".to_string(),
        }
    }

    /// Create with custom model type identifier.
    pub fn with_model_type(mut self, model_type: String) -> Self {
        self.model_type = model_type;
        self
    }

    /// Check if the current buffer is a prefix of one of the tokens.
    fn is_partial_token(&self, text: &str) -> bool {
        (self.config.think_start_token.starts_with(text) && self.config.think_start_token != text)
            || (self.config.think_end_token.starts_with(text)
                && self.config.think_end_token != text)
    }
}

impl ReasoningParser for BaseReasoningParser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        // Check input size against buffer limit
        if text.len() > self.config.max_buffer_size {
            return Err(ParseError::BufferOverflow(text.len()));
        }

        let in_reasoning = self.in_reasoning || text.contains(&self.config.think_start_token);

        if !in_reasoning {
            return Ok(ParserResult::normal(text.to_string()));
        }

        // The text is considered to be in a reasoning block.
        let processed_text = text
            .replace(&self.config.think_start_token, "")
            .trim()
            .to_string();

        if !processed_text.contains(&self.config.think_end_token) {
            // Assume reasoning was truncated before end token
            return Ok(ParserResult::reasoning(processed_text));
        }

        // Extract reasoning content
        let splits: Vec<&str> = processed_text
            .splitn(2, &self.config.think_end_token)
            .collect();
        let reasoning_text = splits.first().unwrap_or(&"").to_string();
        let normal_text = splits
            .get(1)
            .map(|s| s.trim().to_string())
            .unwrap_or_default();

        Ok(ParserResult::new(normal_text, reasoning_text))
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        // Check if adding this text would exceed buffer limit
        if self.buffer.len() + text.len() > self.config.max_buffer_size {
            return Err(ParseError::BufferOverflow(self.buffer.len() + text.len()));
        }

        // Incrementally parse the streaming text
        self.buffer.push_str(text);
        let mut current_text = self.buffer.clone();

        // If the current text is a prefix of a token, keep buffering
        if self.is_partial_token(&current_text) {
            return Ok(ParserResult::default());
        }

        // Strip start token if present
        if !self.stripped_think_start && current_text.contains(&self.config.think_start_token) {
            current_text = current_text.replace(&self.config.think_start_token, "");
            self.buffer = current_text.clone();
            self.stripped_think_start = true;
            self.in_reasoning = true;
        }

        // Handle end of reasoning block
        let think_end_idx = if self.in_reasoning {
            current_text
                .find(&self.config.think_end_token)
                .unwrap_or(current_text.len())
        } else {
            current_text.len()
        };

        if self.in_reasoning && think_end_idx < current_text.len() {
            let reasoning_text = &current_text[..think_end_idx];
            self.buffer.clear();
            self.in_reasoning = false;
            let start_idx = think_end_idx + self.config.think_end_token.len();
            let normal_text = if start_idx < current_text.len() {
                &current_text[start_idx..]
            } else {
                ""
            };
            return Ok(ParserResult::new(
                normal_text.to_string(),
                reasoning_text.trim().to_string(),
            ));
        }

        // Continue with reasoning content
        if self.in_reasoning && self.config.stream_reasoning {
            // Stream the content immediately
            let reasoning_text = current_text;
            self.buffer.clear();
            Ok(ParserResult::reasoning(reasoning_text))
        } else if !self.in_reasoning {
            // If we're not in a reasoning block, return as normal text
            // CRITICAL FIX: Return current_text (with buffer) not just text
            // This prevents buffer loss when partial tokens are followed by normal text
            let normal_text = current_text;
            self.buffer.clear();
            Ok(ParserResult::normal(normal_text))
        } else {
            // If we are in a reasoning block but no end token is found, buffer it
            Ok(ParserResult::default())
        }
    }

    fn reset(&mut self) {
        self.in_reasoning = self.config.initial_in_reasoning;
        self.buffer.clear();
        self.stripped_think_start = false;
    }

    fn model_type(&self) -> &str {
        &self.model_type
    }

    fn is_in_reasoning(&self) -> bool {
        self.in_reasoning
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_parser(
        initial_in_reasoning: bool,
        stream_reasoning: bool,
    ) -> BaseReasoningParser {
        let config = ParserConfig {
            think_start_token: "<think>".to_string(),
            think_end_token: "</think>".to_string(),
            stream_reasoning,
            max_buffer_size: 65536,
            initial_in_reasoning,
        };
        BaseReasoningParser::new(config)
    }

    #[test]
    fn test_detect_and_parse_reasoning() {
        let mut parser = create_test_parser(false, true);
        let result = parser
            .detect_and_parse_reasoning("<think>with reasoning</think> and more text.")
            .unwrap();
        assert_eq!(result.normal_text, "and more text.");
        assert_eq!(result.reasoning_text, "with reasoning");
    }

    #[test]
    fn test_detect_and_parse_no_reasoning() {
        let mut parser = create_test_parser(false, true);
        let result = parser
            .detect_and_parse_reasoning("This is a test without reasoning.")
            .unwrap();
        assert_eq!(result.normal_text, "This is a test without reasoning.");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn test_detect_and_parse_truncated_reasoning() {
        let mut parser = create_test_parser(false, true);
        let result = parser
            .detect_and_parse_reasoning("<think>with truncated reasoning")
            .unwrap();
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "with truncated reasoning");
    }

    #[test]
    fn test_parse_streaming_partial_token() {
        let mut parser = create_test_parser(false, true);
        let result = parser
            .parse_reasoning_streaming_incremental("<thi")
            .unwrap();
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn test_parse_streaming_complete() {
        let mut parser = create_test_parser(false, true);
        let result = parser
            .parse_reasoning_streaming_incremental("<think>with reasoning</think> and more text.")
            .unwrap();
        assert_eq!(result.normal_text, " and more text.");
        assert_eq!(result.reasoning_text, "with reasoning");
    }

    #[test]
    fn test_parse_streaming_no_end_token() {
        let mut parser = create_test_parser(true, true);
        let result = parser
            .parse_reasoning_streaming_incremental("<think>with reasoning")
            .unwrap();
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "with reasoning");
    }

    #[test]
    fn test_initial_in_reasoning_true() {
        // Parser starts with in_reasoning=true (like DeepSeek-R1)
        let mut parser = create_test_parser(true, true);
        let result = parser
            .detect_and_parse_reasoning("no think tags here")
            .unwrap();
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "no think tags here");
    }

    #[test]
    fn test_buffer_loss_bug_fix() {
        // Critical test for buffer preservation
        let mut parser = create_test_parser(false, true);

        // Step 1: Send partial end tag when not in reasoning mode
        let result1 = parser.parse_reasoning_streaming_incremental("</").unwrap();
        assert_eq!(result1.normal_text, "");
        assert_eq!(result1.reasoning_text, "");

        // Step 2: Send normal text that doesn't complete the end tag
        // Must return "</answer" not just "answer"
        let result2 = parser
            .parse_reasoning_streaming_incremental("answer")
            .unwrap();
        assert_eq!(result2.normal_text, "</answer");
        assert_eq!(result2.reasoning_text, "");
    }

    #[test]
    fn test_streaming_with_stream_reasoning_enabled() {
        let mut parser = create_test_parser(false, true);

        // Start reasoning block
        let result1 = parser
            .parse_reasoning_streaming_incremental("<think>reasoning ")
            .unwrap();
        assert_eq!(result1.normal_text, "");
        assert_eq!(result1.reasoning_text, "reasoning ");

        // Continue streaming reasoning
        let result2 = parser
            .parse_reasoning_streaming_incremental("content ")
            .unwrap();
        assert_eq!(result2.normal_text, "");
        assert_eq!(result2.reasoning_text, "content ");

        // End reasoning block
        let result3 = parser
            .parse_reasoning_streaming_incremental("more</think> normal")
            .unwrap();
        assert_eq!(result3.normal_text, " normal");
        assert_eq!(result3.reasoning_text, "more");
    }

    #[test]
    fn test_reset_state() {
        let mut parser = create_test_parser(false, true);

        // Process some text
        parser
            .parse_reasoning_streaming_incremental("<think>reasoning</think> normal")
            .unwrap();

        // Reset and verify state
        parser.reset();
        assert!(!parser.in_reasoning);
        assert!(parser.buffer.is_empty());
        assert!(!parser.stripped_think_start);
    }

    #[test]
    fn test_buffer_overflow_detect_and_parse() {
        let config = ParserConfig {
            max_buffer_size: 10, // Set a very small buffer
            ..Default::default()
        };
        let mut parser = BaseReasoningParser::new(config);

        let large_text = "a".repeat(20);
        let result = parser.detect_and_parse_reasoning(&large_text);

        assert!(result.is_err());
        match result {
            Err(ParseError::BufferOverflow(size)) => {
                assert_eq!(size, 20);
            }
            _ => panic!("Expected BufferOverflow error"),
        }
    }

    #[test]
    fn test_buffer_overflow_streaming() {
        let config = ParserConfig {
            max_buffer_size: 10, // Set a very small buffer
            ..Default::default()
        };
        let mut parser = BaseReasoningParser::new(config);

        // Send a partial token that will be buffered
        let result1 = parser.parse_reasoning_streaming_incremental("<thi");
        assert!(result1.is_ok());
        assert_eq!(result1.unwrap().normal_text, "");

        // Second chunk would exceed buffer
        // Buffer has "<thi" (4 chars) + "this_is_too_large" (17 chars) = 21 total
        let result2 = parser.parse_reasoning_streaming_incremental("this_is_too_large");
        assert!(result2.is_err());
        match result2 {
            Err(ParseError::BufferOverflow(size)) => {
                assert_eq!(size, 21); // 4 + 17
            }
            _ => panic!("Expected BufferOverflow error"),
        }
    }
}

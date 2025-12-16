use std::{collections::HashSet, sync::Arc};

use anyhow::Result;

use super::{
    sequence::Sequence,
    traits::{self, TokenIdType},
};

/// Output from the sequence decoder
#[derive(Debug, Clone, PartialEq)]
pub enum SequenceDecoderOutput {
    /// Normal text output
    Text(String),
    /// Text is being held due to partial stop sequence match
    Held,
    /// Stop sequence matched (hidden - not included in output)
    Stopped,
    /// Stop sequence matched with text (visible - included in output)
    StoppedWithText(String),
}

/// Configuration for stop sequences
#[derive(Debug, Clone, Default)]
pub struct StopSequenceConfig {
    /// Token IDs that trigger a stop
    pub stop_tokens: HashSet<TokenIdType>,
    /// String sequences that trigger a stop
    pub stop_sequences: Vec<String>,
    /// Token IDs for visible stops (included in output)
    pub visible_stop_tokens: HashSet<TokenIdType>,
    /// String sequences for visible stops (included in output)
    pub visible_stop_sequences: Vec<String>,
}

impl StopSequenceConfig {
    /// Builder pattern - add a stop token
    pub fn with_stop_token(mut self, token_id: TokenIdType) -> Self {
        self.stop_tokens.insert(token_id);
        self
    }

    /// Builder pattern - add a stop sequence
    pub fn with_stop_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.stop_sequences.push(sequence.into());
        self
    }

    /// Builder pattern - add a visible stop token
    pub fn with_visible_stop_token(mut self, token_id: TokenIdType) -> Self {
        self.visible_stop_tokens.insert(token_id);
        self
    }

    /// Builder pattern - add a visible stop sequence
    pub fn with_visible_stop_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.visible_stop_sequences.push(sequence.into());
        self
    }
}

/// Decoder that handles stop sequences
pub struct StopSequenceDecoder {
    /// Sequence for incremental decoding (replaces token_buffer + offsets)
    sequence: Sequence,
    config: StopSequenceConfig,
    /// Buffer for partial matches (the "jail")
    jail_buffer: String,
    /// Whether we've stopped
    stopped: bool,
}

impl StopSequenceDecoder {
    /// Create a new stop sequence decoder
    pub fn new(
        tokenizer: Arc<dyn traits::Tokenizer>,
        config: StopSequenceConfig,
        skip_special_tokens: bool,
    ) -> Self {
        StopSequenceDecoder {
            sequence: Sequence::new_with_options(tokenizer, skip_special_tokens),
            config,
            jail_buffer: String::new(),
            stopped: false,
        }
    }

    /// Process a single token
    pub fn process_token(&mut self, token_id: TokenIdType) -> Result<SequenceDecoderOutput> {
        if self.stopped {
            return Ok(SequenceDecoderOutput::Stopped);
        }

        // Check for token-level stops first
        if self.config.stop_tokens.contains(&token_id) {
            self.stopped = true;

            // Flush any jailed text before stopping - use mem::take to avoid clone
            if !self.jail_buffer.is_empty() {
                return Ok(SequenceDecoderOutput::StoppedWithText(std::mem::take(
                    &mut self.jail_buffer,
                )));
            }
            return Ok(SequenceDecoderOutput::Stopped);
        }

        if self.config.visible_stop_tokens.contains(&token_id) {
            self.stopped = true;

            // Include jailed text plus the stop token
            let stop_text = self
                .sequence
                .tokenizer()
                .decode(&[token_id], self.sequence.skip_special_tokens())?;
            let output = format!("{}{}", self.jail_buffer, stop_text);
            self.jail_buffer.clear();
            return Ok(SequenceDecoderOutput::StoppedWithText(output));
        }

        // Use Sequence for incremental decoding
        let new_text = self.sequence.append_token(token_id)?;

        self.jail_buffer.push_str(&new_text);

        // Check for hidden stop sequences
        for stop_seq in &self.config.stop_sequences {
            if let Some(pos) = self.jail_buffer.find(stop_seq) {
                self.stopped = true;
                let output = self.jail_buffer[..pos].to_string();
                self.jail_buffer.clear();
                return Ok(if output.is_empty() {
                    SequenceDecoderOutput::Stopped
                } else {
                    SequenceDecoderOutput::StoppedWithText(output)
                });
            }
        }

        // Check for visible stop sequences
        for stop_seq in &self.config.visible_stop_sequences {
            if let Some(pos) = self.jail_buffer.find(stop_seq) {
                self.stopped = true;
                let end_pos = pos + stop_seq.len();
                let output = self.jail_buffer[..end_pos].to_string();
                self.jail_buffer.clear();
                return Ok(SequenceDecoderOutput::StoppedWithText(output));
            }
        }

        // Check for partial matches: is the end of jail_buffer the start of any stop_seq?
        // This handles stop sequences split across tokens
        let buffer_len = self.jail_buffer.len();
        let mut best_split_pos: Option<usize> = None;

        for stop_seq in self
            .config
            .stop_sequences
            .iter()
            .chain(&self.config.visible_stop_sequences)
        {
            let stop_len = stop_seq.len();

            if stop_len <= 1 || buffer_len == 0 {
                continue;
            }

            let max_len = buffer_len.min(stop_len - 1);

            for len in (1..=max_len).rev() {
                let suffix_start = buffer_len - len;

                if !self.jail_buffer.is_char_boundary(suffix_start) {
                    continue;
                }

                let suffix = &self.jail_buffer[suffix_start..];

                if stop_seq.starts_with(suffix)
                    && best_split_pos.is_none_or(|current| suffix_start < current)
                {
                    best_split_pos = Some(suffix_start);
                    break;
                }
            }
        }

        if let Some(split_pos) = best_split_pos {
            // Hold the partial match, flush the rest
            // Use split_off for zero-copy: keeps [0..split_pos] in place, returns [split_pos..]
            // Then swap so we output the prefix and keep the suffix
            let suffix = self.jail_buffer.split_off(split_pos);
            let to_output = std::mem::replace(&mut self.jail_buffer, suffix);

            if to_output.is_empty() {
                Ok(SequenceDecoderOutput::Held)
            } else {
                Ok(SequenceDecoderOutput::Text(to_output))
            }
        } else {
            // No partial matches - flush everything
            let output = std::mem::take(&mut self.jail_buffer);
            if output.is_empty() {
                Ok(SequenceDecoderOutput::Held)
            } else {
                Ok(SequenceDecoderOutput::Text(output))
            }
        }
    }

    /// Process multiple tokens
    pub fn process_tokens(
        &mut self,
        token_ids: &[TokenIdType],
    ) -> Result<Vec<SequenceDecoderOutput>> {
        // Pre-allocate with exact capacity to avoid reallocations
        let mut outputs = Vec::with_capacity(token_ids.len());
        for &token_id in token_ids {
            outputs.push(self.process_token(token_id)?);
        }
        Ok(outputs)
    }

    /// Flush any held text
    pub fn flush(&mut self) -> SequenceDecoderOutput {
        if !self.jail_buffer.is_empty() {
            // Use mem::take to avoid clone - transfers ownership and leaves empty string
            SequenceDecoderOutput::Text(std::mem::take(&mut self.jail_buffer))
        } else {
            SequenceDecoderOutput::Text(String::new())
        }
    }

    /// Check if decoding has stopped
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Reset the decoder state
    pub fn reset(&mut self) {
        self.jail_buffer.clear();
        self.sequence.clear();
        self.stopped = false;
    }
}

/// Builder for StopSequenceDecoder
pub struct StopSequenceDecoderBuilder {
    tokenizer: Arc<dyn traits::Tokenizer>,
    config: StopSequenceConfig,
    skip_special_tokens: bool,
}

impl StopSequenceDecoderBuilder {
    pub fn new(tokenizer: Arc<dyn traits::Tokenizer>) -> Self {
        StopSequenceDecoderBuilder {
            tokenizer,
            config: StopSequenceConfig::default(),
            skip_special_tokens: true,
        }
    }

    pub fn stop_token(mut self, token_id: TokenIdType) -> Self {
        self.config.stop_tokens.insert(token_id);
        self
    }

    pub fn stop_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.config.stop_sequences.push(sequence.into());
        self
    }

    pub fn visible_stop_token(mut self, token_id: TokenIdType) -> Self {
        self.config.visible_stop_tokens.insert(token_id);
        self
    }

    pub fn visible_stop_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.config.visible_stop_sequences.push(sequence.into());
        self
    }

    pub fn skip_special_tokens(mut self, skip: bool) -> Self {
        self.skip_special_tokens = skip;
        self
    }

    pub fn build(self) -> StopSequenceDecoder {
        StopSequenceDecoder::new(self.tokenizer, self.config, self.skip_special_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::mock::MockTokenizer;

    #[test]
    fn test_stop_token_detection() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_token(999); // <eos> token

        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process tokens before stop
        let result = decoder.process_token(1).unwrap(); // "Hello"
        assert!(matches!(result, SequenceDecoderOutput::Text(_)));

        // Process stop token
        let result = decoder.process_token(999).unwrap(); // <eos>
        assert_eq!(result, SequenceDecoderOutput::Stopped);

        // Further tokens should also return Stopped
        let result = decoder.process_token(2).unwrap();
        assert_eq!(result, SequenceDecoderOutput::Stopped);
    }

    #[test]
    fn test_visible_stop_token() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_visible_stop_token(999);

        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        let result = decoder.process_token(999).unwrap();
        assert!(matches!(result, SequenceDecoderOutput::StoppedWithText(_)));
    }

    #[test]
    fn test_builder_pattern() {
        let tokenizer = Arc::new(MockTokenizer::new());

        let decoder = StopSequenceDecoderBuilder::new(tokenizer)
            .stop_token(999)
            .stop_sequence("STOP")
            .visible_stop_token(1000)
            .skip_special_tokens(true)
            .build();

        assert!(!decoder.is_stopped());
    }

    #[test]
    fn test_incremental_decoding_no_repetition() {
        // This test verifies the critical fix: no repeated output
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default();
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process tokens one by one and collect outputs
        let mut outputs = Vec::new();

        // Token 1: "Hello"
        let result = decoder.process_token(1).unwrap();
        if let SequenceDecoderOutput::Text(text) = result {
            outputs.push(text.clone());
        }

        // Token 2: "world"
        let result = decoder.process_token(2).unwrap();
        if let SequenceDecoderOutput::Text(text) = result {
            outputs.push(text.clone());
        }

        // Token 3: "test"
        let result = decoder.process_token(3).unwrap();
        if let SequenceDecoderOutput::Text(text) = result {
            outputs.push(text.clone());
        }

        // CRITICAL: Each output should be unique (no accumulation)
        // The fix ensures we only output NEW text, not accumulated text
        assert_eq!(outputs.len(), 3);

        for i in 0..outputs.len() {
            for j in i + 1..outputs.len() {
                // No output should contain another (no accumulation)
                assert!(!outputs[j].contains(&outputs[i]));
            }
        }
    }

    #[test]
    fn test_stop_sequence_detection() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_sequence("test");
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process "Hello world"
        decoder.process_token(1).unwrap(); // "Hello"
        decoder.process_token(2).unwrap(); // "world"

        // Process "test" which should trigger stop
        let result = decoder.process_token(3).unwrap(); // "test"

        // Should stop when we hit "test"
        assert!(matches!(
            result,
            SequenceDecoderOutput::Stopped | SequenceDecoderOutput::StoppedWithText(_)
        ));
    }

    #[test]
    fn test_flush_after_partial() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_sequence("NEVER_MATCH");
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process a token
        decoder.process_token(1).unwrap(); // "Hello"

        // Flush should return any remaining text in jail
        let result = decoder.flush();

        // After processing, flush should work
        assert!(matches!(result, SequenceDecoderOutput::Text(_)));
    }

    #[test]
    fn test_reset_functionality() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_token(999);
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process and stop
        decoder.process_token(1).unwrap();
        decoder.process_token(999).unwrap();
        assert!(decoder.is_stopped());

        // Reset should clear everything
        decoder.reset();
        assert!(!decoder.is_stopped());

        // Should be able to process again
        let result = decoder.process_token(2).unwrap();
        assert!(matches!(result, SequenceDecoderOutput::Text(_)));
    }

    #[test]
    fn test_visible_stop_sequence() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_visible_stop_sequence("world");
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process "Hello"
        decoder.process_token(1).unwrap();

        // Process "world" - should include it in output
        let result = decoder.process_token(2).unwrap();

        if let SequenceDecoderOutput::StoppedWithText(text) = result {
            // Should include "world" in the output
            assert!(text.contains("world"));
        } else {
            panic!("Expected StoppedWithText with visible stop sequence");
        }
    }

    #[test]
    fn test_multiple_tokens_processing() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default();
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process multiple tokens at once
        let results = decoder.process_tokens(&[1, 2, 3]).unwrap();

        // Should get results for each token
        assert_eq!(results.len(), 3);

        // Each result should be Text (no stops configured)
        for result in results {
            assert!(matches!(
                result,
                SequenceDecoderOutput::Text(_) | SequenceDecoderOutput::Held
            ));
        }
    }

    #[test]
    fn test_utf8_multibyte_character_boundaries() {
        // This test verifies the fix for the UTF-8 boundary panic
        // The panic occurred when trying to slice jail_buffer at a byte index
        // that was in the middle of a multi-byte UTF-8 character (e.g., '×')
        use crate::tokenizer::mock::MockTokenizer;

        let tokenizer = Arc::new(MockTokenizer::new());

        // Configure stop sequence with a multi-byte character
        let config = StopSequenceConfig::default().with_stop_sequence(" ×");

        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Simulate the scenario: jail_buffer will contain " ×" (space + multiplication sign)
        // The '×' character is UTF-8 encoded as bytes [0xC3, 0x97] (2 bytes)
        // When checking for partial matches, we must not slice in the middle of these bytes

        // This should not panic - the fix ensures we only slice at char boundaries
        let result = decoder.process_token(1); // Will add some text to jail_buffer
        assert!(result.is_ok());

        // Even with multi-byte UTF-8 characters in the buffer, processing should work
        let result = decoder.process_token(2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_utf8_multibyte_delta_character() {
        // Test for: byte index 1 is not a char boundary; it is inside 'Δ' (bytes 0..2) of `Δ`
        // 'Δ' (U+0394 GREEK CAPITAL LETTER DELTA) is encoded as [0xCE, 0x94] (2 bytes)
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_sequence("Δ");

        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process tokens - should not panic when checking partial matches
        let result = decoder.process_token(1);
        assert!(result.is_ok());
        let result = decoder.process_token(2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_utf8_multibyte_degree_character() {
        // Test for: byte index 1 is not a char boundary; it is inside '°' (bytes 0..2) of `°`
        // '°' (U+00B0 DEGREE SIGN) is encoded as [0xC2, 0xB0] (2 bytes)
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_sequence("°");

        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process tokens - should not panic when checking partial matches
        let result = decoder.process_token(1);
        assert!(result.is_ok());
        let result = decoder.process_token(2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_utf8_multibyte_triangle_character() {
        // Test for: byte index 4 is not a char boundary; it is inside '∆' (bytes 2..5) of ` (∆`
        // '∆' (U+2206 INCREMENT) is encoded as [0xE2, 0x88, 0x86] (3 bytes)
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_sequence(" (∆");

        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process tokens - should not panic when checking partial matches
        let result = decoder.process_token(1);
        assert!(result.is_ok());
        let result = decoder.process_token(2);
        assert!(result.is_ok());
        let result = decoder.process_token(3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_utf8_multibyte_en_dash_character() {
        // Test for: byte index 3 is not a char boundary; it is inside '–' (bytes 1..4) of ` –`
        // '–' (U+2013 EN DASH) is encoded as [0xE2, 0x80, 0x93] (3 bytes)
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_sequence(" –");

        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process tokens - should not panic when checking partial matches
        let result = decoder.process_token(1);
        assert!(result.is_ok());
        let result = decoder.process_token(2);
        assert!(result.is_ok());
        let result = decoder.process_token(3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_utf8_multibyte_various_characters() {
        // Comprehensive test with multiple multi-byte UTF-8 characters
        // Tests 2-byte, 3-byte, and 4-byte UTF-8 sequences
        let test_cases = vec![
            ("×", "multiplication sign - 2 bytes"),
            ("Δ", "Greek Delta - 2 bytes"),
            ("°", "degree sign - 2 bytes"),
            ("∆", "increment - 3 bytes"),
            ("–", "en dash - 3 bytes"),
            ("€", "euro sign - 3 bytes"),
            ("中", "Chinese character - 3 bytes"),
            ("🚀", "rocket emoji - 4 bytes"),
            ("💡", "lightbulb emoji - 4 bytes"),
        ];

        for (stop_char, description) in test_cases {
            let tokenizer = Arc::new(MockTokenizer::new());
            let config = StopSequenceConfig::default().with_stop_sequence(stop_char);

            let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

            // Process multiple tokens - should not panic
            for token_id in 1..=5 {
                let result = decoder.process_token(token_id);
                assert!(
                    result.is_ok(),
                    "Failed on {} with token {}",
                    description,
                    token_id
                );
            }
        }
    }
}

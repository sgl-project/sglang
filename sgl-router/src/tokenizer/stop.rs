use super::traits::{self, TokenIdType};
use anyhow::Result;
use std::collections::HashSet;
use std::sync::Arc;

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
    tokenizer: Arc<dyn traits::Tokenizer>,
    config: StopSequenceConfig,
    /// Buffer for partial matches (the "jail")
    jail_buffer: String,
    /// Accumulated tokens
    token_buffer: Vec<TokenIdType>,
    /// Offset where the prefix text starts (for context)
    prefix_offset: usize,
    /// Offset marking the end of previously decoded text
    read_offset: usize,
    /// Whether we've stopped
    stopped: bool,
    skip_special_tokens: bool,
}

impl StopSequenceDecoder {
    /// Create a new stop sequence decoder
    pub fn new(
        tokenizer: Arc<dyn traits::Tokenizer>,
        config: StopSequenceConfig,
        skip_special_tokens: bool,
    ) -> Self {
        StopSequenceDecoder {
            tokenizer,
            config,
            jail_buffer: String::new(),
            token_buffer: Vec::new(),
            prefix_offset: 0,
            read_offset: 0,
            stopped: false,
            skip_special_tokens,
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

            // Flush any jailed text before stopping
            if !self.jail_buffer.is_empty() {
                let output = self.jail_buffer.clone();
                self.jail_buffer.clear();
                return Ok(SequenceDecoderOutput::StoppedWithText(output));
            }
            return Ok(SequenceDecoderOutput::Stopped);
        }

        if self.config.visible_stop_tokens.contains(&token_id) {
            self.stopped = true;

            // Include jailed text plus the stop token
            let stop_text = self
                .tokenizer
                .decode(&[token_id], self.skip_special_tokens)?;
            let output = format!("{}{}", self.jail_buffer, stop_text);
            self.jail_buffer.clear();
            return Ok(SequenceDecoderOutput::StoppedWithText(output));
        }

        // Add token to buffer
        self.token_buffer.push(token_id);

        // Use incremental decoding like DecodeStream
        // First decode the previous context (what we've already output)
        let prefix_text = if self.read_offset > self.prefix_offset {
            self.tokenizer.decode(
                &self.token_buffer[self.prefix_offset..self.read_offset],
                self.skip_special_tokens,
            )?
        } else {
            String::new()
        };

        // Now decode from prefix to current position
        let new_full_text = self.tokenizer.decode(
            &self.token_buffer[self.prefix_offset..],
            self.skip_special_tokens,
        )?;

        // Check for incomplete UTF-8 sequence
        if new_full_text.ends_with("�") {
            // Wait for more tokens to complete the sequence
            return Ok(SequenceDecoderOutput::Held);
        }

        // Calculate only the NEW text since last successful decode
        let new_text = if new_full_text.len() > prefix_text.len() {
            &new_full_text[prefix_text.len()..]
        } else {
            // No new text produced (can happen with special tokens)
            return Ok(SequenceDecoderOutput::Held);
        };

        // Combine jail buffer with new text for checking
        let check_text = format!("{}{}", self.jail_buffer, new_text);

        // Check for complete stop sequences
        for stop_seq in &self.config.stop_sequences {
            if let Some(pos) = check_text.find(stop_seq) {
                self.stopped = true;

                // Output text before the stop sequence
                let output = check_text[..pos].to_string();
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
            if let Some(pos) = check_text.find(stop_seq) {
                self.stopped = true;

                // Include the stop sequence in output
                let end_pos = pos + stop_seq.len();
                let output = check_text[..end_pos].to_string();
                self.jail_buffer.clear();
                return Ok(SequenceDecoderOutput::StoppedWithText(output));
            }
        }

        // Check for partial matches at the end of check_text
        let mut partial_match_len = 0;
        for stop_seq in self
            .config
            .stop_sequences
            .iter()
            .chain(&self.config.visible_stop_sequences)
        {
            // Check all possible suffixes that could be a prefix of stop_seq
            for i in 1..=check_text.len().min(stop_seq.len() - 1) {
                let suffix = &check_text[check_text.len() - i..];
                if stop_seq.starts_with(suffix) {
                    partial_match_len = partial_match_len.max(i);
                }
            }
        }

        if partial_match_len > 0 {
            // Split: output safe text, jail the potential match
            let safe_end = check_text.len() - partial_match_len;
            let safe_text = &check_text[..safe_end];
            self.jail_buffer = check_text[safe_end..].to_string();

            // Update offsets for next iteration
            self.prefix_offset = self.read_offset;
            self.read_offset = self.token_buffer.len();

            if safe_text.is_empty() {
                Ok(SequenceDecoderOutput::Held)
            } else {
                Ok(SequenceDecoderOutput::Text(safe_text.to_string()))
            }
        } else {
            // No partial matches - output everything
            self.jail_buffer.clear();

            // Update offsets for next iteration
            self.prefix_offset = self.read_offset;
            self.read_offset = self.token_buffer.len();

            Ok(SequenceDecoderOutput::Text(check_text))
        }
    }

    /// Process multiple tokens
    pub fn process_tokens(
        &mut self,
        token_ids: &[TokenIdType],
    ) -> Result<Vec<SequenceDecoderOutput>> {
        let mut outputs = Vec::new();
        for &token_id in token_ids {
            outputs.push(self.process_token(token_id)?);
        }
        Ok(outputs)
    }

    /// Flush any held text
    pub fn flush(&mut self) -> SequenceDecoderOutput {
        if !self.jail_buffer.is_empty() {
            let output = self.jail_buffer.clone();
            self.jail_buffer.clear();
            SequenceDecoderOutput::Text(output)
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
        self.token_buffer.clear();
        self.prefix_offset = 0;
        self.read_offset = 0;
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
}

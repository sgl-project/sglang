use std::sync::Arc;

use anyhow::Result;

use super::traits::{TokenIdType, Tokenizer as TokenizerTrait};

/// Maintains state for an ongoing sequence of tokens and their decoded text
/// This provides a cleaner abstraction for managing token sequences
pub struct Sequence {
    /// The tokenizer used for encoding/decoding
    tokenizer: Arc<dyn TokenizerTrait>,

    /// The current sequence of token ids
    token_ids: Vec<TokenIdType>,

    /// The position in the current sequence the last decoded token completed
    prefix_offset: usize,

    /// Current position in the sequence
    read_offset: usize,

    /// Whether to skip special tokens when decoding
    skip_special_tokens: bool,
}

impl std::fmt::Debug for Sequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequence")
            .field("tokenizer", &"Arc<dyn Tokenizer>")
            .field(
                "token_ids",
                &format_args!("{}", {
                    let token_ids = self.token_ids();
                    if token_ids.len() <= 20 {
                        format!("{:?}", token_ids)
                    } else {
                        let first_ten = &token_ids[..10];
                        let last_ten = &token_ids[token_ids.len() - 10..];
                        format!("{:?} ... {:?}", first_ten, last_ten)
                    }
                }),
            )
            .field("prefix_offset", &self.prefix_offset)
            .field("read_offset", &self.read_offset)
            .field("token count", &self.token_ids.len())
            .finish()
    }
}

impl Sequence {
    /// Create a new empty sequence
    pub fn new(tokenizer: Arc<dyn TokenizerTrait>) -> Self {
        Self::new_with_options(tokenizer, false)
    }

    /// Create a new empty sequence with skip_special_tokens option
    pub fn new_with_options(tokenizer: Arc<dyn TokenizerTrait>, skip_special_tokens: bool) -> Self {
        Self {
            tokenizer,
            token_ids: Vec::new(),
            prefix_offset: 0,
            read_offset: 0,
            skip_special_tokens,
        }
    }

    /// Create a sequence with initial tokens
    pub fn with_tokens(tokenizer: Arc<dyn TokenizerTrait>, token_ids: Vec<TokenIdType>) -> Self {
        Self::with_tokens_and_options(tokenizer, token_ids, false)
    }

    /// Create a sequence with initial tokens and skip_special_tokens option
    pub fn with_tokens_and_options(
        tokenizer: Arc<dyn TokenizerTrait>,
        token_ids: Vec<TokenIdType>,
        skip_special_tokens: bool,
    ) -> Self {
        let len = token_ids.len();
        Self {
            tokenizer,
            token_ids,
            prefix_offset: 0,
            read_offset: len,
            skip_special_tokens,
        }
    }

    /// Check if the sequence is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    /// Get the length of the sequence
    #[inline]
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    /// Clear the sequence
    pub fn clear(&mut self) {
        self.token_ids.clear();
        self.prefix_offset = 0;
        self.read_offset = 0;
    }

    /// Append text to the sequence by encoding it
    pub fn append_text(&mut self, input: &str) -> Result<()> {
        let encoding = self.tokenizer.encode(input)?;
        self.token_ids.extend(encoding.token_ids());
        Ok(())
    }

    /// Append a single token to the sequence and return newly decoded text
    /// Based on HuggingFace TGI incremental decoding
    #[inline]
    pub fn append_token(&mut self, token_id: TokenIdType) -> Result<String> {
        // Store the old read offset before adding the new token
        let old_read_offset = self.read_offset;

        self.token_ids.push(token_id);
        self.read_offset = self.token_ids.len();

        // If this is the first token or we're at the beginning, decode everything
        if self.prefix_offset == 0 && old_read_offset == 0 {
            let text = self
                .tokenizer
                .decode(&self.token_ids, self.skip_special_tokens)?;
            if text.ends_with("�") {
                // Incomplete UTF-8 sequence, wait for more tokens
                return Ok(String::new());
            }
            self.prefix_offset = 0;
            return Ok(text);
        }

        // Decode the text up to the previous position
        let prefix_text = self.tokenizer.decode(
            &self.token_ids[self.prefix_offset..old_read_offset],
            self.skip_special_tokens,
        )?;

        // Decode the text including the new token
        let new_text = self.tokenizer.decode(
            &self.token_ids[self.prefix_offset..],
            self.skip_special_tokens,
        )?;

        // Handle multi-byte character boundaries
        let mut prefix_text_len = prefix_text.len();
        while !new_text.is_char_boundary(prefix_text_len) && prefix_text_len > 0 {
            prefix_text_len -= 1;
        }

        if new_text.len() > prefix_text.len() {
            if new_text.ends_with("�") {
                // Incomplete UTF-8 sequence, wait for more tokens
                return Ok(String::new());
            } else {
                // Return the new text portion
                let incremental_text = new_text[prefix_text_len..].to_string().replace("�", "");
                self.prefix_offset = old_read_offset;
                return Ok(incremental_text);
            }
        }

        Ok(String::new())
    }

    /// Get a reference to the tokenizer
    #[inline]
    pub fn tokenizer(&self) -> &Arc<dyn TokenizerTrait> {
        &self.tokenizer
    }

    /// Get the current token ids
    #[inline]
    pub fn token_ids(&self) -> &[TokenIdType] {
        &self.token_ids
    }

    /// Decode the entire sequence to text
    pub fn text(&self) -> Result<String> {
        self.tokenizer
            .decode(&self.token_ids, self.skip_special_tokens)
    }

    /// Get the prefix offset
    #[inline]
    pub fn prefix_offset(&self) -> usize {
        self.prefix_offset
    }

    /// Get the read offset
    #[inline]
    pub fn read_offset(&self) -> usize {
        self.read_offset
    }

    /// Get whether special tokens are skipped during decoding
    #[inline]
    pub fn skip_special_tokens(&self) -> bool {
        self.skip_special_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::mock::MockTokenizer;

    #[test]
    fn test_sequence_new() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let seq = Sequence::new(tokenizer);
        assert!(seq.is_empty());
        assert_eq!(seq.len(), 0);
    }

    #[test]
    fn test_sequence_append_text() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let mut seq = Sequence::new(tokenizer);

        seq.append_text("Hello").unwrap();
        assert!(!seq.is_empty());
        assert!(!seq.is_empty());

        let text = seq.text().unwrap();
        assert_eq!(text, "Hello");
    }

    #[test]
    fn test_sequence_append_token() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let mut seq = Sequence::new(tokenizer.clone());

        // Start with an empty sequence and append token 1 ("Hello")
        let text1 = seq.append_token(1).unwrap();
        assert_eq!(text1, "Hello");

        // Now append token 2 ("world")
        // The mock tokenizer will decode [1, 2] as "Hello world" (with a space)
        let text2 = seq.append_token(2).unwrap();
        // The incremental text should be " world" (with the space that the mock tokenizer adds)
        assert_eq!(text2, " world");

        assert_eq!(seq.text().unwrap(), "Hello world");
    }

    #[test]
    fn test_sequence_clear() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let mut seq = Sequence::new(tokenizer);

        seq.append_text("Hello world").unwrap();
        assert!(!seq.is_empty());

        seq.clear();
        assert!(seq.is_empty());
        assert_eq!(seq.len(), 0);
        assert_eq!(seq.prefix_offset(), 0);
        assert_eq!(seq.read_offset(), 0);
    }

    #[test]
    fn test_sequence_debug() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let mut seq = Sequence::new(tokenizer);

        seq.append_text("Test").unwrap();
        let debug_str = format!("{:?}", seq);
        assert!(debug_str.contains("Sequence"));
        assert!(debug_str.contains("token count"));
    }
}

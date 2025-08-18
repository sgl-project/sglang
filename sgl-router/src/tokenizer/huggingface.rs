use super::traits::{Decoder, Encoder, Encoding, SpecialTokens, Tokenizer as TokenizerTrait};
use crate::metrics::TokenizerMetrics;
use anyhow::{Error, Result};
use std::collections::HashMap;
use std::time::Instant;
use tokenizers::tokenizer::Tokenizer as HfTokenizer;

/// HuggingFace tokenizer wrapper
pub struct HuggingFaceTokenizer {
    tokenizer: HfTokenizer,
    special_tokens: SpecialTokens,
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
}

impl HuggingFaceTokenizer {
    /// Create a tokenizer from a HuggingFace tokenizer JSON file
    pub fn from_file(file_path: &str) -> Result<Self> {
        let tokenizer = HfTokenizer::from_file(file_path)
            .map_err(|e| Error::msg(format!("Failed to load tokenizer: {}", e)))?;

        // Extract special tokens
        let special_tokens = Self::extract_special_tokens(&tokenizer);

        // Build vocab mappings
        let vocab = tokenizer.get_vocab(false);
        let reverse_vocab: HashMap<u32, String> = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();

        Ok(HuggingFaceTokenizer {
            tokenizer,
            special_tokens,
            vocab,
            reverse_vocab,
        })
    }

    /// Create from an existing HuggingFace tokenizer
    pub fn from_tokenizer(tokenizer: HfTokenizer) -> Self {
        let special_tokens = Self::extract_special_tokens(&tokenizer);
        let vocab = tokenizer.get_vocab(false);
        let reverse_vocab: HashMap<u32, String> = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();

        HuggingFaceTokenizer {
            tokenizer,
            special_tokens,
            vocab,
            reverse_vocab,
        }
    }

    /// Extract special tokens from the tokenizer
    fn extract_special_tokens(tokenizer: &HfTokenizer) -> SpecialTokens {
        // Try to get special tokens from the tokenizer
        // This is a simplified version - actual implementation would need to handle various formats
        let vocab = tokenizer.get_vocab(true);

        let find_token = |patterns: &[&str]| -> Option<String> {
            for pattern in patterns {
                if vocab.contains_key(*pattern) {
                    return Some(pattern.to_string());
                }
            }
            None
        };

        SpecialTokens {
            bos_token: find_token(&["<s>", "<|startoftext|>", "<BOS>", "[CLS]"]),
            eos_token: find_token(&["</s>", "<|endoftext|>", "<EOS>", "[SEP]"]),
            unk_token: find_token(&["<unk>", "<UNK>", "[UNK]"]),
            sep_token: find_token(&["[SEP]", "<sep>", "<SEP>"]),
            pad_token: find_token(&["<pad>", "<PAD>", "[PAD]"]),
            cls_token: find_token(&["[CLS]", "<cls>", "<CLS>"]),
            mask_token: find_token(&["[MASK]", "<mask>", "<MASK>"]),
            additional_special_tokens: vec![],
        }
    }

    /// Apply chat template if available
    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        // This is a placeholder - actual implementation would handle templates
        let mut result = String::new();
        for msg in messages {
            result.push_str(&format!("{}: {}\n", msg.role, msg.content));
        }
        Ok(result)
    }
}

impl Encoder for HuggingFaceTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        let start = Instant::now();

        TokenizerMetrics::record_encode_request("huggingface");
        TokenizerMetrics::record_chars_per_encode(input.len());

        self.tokenizer
            .encode(input, false)
            .map_err(|e| {
                TokenizerMetrics::record_encode_error("encoding_failed");
                Error::msg(format!("Encoding failed: {}", e))
            })
            .map(|encoding| {
                TokenizerMetrics::record_tokens_per_encode(encoding.get_ids().len());
                TokenizerMetrics::record_encode_duration(start.elapsed());
                Encoding::Hf(Box::new(encoding))
            })
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        let start = Instant::now();

        let encodings = self
            .tokenizer
            .encode_batch(inputs.to_vec(), false)
            .map_err(|e| {
                TokenizerMetrics::record_encode_error("batch_encoding_failed");
                Error::msg(format!("Batch encoding failed: {}", e))
            })?;

        TokenizerMetrics::record_encode_batch_duration(start.elapsed(), inputs.len());

        Ok(encodings
            .into_iter()
            .map(|e| Encoding::Hf(Box::new(e)))
            .collect())
    }
}

impl Decoder for HuggingFaceTokenizer {
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let start = Instant::now();

        TokenizerMetrics::record_decode_request("huggingface");
        TokenizerMetrics::record_tokens_per_decode(token_ids.len());

        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| {
                TokenizerMetrics::record_decode_error("decoding_failed");
                Error::msg(format!("Decoding failed: {}", e))
            })
            .inspect(|_| {
                TokenizerMetrics::record_decode_duration(start.elapsed());
            })
    }
}

impl TokenizerTrait for HuggingFaceTokenizer {
    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.reverse_vocab.get(&id).cloned()
    }
}

/// Represents a chat message for template application
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        ChatMessage {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_creation() {
        let msg = ChatMessage::system("You are a helpful assistant");
        assert_eq!(msg.role, "system");
        assert_eq!(msg.content, "You are a helpful assistant");

        let user_msg = ChatMessage::user("Hello!");
        assert_eq!(user_msg.role, "user");

        let assistant_msg = ChatMessage::assistant("Hi there!");
        assert_eq!(assistant_msg.role, "assistant");
    }

    // Note: Actual tokenizer tests would require a real tokenizer file
    // These would be integration tests rather than unit tests
}

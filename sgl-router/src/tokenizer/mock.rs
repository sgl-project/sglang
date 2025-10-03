//! Mock tokenizer implementation for testing

use super::traits::{Decoder, Encoder, Encoding, SpecialTokens, Tokenizer as TokenizerTrait};
use anyhow::Result;
use std::collections::HashMap;

/// Mock tokenizer for testing purposes
pub struct MockTokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    special_tokens: SpecialTokens,
}

impl Default for MockTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MockTokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();

        // Add some basic tokens
        let tokens = vec![
            ("Hello", 1),
            ("world", 2),
            ("test", 3),
            ("token", 4),
            (" ", 5),
            (".", 6),
            ("<eos>", 999),
            ("<bos>", 1000),
        ];

        for (token, id) in tokens {
            vocab.insert(token.to_string(), id);
            reverse_vocab.insert(id, token.to_string());
        }

        let special_tokens = SpecialTokens {
            bos_token: Some("<bos>".to_string()),
            eos_token: Some("<eos>".to_string()),
            unk_token: Some("<unk>".to_string()),
            sep_token: None,
            pad_token: None,
            cls_token: None,
            mask_token: None,
            additional_special_tokens: vec![],
        };

        Self {
            vocab,
            reverse_vocab,
            special_tokens,
        }
    }
}

impl Encoder for MockTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        // Simple word-based tokenization for testing
        let tokens: Vec<u32> = input
            .split_whitespace()
            .filter_map(|word| self.vocab.get(word).copied())
            .collect();

        Ok(Encoding::Sp(tokens))
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.iter().map(|input| self.encode(input)).collect()
    }
}

impl Decoder for MockTokenizer {
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let tokens: Vec<String> = token_ids
            .iter()
            .filter_map(|id| {
                self.reverse_vocab.get(id).and_then(|token| {
                    if skip_special_tokens && (token == "<eos>" || token == "<bos>") {
                        None
                    } else {
                        Some(token.clone())
                    }
                })
            })
            .collect();

        Ok(tokens.join(" "))
    }
}

impl TokenizerTrait for MockTokenizer {
    fn vocab_size(&self) -> usize {
        self.vocab.len()
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

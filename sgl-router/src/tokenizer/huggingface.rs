use super::traits::{
    Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer as TokenizerTrait,
};
use anyhow::{Error, Result};
use std::collections::HashMap;
use tokenizers::tokenizer::Tokenizer as HfTokenizer;

use super::chat_template::{ChatMessage, ChatTemplateProcessor};

/// HuggingFace tokenizer wrapper
pub struct HuggingFaceTokenizer {
    tokenizer: HfTokenizer,
    special_tokens: SpecialTokens,
    vocab: HashMap<String, TokenIdType>,
    reverse_vocab: HashMap<TokenIdType, String>,
    chat_template: Option<String>,
}

impl HuggingFaceTokenizer {
    /// Create a tokenizer from a HuggingFace tokenizer JSON file
    pub fn from_file(file_path: &str) -> Result<Self> {
        Self::from_file_with_chat_template(file_path, None)
    }

    /// Create a tokenizer from a HuggingFace tokenizer JSON file with an optional chat template
    pub fn from_file_with_chat_template(
        file_path: &str,
        chat_template_path: Option<&str>,
    ) -> Result<Self> {
        let tokenizer = HfTokenizer::from_file(file_path)
            .map_err(|e| Error::msg(format!("Failed to load tokenizer: {}", e)))?;

        // Extract special tokens
        let special_tokens = Self::extract_special_tokens(&tokenizer);

        // Build vocab mappings
        let vocab = tokenizer.get_vocab(false);
        let reverse_vocab: HashMap<TokenIdType, String> = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();

        // Load chat template
        let chat_template = if let Some(template_path) = chat_template_path {
            // Load from specified .jinja file
            Self::load_chat_template_from_file(template_path)?
        } else {
            // Try to load from tokenizer_config.json
            Self::load_chat_template(file_path)
        };

        Ok(HuggingFaceTokenizer {
            tokenizer,
            special_tokens,
            vocab,
            reverse_vocab,
            chat_template,
        })
    }

    /// Create from an existing HuggingFace tokenizer
    pub fn from_tokenizer(tokenizer: HfTokenizer) -> Self {
        let special_tokens = Self::extract_special_tokens(&tokenizer);
        let vocab = tokenizer.get_vocab(false);
        let reverse_vocab: HashMap<TokenIdType, String> = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();

        HuggingFaceTokenizer {
            tokenizer,
            special_tokens,
            vocab,
            reverse_vocab,
            chat_template: None,
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

    /// Try to load chat template from tokenizer_config.json
    fn load_chat_template(tokenizer_path: &str) -> Option<String> {
        // Try to find tokenizer_config.json in the same directory
        let path = std::path::Path::new(tokenizer_path);
        let dir = path.parent()?;
        let config_path = dir.join("tokenizer_config.json");

        if config_path.exists() {
            if let Ok(template) =
                super::chat_template::load_chat_template_from_config(config_path.to_str()?)
            {
                return template;
            }
        }
        None
    }

    /// Load chat template from a .jinja file
    fn load_chat_template_from_file(template_path: &str) -> Result<Option<String>> {
        use std::fs;

        let content = fs::read_to_string(template_path)
            .map_err(|e| Error::msg(format!("Failed to read chat template file: {}", e)))?;

        // Clean up the template (similar to Python implementation)
        let template = content.trim().replace("\\n", "\n");

        Ok(Some(template))
    }

    /// Set or override the chat template
    pub fn set_chat_template(&mut self, template: String) {
        self.chat_template = Some(template);
    }

    /// Apply chat template if available
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        if let Some(ref template) = self.chat_template {
            let processor = ChatTemplateProcessor::new(
                template.clone(),
                self.special_tokens.bos_token.clone(),
                self.special_tokens.eos_token.clone(),
            );
            processor.apply_chat_template(messages, add_generation_prompt)
        } else {
            // Fallback to simple formatting if no template is available
            let mut result = String::new();
            for msg in messages {
                result.push_str(&format!("{}: {}\n", msg.role, msg.content));
            }
            if add_generation_prompt {
                result.push_str("assistant: ");
            }
            Ok(result)
        }
    }
}

impl Encoder for HuggingFaceTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        self.tokenizer
            .encode(input, false)
            .map_err(|e| Error::msg(format!("Encoding failed: {}", e)))
            .map(|encoding| Encoding::Hf(Box::new(encoding)))
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        let encodings = self
            .tokenizer
            .encode_batch(inputs.to_vec(), false)
            .map_err(|e| Error::msg(format!("Batch encoding failed: {}", e)))?;

        Ok(encodings
            .into_iter()
            .map(|e| Encoding::Hf(Box::new(e)))
            .collect())
    }
}

impl Decoder for HuggingFaceTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| Error::msg(format!("Decoding failed: {}", e)))
    }
}

impl TokenizerTrait for HuggingFaceTokenizer {
    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn token_to_id(&self, token: &str) -> Option<TokenIdType> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: TokenIdType) -> Option<String> {
        self.reverse_vocab.get(&id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::ChatMessage;

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

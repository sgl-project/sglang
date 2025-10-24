use std::collections::HashMap;

use anyhow::{Error, Result};
use tokenizers::tokenizer::Tokenizer as HfTokenizer;

use super::{
    chat_template::{
        detect_chat_template_content_format, ChatTemplateContentFormat, ChatTemplateParams,
        ChatTemplateProcessor,
    },
    traits::{Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer as TokenizerTrait},
};

/// HuggingFace tokenizer wrapper
pub struct HuggingFaceTokenizer {
    tokenizer: HfTokenizer,
    special_tokens: SpecialTokens,
    vocab: HashMap<String, TokenIdType>,
    reverse_vocab: HashMap<TokenIdType, String>,
    chat_template: Option<String>,
    /// Detected chat template content format (computed once at initialization)
    content_format: ChatTemplateContentFormat,
}

impl HuggingFaceTokenizer {
    /// Create a tokenizer from a HuggingFace tokenizer JSON file
    pub fn from_file(file_path: &str) -> Result<Self> {
        // Try to auto-discover chat template if not explicitly provided
        let path = std::path::Path::new(file_path);
        let chat_template_path = path
            .parent()
            .and_then(crate::tokenizer::factory::discover_chat_template_in_dir);
        Self::from_file_with_chat_template(file_path, chat_template_path.as_deref())
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

        // Build vocab mappings (include special tokens to get added_tokens like <|im_start|>)
        let vocab = tokenizer.get_vocab(true); // true = include special tokens and added_tokens
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

        // Detect content format once at initialization
        let content_format = if let Some(ref template) = chat_template {
            detect_chat_template_content_format(template)
        } else {
            ChatTemplateContentFormat::String // Default if no template
        };

        Ok(HuggingFaceTokenizer {
            tokenizer,
            special_tokens,
            vocab,
            reverse_vocab,
            chat_template,
            content_format,
        })
    }

    /// Create from an existing HuggingFace tokenizer
    pub fn from_tokenizer(tokenizer: HfTokenizer) -> Self {
        let special_tokens = Self::extract_special_tokens(&tokenizer);
        let vocab = tokenizer.get_vocab(true); // true = include special tokens and added_tokens
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
            content_format: ChatTemplateContentFormat::String, // Default
        }
    }

    /// Extract special tokens from the tokenizer
    fn extract_special_tokens(tokenizer: &HfTokenizer) -> SpecialTokens {
        // Get vocab with special tokens included (added_tokens like <|im_start|>)
        let vocab = tokenizer.get_vocab(true);

        let find_token = |patterns: &[&str]| -> Option<String> {
            for pattern in patterns {
                if vocab.contains_key(*pattern) {
                    return Some(pattern.to_string());
                }
            }
            None
        };

        // Extract additional special tokens using the tokenizers library API
        let additional_special_tokens: Vec<String> = tokenizer
            .get_added_tokens_decoder()
            .iter()
            .filter(|(_id, token)| token.special) // Only tokens marked as special: true
            .map(|(_id, token)| token.content.clone())
            .collect();

        SpecialTokens {
            bos_token: find_token(&["<s>", "<|startoftext|>", "<BOS>", "[CLS]"]),
            eos_token: find_token(&["</s>", "<|endoftext|>", "<EOS>", "[SEP]"]),
            unk_token: find_token(&["<unk>", "<UNK>", "[UNK]"]),
            sep_token: find_token(&["[SEP]", "<sep>", "<SEP>"]),
            pad_token: find_token(&["<pad>", "<PAD>", "[PAD]"]),
            cls_token: find_token(&["[CLS]", "<cls>", "<CLS>"]),
            mask_token: find_token(&["[MASK]", "<mask>", "<MASK>"]),
            additional_special_tokens,
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

    /// Load chat template from a file (.jinja or .json containing Jinja)
    fn load_chat_template_from_file(template_path: &str) -> Result<Option<String>> {
        use std::fs;

        let content = fs::read_to_string(template_path)
            .map_err(|e| Error::msg(format!("Failed to read chat template file: {}", e)))?;

        // Check if it's a JSON file containing a Jinja template
        if template_path.ends_with(".json") {
            // Parse JSON and extract the template string
            let json_value: serde_json::Value = serde_json::from_str(&content)
                .map_err(|e| Error::msg(format!("Failed to parse chat_template.json: {}", e)))?;

            if let Some(template_str) = json_value.as_str() {
                return Ok(Some(template_str.to_string()));
            } else if let Some(obj) = json_value.as_object() {
                if let Some(template_value) = obj.get("chat_template") {
                    if let Some(template_str) = template_value.as_str() {
                        return Ok(Some(template_str.to_string()));
                    }
                }
            }

            return Err(Error::msg(
                "chat_template.json does not contain a valid template",
            ));
        }

        // Otherwise it's a plain .jinja file
        // Clean up the template (similar to Python implementation)
        let template = content.trim().replace("\\n", "\n");

        Ok(Some(template))
    }

    /// Set or override the chat template
    pub fn set_chat_template(&mut self, template: String) {
        // Detect format for the new template
        self.content_format = detect_chat_template_content_format(&template);
        self.chat_template = Some(template);
    }

    /// Get the content format expected by the chat template
    pub fn chat_template_content_format(&self) -> ChatTemplateContentFormat {
        self.content_format
    }

    /// Apply chat template if available
    ///
    /// Takes transformed JSON Values (already transformed based on content format)
    pub fn apply_chat_template(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<String> {
        if let Some(ref template) = self.chat_template {
            let processor = ChatTemplateProcessor::new(template.clone());
            processor.apply_chat_template(messages, params)
        } else {
            Err(Error::msg(
                "Cannot use chat template functions because tokenizer.chat_template is not set and no template \
                argument was passed! For information about writing templates and setting the \
                tokenizer.chat_template attribute, please see the documentation at \
                https://huggingface.co/docs/transformers/main/en/chat_templating"
            ))
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    // Note: Actual tokenizer tests would require a real tokenizer file
    // These would be integration tests rather than unit tests
}

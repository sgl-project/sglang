use anyhow::{Error, Result};
use tiktoken_rs::{cl100k_base, p50k_base, p50k_edit, r50k_base, CoreBPE};

use super::traits::{
    Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer as TokenizerTrait,
};

/// Tiktoken tokenizer wrapper for OpenAI GPT models
pub struct TiktokenTokenizer {
    tokenizer: CoreBPE,
    #[allow(dead_code)]
    model: TiktokenModel,
    special_tokens: SpecialTokens,
    vocab_size: usize,
}

/// Supported Tiktoken models
#[derive(Debug, Clone, Copy)]
pub enum TiktokenModel {
    /// GPT-4, GPT-3.5-turbo, text-embedding-ada-002
    Cl100kBase,
    /// Codex models, text-davinci-002, text-davinci-003
    P50kBase,
    /// Use for edit models like text-davinci-edit-001, code-davinci-edit-001
    P50kEdit,
    /// GPT-3 models like davinci
    R50kBase,
}

impl TiktokenTokenizer {
    /// Create a new Tiktoken tokenizer for the specified model
    pub fn new(model: TiktokenModel) -> Result<Self> {
        let tokenizer =
            match model {
                TiktokenModel::Cl100kBase => cl100k_base()
                    .map_err(|e| Error::msg(format!("Failed to load cl100k_base: {}", e)))?,
                TiktokenModel::P50kBase => p50k_base()
                    .map_err(|e| Error::msg(format!("Failed to load p50k_base: {}", e)))?,
                TiktokenModel::P50kEdit => p50k_edit()
                    .map_err(|e| Error::msg(format!("Failed to load p50k_edit: {}", e)))?,
                TiktokenModel::R50kBase => r50k_base()
                    .map_err(|e| Error::msg(format!("Failed to load r50k_base: {}", e)))?,
            };

        // Extract special tokens (tiktoken-rs doesn't expose them directly)
        // We'll use common ones for GPT models
        let special_tokens = Self::get_special_tokens_for_model(model);

        // Get vocabulary size (this is an approximation)
        let vocab_size = match model {
            TiktokenModel::Cl100kBase => 100256, // cl100k has ~100k tokens
            TiktokenModel::P50kBase | TiktokenModel::P50kEdit => 50281, // p50k has ~50k tokens
            TiktokenModel::R50kBase => 50257,    // r50k has ~50k tokens
        };

        Ok(TiktokenTokenizer {
            tokenizer,
            model,
            special_tokens,
            vocab_size,
        })
    }

    /// Create a tokenizer from a model string (e.g., "gpt-4", "gpt-3.5-turbo")
    pub fn from_model_name(model_name: &str) -> Result<Self> {
        let model = Self::model_from_name(model_name)?;
        Self::new(model)
    }

    /// Determine the appropriate model from a model name
    fn model_from_name(model_name: &str) -> Result<TiktokenModel> {
        // Based on OpenAI's model-to-encoding mapping
        if model_name.contains("gpt-4")
            || model_name.contains("gpt-3.5")
            || model_name.contains("turbo")
        {
            Ok(TiktokenModel::Cl100kBase)
        } else if model_name.contains("davinci-002")
            || model_name.contains("davinci-003")
            || model_name.contains("codex")
        {
            Ok(TiktokenModel::P50kBase)
        } else if model_name.contains("edit") {
            Ok(TiktokenModel::P50kEdit)
        } else if model_name.contains("davinci")
            || model_name.contains("curie")
            || model_name.contains("babbage")
            || model_name.contains("ada")
        {
            Ok(TiktokenModel::R50kBase)
        } else {
            // Return an error for unrecognized model names to prevent silent failures
            Err(anyhow::anyhow!(
                "Unrecognized OpenAI model name: '{}'. Expected GPT-3, GPT-3.5, GPT-4, or related model names",
                model_name
            ))
        }
    }

    /// Get special tokens for a specific model
    fn get_special_tokens_for_model(model: TiktokenModel) -> SpecialTokens {
        // These are common special tokens for GPT models
        // The actual token IDs might vary by model
        match model {
            TiktokenModel::Cl100kBase => SpecialTokens {
                bos_token: Some("<|endoftext|>".to_string()),
                eos_token: Some("<|endoftext|>".to_string()),
                unk_token: None,
                sep_token: None,
                pad_token: Some("<|endoftext|>".to_string()),
                cls_token: None,
                mask_token: None,
                additional_special_tokens: vec![
                    "<|fim_prefix|>".to_string(),
                    "<|fim_middle|>".to_string(),
                    "<|fim_suffix|>".to_string(),
                    "<|endofprompt|>".to_string(),
                ],
            },
            _ => SpecialTokens {
                bos_token: Some("<|endoftext|>".to_string()),
                eos_token: Some("<|endoftext|>".to_string()),
                unk_token: None,
                sep_token: None,
                pad_token: Some("<|endoftext|>".to_string()),
                cls_token: None,
                mask_token: None,
                additional_special_tokens: vec![],
            },
        }
    }
}

impl Encoder for TiktokenTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        let tokens = self.tokenizer.encode_ordinary(input);
        Ok(Encoding::Tiktoken(tokens))
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.iter().map(|input| self.encode(input)).collect()
    }
}

impl Decoder for TiktokenTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], _skip_special_tokens: bool) -> Result<String> {
        // tiktoken-rs 0.7.0 now uses u32 (Rank type)
        self.tokenizer
            .decode(token_ids.to_vec())
            .map_err(|e| Error::msg(format!("Decoding failed: {}", e)))
    }
}

impl TokenizerTrait for TiktokenTokenizer {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn token_to_id(&self, _token: &str) -> Option<TokenIdType> {
        // Tiktoken doesn't provide direct token-to-id mapping
        // We'd need to encode the token and check if it produces a single ID
        None
    }

    fn id_to_token(&self, _id: TokenIdType) -> Option<String> {
        // Tiktoken doesn't provide direct id-to-token mapping
        // We can only decode IDs to text
        None
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiktoken_creation() {
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();
        assert_eq!(tokenizer.vocab_size(), 100256);
    }

    #[test]
    fn test_model_from_name() {
        assert!(matches!(
            TiktokenTokenizer::model_from_name("gpt-4").unwrap(),
            TiktokenModel::Cl100kBase
        ));
        assert!(matches!(
            TiktokenTokenizer::model_from_name("gpt-3.5-turbo").unwrap(),
            TiktokenModel::Cl100kBase
        ));
        assert!(matches!(
            TiktokenTokenizer::model_from_name("text-davinci-003").unwrap(),
            TiktokenModel::P50kBase
        ));
        assert!(matches!(
            TiktokenTokenizer::model_from_name("text-davinci-edit-001").unwrap(),
            TiktokenModel::P50kEdit
        ));
        assert!(matches!(
            TiktokenTokenizer::model_from_name("davinci").unwrap(),
            TiktokenModel::R50kBase
        ));
    }

    #[test]
    fn test_encode_decode() {
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();

        let text = "Hello, world!";
        let encoding = tokenizer.encode(text).unwrap();

        let decoded = tokenizer.decode(encoding.token_ids(), false).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_batch_encode() {
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();

        let texts = vec!["Hello", "World", "Test"];
        let encodings = tokenizer.encode_batch(&texts).unwrap();

        assert_eq!(encodings.len(), 3);
        for (i, encoding) in encodings.iter().enumerate() {
            let decoded = tokenizer.decode(encoding.token_ids(), false).unwrap();
            assert_eq!(decoded, texts[i]);
        }
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();
        let special_tokens = tokenizer.get_special_tokens();

        assert!(special_tokens.eos_token.is_some());
        assert_eq!(special_tokens.eos_token.as_ref().unwrap(), "<|endoftext|>");
    }

    #[test]
    fn test_unrecognized_model_name_returns_error() {
        let result = TiktokenTokenizer::from_model_name("distilgpt-2");
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Unrecognized OpenAI model name"));
        }

        let result = TiktokenTokenizer::from_model_name("bert-base-uncased");
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Unrecognized OpenAI model name"));
        }

        let result = TiktokenTokenizer::from_model_name("llama-7b");
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Unrecognized OpenAI model name"));
        }
    }

    #[test]
    fn test_recognized_model_names() {
        assert!(TiktokenTokenizer::from_model_name("gpt-4").is_ok());
        assert!(TiktokenTokenizer::from_model_name("gpt-3.5-turbo").is_ok());
        assert!(TiktokenTokenizer::from_model_name("text-davinci-003").is_ok());
        assert!(TiktokenTokenizer::from_model_name("code-davinci-002").is_ok());
        assert!(TiktokenTokenizer::from_model_name("text-curie-001").is_ok());
        assert!(TiktokenTokenizer::from_model_name("text-babbage-001").is_ok());
        assert!(TiktokenTokenizer::from_model_name("text-ada-001").is_ok());
    }
}

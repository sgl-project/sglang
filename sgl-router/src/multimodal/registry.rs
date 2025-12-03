use std::collections::HashMap;

use once_cell::sync::Lazy;
use serde_json::{json, Value};
use thiserror::Error;

use super::types::{ImageSize, Modality, PromptReplacement, TokenId};
use crate::tokenizer::traits::Tokenizer as TokenizerTrait;

#[derive(Debug, Error)]
pub enum ModelRegistryError {
    #[error("unsupported model: {0}")]
    UnsupportedModel(String),
    #[error("token '{token}' not found in tokenizer vocabulary")]
    TokenNotFound { token: String },
    #[error("missing config field '{field}'")]
    MissingConfigField { field: String },
}

pub type RegistryResult<T> = Result<T, ModelRegistryError>;

/// Metadata about the current model used to derive tokenizer/config dependent fields.
pub struct ModelMetadata<'a> {
    pub model_id: &'a str,
    pub tokenizer: &'a dyn TokenizerTrait,
    pub config: &'a Value,
}

impl<'a> ModelMetadata<'a> {
    pub fn token_id(&self, token: &str) -> RegistryResult<TokenId> {
        self.tokenizer
            .token_to_id(token)
            .map(|id| id as TokenId)
            .ok_or_else(|| ModelRegistryError::TokenNotFound {
                token: token.to_string(),
            })
    }

    pub fn config_u32(&self, path: &[&str]) -> Option<u32> {
        Self::find_value(self.config, path).and_then(|value| value.as_u64().map(|v| v as u32))
    }

    fn find_value<'v>(value: &'v Value, path: &[&str]) -> Option<&'v Value> {
        let mut current = value;
        for key in path {
            current = current.get(*key)?;
        }
        Some(current)
    }
}

pub trait ModelProcessorSpec: Send + Sync {
    fn name(&self) -> &'static str;
    fn matches(&self, metadata: &ModelMetadata) -> bool;
    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String>;
    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId>;
    fn modality_limits(&self, metadata: &ModelMetadata)
        -> RegistryResult<HashMap<Modality, usize>>;
    fn processor_kwargs(&self, metadata: &ModelMetadata) -> RegistryResult<Value>;
    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        image_sizes: &[ImageSize],
    ) -> RegistryResult<Vec<PromptReplacement>>;
}

pub struct ModelRegistry {
    specs: Vec<LazySpec>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            specs: vec![
                LazySpec::new("llava", || Box::new(LlavaSpec)),
                LazySpec::new("qwen_vl", || Box::new(QwenVLVisionSpec)),
                LazySpec::new("phi3_v", || Box::new(Phi3VisionSpec)),
            ],
        }
    }

    pub fn lookup<'a>(&'a self, metadata: &ModelMetadata) -> Option<&'a dyn ModelProcessorSpec> {
        for spec in &self.specs {
            let spec_ref = spec.get();
            if spec_ref.matches(metadata) {
                return Some(spec_ref);
            }
        }
        None
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

struct LazySpec {
    inner: Lazy<Box<dyn ModelProcessorSpec>>,
}

impl LazySpec {
    fn new(_id: &'static str, factory: fn() -> Box<dyn ModelProcessorSpec>) -> Self {
        Self {
            inner: Lazy::new(factory),
        }
    }

    fn get(&self) -> &dyn ModelProcessorSpec {
        self.inner.as_ref()
    }
}

struct LlavaSpec;

impl LlavaSpec {
    fn patch_size(metadata: &ModelMetadata) -> u32 {
        metadata
            .config_u32(&["vision_config", "patch_size"])
            .unwrap_or(14)
    }

    fn tokens_per_image(metadata: &ModelMetadata, size: &ImageSize) -> usize {
        let patch = Self::patch_size(metadata);
        let cols = size.width.div_ceil(patch) as usize;
        let rows = size.height.div_ceil(patch) as usize;
        cols * rows
    }
}

impl ModelProcessorSpec for LlavaSpec {
    fn name(&self) -> &'static str {
        "llava"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata.model_id.to_ascii_lowercase().contains("llava")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<image>".to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        if let Some(value) = metadata.config_u32(&["image_token_index"]) {
            return Ok(value as TokenId);
        }
        metadata.token_id("<image>")
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 4)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        image_sizes: &[ImageSize],
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let token_id = self.placeholder_token_id(metadata)?;
        let token = self.placeholder_token(metadata)?;
        Ok(image_sizes
            .iter()
            .map(|size| {
                let count = Self::tokens_per_image(metadata, size);
                PromptReplacement::repeated(Modality::Image, &token, token_id, count)
            })
            .collect())
    }
}

struct QwenVLVisionSpec;

impl QwenVLVisionSpec {
    fn pad_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["vision_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "vision_token_id".to_string(),
            })
    }

    fn start_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["vision_start_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "vision_start_token_id".to_string(),
            })
    }

    fn patch_grid(metadata: &ModelMetadata, size: &ImageSize) -> (usize, usize) {
        let patch = metadata
            .config_u32(&["vision_config", "patch_size"])
            .unwrap_or(14);
        let cols = size.width.div_ceil(patch) as usize;
        let rows = size.height.div_ceil(patch) as usize;
        (rows, cols)
    }
}

impl ModelProcessorSpec for QwenVLVisionSpec {
    fn name(&self) -> &'static str {
        "qwen_vl"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata.model_id.to_ascii_lowercase().contains("qwen")
            && metadata.model_id.to_ascii_lowercase().contains("vl")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<image>".to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["image_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "image_token_id".to_string(),
            })
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 10)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        image_sizes: &[ImageSize],
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let start_token_id = Self::start_token_id(metadata)?;
        let pad_token_id = Self::pad_token_id(metadata)?;
        let placeholder_token = self.placeholder_token(metadata)?;
        Ok(image_sizes
            .iter()
            .map(|size| {
                let (rows, cols) = Self::patch_grid(metadata, size);
                let pad_len = rows * cols;
                let mut tokens = Vec::with_capacity(pad_len + 1);
                tokens.push(start_token_id);
                tokens.extend(std::iter::repeat_n(pad_token_id, pad_len));
                PromptReplacement::sequence(Modality::Image, &placeholder_token, tokens)
            })
            .collect())
    }
}

struct Phi3VisionSpec;

impl Phi3VisionSpec {
    fn tokens_per_image(metadata: &ModelMetadata) -> usize {
        metadata
            .config_u32(&["img_processor", "num_img_tokens"])
            .unwrap_or(256) as usize
    }
}

impl ModelProcessorSpec for Phi3VisionSpec {
    fn name(&self) -> &'static str {
        "phi3_v"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("phi") && id.contains("vision")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<image>".to_owned())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.token_id("<image>")
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 4)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        image_sizes: &[ImageSize],
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let token_id = self.placeholder_token_id(metadata)?;
        let token = self.placeholder_token(metadata)?;
        let count = Self::tokens_per_image(metadata);
        Ok(image_sizes
            .iter()
            .map(|_| PromptReplacement::repeated(Modality::Image, &token, token_id, count))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::*;
    use crate::tokenizer::traits::{
        Decoder, Encoder, Encoding, SpecialTokens, Tokenizer as TokenizerTrait,
    };

    struct TestTokenizer {
        vocab: HashMap<String, u32>,
    }

    impl TestTokenizer {
        fn new(pairs: &[(&str, u32)]) -> Self {
            let vocab = pairs
                .iter()
                .map(|(token, id)| ((*token).to_string(), *id))
                .collect();
            Self { vocab }
        }
    }

    impl Encoder for TestTokenizer {
        fn encode(&self, _input: &str) -> anyhow::Result<Encoding> {
            Ok(Encoding::Sp(Vec::new()))
        }

        fn encode_batch(&self, inputs: &[&str]) -> anyhow::Result<Vec<Encoding>> {
            inputs.iter().map(|_| self.encode("")).collect()
        }
    }

    impl Decoder for TestTokenizer {
        fn decode(&self, _token_ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
            Ok(String::new())
        }
    }

    impl TokenizerTrait for TestTokenizer {
        fn vocab_size(&self) -> usize {
            self.vocab.len()
        }

        fn get_special_tokens(&self) -> &SpecialTokens {
            static TOKENS: Lazy<SpecialTokens> = Lazy::new(|| SpecialTokens {
                bos_token: None,
                eos_token: None,
                unk_token: None,
                sep_token: None,
                pad_token: None,
                cls_token: None,
                mask_token: None,
                additional_special_tokens: vec![],
            });
            &TOKENS
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.vocab.get(token).copied()
        }

        fn id_to_token(&self, _id: u32) -> Option<String> {
            None
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn llava_prompt_replacement_uses_config_ids() {
        let tokenizer = TestTokenizer::new(&[("<image>", 32000)]);
        let config = json!({
            "model_type": "llava",
            "image_token_index": 32000,
            "vision_config": {"patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "llava-v1.5",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("llava spec");
        let replacements = spec
            .prompt_replacements(&metadata, &[ImageSize::new(336, 336)])
            .unwrap();
        assert_eq!(replacements[0].tokens.len(), 576);
    }

    #[test]
    fn qwen_vision_uses_config_token_ids() {
        let tokenizer = TestTokenizer::new(&[("<image>", 999)]);
        let config = json!({
            "model_type": "qwen2_vl",
            "vision_start_token_id": 151652,
            "vision_token_id": 151654,
            "image_token_id": 151655,
            "vision_config": {"patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "Qwen2-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen spec");
        let replacements = spec
            .prompt_replacements(&metadata, &[ImageSize::new(448, 448)])
            .unwrap();
        // 448/14 = 32 patches => 1024 pad tokens + 1 start token
        assert_eq!(replacements[0].tokens.len(), 1025);
        assert_eq!(replacements[0].tokens[0], 151652);
        assert_eq!(replacements[0].tokens[1], 151654);
    }

    #[test]
    fn phi3_uses_num_img_tokens() {
        let tokenizer = TestTokenizer::new(&[("<image>", 555)]);
        let config = json!({
            "model_type": "phi3_v",
            "img_processor": {"num_img_tokens": 144}
        });
        let metadata = ModelMetadata {
            model_id: "Phi-3-vision",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("phi3 spec");
        let replacements = spec
            .prompt_replacements(&metadata, &[ImageSize::new(336, 336)])
            .unwrap();
        assert_eq!(replacements[0].tokens.len(), 144);
        assert_eq!(replacements[0].tokens[0], 555);
    }
}

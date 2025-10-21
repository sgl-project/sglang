use std::{ops::Deref, sync::Arc};

use anyhow::Result;

pub mod cache;
pub mod factory;
pub mod hub;
pub mod mock;
pub mod sequence;
pub mod stop;
pub mod stream;
pub mod traits;

// Feature-gated modules

pub mod chat_template;

pub mod huggingface;

pub mod tiktoken;

#[cfg(test)]
mod tests;

// Re-exports
pub use cache::{CacheConfig, CacheStats, CachedTokenizer, TokenizerFingerprint};
pub use factory::{
    create_tokenizer, create_tokenizer_async, create_tokenizer_async_with_chat_template,
    create_tokenizer_from_file, create_tokenizer_with_chat_template,
    create_tokenizer_with_chat_template_blocking, TokenizerType,
};
pub use huggingface::HuggingFaceTokenizer;
pub use sequence::Sequence;
pub use stop::{SequenceDecoderOutput, StopSequenceConfig, StopSequenceDecoder};
pub use stream::DecodeStream;
pub use tiktoken::{TiktokenModel, TiktokenTokenizer};
pub use traits::{Decoder, Encoder, Encoding, SpecialTokens, Tokenizer as TokenizerTrait};

/// Main tokenizer wrapper that provides a unified interface for different tokenizer implementations
#[derive(Clone)]
pub struct Tokenizer(Arc<dyn traits::Tokenizer>);

impl Tokenizer {
    /// Create a tokenizer from a file path
    pub fn from_file(file_path: &str) -> Result<Tokenizer> {
        Ok(Tokenizer(create_tokenizer_from_file(file_path)?))
    }

    /// Create a tokenizer from a file path with an optional chat template
    pub fn from_file_with_chat_template(
        file_path: &str,
        chat_template_path: Option<&str>,
    ) -> Result<Tokenizer> {
        Ok(Tokenizer(create_tokenizer_with_chat_template(
            file_path,
            chat_template_path,
        )?))
    }

    /// Create a tokenizer from an Arc<dyn Tokenizer>
    pub fn from_arc(tokenizer: Arc<dyn traits::Tokenizer>) -> Self {
        Tokenizer(tokenizer)
    }

    /// Create a stateful sequence object for decoding token_ids into text
    pub fn decode_stream(
        &self,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> DecodeStream {
        DecodeStream::new(self.0.clone(), prompt_token_ids, skip_special_tokens)
    }

    /// Direct encode method
    pub fn encode(&self, input: &str) -> Result<Encoding> {
        self.0.encode(input)
    }

    /// Direct batch encode method
    pub fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        self.0.encode_batch(inputs)
    }

    /// Direct decode method
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.0.decode(token_ids, skip_special_tokens)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }

    /// Get special tokens
    pub fn get_special_tokens(&self) -> &SpecialTokens {
        self.0.get_special_tokens()
    }

    /// Convert token string to ID
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }

    /// Convert ID to token string
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id)
    }
}

impl Deref for Tokenizer {
    type Target = Arc<dyn traits::Tokenizer>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Arc<dyn traits::Tokenizer>> for Tokenizer {
    fn from(tokenizer: Arc<dyn traits::Tokenizer>) -> Self {
        Tokenizer(tokenizer)
    }
}

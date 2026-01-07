use std::{ops::Deref, sync::Arc};

use anyhow::Result;

pub mod cache;
pub mod factory;
pub mod hub;
pub mod mock;
pub mod registry;
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

// Internal imports for Tokenizer struct
use factory::{create_tokenizer_from_file, create_tokenizer_with_chat_template};
// Re-export types used outside this module
pub use huggingface::HuggingFaceTokenizer;
pub use registry::TokenizerRegistry;
pub use stop::StopSequenceDecoder;
pub use stream::DecodeStream;
pub use traits::{Decoder, Encoder, Encoding, SpecialTokens};

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
    ///
    /// Set `add_special_tokens` to `true` for embeddings (to add BOS/EOS tokens configured in tokenizer_config.json),
    /// or `false` for chat completion (where the chat template handles special tokens).
    pub fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
        self.0.encode(input, add_special_tokens)
    }

    /// Direct batch encode method
    ///
    /// Set `add_special_tokens` to `true` for embeddings (to add BOS/EOS tokens configured in tokenizer_config.json),
    /// or `false` for chat completion (where the chat template handles special tokens).
    pub fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        self.0.encode_batch(inputs, add_special_tokens)
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

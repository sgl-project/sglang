use anyhow::Result;

/// Core encoding trait - separate from decoding for modularity
pub trait Encoder: Send + Sync {
    fn encode(&self, input: &str) -> Result<Encoding>;
    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>>;
}

/// Core decoding trait - can be implemented independently
pub trait Decoder: Send + Sync {
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String>;
}

/// Combined tokenizer trait
pub trait Tokenizer: Encoder + Decoder {
    fn vocab_size(&self) -> usize;
    fn get_special_tokens(&self) -> &SpecialTokens;
    fn token_to_id(&self, token: &str) -> Option<u32>;
    fn id_to_token(&self, id: u32) -> Option<String>;
}

/// Contains the results of tokenizing text: token IDs, string tokens, and their spans
#[derive(Debug, Clone)]
pub enum Encoding {
    /// Hugging Face
    Hf(Box<tokenizers::tokenizer::Encoding>),
    /// Sentence Piece
    Sp(Vec<u32>),
    /// Tiktoken (for GPT models)
    Tiktoken(Vec<usize>),
}

impl Encoding {
    pub fn token_ids(&self) -> Vec<u32> {
        match self {
            Encoding::Hf(inner) => inner.get_ids().to_vec(),
            Encoding::Sp(inner) => inner.clone(),
            Encoding::Tiktoken(inner) => inner.iter().map(|&id| id as u32).collect(),
        }
    }

    pub fn token_ids_ref(&self) -> &[u32] {
        match self {
            Encoding::Hf(inner) => inner.get_ids(),
            Encoding::Sp(inner) => inner,
            Encoding::Tiktoken(_) => {
                // Tiktoken uses usize, we can't return a reference to u32
                // This is a limitation - callers should use token_ids() for Tiktoken
                &[]
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unk_token: Option<String>,
    pub sep_token: Option<String>,
    pub pad_token: Option<String>,
    pub cls_token: Option<String>,
    pub mask_token: Option<String>,
    pub additional_special_tokens: Vec<String>,
}

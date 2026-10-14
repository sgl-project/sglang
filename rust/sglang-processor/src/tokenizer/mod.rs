//! Tokenizer loading, encoding and decoding over Dynamo tokenizers.

use crate::ProcessorError as Error;
use crate::model_files::resolve_tokenizer_file;

/// Pluggable text→token-ids backend. `Send + Sync` so one instance is shared
/// (read-only) across all pinned workers.
pub trait TextTokenizer: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>, Error>;

    fn encode_segments(
        &self,
        segments: &[dynamo_tokenizers::EncodeSegment<'_>],
        add_special_tokens: bool,
    ) -> Result<Vec<i32>, Error> {
        let text = segments
            .iter()
            .map(|segment| segment.text)
            .collect::<String>();
        self.encode(&text, add_special_tokens)
    }

    /// Decode token IDs back to text. Encode-only backends keep the default.
    fn decode(&self, _token_ids: &[u32], _skip_special_tokens: bool) -> Result<String, Error> {
        Err(Error::Internal(
            "the configured tokenizer does not support decoding".into(),
        ))
    }
}

/// Real tokenizer over two already-loaded dynamo handles. Dynamo fixes
/// `add_special_tokens` when loading, so selecting the mode at request time
/// requires one handle for each setting.
pub struct DynamoTokenizer {
    without_specials: dynamo_tokenizers::Tokenizer,
    with_specials: dynamo_tokenizers::Tokenizer,
}

/// Load the tokenizer shared (Arc-backed) by the encode pool and detok shards.
/// `tokenizer_path` is a tokenizer file, a model dir, or an HF Hub repo id
/// (resolved from the local cache — no network).
pub fn load_tokenizer(
    tokenizer_path: Option<&str>,
    revision: Option<&str>,
    add_special_tokens: bool,
) -> Result<dynamo_tokenizers::Tokenizer, String> {
    let path =
        tokenizer_path.ok_or_else(|| "no tokenizer configured: set tokenizer_path".to_string())?;
    let file = resolve_tokenizer_file(path, revision).ok_or_else(|| {
        format!(
            "no supported tokenizer file found for '{path}' (expected tokenizer.json, tiktoken.model, or *.tiktoken)"
        )
    })?;
    let tokenizer = dynamo_tokenizers::Tokenizer::from_file_with_options(
        &file,
        dynamo_tokenizers::TokenizerOptions { add_special_tokens },
    )
    .map_err(|e| format!("tokenizer load failed ({file}): {e}"))?;
    tracing::info!(%path, "loaded tokenizer");
    Ok(tokenizer)
}

impl DynamoTokenizer {
    pub fn new(
        without_specials: dynamo_tokenizers::Tokenizer,
        with_specials: dynamo_tokenizers::Tokenizer,
    ) -> Self {
        Self {
            without_specials,
            with_specials,
        }
    }
}

impl TextTokenizer for DynamoTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>, Error> {
        let encoding = if add_special_tokens {
            &self.with_specials
        } else {
            &self.without_specials
        }
        .encode(text)
        .map_err(|e| Error::Tokenize(e.to_string()))?;
        // Vocab ids are non-negative and fit in i32.
        Ok(encoding.token_ids().iter().map(|&id| id as i32).collect())
    }

    fn encode_segments(
        &self,
        segments: &[dynamo_tokenizers::EncodeSegment<'_>],
        add_special_tokens: bool,
    ) -> Result<Vec<i32>, Error> {
        // Segments that all allow special tokens encode one by one, which keeps their
        // boundaries on backends without segmented encoding, such as Hugging Face's.
        if segments.iter().all(|segment| segment.allow_special) {
            let mut ids = Vec::new();
            for (index, segment) in segments.iter().enumerate() {
                ids.extend(self.encode(segment.text, add_special_tokens && index == 0)?);
            }
            return Ok(ids);
        }
        let encoding = if add_special_tokens {
            &self.with_specials
        } else {
            &self.without_specials
        }
        .encode_segments(segments)
        .map_err(|error| Error::Tokenize(error.to_string()))?;
        Ok(encoding.token_ids().iter().map(|&id| id as i32).collect())
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String, Error> {
        // `add_special_tokens` only affects encoding, so either handle decodes.
        self.without_specials
            .decode(token_ids, skip_special_tokens)
            .map(String::from)
            .map_err(|error| Error::InvalidRequest(format!("Error decoding tokens: {error}")))
    }
}

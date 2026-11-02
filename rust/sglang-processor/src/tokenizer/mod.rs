//! Tokenizer loading and encoding over Dynamo tokenizers.

use crate::ProcessorError as Error;
use crate::model_files::resolve_tokenizer_file;

/// Pluggable text→token-ids backend. `Send + Sync` so one instance is shared
/// (read-only) across all pinned workers.
pub trait TextTokenizer: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>, Error>;

    /// Python's `tokenizer.bos_token_id`.
    fn bos_token_id(&self) -> Option<i32> {
        None
    }

    /// Python's chat encode: the template without specials, then the continuation
    /// minus one leading BOS (`_append_assistant_prefix_to_prompt_ids`).
    fn encode_chat(
        &self,
        prompt: &[dynamo_tokenizers::EncodeSegment<'_>],
        continuation: &str,
    ) -> Result<Vec<i32>, Error> {
        let mut ids = self.encode_segments(prompt, false)?;
        if !continuation.is_empty() {
            let suffix = self.encode(continuation, true)?;
            let bos = self.bos_token_id();
            ids.extend_from_slice(
                bos.and_then(|bos| suffix.strip_prefix(&[bos]))
                    .unwrap_or(&suffix),
            );
        }
        Ok(ids)
    }

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
}

/// Real tokenizer over two already-loaded dynamo handles. Dynamo fixes
/// `add_special_tokens` when loading, so selecting the mode at request time
/// requires one handle for each setting.
pub struct DynamoTokenizer {
    without_specials: dynamo_tokenizers::Tokenizer,
    with_specials: dynamo_tokenizers::Tokenizer,
    bos: Option<i32>,
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
    /// Over loaded handles, with no configured BOS for
    /// [`TextTokenizer::encode_chat`] to drop; [`Self::load`] reads it.
    pub fn new(
        without_specials: dynamo_tokenizers::Tokenizer,
        with_specials: dynamo_tokenizers::Tokenizer,
    ) -> Self {
        Self {
            without_specials,
            with_specials,
            bos: None,
        }
    }

    /// Load the tokenizer at `path` (see [`load_tokenizer`]) in both
    /// special-token modes, with the BOS its `tokenizer_config.json` names.
    pub fn load(path: &str, revision: Option<&str>) -> Result<Self, String> {
        let mut tokenizer = Self::new(
            load_tokenizer(Some(path), revision, false)?,
            load_tokenizer(Some(path), revision, true)?,
        );
        tokenizer.bos = configured_bos(&tokenizer.without_specials, path, revision);
        Ok(tokenizer)
    }

    /// The handle that adds the tokenizer's specials, which decoding shares.
    pub fn tokenizer(&self) -> &dynamo_tokenizers::Tokenizer {
        &self.with_specials
    }
}

/// `bos_token` from `tokenizer_config.json`, looked up as Python's `bos_token_id`;
/// tiktoken has no vocab lookup, so it encodes the token as a special, which must
/// give one id.
fn configured_bos(
    tokenizer: &dynamo_tokenizers::Tokenizer,
    path: &str,
    revision: Option<&str>,
) -> Option<i32> {
    let config = crate::resolve_model_file(path, revision, "tokenizer_config.json")?;
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(config).ok()?).ok()?;
    let bos = &config["bos_token"];
    let bos = bos.as_str().or_else(|| bos["content"].as_str())?;
    if let Ok(Some(id)) = tokenizer.token_to_id(bos) {
        return i32::try_from(id).ok();
    }
    let ids = tokenizer
        .encode_segments(&[dynamo_tokenizers::EncodeSegment::new(bos, true)])
        .ok()?;
    match ids.token_ids() {
        &[id] => i32::try_from(id).ok(),
        _ => None,
    }
}

impl TextTokenizer for DynamoTokenizer {
    fn bos_token_id(&self) -> Option<i32> {
        self.bos
    }

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
}

#[cfg(test)]
mod tests {
    use dynamo_tokenizers::EncodeSegment;

    use super::*;

    /// With a configured, auto-added `<s>`, the template encodes without it and the
    /// continuation loses one leading BOS, as on Python.
    #[test]
    fn chat_encode_drops_one_configured_bos() {
        let dir = std::env::temp_dir().join(format!("processor-bos-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tokenizer = serde_json::json!({
            "added_tokens": [{"id": 0, "content": "<s>", "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true}],
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": {"type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [{"Sequence": {"id": "A", "type_id": 0}}],
                "special_tokens": {"<s>": {"id": "<s>", "ids": [0], "tokens": ["<s>"]}}},
            "model": {"type": "WordLevel", "unk_token": "[UNK]",
                "vocab": {"<s>": 0, "[UNK]": 1, "Hi": 2, "It": 3}},
        });
        let file = dir.join("tokenizer.json");
        std::fs::write(&file, tokenizer.to_string()).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), r#"{"bos_token": "<s>"}"#).unwrap();
        let tokenizer = DynamoTokenizer::load(file.to_str().unwrap(), None).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(tokenizer.bos_token_id(), Some(0));

        let prompt = [EncodeSegment::new("Hi", true)];
        assert_eq!(tokenizer.encode_chat(&prompt, "It").unwrap(), vec![2, 3]);
        // Only one BOS is dropped, as Python's: the auto-added one here.
        assert_eq!(
            tokenizer.encode_chat(&prompt, "<s><s>It").unwrap(),
            vec![2, 0, 0, 3]
        );
    }
}

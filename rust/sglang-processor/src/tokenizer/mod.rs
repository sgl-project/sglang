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
    /// Set when Python restores `add_bos_token` (see [`restores_add_bos`]): an
    /// encode with specials is then this BOS plus the encode without them.
    restored_bos: Option<i32>,
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
            restored_bos: None,
        }
    }

    /// Load the tokenizer at `path` (see [`load_tokenizer`]) in both
    /// special-token modes, with the BOS its `tokenizer_config.json` names.
    pub fn load(path: &str, revision: Option<&str>) -> Result<Self, String> {
        let mut tokenizer = Self::new(
            load_tokenizer(Some(path), revision, false)?,
            load_tokenizer(Some(path), revision, true)?,
        );
        let config = crate::resolve_model_file(path, revision, "tokenizer_config.json")
            .and_then(|file| std::fs::read_to_string(file).ok())
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok());
        if let Some(config) = config {
            tokenizer.bos = configured_bos(&tokenizer.without_specials, &config);
            tokenizer.restored_bos = tokenizer.bos.filter(|_| restores_add_bos(&config));
        }
        Ok(tokenizer)
    }

    /// The handle to encode with, and the BOS to put before its ids.
    fn handle(&self, add_special_tokens: bool) -> (&dynamo_tokenizers::Tokenizer, Option<i32>) {
        match (add_special_tokens, self.restored_bos) {
            (false, _) => (&self.without_specials, None),
            (true, None) => (&self.with_specials, None),
            (true, Some(bos)) => (&self.without_specials, Some(bos)),
        }
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
    config: &serde_json::Value,
) -> Option<i32> {
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

/// Tokenizer classes whose `add_bos_token` Python's `_fix_v5_add_bos_eos_token`
/// (`python/sglang/srt/hf_transformers/tokenizer.py`) restores; keep in sync.
const RESTORED_ADD_BOS_CLASSES: &[&str] = &[
    "LlamaTokenizer",
    "LlamaTokenizerFast",
    "CodeLlamaTokenizer",
    "CodeLlamaTokenizerFast",
    "GemmaTokenizer",
    "GemmaTokenizerFast",
    "CohereTokenizerFast",
];

/// Whether Python restores `add_bos_token=True`, which rebuilds the post-processor
/// to `<bos> $A` whatever `tokenizer.json` holds. A missing or null flag takes the
/// transformers v4 default of these classes, true.
fn restores_add_bos(config: &serde_json::Value) -> bool {
    config["tokenizer_class"]
        .as_str()
        .is_some_and(|class| RESTORED_ADD_BOS_CLASSES.contains(&class))
        && config["add_bos_token"].as_bool().unwrap_or(true)
}

/// Vocab ids are non-negative and fit in i32.
fn with_bos(bos: Option<i32>, encoding: &dynamo_tokenizers::Encoding) -> Vec<i32> {
    bos.into_iter()
        .chain(encoding.token_ids().iter().map(|&id| id as i32))
        .collect()
}

impl TextTokenizer for DynamoTokenizer {
    fn bos_token_id(&self) -> Option<i32> {
        self.bos
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>, Error> {
        let (handle, bos) = self.handle(add_special_tokens);
        let encoding = handle
            .encode(text)
            .map_err(|e| Error::Tokenize(e.to_string()))?;
        Ok(with_bos(bos, &encoding))
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
        let (handle, bos) = self.handle(add_special_tokens);
        let encoding = handle
            .encode_segments(segments)
            .map_err(|error| Error::Tokenize(error.to_string()))?;
        Ok(with_bos(bos, &encoding))
    }
}

#[cfg(test)]
mod tests {
    use dynamo_tokenizers::EncodeSegment;

    use super::*;

    /// A WordLevel tokenizer over `<s>`=0, `Hi`=2, `It`=3, whose post-processor adds
    /// `<s>` only when `adds_bos`, loaded next to `config` as `tokenizer_config.json`.
    fn load_fixture(name: &str, adds_bos: bool, config: &str) -> DynamoTokenizer {
        let dir = std::env::temp_dir().join(format!("processor-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bos = serde_json::json!({"SpecialToken": {"id": "<s>", "type_id": 0}});
        let sequence = serde_json::json!({"Sequence": {"id": "A", "type_id": 0}});
        let single = if adds_bos {
            vec![bos, sequence.clone()]
        } else {
            vec![sequence.clone()]
        };
        let tokenizer = serde_json::json!({
            "added_tokens": [{"id": 0, "content": "<s>", "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true}],
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": {"type": "TemplateProcessing", "single": single, "pair": [sequence],
                "special_tokens": {"<s>": {"id": "<s>", "ids": [0], "tokens": ["<s>"]}}},
            "model": {"type": "WordLevel", "unk_token": "[UNK]",
                "vocab": {"<s>": 0, "[UNK]": 1, "Hi": 2, "It": 3}},
        });
        let file = dir.join("tokenizer.json");
        std::fs::write(&file, tokenizer.to_string()).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), config).unwrap();
        let tokenizer = DynamoTokenizer::load(file.to_str().unwrap(), None).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        tokenizer
    }

    /// With a configured, auto-added `<s>`, the template encodes without it and the
    /// continuation loses one leading BOS, as on Python.
    #[test]
    fn chat_encode_drops_one_configured_bos() {
        let tokenizer = load_fixture("chat", true, r#"{"bos_token": "<s>"}"#);
        assert_eq!(tokenizer.bos_token_id(), Some(0));

        let prompt = [EncodeSegment::new("Hi", true)];
        assert_eq!(tokenizer.encode_chat(&prompt, "It").unwrap(), vec![2, 3]);
        // Only one BOS is dropped, as Python's: the auto-added one here.
        assert_eq!(
            tokenizer.encode_chat(&prompt, "<s><s>It").unwrap(),
            vec![2, 0, 0, 3]
        );
    }

    /// Gemma's `tokenizer.json` adds no BOS, but Python restores `add_bos_token` for
    /// its class, so an encode with specials starts with exactly one BOS, as on
    /// Python. An explicit `false` or an unlisted class keeps `tokenizer.json`'s.
    #[test]
    fn restored_add_bos_token_prepends_one_bos() {
        let gemma = r#"{"bos_token": "<s>", "tokenizer_class": "GemmaTokenizer"}"#;
        let tokenizer = load_fixture("gemma", false, gemma);
        assert_eq!(tokenizer.encode("Hi It", true).unwrap(), vec![0, 2, 3]);
        assert_eq!(tokenizer.encode("Hi It", false).unwrap(), vec![2, 3]);
        // BOS once, before the first segment only.
        let segments = [
            EncodeSegment::new("Hi", true),
            EncodeSegment::new("It", true),
        ];
        assert_eq!(
            tokenizer.encode_segments(&segments, true).unwrap(),
            vec![0, 2, 3]
        );
        let prompt = [EncodeSegment::new("Hi", true)];
        assert_eq!(tokenizer.encode_chat(&prompt, "It").unwrap(), vec![2, 3]);
        // Python replaces the post-processor, so one that already adds BOS gives one.
        let tokenizer = load_fixture("gemma-bos", true, gemma);
        assert_eq!(tokenizer.encode("Hi", true).unwrap(), vec![0, 2]);

        for (name, config) in [
            (
                "explicit-false",
                r#"{"bos_token": "<s>", "tokenizer_class": "GemmaTokenizer", "add_bos_token": false}"#,
            ),
            (
                "unlisted",
                r#"{"bos_token": "<s>", "tokenizer_class": "PreTrainedTokenizerFast"}"#,
            ),
        ] {
            let tokenizer = load_fixture(name, false, config);
            assert_eq!(
                tokenizer.encode("Hi It", true).unwrap(),
                vec![2, 3],
                "{name}"
            );
        }
    }
}

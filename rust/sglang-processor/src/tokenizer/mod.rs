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
    /// Prepend BOS in [`TextTokenizer::encode`] when `add_special_tokens` is
    /// set: Python's `_fix_v5_add_bos_eos_token` restores `add_bos_token` for
    /// the v4 BOS-flag classes, but the `tokenizer.json` post-processor of
    /// those models adds no BOS, so the Rust side must add it itself.
    add_bos: bool,
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
    /// [`TextTokenizer::encode_chat`] to drop and no BOS prepend;
    /// [`Self::load`] reads both. Only used for test tokenizers that add no
    /// specials.
    pub fn new(
        without_specials: dynamo_tokenizers::Tokenizer,
        with_specials: dynamo_tokenizers::Tokenizer,
    ) -> Self {
        Self {
            without_specials,
            with_specials,
            bos: None,
            add_bos: false,
        }
    }

    /// Load the tokenizer at `path` (see [`load_tokenizer`]) in both
    /// special-token modes, with the BOS its `tokenizer_config.json` names
    /// and the v4 `add_bos_token` default for its tokenizer class.
    pub fn load(path: &str, revision: Option<&str>) -> Result<Self, String> {
        let mut tokenizer = Self::new(
            load_tokenizer(Some(path), revision, false)?,
            load_tokenizer(Some(path), revision, true)?,
        );
        let config = read_tokenizer_config(path, revision);
        tokenizer.bos = config
            .as_ref()
            .and_then(|config| configured_bos(&tokenizer.without_specials, config));
        // Only prepend when the BOS id is known; otherwise there is nothing
        // to prepend, and chat stripping stays a no-op as before.
        tokenizer.add_bos =
            tokenizer.bos.is_some() && config.as_ref().is_some_and(v5_add_bos_token);
        Ok(tokenizer)
    }

    /// The handle that adds the tokenizer's specials, which decoding shares.
    pub fn tokenizer(&self) -> &dynamo_tokenizers::Tokenizer {
        &self.with_specials
    }
}

/// `tokenizer_config.json` for the model at `path`, if the file resolves.
fn read_tokenizer_config(path: &str, revision: Option<&str>) -> Option<serde_json::Value> {
    let config = crate::resolve_model_file(path, revision, "tokenizer_config.json")?;
    serde_json::from_str(&std::fs::read_to_string(config).ok()?).ok()
}

/// Whether Python's `_fix_v5_add_bos_eos_token` restores `add_bos_token`
/// for this tokenizer: only the tokenizer classes that supported the flag
/// in transformers v4 get it, defaulting to true when the config omits the
/// key or sets it to null; an explicit `false` opts out. Mirrors
/// `python/sglang/srt/hf_transformers/tokenizer.py`.
fn v5_add_bos_token(config: &serde_json::Value) -> bool {
    const V4_CLASSES_WITH_BOS_EOS_FLAGS: [&str; 7] = [
        "LlamaTokenizer",
        "LlamaTokenizerFast",
        "CodeLlamaTokenizer",
        "CodeLlamaTokenizerFast",
        "GemmaTokenizer",
        "GemmaTokenizerFast",
        "CohereTokenizerFast",
    ];
    let class = config["tokenizer_class"].as_str().unwrap_or("");
    if !V4_CLASSES_WITH_BOS_EOS_FLAGS.contains(&class) {
        return false;
    }
    config["add_bos_token"].as_bool().unwrap_or(true)
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
        let mut ids: Vec<i32> = encoding.token_ids().iter().map(|&id| id as i32).collect();
        // Transformers v5 dropped `add_bos_token` for the v4 BOS-flag classes
        // while their `tokenizer.json` post-processor adds no BOS; Python
        // restores the flag, so prepend it here when the post-processor did
        // not. A leading BOS is never duplicated, and `encode_chat` still
        // strips exactly one from the continuation.
        if add_special_tokens
            && self.add_bos
            && let Some(bos) = self.bos
            && ids.first() != Some(&bos)
        {
            ids.insert(0, bos);
        }
        Ok(ids)
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
        // The post-processor already adds BOS, so text encoding must not
        // add a second one.
        assert_eq!(tokenizer.encode("Hi", true).unwrap(), vec![0, 2]);
        assert_eq!(tokenizer.encode("Hi", false).unwrap(), vec![2]);
    }

    /// A Gemma 4-like tokenizer: the post-processor adds no special tokens
    /// and `tokenizer_config.json` names a v4 BOS-flag class with no
    /// `add_bos_token` key, so the v4 default (`true`) applies.
    /// `bos_id` is 2, as in Gemma 4's vocab.
    fn gemma_like_tokenizer(config_json: &str) -> DynamoTokenizer {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "processor-bos-gemma-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let tokenizer = serde_json::json!({
            "added_tokens": [{"id": 2, "content": "<bos>", "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true}],
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": {"type": "TemplateProcessing",
                "single": [{"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [{"Sequence": {"id": "A", "type_id": 0}},
                    {"Sequence": {"id": "B", "type_id": 1}}],
                "special_tokens": {}},
            "model": {"type": "WordLevel", "unk_token": "[UNK]",
                "vocab": {"[UNK]": 0, "Hi": 1, "<bos>": 2, "It": 3}},
        });
        let file = dir.join("tokenizer.json");
        std::fs::write(&file, tokenizer.to_string()).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), config_json).unwrap();
        let tokenizer = DynamoTokenizer::load(file.to_str().unwrap(), None).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        tokenizer
    }

    /// sglang#49519: a text prompt must get the BOS that Python's
    /// `_fix_v5_add_bos_eos_token` restores for Gemma tokenizers.
    #[test]
    fn text_encode_prepends_bos_for_gemma_class() {
        let tokenizer =
            gemma_like_tokenizer(r#"{"tokenizer_class": "GemmaTokenizer", "bos_token": "<bos>"}"#);
        assert_eq!(tokenizer.bos_token_id(), Some(2));
        assert_eq!(tokenizer.encode("Hi", true).unwrap(), vec![2, 1]);
        assert_eq!(tokenizer.encode("Hi", false).unwrap(), vec![1]);
    }

    /// The prepend must not leak into chat encoding: the template prompt
    /// stays BOS-free and the continuation still loses exactly one BOS.
    #[test]
    fn chat_encode_still_drops_one_bos_for_gemma_class() {
        let tokenizer =
            gemma_like_tokenizer(r#"{"tokenizer_class": "GemmaTokenizer", "bos_token": "<bos>"}"#);
        let prompt = [EncodeSegment::new("Hi", true)];
        assert_eq!(tokenizer.encode_chat(&prompt, "It").unwrap(), vec![1, 3]);
    }

    /// Classes outside Python's v4 BOS-flag list (e.g. Qwen2) are unaffected.
    #[test]
    fn text_encode_skips_bos_for_unlisted_class() {
        let tokenizer =
            gemma_like_tokenizer(r#"{"tokenizer_class": "Qwen2Tokenizer", "bos_token": "<bos>"}"#);
        assert_eq!(tokenizer.encode("Hi", true).unwrap(), vec![1]);
    }

    /// An explicit `add_bos_token: false` in the config is respected.
    #[test]
    fn text_encode_skips_bos_when_config_disables_it() {
        let tokenizer = gemma_like_tokenizer(
            r#"{"tokenizer_class": "GemmaTokenizer", "bos_token": "<bos>", "add_bos_token": false}"#,
        );
        assert_eq!(tokenizer.encode("Hi", true).unwrap(), vec![1]);
    }
}

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

/// `--tokenizer-backend`: the library that encodes text.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TokenizerBackend {
    #[default]
    HuggingFace,
    /// The `fastokens` crate; decoding stays on Hugging Face's `tokenizers`.
    Fastokens,
}

impl std::str::FromStr for TokenizerBackend {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, String> {
        match name {
            "huggingface" => Ok(Self::HuggingFace),
            "fastokens" => Ok(Self::Fastokens),
            _ => Err(format!("unknown tokenizer backend '{name}'")),
        }
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
/// (resolved from the local cache — no network). `Fastokens` loads the
/// `tokenizer.json` and encodes without the post-processor's special tokens, so
/// it takes only `add_special_tokens = false`.
pub fn load_tokenizer(
    tokenizer_path: Option<&str>,
    revision: Option<&str>,
    add_special_tokens: bool,
    backend: TokenizerBackend,
) -> Result<dynamo_tokenizers::Tokenizer, String> {
    let path =
        tokenizer_path.ok_or_else(|| "no tokenizer configured: set tokenizer_path".to_string())?;
    let file = resolve_tokenizer_file(path, revision).ok_or_else(|| {
        format!(
            "no supported tokenizer file found for '{path}' (expected tokenizer.json, tiktoken.model, or *.tiktoken)"
        )
    })?;
    let tokenizer = match backend {
        TokenizerBackend::HuggingFace => dynamo_tokenizers::Tokenizer::from_file_with_options(
            &file,
            dynamo_tokenizers::TokenizerOptions { add_special_tokens },
        )
        .map_err(|e| format!("tokenizer load failed ({file}): {e}"))?,
        TokenizerBackend::Fastokens => {
            if add_special_tokens {
                return Err("fastokens encodes without special tokens".to_string());
            }
            if !file.ends_with(".json") {
                return Err(format!(
                    "the fastokens tokenizer backend needs a tokenizer.json, and '{path}' has none"
                ));
            }
            let fast = dynamo_tokenizers::FastTokenizer::from_file(&file).map_err(|e| {
                format!(
                    "fastokens failed to load {file}: {e}. Re-run without --tokenizer-backend=fastokens"
                )
            })?;
            std::sync::Arc::new(fast).into()
        }
    };
    tracing::info!(%path, ?backend, "loaded tokenizer");
    Ok(tokenizer)
}

/// Whether the `tokenizer.json` post-processor adds tokens to a single sequence,
/// as Hugging Face's `num_special_tokens_to_add(false)`. A processor type this
/// does not know counts as adding.
fn post_processor_adds_tokens(processor: &serde_json::Value) -> bool {
    match processor["type"].as_str() {
        None | Some("ByteLevel") => false,
        Some("TemplateProcessing") => processor["single"]
            .as_array()
            .is_none_or(|items| items.iter().any(|item| item.get("Sequence").is_none())),
        Some("Sequence") => processor["processors"]
            .as_array()
            .is_none_or(|items| items.iter().any(post_processor_adds_tokens)),
        Some(_) => true,
    }
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
        Self::load_with_backend(path, revision, TokenizerBackend::HuggingFace)
    }

    /// [`Self::load`], encoding on `backend`. Fastokens adds no special tokens, so
    /// it also serves `add_special_tokens` only when the tokenizer adds none.
    pub fn load_with_backend(
        path: &str,
        revision: Option<&str>,
        backend: TokenizerBackend,
    ) -> Result<Self, String> {
        let without_specials = load_tokenizer(Some(path), revision, false, backend)?;
        let with_specials = match backend {
            TokenizerBackend::Fastokens if !Self::adds_special_tokens(path, revision)? => {
                without_specials.clone()
            }
            // Fastokens skips the post-processor, so its tokens come from Hugging Face.
            _ => load_tokenizer(Some(path), revision, true, TokenizerBackend::HuggingFace)?,
        };
        let mut tokenizer = Self::new(without_specials, with_specials);
        tokenizer.bos = configured_bos(&tokenizer.without_specials, path, revision);
        Ok(tokenizer)
    }

    /// Whether the post-processor of the `tokenizer.json` at `path` adds tokens.
    fn adds_special_tokens(path: &str, revision: Option<&str>) -> Result<bool, String> {
        let file = resolve_tokenizer_file(path, revision)
            .ok_or_else(|| format!("no tokenizer file found for '{path}'"))?;
        #[derive(serde::Deserialize)]
        struct Spec {
            #[serde(default)]
            post_processor: serde_json::Value,
        }
        let spec: Spec = serde_json::from_str(
            &std::fs::read_to_string(&file).map_err(|e| format!("read {file}: {e}"))?,
        )
        .map_err(|e| format!("parse {file}: {e}"))?;
        Ok(post_processor_adds_tokens(&spec.post_processor))
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

    /// A BPE `tokenizer.json` whose post-processor adds `<s>` when `bos` is set.
    fn write_bpe_tokenizer(name: &str, bos: bool) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("processor-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let words = ["a", "b", "c", "\u{2581}", "ab", "\u{2581}ab", "bc", "<s>"];
        let vocab: serde_json::Map<String, serde_json::Value> = words
            .iter()
            .enumerate()
            .map(|(id, word)| (word.to_string(), id.into()))
            .collect();
        let post_processor = if bos {
            serde_json::json!({"type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [{"Sequence": {"id": "A", "type_id": 0}}],
                "special_tokens": {"<s>": {"id": "<s>", "ids": [7], "tokens": ["<s>"]}}})
        } else {
            serde_json::Value::Null
        };
        let tokenizer = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [{"id": 7, "content": "<s>", "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true}],
            "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"},
            "pre_tokenizer": {"type": "Split", "pattern": {"String": " "},
                "behavior": "MergedWithPrevious", "invert": false},
            "post_processor": post_processor,
            "decoder": null,
            "model": {"type": "BPE", "dropout": null, "unk_token": null,
                "continuing_subword_prefix": null, "end_of_word_suffix": null,
                "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                "vocab": vocab,
                "merges": [["a", "b"], ["\u{2581}", "ab"], ["b", "c"]]},
        });
        std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
        dir
    }

    /// Fastokens adds no special tokens, so a tokenizer that adds one must keep
    /// `add_special_tokens` encodes on Hugging Face.
    #[test]
    fn fastokens_ids_match_huggingface_in_both_special_token_modes() {
        for bos in [false, true] {
            let dir = write_bpe_tokenizer(if bos { "fast-bos" } else { "fast" }, bos);
            let path = dir.to_str().unwrap();
            let reference = DynamoTokenizer::load(path, None).unwrap();
            let fast = DynamoTokenizer::load_with_backend(path, None, TokenizerBackend::Fastokens)
                .unwrap();
            std::fs::remove_dir_all(&dir).ok();
            for text in ["ab abc", "<s>ab  ab<s> bc a"] {
                for add_special_tokens in [false, true] {
                    assert_eq!(
                        fast.encode(text, add_special_tokens).unwrap(),
                        reference.encode(text, add_special_tokens).unwrap(),
                        "bos={bos} text={text:?} add_special_tokens={add_special_tokens}"
                    );
                }
            }
            assert_eq!(fast.encode("ab abc", false).unwrap(), vec![4, 5, 2]);
            assert_eq!(
                fast.encode("ab", true).unwrap(),
                if bos { vec![7, 4] } else { vec![4] }
            );
            let segments = [
                EncodeSegment::new("<s>ab", true),
                EncodeSegment::new(" bc", true),
            ];
            assert_eq!(
                fast.encode_segments(&segments, true).unwrap(),
                reference.encode_segments(&segments, true).unwrap()
            );
        }
    }

    /// The post-processor rule agrees with Hugging Face's count of added tokens.
    #[test]
    fn post_processor_rule_matches_huggingface_count() {
        let bos = serde_json::json!({"type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}],
            "special_tokens": {"<s>": {"id": "<s>", "ids": [7], "tokens": ["<s>"]}}});
        let plain = serde_json::json!({"type": "TemplateProcessing",
            "single": [{"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}],
            "special_tokens": {}});
        let byte_level = serde_json::json!({"type": "ByteLevel", "add_prefix_space": false,
            "trim_offsets": false, "use_regex": false});
        let processors = [
            serde_json::Value::Null,
            byte_level.clone(),
            plain.clone(),
            bos.clone(),
            serde_json::json!({"type": "Sequence", "processors": [byte_level.clone(), plain]}),
            serde_json::json!({"type": "Sequence", "processors": [byte_level, bos]}),
            serde_json::json!({"type": "RobertaProcessing", "sep": ["<s>", 7], "cls": ["<s>", 7],
                "trim_offsets": true, "add_prefix_space": false}),
        ];
        for (i, processor) in processors.into_iter().enumerate() {
            let dir = write_bpe_tokenizer(&format!("post-{i}"), false);
            let file = dir.join("tokenizer.json");
            let mut spec: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
            spec["post_processor"] = processor.clone();
            std::fs::write(&file, spec.to_string()).unwrap();
            let path = dir.to_str().unwrap();
            let added = load_tokenizer(Some(path), None, true, TokenizerBackend::HuggingFace)
                .unwrap()
                .num_special_tokens_added()
                .unwrap();
            let adds = DynamoTokenizer::adds_special_tokens(path, None).unwrap();
            std::fs::remove_dir_all(&dir).ok();
            assert_eq!(adds, added > 0, "{processor}");
        }
    }

    #[test]
    fn fastokens_needs_a_tokenizer_json() {
        let dir = std::env::temp_dir().join(format!("processor-fast-none-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiktoken.model");
        std::fs::write(&path, "").unwrap();
        let error = load_tokenizer(path.to_str(), None, false, TokenizerBackend::Fastokens)
            .err()
            .unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert!(error.contains("needs a tokenizer.json"), "{error}");
    }
}

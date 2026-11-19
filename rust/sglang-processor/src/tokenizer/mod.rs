//! Tokenizer loading and encoding over Dynamo tokenizers.

use std::{path::Path, sync::Arc};

use dynamo_tokenizers::traits::Tokenizer as _;
use dynamo_tokenizers::{HuggingFaceTokenizer, Tokenizer, TokenizerOptions};
use tokenizers::PostProcessor as _;

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

/// The default Dynamo handles also provide decoding and fallback for fastokens.
pub struct DynamoTokenizer {
    without_specials: Tokenizer,
    with_specials: Tokenizer,
    fast: Option<Box<fastokens::Tokenizer>>,
    bos: Option<i32>,
}

/// Load the default tokenizer from a file, model directory, or HF Hub repo id
/// (resolved from the local cache).
pub fn load_tokenizer(
    tokenizer_path: Option<&str>,
    revision: Option<&str>,
    add_special_tokens: bool,
) -> Result<Tokenizer, String> {
    let path =
        tokenizer_path.ok_or_else(|| "no tokenizer configured: set tokenizer_path".to_string())?;
    load_resolved_tokenizer(&tokenizer_file(path, revision)?, add_special_tokens)
}

fn tokenizer_file(path: &str, revision: Option<&str>) -> Result<String, String> {
    resolve_tokenizer_file(path, revision).ok_or_else(|| {
        format!(
            "no supported tokenizer file found for '{path}' (expected tokenizer.json, tiktoken.model, or *.tiktoken)"
        )
    })
}

fn load_resolved_tokenizer(file: &str, add_special_tokens: bool) -> Result<Tokenizer, String> {
    let tokenizer =
        Tokenizer::from_file_with_options(file, TokenizerOptions { add_special_tokens })
            .map_err(|e| format!("tokenizer load failed ({file}): {e}"))?;
    tracing::info!(%file, "loaded tokenizer");
    Ok(tokenizer)
}

/// Without an UNK id or complete byte coverage, fastokens can return
/// different merges around characters the BPE vocabulary does not cover.
fn bpe_covers_input(spec: &serde_json::Value) -> bool {
    let model = &spec["model"];
    if model["type"] != "BPE" || model["unk_token"].is_string() {
        return true;
    }
    let vocab = &model["vocab"];
    if model["byte_fallback"] == true
        && (0..=255).all(|byte| vocab.get(format!("<0x{byte:02X}>")).is_some())
    {
        return true;
    }

    let mut pre_tokenizer = &spec["pre_tokenizer"];
    while pre_tokenizer["type"] == "Sequence" {
        let Some(last) = pre_tokenizer["pretokenizers"]
            .as_array()
            .and_then(|steps| steps.last())
        else {
            return false;
        };
        pre_tokenizer = last;
    }
    pre_tokenizer["type"] == "ByteLevel"
        && model["continuing_subword_prefix"].is_null()
        && model["end_of_word_suffix"].is_null()
        && tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
            .into_iter()
            .all(|ch| vocab.get(ch.to_string()).is_some())
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
            fast: None,
            bos: None,
        }
    }

    /// Load the tokenizer at `path` (see [`load_tokenizer`]) in both
    /// special-token modes, with the BOS its `tokenizer_config.json` names.
    pub fn load(path: &str, revision: Option<&str>) -> Result<Self, String> {
        Self::load_with_backend(path, revision, TokenizerBackend::HuggingFace)
    }

    /// Load the chosen encoder and keep the checkpoint's default tokenizer for
    /// decoding and for inputs the fast backend cannot encode.
    pub fn load_with_backend(
        path: &str,
        revision: Option<&str>,
        backend: TokenizerBackend,
    ) -> Result<Self, String> {
        let file = tokenizer_file(path, revision)?;
        let mut tokenizer = if backend == TokenizerBackend::Fastokens && file.ends_with(".json") {
            Self::load_fastokens(Path::new(&file))?
        } else {
            if backend == TokenizerBackend::Fastokens {
                tracing::warn!(%file, "using the checkpoint's tiktoken backend with --tokenizer-backend fastokens");
            }
            Self::new(
                load_resolved_tokenizer(&file, false)?,
                load_resolved_tokenizer(&file, true)?,
            )
        };
        tokenizer.bos = configured_bos(&tokenizer.without_specials, path, revision);
        Ok(tokenizer)
    }

    fn load_fastokens(file: &Path) -> Result<Self, String> {
        let data = std::fs::read(file).map_err(|e| format!("read {}: {e}", file.display()))?;
        let mut hf = tokenizers::Tokenizer::from_bytes(&data)
            .map_err(|e| format!("tokenizer load failed ({}): {e}", file.display()))?;
        if let Some(model_dir) = file.parent() {
            dynamo_tokenizers::hf::merge_special_tokens_from_config(&mut hf, model_dir);
        }

        // Use the same added tokens (including IDs and matching flags) as the
        // HF decoder and fallback. Fastokens' file loader merges config entries
        // using different rules, so construct it from the effective vocabulary.
        let mut spec: serde_json::Value =
            serde_json::from_slice(&data).map_err(|e| format!("parse {}: {e}", file.display()))?;
        let mut added_tokens: Vec<_> = hf.get_added_tokens_decoder().into_iter().collect();
        added_tokens.sort_unstable_by_key(|(id, _)| *id);
        spec["added_tokens"] = added_tokens
            .into_iter()
            .map(|(id, token)| {
                serde_json::json!({
                    "id": id, "content": token.content, "special": token.special,
                    "normalized": token.normalized, "single_word": token.single_word,
                    "lstrip": token.lstrip, "rstrip": token.rstrip,
                })
            })
            .collect::<Vec<_>>()
            .into();
        let fast = if !bpe_covers_input(&spec) {
            tracing::warn!(path = %file.display(), "fastokens needs an UNK token or full byte coverage for BPE; using Hugging Face");
            None
        } else {
            match fastokens::Tokenizer::from_json(spec) {
                Ok(fast) => {
                    tracing::info!(path = %file.display(), "loaded fastokens encoder");
                    Some(Box::new(fast))
                }
                Err(error) => {
                    tracing::warn!(path = %file.display(), %error, "fastokens could not load tokenizer; using Hugging Face");
                    None
                }
            }
        };

        let adds_specials = hf
            .get_post_processor()
            .is_some_and(|processor| processor.added_tokens(false) != 0);
        let (without_specials, with_specials) = if adds_specials {
            let plain = Arc::new(HuggingFaceTokenizer::from_tokenizer(hf.clone())).into();
            let special = Arc::new(HuggingFaceTokenizer::from_tokenizer(hf).with_options(
                TokenizerOptions {
                    add_special_tokens: true,
                },
            ))
            .into();
            (plain, special)
        } else {
            let plain: Tokenizer = Arc::new(HuggingFaceTokenizer::from_tokenizer(hf)).into();
            (plain.clone(), plain)
        };
        Ok(Self {
            without_specials,
            with_specials,
            fast,
            bos: None,
        })
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
        if let Some(fast) = &self.fast
            && let Ok(ids) = fast.encode_with_special_tokens(text, add_special_tokens)
        {
            return Ok(ids.into_iter().map(|id| id as i32).collect());
        }
        // An input rejected by the fast encoder may still work on Hugging Face.
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
        // The checkpoint tokenizer enforces Dynamo's control-token checks for
        // ordinary segments, including IDs produced by base-vocabulary merges.
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
    fn write_bpe_tokenizer(name: &str, bos: bool, unk: Option<&str>) -> std::path::PathBuf {
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
            "model": {"type": "BPE", "dropout": null, "unk_token": unk,
                "continuing_subword_prefix": null, "end_of_word_suffix": null,
                "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                "vocab": vocab,
                "merges": [["a", "b"], ["\u{2581}", "ab"], ["b", "c"]]},
        });
        std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
        dir
    }

    /// Exercise the fast encoder with and without a BOS post-processor.
    #[test]
    fn fastokens_ids_match_huggingface_in_both_special_token_modes() {
        for bos in [false, true] {
            let dir = write_bpe_tokenizer(if bos { "fast-bos" } else { "fast" }, bos, Some("c"));
            let path = dir.to_str().unwrap();
            let reference = DynamoTokenizer::load(path, None).unwrap();
            let fast = DynamoTokenizer::load_with_backend(path, None, TokenizerBackend::Fastokens)
                .unwrap();
            std::fs::remove_dir_all(&dir).ok();
            assert!(fast.fast.is_some(), "fixture must use the fast encoder");
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

    #[test]
    fn fastokens_matches_huggingface_added_token_matching() {
        let dir = write_bpe_tokenizer("fast-matching", true, Some("c"));
        let file = dir.join("tokenizer.json");
        let mut spec: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        spec["normalizer"] = serde_json::json!({
            "type": "Replace", "pattern": {"String": "z"}, "content": "m"
        });
        let added = spec["added_tokens"].as_array_mut().unwrap();
        added.extend([
            serde_json::json!({"id": 12, "content": "<x>", "single_word": true, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true}),
            serde_json::json!({"id": 13, "content": "<z>", "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": true, "special": true}),
        ]);
        spec["model"]["vocab"].as_object_mut().unwrap().extend([
            ("<".into(), 8.into()),
            ("x".into(), 9.into()),
            (">".into(), 10.into()),
            ("m".into(), 11.into()),
        ]);
        std::fs::write(&file, spec.to_string()).unwrap();

        for text in ["a<x>b", "<x>", "a<z>b", "a<m>b"] {
            assert_fastokens_parity(&dir, text);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn fastokens_uses_the_huggingface_added_token_set() {
        let dir = write_bpe_tokenizer("fast-config", true, Some("c"));
        let config = dir.join("tokenizer_config.json");
        // A non-special config-only token is ignored by the default Rust loader.
        std::fs::write(
            &config,
            r#"{"added_tokens_decoder": {
            "8": {"content": "abc", "special": false, "normalized": false}
        }}"#,
        )
        .unwrap();
        assert_fastokens_parity(&dir, "abc");

        // The same entry is loaded when marked special, using the effective HF ID.
        std::fs::write(
            &config,
            r#"{"added_tokens_decoder": {
            "7": {"content": "abc", "special": true, "normalized": false}
        }}"#,
        )
        .unwrap();
        assert_fastokens_parity(&dir, "abc");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn fastokens_matches_huggingface_on_out_of_vocab_input() {
        let dir = write_bpe_tokenizer("fast-oov", true, None);
        // HF drops the unknown character and merges the remaining symbols.
        // Without byte coverage or an UNK id, fastokens can differ here.
        let tokenizer = DynamoTokenizer::load_with_backend(
            dir.to_str().unwrap(),
            None,
            TokenizerBackend::Fastokens,
        )
        .unwrap();
        assert!(tokenizer.fast.is_none());
        assert_eq!(tokenizer.encode("aqb", false).unwrap(), [4]);
        assert_eq!(tokenizer.encode("aqb", true).unwrap(), [7, 4]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn fastokens_handles_complete_byte_fallback_without_an_unk_id() {
        let dir = write_bpe_tokenizer("fast-byte-coverage", false, None);
        let file = dir.join("tokenizer.json");
        let mut spec: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        spec["model"]["byte_fallback"] = true.into();
        let vocab = spec["model"]["vocab"].as_object_mut().unwrap();
        let first_byte_id = vocab.len();
        for byte in 0..=255 {
            vocab.insert(format!("<0x{byte:02X}>"), (first_byte_id + byte).into());
        }
        std::fs::write(file, spec.to_string()).unwrap();

        assert_fastokens_parity(&dir, "ab q 🚀");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn fastokens_preserves_the_checkpoint_tokenizers_segment_validation() {
        let dir = write_bpe_tokenizer("fast-segments", false, Some("c"));
        let file = dir.join("tokenizer.json");
        let mut spec: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        spec["added_tokens"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "id": 4, "content": "ab", "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": false
            }));
        std::fs::write(file, spec.to_string()).unwrap();

        let tokenizer = DynamoTokenizer::load_with_backend(
            dir.to_str().unwrap(),
            None,
            TokenizerBackend::Fastokens,
        )
        .unwrap();
        assert!(tokenizer.fast.is_some());
        let segments = [EncodeSegment::control("<s>"), EncodeSegment::ordinary("ab")];
        let error = tokenizer.encode_segments(&segments, false).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("ordinary segment encoded to control-token id 4")
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn assert_fastokens_parity(dir: &std::path::Path, text: &str) {
        let path = dir.to_str().unwrap();
        let reference = DynamoTokenizer::load(path, None).unwrap();
        let fast =
            DynamoTokenizer::load_with_backend(path, None, TokenizerBackend::Fastokens).unwrap();
        let encoder = fast
            .fast
            .as_ref()
            .expect("fixture must use the fast encoder");
        for add_special_tokens in [false, true] {
            let expected = reference.encode(text, add_special_tokens).unwrap();
            let ids = encoder
                .encode_with_special_tokens(text, add_special_tokens)
                .unwrap();
            assert!(
                ids.into_iter().map(|id| id as i32).eq(expected),
                "text={text:?} add_special_tokens={add_special_tokens}"
            );
        }
    }

    #[test]
    fn fastokens_keeps_the_native_tiktoken_backend() {
        let dir = write_bpe_tokenizer("fast-tiktoken", false, None);
        std::fs::write(dir.join("tiktoken.model"), "YQ== 0\nYg== 1\n").unwrap();
        std::fs::write(dir.join("config.json"), r#"{"model_type": "kimi_k2"}"#).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"tokenizer_class": "TikTokenTokenizer"}"#,
        )
        .unwrap();

        let tokenizer = DynamoTokenizer::load_with_backend(
            dir.to_str().unwrap(),
            None,
            TokenizerBackend::Fastokens,
        )
        .unwrap();
        assert!(tokenizer.fast.is_none());
        assert_eq!(tokenizer.encode("ab", true).unwrap(), [0, 1]);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

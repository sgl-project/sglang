//! The tokenizer as SGLang's Python path uses it, over any Dynamo backend.
//!
//! Dynamo picks the backend and encodes bytes; this layer reproduces what
//! transformers' `tokenizer(text)` and SGLang's tokenizer manager do on top:
//! special tokens per call, the chat continuation's BOS, and tiktoken chunking.

use std::fmt::Display;
use std::{fs, slice};

use dynamo_tokenizers::traits::DecodeResult;
use dynamo_tokenizers::{
    EncodeSegment, Tokenizer, TokenizerOptions, create_tokenizer_from_file,
    create_tokenizer_from_file_with_options, traits,
};
use serde_json::{Value, json};

use crate::ProcessorError as Error;
use crate::token_pieces::TokenPieces;

fn tokenize_error(error: impl Display) -> Error {
    Error::Tokenize(error.to_string())
}

/// Tokenizer classes whose BOS SGLang restores from `add_bos_token`
/// (`_fix_v5_add_bos_eos_token`), rebuilding their post-processor without EOS.
const BOS_FLAG_CLASSES: [&str; 7] = [
    "LlamaTokenizer",
    "LlamaTokenizerFast",
    "CodeLlamaTokenizer",
    "CodeLlamaTokenizerFast",
    "GemmaTokenizer",
    "GemmaTokenizerFast",
    "CohereTokenizerFast",
];

/// Whether SGLang keeps tokenizer.json's normalizer: transformers v5 rebuilds these
/// classes with its own, and `_fix_v5_tokenizer_components` restores only the pre-tokenizer.
fn engine_keeps_normalizer(class: &str, normalizer: &Value) -> bool {
    let steps = match &normalizer["normalizers"] {
        Value::Array(steps) => steps.as_slice(),
        _ if normalizer.is_null() => &[],
        _ => slice::from_ref(normalizer),
    };
    let gemma = json!({"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"});
    match class.trim_end_matches("Fast") {
        "LlamaTokenizer" => steps.is_empty(),
        "XLMRobertaTokenizer" => steps.iter().all(|step| step["type"] == "Precompiled"),
        "GemmaTokenizer" => *steps == [gemma],
        // Rewritten for infilling; unverified.
        "CodeLlamaTokenizer" => false,
        _ => true,
    }
}

/// Special tokens SGLang's `tokenizer(text)` puts around a prompt, which a
/// Dynamo tokenizer loaded without special tokens leaves out.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PromptAffixes {
    pub prefix: Vec<u32>,
    pub suffix: Vec<u32>,
    /// EOS that SGLang appends to an EmbeddingGemma prompt not already ending in it.
    pub eos: Option<u32>,
}

impl PromptAffixes {
    /// BOS per `add_bos_token` (default true) for [`BOS_FLAG_CLASSES`], else the
    /// tokenizer.json post-processor's, and EmbeddingGemma's EOS. Tiktoken models
    /// add none. Errs when SGLang's tokens cannot be reproduced.
    ///
    /// `tokenizer_file` is the local file the backend was loaded from; the configs
    /// are the checkpoint's `tokenizer_config.json` and `config.json` (`Null` when absent).
    pub fn derive(
        tokenizer_file: &str,
        tokenizer_config: &Value,
        model_config: &Value,
    ) -> Result<Self, Error> {
        if !tokenizer_file.ends_with(".json") {
            return Ok(Self::default());
        }
        if !tokenizer_config.is_object() {
            return Err(tokenize_error(
                "no tokenizer_config.json beside the tokenizer",
            ));
        }
        let file: Value = fs::read_to_string(tokenizer_file)
            .map_err(tokenize_error)
            .and_then(|text| serde_json::from_str(&text).map_err(tokenize_error))?;
        let class = tokenizer_config["tokenizer_class"]
            .as_str()
            .unwrap_or_default();
        if !engine_keeps_normalizer(class, &file["normalizer"]) {
            return Err(tokenize_error(format!(
                "transformers replaces the {class} normalizer"
            )));
        }
        let plain = create_tokenizer_from_file(tokenizer_file).map_err(tokenize_error)?;
        let ids = |t: &dyn traits::Tokenizer, text: &str| -> Result<Vec<u32>, Error> {
            Ok(t.encode(text).map_err(tokenize_error)?.token_ids().to_vec())
        };
        let special = |key: &str| -> Result<Option<u32>, Error> {
            let config = &tokenizer_config[key];
            let Some(token) = config.as_str().or(config["content"].as_str()) else {
                return Ok(None);
            };
            match ids(plain.as_ref(), token)?.as_slice() {
                [id] => Ok(Some(*id)),
                _ => Err(tokenize_error(format!(
                    "{key} {token:?} is not a single token"
                ))),
            }
        };
        // SGLang's `is_embedding_gemma`, whose `_tokenize_texts` ends each prompt with EOS.
        let embedding_gemma = model_config["model_type"] == "gemma3_text"
            && model_config["use_bidirectional_attention"] == true;
        let required =
            |key: &str, missing: &str| special(key)?.ok_or_else(|| tokenize_error(missing));
        let eos = embedding_gemma
            .then(|| required("eos_token", "EmbeddingGemma declares no eos_token"))
            .transpose()?;
        if BOS_FLAG_CLASSES.contains(&class) {
            let add_bos = tokenizer_config["add_bos_token"].as_bool().unwrap_or(true);
            let prefix = add_bos
                .then(|| required("bos_token", "add_bos_token is set without a bos_token"))
                .transpose()?
                .into_iter()
                .collect();
            return Ok(Self {
                prefix,
                suffix: Vec::new(),
                eos,
            });
        }
        let options = TokenizerOptions {
            add_special_tokens: true,
        };
        let with_specials = create_tokenizer_from_file_with_options(tokenizer_file, options)
            .map_err(tokenize_error)?;
        let full = ids(with_specials.as_ref(), "a")?;
        let bare = ids(plain.as_ref(), "a")?;
        if bare.is_empty() {
            return Err(tokenize_error("the tokenizer drops the probe text"));
        }
        let start = full
            .windows(bare.len())
            .position(|window| window == bare)
            .ok_or_else(|| tokenize_error("the post-processor rewrites the prompt"))?;
        Ok(Self {
            prefix: full[..start].to_vec(),
            suffix: full[start + bare.len()..].to_vec(),
            eos,
        })
    }

    /// `tokenizer(text, add_special_tokens=True)` from the ids encoded without them.
    fn wrap(&self, ids: &[u32]) -> Vec<u32> {
        [self.prefix.as_slice(), ids, &self.suffix].concat()
    }
}

/// Tokenizer behavior SGLang's Python serving path has, over a Dynamo backend
/// that encodes without special tokens (Dynamo's default).
pub struct SglangTokenizer {
    inner: Tokenizer,
    /// Tiktoken checkpoints encode in `tokenization_kimi.py`'s chunks.
    chunked: bool,
    /// `None` when the backend cannot reproduce `add_special_tokens=True`.
    affixes: Option<PromptAffixes>,
    /// Python's `tokenizer.bos_token_id`.
    bos: Option<u32>,
    /// `convert_ids_to_tokens`, from `tokenizer.json`; `None` for tiktoken.
    pieces: Option<TokenPieces>,
}

impl SglangTokenizer {
    /// `tokenizer_file` is the file `inner` was loaded from; its kind picks the chunking.
    pub fn new(inner: Tokenizer, tokenizer_file: &str) -> Self {
        Self {
            inner,
            chunked: !tokenizer_file.ends_with(".json"),
            affixes: None,
            bos: None,
            pieces: None,
        }
    }

    pub fn with_prompt_affixes(mut self, affixes: PromptAffixes) -> Self {
        self.affixes = Some(affixes);
        self
    }

    pub fn with_token_pieces(mut self, tokenizer_json: &Value) -> Self {
        self.pieces = Some(TokenPieces::from_tokenizer_json(tokenizer_json));
        self
    }

    pub fn with_bos_token(mut self, tokenizer_config: &Value) -> Self {
        let bos = &tokenizer_config["bos_token"];
        self.bos = bos
            .as_str()
            .or_else(|| bos["content"].as_str())
            .filter(|token| !token.is_empty())
            .and_then(|token| self.token_id(token));
        self
    }

    /// Tiktoken has no vocab lookup, so the token must encode to one id.
    fn token_id(&self, token: &str) -> Option<u32> {
        if let Ok(Some(id)) = self.inner.token_to_id(token) {
            return Some(id);
        }
        match self.inner.encode(token).ok()?.token_ids() {
            &[id] => Some(id),
            _ => None,
        }
    }

    /// The backend, for callers that need Dynamo's own encode (raw-text hashing,
    /// `/tokenize`) or its streaming decode.
    pub fn dynamo(&self) -> &Tokenizer {
        &self.inner
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos
    }

    pub fn knows_special_tokens(&self) -> bool {
        self.affixes.is_some()
    }

    /// Python's `tokenizer.encode(text, add_special_tokens=...)`.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, Error> {
        let ids = if self.chunked {
            self.encode_chunked(&[EncodeSegment::control(text)])?
        } else {
            self.inner
                .encode(text)
                .map_err(tokenize_error)?
                .token_ids()
                .to_vec()
        };
        self.add_special_tokens(ids, add_special_tokens)
    }

    /// A raw prompt as SGLang's `_tokenize_texts` encodes it: special tokens
    /// included, and EmbeddingGemma's EOS once at the end.
    pub fn encode_prompt(&self, text: &str) -> Result<Vec<u32>, Error> {
        let mut ids = self.encode(text, true)?;
        if let Some(eos) = self.affixes.as_ref().and_then(|a| a.eos)
            && ids.last() != Some(&eos)
        {
            ids.push(eos);
        }
        Ok(ids)
    }

    /// Rendered segments, keeping their trust boundaries.
    pub fn encode_segments(
        &self,
        segments: &[EncodeSegment<'_>],
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, Error> {
        let ids = if self.chunked {
            self.encode_chunked(segments)?
        } else if segments.iter().all(|segment| segment.allow_special) {
            // One by one keeps the boundaries on backends without segmented
            // encoding, such as Hugging Face's.
            segments
                .iter()
                .map(|segment| self.encode(segment.text, false))
                .collect::<Result<Vec<_>, _>>()?
                .concat()
        } else {
            self.inner
                .encode_segments(segments)
                .map_err(tokenize_error)?
                .token_ids()
                .to_vec()
        };
        self.add_special_tokens(ids, add_special_tokens)
    }

    /// Python's chat encode: the template without specials, then the continuation
    /// with them minus one leading BOS (`_append_assistant_prefix_to_prompt_ids`).
    pub fn encode_chat(
        &self,
        prompt: &[EncodeSegment<'_>],
        continuation: &str,
    ) -> Result<Vec<u32>, Error> {
        let mut ids = self.encode_segments(prompt, false)?;
        if continuation.is_empty() {
            return Ok(ids);
        }
        // Unknown affixes encode the continuation bare instead of failing the chat,
        // so routing still gets its ids.
        let suffix = self.encode(continuation, self.knows_special_tokens())?;
        let suffix = self
            .bos
            .and_then(|bos| suffix.strip_prefix(&[bos]))
            .unwrap_or(&suffix);
        ids.extend_from_slice(suffix);
        Ok(ids)
    }

    /// The full text of `ids`, a trailing partial character included (as U+FFFD).
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Error> {
        match self
            .inner
            .decode(ids, skip_special_tokens)
            .map_err(tokenize_error)?
        {
            DecodeResult::Complete(text) | DecodeResult::Partial(text) => Ok(text),
        }
    }

    /// The bytes of `convert_ids_to_tokens(id)` for a byte-level BPE tokenizer.
    pub fn byte_level_bytes(&self, id: u32) -> Option<Vec<u8>> {
        self.pieces.as_ref()?.byte_level_bytes(id)
    }

    fn add_special_tokens(&self, ids: Vec<u32>, add: bool) -> Result<Vec<u32>, Error> {
        match (add, &self.affixes) {
            (false, _) => Ok(ids),
            (true, Some(affixes)) => Ok(affixes.wrap(&ids)),
            (true, None) => Err(tokenize_error(
                "the tokenizer's special tokens cannot be reproduced",
            )),
        }
    }

    fn encode_chunked(&self, segments: &[EncodeSegment<'_>]) -> Result<Vec<u32>, Error> {
        let chunks: Vec<_> = segments.iter().flat_map(kimi_chunks).collect();
        Ok(self
            .inner
            .encode_segments(&chunks)
            .map_err(tokenize_error)?
            .token_ids()
            .to_vec())
    }
}

/// `tokenization_kimi.py` encodes 400k-char windows, each split after 25k
/// consecutive (non-)whitespace chars, and BPE is not chunk-invariant.
fn kimi_chunks<'a>(segment: &EncodeSegment<'a>) -> Vec<EncodeSegment<'a>> {
    let text = segment.text;
    let mut chunks = Vec::new();
    let (mut start, mut run, mut was_space) = (0, 0, false);
    for (count, (offset, ch)) in text.char_indices().enumerate() {
        // Python's `str.isspace` also covers U+001C..U+001F.
        let space = ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch);
        run = if space == was_space { run + 1 } else { 1 };
        if (count > 0 && count % 400_000 == 0) || run > 25_000 {
            chunks.push(EncodeSegment::new(
                &text[start..offset],
                segment.allow_special,
            ));
            (start, run) = (offset, 1);
        }
        was_space = space;
    }
    chunks.push(EncodeSegment::new(&text[start..], segment.allow_special));
    chunks
}

/// Lets `sglang-server`'s tokenizer pool run on this type unchanged.
impl super::TextTokenizer for SglangTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>, Error> {
        SglangTokenizer::encode(self, text, add_special_tokens).map(widen)
    }

    fn bos_token_id(&self) -> Option<i32> {
        self.bos.and_then(|id| i32::try_from(id).ok())
    }

    fn encode_chat(
        &self,
        prompt: &[EncodeSegment<'_>],
        continuation: &str,
    ) -> Result<Vec<i32>, Error> {
        SglangTokenizer::encode_chat(self, prompt, continuation).map(widen)
    }

    fn encode_segments(
        &self,
        segments: &[EncodeSegment<'_>],
        add_special_tokens: bool,
    ) -> Result<Vec<i32>, Error> {
        SglangTokenizer::encode_segments(self, segments, add_special_tokens).map(widen)
    }
}

/// Vocab ids are non-negative and fit in i32.
fn widen(ids: Vec<u32>) -> Vec<i32> {
    ids.into_iter().map(|id| id as i32).collect()
}

#[cfg(feature = "openai")]
impl crate::openai::OpenAiTokenizer for SglangTokenizer {
    fn decode(&self, ids: &[u32]) -> Option<String> {
        SglangTokenizer::decode(self, ids, true).ok()
    }

    fn byte_level_bytes(&self, id: u32) -> Option<Vec<u8>> {
        SglangTokenizer::byte_level_bytes(self, id)
    }
}

#[cfg(test)]
mod tests {
    use std::env::temp_dir;
    use std::path::PathBuf;
    use std::process;

    use super::*;

    /// A temp dir holding the tiny tokenizer with `normalizer` and `post_processor`.
    struct TinyTokenizer(PathBuf);

    impl TinyTokenizer {
        fn new(name: &str, normalizer: Value, post_processor: Value) -> Self {
            let mut data: Value =
                serde_json::from_str(include_str!("../../tests/fixtures/tiny_tokenizer.json"))
                    .unwrap();
            data["normalizer"] = normalizer;
            data["post_processor"] = post_processor;
            let dir = temp_dir().join(format!("processor-affixes-{name}-{}", process::id()));
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("tokenizer.json"), data.to_string()).unwrap();
            Self(dir)
        }

        fn file(&self) -> String {
            self.0.join("tokenizer.json").to_string_lossy().into_owned()
        }

        fn affixes(&self, config: &Value, model: &Value) -> Result<PromptAffixes, Error> {
            PromptAffixes::derive(&self.file(), config, model)
        }
    }

    impl Drop for TinyTokenizer {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).ok();
        }
    }

    fn template() -> Value {
        json!({"type": "TemplateProcessing", "pair": [], "single": [
            {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}}],
            "special_tokens": {"<|endoftext|>": {"id": "<|endoftext|>", "ids": [256], "tokens": ["<|endoftext|>"]}}})
    }

    #[test]
    fn affixes_match_the_engines_special_tokens() {
        let llama = json!({"tokenizer_class": "LlamaTokenizerFast", "bos_token": "<|endoftext|>"});
        let mut no_bos = llama.clone();
        no_bos["add_bos_token"] = false.into();
        let embedding_gemma =
            json!({"model_type": "gemma3_text", "use_bidirectional_attention": true});
        let eos = json!({"eos_token": {"content": "<|endoftext|>"}});
        let plain = TinyTokenizer::new("plain", Value::Null, Value::Null);
        assert!(plain.affixes(&json!({}), &embedding_gemma).is_err());
        let templated = TinyTokenizer::new("templated", Value::Null, template());
        for (tokenizer, config, model, prefix, eos) in [
            (&plain, json!({}), json!({}), vec![], None),
            (&templated, json!({}), json!({}), vec![256], None),
            // SGLang adds BOS by `add_bos_token`, ignoring the post-processor.
            (&plain, llama, json!({}), vec![256], None),
            (&templated, no_bos, json!({}), vec![], None),
            (&plain, eos, embedding_gemma, vec![], Some(256)),
        ] {
            let expected = PromptAffixes {
                prefix,
                suffix: vec![],
                eos,
            };
            assert_eq!(tokenizer.affixes(&config, &model).unwrap(), expected);
        }
        // Without the config or its BOS, the engine's special tokens are unknown.
        for config in [
            Value::Null,
            json!({"tokenizer_class": "LlamaTokenizerFast"}),
        ] {
            assert!(plain.affixes(&config, &json!({})).is_err());
        }
    }

    #[test]
    fn affixes_refuse_normalizers_the_engine_replaces() {
        let legacy = json!({"type": "Sequence", "normalizers": [
            {"type": "Prepend", "prepend": "\u{2581}"},
            {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"}]});
        let collapse = json!({"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "});
        let gemma = json!({"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"});
        for (class, normalizer, kept) in [
            ("LlamaTokenizerFast", legacy.clone(), false),
            ("PreTrainedTokenizerFast", legacy, true),
            ("XLMRobertaTokenizer", collapse, false),
            ("GemmaTokenizer", gemma, true),
        ] {
            let tokenizer = TinyTokenizer::new(class, normalizer, Value::Null);
            let config = json!({"tokenizer_class": class, "add_bos_token": false});
            assert_eq!(
                tokenizer.affixes(&config, &json!({})).is_ok(),
                kept,
                "{class}"
            );
        }
    }

    fn tiny(name: &str, affixes: PromptAffixes) -> (TinyTokenizer, SglangTokenizer) {
        let dir = TinyTokenizer::new(name, Value::Null, Value::Null);
        let file = dir.file();
        let inner = Tokenizer::from_file(&file).unwrap();
        let tokenizer = SglangTokenizer::new(inner, &file)
            .with_prompt_affixes(affixes)
            .with_bos_token(&json!({"bos_token": "<|endoftext|>"}));
        (dir, tokenizer)
    }

    #[test]
    fn embedding_gemma_eos_ends_the_prompt_once() {
        let (_dir, tokenizer) = tiny(
            "gemma-eos",
            PromptAffixes {
                eos: Some(256),
                ..Default::default()
            },
        );
        let ids = tokenizer.encode_prompt("a").unwrap();
        assert_eq!(ids.last(), Some(&256));
        assert_eq!(tokenizer.encode_prompt("a<|endoftext|>").unwrap(), ids);
    }

    #[test]
    fn chat_continuation_drops_one_bos() {
        let (_dir, tokenizer) = tiny(
            "continuation-bos",
            PromptAffixes {
                prefix: vec![256],
                ..Default::default()
            },
        );
        assert_eq!(tokenizer.bos_token_id(), Some(256));
        let prompt = [EncodeSegment::control("ab")];
        let template = tokenizer.encode_segments(&prompt, false).unwrap();
        let text = tokenizer.encode("c", false).unwrap();
        assert_eq!(
            tokenizer.encode_chat(&prompt, "c").unwrap(),
            [template.as_slice(), &text].concat()
        );
        // Only the affix BOS goes; one the continuation spells out stays.
        let spelled = tokenizer.encode_chat(&prompt, "<|endoftext|>c").unwrap();
        assert_eq!(spelled, [template.as_slice(), &[256], &text].concat());
    }

    #[test]
    fn special_tokens_wrap_a_segmented_prompt_once() {
        let (_dir, tokenizer) = tiny(
            "segments-once",
            PromptAffixes {
                prefix: vec![256],
                suffix: vec![256],
                eos: None,
            },
        );
        let segments = [EncodeSegment::control("a"), EncodeSegment::control("b")];
        let bare = tokenizer.encode_segments(&segments, false).unwrap();
        assert_eq!(
            tokenizer.encode_segments(&segments, true).unwrap(),
            [&[256], bare.as_slice(), &[256]].concat()
        );
    }

    #[test]
    fn special_tokens_need_known_affixes() {
        let dir = TinyTokenizer::new("unknown", Value::Null, Value::Null);
        let file = dir.file();
        let tokenizer = SglangTokenizer::new(Tokenizer::from_file(&file).unwrap(), &file);
        assert!(!tokenizer.knows_special_tokens());
        assert!(tokenizer.encode("a", true).is_err());
        assert!(tokenizer.encode("a", false).is_ok());
    }
}

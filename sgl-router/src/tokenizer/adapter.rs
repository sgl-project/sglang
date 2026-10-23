// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use dynamo_tokenizers::{
    create_tokenizer_from_file, create_tokenizer_from_file_with_options, traits,
    traits::DecodeResult, CacheTokenUsage, CachedTokenizer, FastTokenizer, Tokenizer,
    TokenizerOptions,
};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::stats::{EncodeBackend, L1Counters, L1State, TokenizerStats};
use crate::config::{TokenizerBackend, TokenizerConfig};

/// Load a local tokenizer file or Hugging Face repo with the default HF, uncached encoder.
pub fn load(source: &str) -> Result<Arc<Tokenizer>> {
    load_with(source, TokenizerConfig::default()).map(|(t, _)| t)
}

/// Load a local tokenizer file or Hugging Face repo, honoring HF cache/auth settings, with the
/// configured encode backend and L1 cache. Tiktoken `.model` files also require sibling
/// config.json and tokenizer_config.json.
pub fn load_with(
    source: &str,
    cfg: TokenizerConfig,
) -> Result<(Arc<Tokenizer>, Arc<TokenizerStats>)> {
    let path = resolve(source)?;
    build(&path, cfg).with_context(|| format!("load tokenizer for {source}"))
}

/// Local tokenizer file for a path or an HF repo id.
fn resolve(source: &str) -> Result<String> {
    if Path::new(source).is_file() || looks_like_path(source) {
        return Ok(source.to_owned());
    }
    download_tokenizer(source)?
        .into_os_string()
        .into_string()
        .map_err(|_| anyhow::anyhow!("downloaded tokenizer path is not valid UTF-8"))
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
fn engine_keeps_normalizer(class: &str, normalizer: &serde_json::Value) -> bool {
    let steps = match &normalizer["normalizers"] {
        serde_json::Value::Array(steps) => steps.as_slice(),
        _ if normalizer.is_null() => &[],
        _ => std::slice::from_ref(normalizer),
    };
    let gemma =
        serde_json::json!({"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"});
    match class.trim_end_matches("Fast") {
        "LlamaTokenizer" => steps.is_empty(),
        "XLMRobertaTokenizer" => steps.iter().all(|step| step["type"] == "Precompiled"),
        "GemmaTokenizer" => *steps == [gemma],
        // Rewritten for infilling; unverified.
        "CodeLlamaTokenizer" => false,
        _ => true,
    }
}

/// Special tokens SGLang's `tokenizer(text)` puts around a prompt and [`encode`] leaves out.
#[derive(Debug, Default, PartialEq)]
pub struct PromptAffixes {
    pub prefix: Vec<u32>,
    pub suffix: Vec<u32>,
    /// EOS that SGLang appends to an EmbeddingGemma prompt not already ending in it.
    pub eos: Option<u32>,
}

impl PromptAffixes {
    /// [`encode`]'s `ids` as the engine tokenizes the same text.
    pub fn apply(&self, ids: &[u32]) -> Vec<u32> {
        let mut ids = [self.prefix.as_slice(), ids, &self.suffix].concat();
        if let Some(eos) = self.eos.filter(|&eos| ids.last() != Some(&eos)) {
            ids.push(eos);
        }
        ids
    }
}

/// BOS per `add_bos_token` (default true) for [`BOS_FLAG_CLASSES`], else the
/// tokenizer.json post-processor's, and EmbeddingGemma's EOS. Tiktoken models add
/// none. Errs when the router cannot reproduce the engine's tokens.
pub fn prompt_affixes(source: &str, files: &ModelFiles) -> Result<PromptAffixes> {
    let path = resolve(source)?;
    if !path.ends_with(".json") {
        return Ok(Default::default());
    }
    // Through `files`, which downloads it: a cold HF cache holds only tokenizer.json.
    // A missing or failed download leaves the special tokens unknown, so /generate keeps text.
    files.ensure_downloaded("tokenizer_config.json")?;
    let config = files
        .json("tokenizer_config.json")?
        .context("no tokenizer_config.json beside the tokenizer")?;
    let file: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let class = config["tokenizer_class"].as_str().unwrap_or_default();
    anyhow::ensure!(
        engine_keeps_normalizer(class, &file["normalizer"]),
        "transformers replaces the {class} normalizer"
    );
    let plain = create_tokenizer_from_file(&path)?;
    let ids = |t: &dyn traits::Tokenizer, text: &str| -> Result<Vec<u32>> {
        Ok(t.encode(text)?.token_ids().to_vec())
    };
    let special = |key: &str| -> Result<Option<u32>> {
        let Some(token) = config[key].as_str().or(config[key]["content"].as_str()) else {
            return Ok(None);
        };
        match ids(plain.as_ref(), token)?.as_slice() {
            [id] => Ok(Some(*id)),
            _ => anyhow::bail!("{key} {token:?} is not a single token"),
        }
    };
    // SGLang's `is_embedding_gemma`, whose `_tokenize_texts` ends each prompt with EOS.
    files.ensure_downloaded("config.json")?;
    let model = files.json("config.json")?.unwrap_or_default();
    let embedding_gemma =
        model["model_type"] == "gemma3_text" && model["use_bidirectional_attention"] == true;
    let eos = embedding_gemma
        .then(|| special("eos_token")?.context("EmbeddingGemma declares no eos_token"))
        .transpose()?;
    if BOS_FLAG_CLASSES.contains(&class) {
        let add_bos = config["add_bos_token"].as_bool().unwrap_or(true);
        let bos = add_bos
            .then(|| special("bos_token")?.context("add_bos_token is set without a bos_token"))
            .transpose()?;
        let prefix = bos.into_iter().collect();
        return Ok(PromptAffixes {
            prefix,
            suffix: Vec::new(),
            eos,
        });
    }
    let options = TokenizerOptions {
        add_special_tokens: true,
    };
    let full = ids(
        create_tokenizer_from_file_with_options(&path, options)?.as_ref(),
        "a",
    )?;
    let bare = ids(plain.as_ref(), "a")?;
    anyhow::ensure!(!bare.is_empty(), "the tokenizer drops the probe text");
    let start = full
        .windows(bare.len())
        .position(|window| window == bare)
        .context("the post-processor rewrites the prompt")?;
    let (prefix, suffix) = (full[..start].to_vec(), full[start + bare.len()..].to_vec());
    Ok(PromptAffixes {
        prefix,
        suffix,
        eos,
    })
}

fn build(path: &str, cfg: TokenizerConfig) -> Result<(Arc<Tokenizer>, Arc<TokenizerStats>)> {
    let (inner, backend) = encoder(path, cfg.backend)?;
    let l1 = Arc::new(L1Counters::default());
    let (tokenizer, l1_state) = if cfg.l1_cache_mb == 0 {
        (Tokenizer::from(inner), L1State::Off)
    } else {
        let specials = boundary_tokens(path)?;
        if specials.is_empty() {
            tracing::warn!(
                path,
                "--tokenizer-l1-cache-mb is set but the tokenizer declares no safely splittable \
                 special tokens; the L1 cache is inert"
            );
            (Tokenizer::from(inner), L1State::DisabledNoSpecials)
        } else {
            let bytes = cfg.l1_cache_mb.saturating_mul(1 << 20);
            let cached = with_l1(inner, specials, bytes, &l1)?;
            (Tokenizer::from(Arc::new(cached)), L1State::Active)
        }
    };
    let stats = TokenizerStats {
        backend,
        l1_state,
        l1,
    };
    Ok((Arc::new(tokenizer), Arc::new(stats)))
}

/// Wrap `inner` in the L1 prefix cache, extending on partial hits so each turn of a growing
/// conversation reuses all earlier turns, and feed `counters` from its observers.
fn with_l1(
    inner: Arc<dyn traits::Tokenizer>,
    specials: Vec<String>,
    max_memory_bytes: usize,
    counters: &Arc<L1Counters>,
) -> Result<CachedTokenizer> {
    let usage = Arc::clone(counters);
    Ok(CachedTokenizer::new(inner, specials, max_memory_bytes)?
        .with_extend(true)
        .with_token_observer(Arc::new(move |u: CacheTokenUsage| {
            usage
                .cached_tokens
                .fetch_add(u.cached_tokens as u64, Ordering::Relaxed);
            usage
                .encoded_tokens
                .fetch_add(u.uncached_tokens as u64, Ordering::Relaxed);
        })))
}

/// Build the encoder for `backend`; `fast` falls back to HF when fastokens cannot load the file.
fn encoder(
    path: &str,
    backend: TokenizerBackend,
) -> Result<(Arc<dyn traits::Tokenizer>, EncodeBackend)> {
    if !path.ends_with(".json") {
        if backend == TokenizerBackend::Fast {
            tracing::warn!(
                path,
                "fastokens needs a tokenizer.json; encoding on tiktoken"
            );
        }
        return Ok((create_tokenizer_from_file(path)?, EncodeBackend::Tiktoken));
    }
    if backend == TokenizerBackend::Fast {
        match FastTokenizer::from_file(path) {
            Ok(t) => return Ok((Arc::new(t), EncodeBackend::Fast)),
            Err(e) => tracing::warn!(path, error = %format!("{e:#}"),
                "fastokens cannot load this tokenizer; encoding on hf"),
        }
        return Ok((
            create_tokenizer_from_file(path)?,
            EncodeBackend::FastFallbackHf,
        ));
    }
    Ok((create_tokenizer_from_file(path)?, EncodeBackend::Hf))
}

/// L1 matches literal spellings, so only unconditional, non-overlapping added tokens
/// are safe boundaries. Consider all added tokens as competitors, including ordinary
/// tokens that can consume a candidate spelling as part of a longer match.
fn boundary_tokens(path: &str) -> Result<Vec<String>> {
    #[derive(serde::Deserialize)]
    struct AddedToken {
        content: String,
        #[serde(default)]
        special: bool,
        #[serde(default)]
        single_word: bool,
        normalized: Option<bool>,
        #[serde(default)]
        lstrip: bool,
        #[serde(default)]
        rstrip: bool,
    }
    #[derive(serde::Deserialize)]
    struct TokenizerJson {
        #[serde(default)]
        added_tokens: Vec<AddedToken>,
    }
    if !path.ends_with(".json") {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    let parsed: TokenizerJson =
        serde_json::from_str(&text).with_context(|| format!("parse added_tokens in {path}"))?;
    let safe = |t: &AddedToken| {
        t.special
            && t.normalized == Some(false)
            && !t.single_word
            && !t.lstrip
            && !t.rstrip
            && !t.content.is_empty()
    };
    // HF also merges special-token declarations from the sibling config. A declaration
    // there can change matching flags or introduce a token overlapping a JSON boundary.
    // Match HF's treatment of this optional file: ignore missing/invalid config entries.
    let config_tokens: Vec<AddedToken> = Path::new(path)
        .parent()
        .and_then(|dir| std::fs::read_to_string(dir.join("tokenizer_config.json")).ok())
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|config| config.get("added_tokens_decoder")?.as_object().cloned())
        .into_iter()
        .flat_map(|tokens| tokens.into_values())
        .filter_map(|value| serde_json::from_value::<AddedToken>(value).ok())
        .filter(|t| t.special)
        .collect();
    let tokens = &parsed.added_tokens;
    Ok(tokens
        .iter()
        .filter(|t| safe(t) && !suffix_overlaps(&t.content, &t.content))
        .filter(|t| {
            tokens.iter().chain(&config_tokens).all(|other| {
                if other.content == t.content {
                    return safe(other);
                }
                other.content.is_empty()
                    || !(t.content.contains(&other.content)
                        || other.content.contains(&t.content)
                        || suffix_overlaps(&t.content, &other.content)
                        || suffix_overlaps(&other.content, &t.content))
            })
        })
        .map(|t| t.content.clone())
        .collect())
}

/// Whether a proper suffix of `left` can start `right` (including self-overlap).
fn suffix_overlaps(left: &str, right: &str) -> bool {
    left.char_indices()
        .skip(1)
        .any(|(start, _)| right.starts_with(&left[start..]))
}

/// Treat `source` as a filesystem path (rather than a HuggingFace repo id)
/// when it has a path-like shape — an absolute/relative prefix or a
/// tokenizer-file suffix. HF repo ids are `namespace/name` with none of these markers, so a
/// missing local file like `/models/tok.json` reports a load error instead of
/// silently attempting a (doomed) network fetch.
fn looks_like_path(source: &str) -> bool {
    source.starts_with('/')
        || source.starts_with("./")
        || source.starts_with("../")
        || source.starts_with('~')
        || source.ends_with(".json")
        || source.ends_with(".model")
}

/// Keep the tokenizer.json path unchanged; tiktoken models additionally need
/// their configuration siblings in the same HF snapshot directory.
fn download_tokenizer(repo_id: &str) -> Result<std::path::PathBuf> {
    if let Ok(path) = download_repo_file(repo_id, "tokenizer.json") {
        return Ok(path);
    }
    let path = download_repo_file(repo_id, "tiktoken.model").with_context(|| {
        format!(
            "download tokenizer.json or tiktoken.model for HuggingFace repo {repo_id:?} \
             (pass --tokenizer-path with a local tokenizer file, or set HF_TOKEN \
             for a gated/private repo)"
        )
    })?;
    for sibling in ["config.json", "tokenizer_config.json"] {
        download_repo_file(repo_id, sibling)?;
    }
    Ok(path)
}

/// Download `file` from a HuggingFace repo id and return the cached local path.
fn download_repo_file(repo_id: &str, file: &str) -> Result<std::path::PathBuf> {
    use hf_hub::api::sync::ApiBuilder;
    let api = ApiBuilder::from_env()
        .build()
        .context("initialize HuggingFace Hub client")?;
    api.model(repo_id.to_string())
        .get(file)
        .with_context(|| format!("download {file} for HuggingFace repo {repo_id:?}"))
}

/// List the files an HF repo ships. `None` (with a warning) when the listing
/// fails, e.g. offline with a warm cache; the caller then attempts each
/// download individually, which is cache-first.
fn list_repo_files(repo_id: &str) -> Option<std::collections::HashSet<String>> {
    use hf_hub::api::sync::ApiBuilder;
    let listing = ApiBuilder::from_env()
        .build()
        .context("initialize HuggingFace Hub client")
        .and_then(|api| {
            api.model(repo_id.to_string())
                .info()
                .context("list repo files")
        });
    match listing {
        Ok(info) => Some(info.siblings.into_iter().map(|s| s.rfilename).collect()),
        Err(e) => {
            tracing::warn!(repo = %repo_id, error = %format!("{e:#}"),
                "could not list HuggingFace repo files; trying sibling downloads individually");
            None
        }
    }
}

/// Files co-located with the tokenizer named by `source` (the same value passed
/// to [`load`]): siblings of a local `tokenizer.json`, or files of the same HF
/// repo. The repo is listed once so only files it ships are downloaded; a file
/// the model lacks resolves to `None` without a network round-trip.
pub struct ModelFiles {
    source: String,
    /// Directory of a local `tokenizer.json`; `None` for an HF repo id.
    local_dir: Option<std::path::PathBuf>,
    /// Repo listing; `None` when it failed and downloads are attempted blindly.
    repo_files: Option<std::collections::HashSet<String>>,
}

impl ModelFiles {
    pub fn open(source: &str) -> Self {
        let local_dir = (Path::new(source).is_file() || looks_like_path(source)).then(|| {
            Path::new(source)
                .parent()
                .map_or_else(Default::default, Path::to_path_buf)
        });
        let repo_files = match local_dir {
            Some(_) => None,
            None => list_repo_files(source),
        };
        Self {
            source: source.to_owned(),
            local_dir,
            repo_files,
        }
    }

    fn path(&self, file: &str) -> Option<std::path::PathBuf> {
        let path = match &self.local_dir {
            Some(dir) => dir.join(file),
            None => {
                if self
                    .repo_files
                    .as_ref()
                    .is_some_and(|files| !files.contains(file))
                {
                    return None;
                }
                match download_repo_file(&self.source, file) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!(repo = %self.source, %file, error = %format!("{e:#}"),
                            "could not download; chat-formatter detection may be degraded for this \
                             model (check HF_TOKEN / network for a gated or private repo)");
                        return None;
                    }
                }
            }
        };
        path.is_file().then_some(path)
    }

    /// Fail if the model may ship `file` but it cannot be downloaded,
    /// which the readers below only warn about and report as absent.
    pub fn ensure_downloaded(&self, file: &str) -> Result<()> {
        if self.local_dir.is_none()
            && self
                .repo_files
                .as_ref()
                .is_none_or(|files| files.contains(file))
        {
            download_repo_file(&self.source, file)?;
        }
        Ok(())
    }

    /// Read the text `file`; `None` when the model ships no such file.
    pub fn text(&self, file: &str) -> Result<Option<String>> {
        self.path(file)
            .map(|p| std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display())))
            .transpose()
    }

    /// Parse the JSON `file`; `None` when the model ships no such file.
    pub fn json(&self, file: &str) -> Result<Option<serde_json::Value>> {
        self.text(file)?
            .map(|text| serde_json::from_str(&text).with_context(|| format!("parse {file}")))
            .transpose()
    }
}

pub fn encode(t: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = t.encode(text).context("encode")?;
    Ok(enc.token_ids().to_vec())
}

/// Decode token ids to a complete UTF-8 string.
///
/// Non-streaming callers (e.g. `/v1/detokenize`) get the full result either way:
/// - `DecodeResult::Complete(s)` — the token sequence ends on a codepoint boundary.
/// - `DecodeResult::Partial(s)` — the token sequence ends mid-codepoint; `s` ends
///   in U+FFFD. We return `s` as-is so the client sees the closest-possible string.
///
/// Streaming callers should NOT use this; they should consume `DecodeResult`
/// directly and withhold the trailing U+FFFD until the next decode produces a
/// `Complete` result.
pub fn decode_complete(t: &Tokenizer, ids: &[u32], skip_special: bool) -> Result<String> {
    let res = t.decode(ids, skip_special).context("decode")?;
    Ok(match res {
        DecodeResult::Complete(s) => s,
        DecodeResult::Partial(s) => {
            tracing::debug!(
                n_tokens = ids.len(),
                trailing_bytes = s.len(),
                "decode_complete: tokenizer returned Partial for non-streaming call"
            );
            s
        }
    })
}

#[cfg(test)]
mod model_files_tests {
    use super::ModelFiles;
    use serde_json::json;

    #[test]
    fn reads_sibling_json_and_template_files() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = dir.path().join("tokenizer.json");
        std::fs::write(&tokenizer, "{}").unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        std::fs::write(dir.path().join("chat_template.jinja"), "{{ messages }}").unwrap();

        let files = ModelFiles::open(tokenizer.to_str().unwrap());
        assert_eq!(
            files.json("config.json").unwrap(),
            Some(json!({"model_type":"llama"}))
        );
        assert_eq!(
            files.text("chat_template.jinja").unwrap().as_deref(),
            Some("{{ messages }}")
        );
        assert!(files.json("tokenizer_config.json").unwrap().is_none());
        assert!(files.text("missing.jinja").unwrap().is_none());
    }

    #[test]
    fn invalid_json_is_an_error_instead_of_a_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = dir.path().join("tokenizer.json");
        std::fs::write(&tokenizer, "{}").unwrap();
        std::fs::write(dir.path().join("config.json"), "invalid JSON").unwrap();

        let files = ModelFiles::open(tokenizer.to_str().unwrap());
        let error = files.json("config.json").unwrap_err();
        assert!(error.to_string().contains("config.json"));
    }
}

#[cfg(test)]
mod prompt_affix_tests {
    use super::{prompt_affixes, ModelFiles, PromptAffixes};
    use anyhow::Result;
    use serde_json::{json, Value};

    /// `prompt_affixes` of the tiny tokenizer with `normalizer`, next to `config`
    /// (unless null) and the `model` config.
    fn affixes(
        normalizer: Value,
        post_processor: Value,
        config: Value,
        model: Value,
    ) -> Result<PromptAffixes> {
        let mut data: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/tiny_tokenizer.json")).unwrap();
        data["normalizer"] = normalizer;
        data["post_processor"] = post_processor;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, data.to_string()).unwrap();
        if !config.is_null() {
            std::fs::write(dir.path().join("tokenizer_config.json"), config.to_string()).unwrap();
        }
        std::fs::write(dir.path().join("config.json"), model.to_string()).unwrap();
        let path = path.to_str().unwrap();
        prompt_affixes(path, &ModelFiles::open(path))
    }

    #[test]
    fn matches_the_engines_special_tokens() {
        let template = json!({"type": "TemplateProcessing", "pair": [], "single": [
            {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}}],
            "special_tokens": {"<|endoftext|>": {"id": "<|endoftext|>", "ids": [256], "tokens": ["<|endoftext|>"]}}});
        let llama = json!({"tokenizer_class": "LlamaTokenizerFast", "bos_token": "<|endoftext|>"});
        let mut no_bos = llama.clone();
        no_bos["add_bos_token"] = false.into();
        let embedding_gemma =
            json!({"model_type": "gemma3_text", "use_bidirectional_attention": true});
        let eos = json!({"eos_token": {"content": "<|endoftext|>"}});
        assert!(affixes(Value::Null, Value::Null, json!({}), embedding_gemma.clone()).is_err());
        for (post_processor, config, model, prefix, eos) in [
            (Value::Null, json!({}), json!({}), vec![], None),
            (template.clone(), json!({}), json!({}), vec![256], None),
            // SGLang adds BOS by `add_bos_token`, ignoring the post-processor.
            (Value::Null, llama, json!({}), vec![256], None),
            (template, no_bos, json!({}), vec![], None),
            (Value::Null, eos, embedding_gemma, vec![], Some(256)),
        ] {
            let affixes = affixes(Value::Null, post_processor, config, model).unwrap();
            let suffix = vec![];
            assert_eq!(
                affixes,
                PromptAffixes {
                    prefix,
                    suffix,
                    eos
                }
            );
        }
        // Without the config or its BOS, the engine's special tokens are unknown.
        let unknown_bos = json!({"tokenizer_class": "LlamaTokenizerFast"});
        for config in [Value::Null, unknown_bos] {
            assert!(affixes(Value::Null, Value::Null, config, json!({})).is_err());
        }
    }

    #[test]
    fn refuses_normalizers_the_engine_replaces() {
        let legacy = json!({"type": "Sequence", "normalizers": [
            {"type": "Prepend", "prepend": "\u{2581}"},
            {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"}]});
        let collapse = json!({"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "});
        for (class, normalizer, kept) in [
            ("LlamaTokenizerFast", legacy.clone(), false),
            ("PreTrainedTokenizerFast", legacy, true),
            ("XLMRobertaTokenizer", collapse, false),
            (
                "GemmaTokenizer",
                json!({"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"}),
                true,
            ),
        ] {
            let config = json!({"tokenizer_class": class, "add_bos_token": false});
            let affixes = affixes(normalizer, Value::Null, config, json!({}));
            assert_eq!(affixes.is_ok(), kept, "{class}");
        }
    }

    #[test]
    fn embedding_gemma_eos_ends_the_prompt_once() {
        let affixes = PromptAffixes {
            eos: Some(1),
            ..Default::default()
        };
        assert_eq!(
            (affixes.apply(&[7]), affixes.apply(&[7, 1])),
            (vec![7, 1], vec![7, 1])
        );
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use serde_json::{json, Value};

    fn added(content: &str) -> Value {
        json!({"content": content, "special": true, "single_word": false,
            "normalized": false, "lstrip": false, "rstrip": false})
    }

    /// Keep an independent safe boundary so both the cold path and an actual cache hit
    /// exercise the unsafe spelling. Compare each backend with its own uncached encoder.
    fn check(mut tokens: Vec<Value>, config: Option<Value>, text: &str, keep_safe: bool) {
        if keep_safe {
            tokens.insert(0, added("[SAFE]"));
        }
        for (i, token) in tokens.iter_mut().enumerate() {
            token["id"] = json!(257 + i);
        }
        let mut data: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/tiny_tokenizer.json")).unwrap();
        data["model"]["type"] = json!("BPE");
        data["added_tokens"] = tokens.into();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, data.to_string()).unwrap();
        if let Some(config) = config {
            std::fs::write(dir.path().join("tokenizer_config.json"), config.to_string()).unwrap();
        }
        let path = path.to_str().unwrap();
        let expected_boundaries = if keep_safe { vec!["[SAFE]"] } else { vec![] };
        assert_eq!(boundary_tokens(path).unwrap(), expected_boundaries);
        let text = if keep_safe {
            format!("[SAFE]{text}")
        } else {
            text.to_owned()
        };
        for backend in [TokenizerBackend::Hf, TokenizerBackend::Fast] {
            let (plain, _) = load_with(
                path,
                TokenizerConfig {
                    backend,
                    l1_cache_mb: 0,
                },
            )
            .unwrap();
            let expected = encode(&plain, &text).unwrap();
            let (cached, stats) = load_with(
                path,
                TokenizerConfig {
                    backend,
                    l1_cache_mb: 1,
                },
            )
            .unwrap();
            assert_eq!(
                stats.backend(),
                match backend {
                    TokenizerBackend::Hf => EncodeBackend::Hf,
                    TokenizerBackend::Fast => EncodeBackend::Fast,
                }
            );
            for pass in 0..3 {
                assert_eq!(
                    encode(&cached, &text).unwrap(),
                    expected,
                    "{backend:?}, pass {pass}: {text}"
                );
            }
            if keep_safe {
                assert!(stats.l1_tokens().0 > 0, "must exercise a cache hit");
            } else {
                assert_eq!(stats.l1_state(), L1State::DisabledNoSpecials);
                assert_eq!(stats.l1_tokens(), (0, 0));
            }
        }
    }

    #[test]
    fn l1_excludes_whole_word_boundaries() {
        let mut cat = added("cat");
        cat["single_word"] = json!(true);
        check(vec![cat], None, "catfish", true);
    }

    #[test]
    fn l1_excludes_overlapping_added_tokens() {
        check(
            vec![added("<x>"), added("<x>long")],
            None,
            "<x>longtail",
            true,
        );
        let mut ordinary = added("<x>long");
        ordinary["special"] = json!(false);
        check(vec![added("<x>"), ordinary], None, "<x>longtail", true);
        check(
            vec![added("<x>"), added(">tail")],
            None,
            "<x>tailmore",
            true,
        );
        check(vec![added("aba")], None, "ababa!", true);
        check(vec![added("猫猫")], None, "猫猫猫!", true);
    }

    #[test]
    fn l1_checks_sibling_special_token_declarations() {
        let mut cat = added("cat");
        cat["single_word"] = json!(true);
        check(
            vec![added("cat")],
            Some(json!({"added_tokens_decoder": {"258": cat}})),
            "catfish",
            true,
        );
        check(
            vec![added("<x>")],
            Some(json!({"added_tokens_decoder": {"259": added("<x>long")}})),
            "<x>longtail",
            true,
        );
    }

    #[test]
    fn l1_passes_through_without_safe_boundaries() {
        let mut cat = added("cat");
        cat["single_word"] = json!(true);
        check(vec![cat], None, "catfish", false);
    }
}

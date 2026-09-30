// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use dynamo_tokenizers::{
    create_tokenizer_from_file, traits, traits::DecodeResult, CacheTokenUsage, CachedTokenizer,
    FastTokenizer, Tokenizer,
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
    if Path::new(source).is_file() || looks_like_path(source) {
        return build(source, cfg).with_context(|| format!("load tokenizer from {source}"));
    }
    let downloaded = download_tokenizer(source)?;
    let path = downloaded
        .to_str()
        .context("downloaded tokenizer path is not valid UTF-8")?;
    build(path, cfg).with_context(|| format!("load downloaded tokenizer for {source}"))
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

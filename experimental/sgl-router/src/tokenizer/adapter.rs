// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use dynamo_tokenizers::{traits::DecodeResult, Tokenizer};
use std::path::Path;
use std::sync::Arc;

/// Load a tokenizer from `source`, which is either a local `tokenizer.json`
/// path or a HuggingFace repo id.
///
/// An existing local file (or anything with a filesystem-path shape) is
/// loaded directly via `Tokenizer::from_file`. Otherwise `source` is treated
/// as a HuggingFace repo id and its `tokenizer.json` is downloaded (once, at
/// startup) into the HF cache, honoring `HF_TOKEN` / `HF_HOME` /
/// `HF_HUB_OFFLINE`. `dynamo_tokenizers` itself has no HF-download path, so
/// the fetch is done here via `hf-hub`.
pub fn load(source: &str) -> Result<Arc<Tokenizer>> {
    if Path::new(source).is_file() || looks_like_path(source) {
        return Tokenizer::from_file(source)
            .map(Arc::new)
            .with_context(|| format!("load tokenizer from {source}"));
    }
    let downloaded = download_tokenizer_json(source)?;
    let path = downloaded
        .to_str()
        .context("downloaded tokenizer path is not valid UTF-8")?;
    Tokenizer::from_file(path)
        .map(Arc::new)
        .with_context(|| format!("load downloaded tokenizer for {source}"))
}

/// Treat `source` as a filesystem path (rather than a HuggingFace repo id)
/// when it has a path-like shape — an absolute/relative prefix or a `.json`
/// suffix. HF repo ids are `namespace/name` with none of these markers, so a
/// missing local file like `/models/tok.json` reports a load error instead of
/// silently attempting a (doomed) network fetch.
fn looks_like_path(source: &str) -> bool {
    source.starts_with('/')
        || source.starts_with("./")
        || source.starts_with("../")
        || source.starts_with('~')
        || source.ends_with(".json")
}

/// Download `tokenizer.json` for a HuggingFace repo id and return the cached
/// local path, adding an actionable error context. The actual fetch (blocking
/// `ureq`, `from_env` so `HF_TOKEN` / `HF_HOME` / endpoint overrides apply)
/// lives in [`download_repo_file`].
fn download_tokenizer_json(repo_id: &str) -> Result<std::path::PathBuf> {
    download_repo_file(repo_id, "tokenizer.json").with_context(|| {
        format!(
            "download tokenizer.json for HuggingFace repo {repo_id:?} \
             (pass --tokenizer-path with a local tokenizer.json, or set HF_TOKEN \
             for a gated/private repo)"
        )
    })
}

/// Download `file` from a HuggingFace repo id and return the cached local path.
/// Shared by `tokenizer.json` (required) and `tokenizer_config.json` (optional).
fn download_repo_file(repo_id: &str, file: &str) -> Result<std::path::PathBuf> {
    use hf_hub::api::sync::ApiBuilder;
    let api = ApiBuilder::from_env()
        .build()
        .context("initialize HuggingFace Hub client")?;
    api.model(repo_id.to_string())
        .get(file)
        .with_context(|| format!("download {file} for HuggingFace repo {repo_id:?}"))
}

/// Load the `tokenizer_config.json` co-located with the tokenizer named by
/// `source` (the same value passed to [`load`]). For a local
/// `.../tokenizer.json` path this is the sibling file; for an HF repo id it is
/// downloaded from the same repo.
///
/// Returns `Ok(None)` when the model ships no `tokenizer_config.json` (rare but
/// valid) — the caller then has no chat template and routes via raw prompt text.
pub fn load_tokenizer_config(source: &str) -> Result<Option<serde_json::Value>> {
    let path = if Path::new(source).is_file() || looks_like_path(source) {
        match Path::new(source).parent() {
            Some(dir) => dir.join("tokenizer_config.json"),
            None => return Ok(None),
        }
    } else {
        // HF repo id. The download error type doesn't distinguish a genuine
        // 404 (repo ships no tokenizer_config.json — benign) from auth/network
        // failures (wrong/expired HF_TOKEN, gated repo, timeout), so warn with
        // the cause rather than asserting the benign case at debug: a swallowed
        // auth error here silently disables chat-template routing.
        match download_repo_file(source, "tokenizer_config.json") {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(repo = %source, error = %e,
                    "could not download tokenizer_config.json; chat-template routing disabled for this model \
                     (expected if the repo ships none — otherwise check HF_TOKEN / network for a gated or private repo)");
                return Ok(None);
            }
        }
    };
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path)
        .with_context(|| format!("read tokenizer_config.json at {}", path.display()))?;
    let value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse tokenizer_config.json at {}", path.display()))?;
    Ok(Some(value))
}

/// Load the added-token strings from the `tokenizer.json` named by `source`
/// (the same value passed to [`load`]) — the structural markers the segment
/// cache splits on (`<|im_start|>`, role headers, `<|vision_start|>`, ...).
///
/// These live in tokenizer.json's `added_tokens` array, NOT in
/// tokenizer_config.json's `special_tokens_map` (which carries only the 7 named
/// bos/eos/... tokens). An added token is a HuggingFace hard split boundary (the
/// tokenizer pre-splits input at them before BPE), so splitting a prompt there
/// is byte-exact — EXCEPT for tokens with `lstrip`/`rstrip`, which absorb
/// adjacent whitespace across the boundary and so are NOT byte-exact split
/// points; those are excluded here. (`normalized`/`single_word` tokens are kept
/// — they don't move the boundary; validated empirically, e.g. DeepSeek-V4's
/// `normalized` `<｜User｜>`/`<｜Assistant｜>`, with the self-check as backstop.)
/// [`super::segment`] additionally probe-filters
/// the survivors to the tokens the chat template actually emits and runs a
/// startup self-check (which also catches Metaspace/normalizer models where
/// per-segment encoding diverges — those self-disable, staying correct).
///
/// Returns an empty vec — segmentation then disables itself — on any
/// missing/unreadable/unparsable file, never an error: an absent marker set is
/// a graceful "don't segment", not a fatal condition.
pub fn load_added_special_tokens(source: &str) -> Vec<String> {
    let path = if Path::new(source).is_file() || looks_like_path(source) {
        std::path::PathBuf::from(source)
    } else {
        match download_tokenizer_json(source) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(repo = %source, error = %e,
                    "could not resolve tokenizer.json for segment markers; segment cache disabled for this model");
                return Vec::new();
            }
        }
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e,
                "read tokenizer.json for segment markers failed; segment cache disabled");
            return Vec::new();
        }
    };
    let value: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e,
                "parse tokenizer.json for segment markers failed; segment cache disabled");
            return Vec::new();
        }
    };
    value
        .get("added_tokens")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    // Exclude whitespace-absorbing tokens: `lstrip`/`rstrip`
                    // pull adjacent spaces into the token, so splitting at them
                    // is not byte-exact (the space crosses the boundary).
                    let lstrip = t.get("lstrip").and_then(|b| b.as_bool()).unwrap_or(false);
                    let rstrip = t.get("rstrip").and_then(|b| b.as_bool()).unwrap_or(false);
                    if lstrip || rstrip {
                        return None;
                    }
                    t.get("content").and_then(|c| c.as_str())
                })
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
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
mod tests {
    use super::*;

    /// `load_added_special_tokens` keeps clean/normalized markers but excludes
    /// whitespace-absorbing (`lstrip`/`rstrip`) tokens, which are not byte-exact
    /// split points, and drops empty content.
    #[test]
    fn added_tokens_excludes_lstrip_rstrip() {
        let markers = load_added_special_tokens("tests/fixtures/added_tokens_flags.json");
        assert!(markers.contains(&"<|clean|>".to_string()));
        assert!(
            markers.contains(&"<|normalized|>".to_string()),
            "normalized (non-strip) tokens are kept"
        );
        for excluded in ["<|lstrip_tok|>", "<|rstrip_tok|>", "<|both_strip|>", ""] {
            assert!(
                !markers.contains(&excluded.to_string()),
                "{excluded:?} must be excluded (lstrip/rstrip/empty)"
            );
        }
        assert_eq!(markers.len(), 2, "only <|clean|> + <|normalized|> survive");
    }

    /// A missing/unreadable file yields no markers (segmentation self-disables),
    /// never a panic/error.
    #[test]
    fn added_tokens_missing_file_is_empty() {
        assert!(load_added_special_tokens("tests/fixtures/does_not_exist.json").is_empty());
    }
}

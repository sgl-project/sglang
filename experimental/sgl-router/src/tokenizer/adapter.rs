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

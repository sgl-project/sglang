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
/// local path. Uses the blocking `ureq` API (this runs once at startup,
/// before the server begins serving) and `from_env` so `HF_TOKEN` /
/// `HF_HOME` / endpoint overrides are honored.
fn download_tokenizer_json(repo_id: &str) -> Result<std::path::PathBuf> {
    use hf_hub::api::sync::ApiBuilder;
    let api = ApiBuilder::from_env()
        .build()
        .context("initialize HuggingFace Hub client")?;
    api.model(repo_id.to_string())
        .get("tokenizer.json")
        .with_context(|| {
            format!(
                "download tokenizer.json for HuggingFace repo {repo_id:?} \
                 (pass --tokenizer-path with a local tokenizer.json, or set HF_TOKEN \
                 for a gated/private repo)"
            )
        })
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

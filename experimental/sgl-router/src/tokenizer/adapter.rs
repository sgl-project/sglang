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

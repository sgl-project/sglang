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
/// startup) into the HF cache. `hf-hub` resolves `HF_TOKEN`, `HF_TOKEN_PATH`,
/// `HF_HOME`, `HF_HUB_CACHE`, and `HF_ENDPOINT`. Cached files are used without
/// contacting the Hub; `HF_HUB_OFFLINE` is not interpreted. Downloads happen
/// here, before passing a local path to `dynamo_tokenizers`.
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
/// local path, adding an actionable error context.
fn download_tokenizer_json(repo_id: &str) -> Result<std::path::PathBuf> {
    download_repo_file(repo_id, "tokenizer.json").with_context(|| {
        format!(
            "download tokenizer.json for HuggingFace repo {repo_id:?} \
             (pass --tokenizer-path with a local tokenizer.json, or set HF_TOKEN \
             for a gated/private repo)"
        )
    })
}

/// Build the shared client for downloads and repository listings.
fn hub_client() -> Result<hf_hub::HFClientSync> {
    use hf_hub::HFClient;

    let mut builder = HFClient::builder();
    // Match hf-hub's implicit-auth opt-out before applying an explicit override.
    if !std::env::var("HF_HUB_DISABLE_IMPLICIT_TOKEN").is_ok_and(|v| !v.is_empty()) {
        match std::env::var("HF_TOKEN") {
            Ok(token) if !token.is_empty() => builder = builder.token(normalize_hf_token(&token)?),
            Err(std::env::VarError::NotUnicode(_)) => anyhow::bail!("HF_TOKEN is not valid UTF-8"),
            _ => {} // hf-hub resolves the token file when HF_TOKEN is unset or empty.
        }
    }
    builder
        .build_sync()
        .context("initialize HuggingFace Hub client")
}

/// Download a repo file, using an existing cached copy before contacting the Hub.
fn download_repo_file(repo_id: &str, file: &str) -> Result<std::path::PathBuf> {
    use hf_hub::{split_id, HFError};

    let api = hub_client()?;
    let (owner, name) = split_id(repo_id);
    let repo = api.model(owner, name);
    // Preserve 0.4's cache-first `get`: 1.0 otherwise revalidates with the Hub.
    let cached = repo
        .download_file()
        .filename(file)
        .local_files_only(true)
        .send();
    match cached {
        Err(HFError::LocalEntryNotFound { .. }) => repo.download_file().filename(file).send(),
        result => result,
    }
    .with_context(|| format!("download {file} for HuggingFace repo {repo_id:?}"))
}

fn normalize_hf_token(token: &str) -> Result<&str> {
    let token = token.trim();
    anyhow::ensure!(
        !token.is_empty() && token.bytes().all(|b| (0x21..=0x7e).contains(&b)),
        "HF_TOKEN is blank or contains invalid bytes; provide a valid token or unset it to use a token file"
    );
    // hf-hub 1.0 neither trims env tokens nor reports invalid header values.
    // Reject them without including the credential in an error or log message.
    Ok(token)
}

/// List the files an HF repo ships. `None` (with a warning) when the listing
/// fails, e.g. offline with a warm cache; the caller then attempts each
/// download individually, which is cache-first.
fn list_repo_files(repo_id: &str) -> Option<std::collections::HashSet<String>> {
    let (owner, name) = hf_hub::split_id(repo_id);
    let listing = hub_client().and_then(|api| {
        api.model(owner, name)
            .info()
            .send()
            .context("list repo files")
    });
    match listing {
        Ok(info) => info
            .siblings
            .map(|files| files.into_iter().map(|s| s.rfilename).collect()),
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
mod hf_token_tests {
    use super::normalize_hf_token;

    #[test]
    fn trims_valid_tokens() {
        for token in ["hf_example", "  hf_example\n", "\thf_example\r\n"] {
            assert_eq!(normalize_hf_token(token).unwrap(), "hf_example");
        }
    }

    #[test]
    fn rejects_invalid_tokens_without_echoing_them() {
        for token in [
            "",
            " \n\t",
            "hf_secret\nvalue",
            "hf_secret value",
            "hf_secret\u{7f}",
            "hf_é_secret",
        ] {
            let error = normalize_hf_token(token).unwrap_err().to_string();
            assert!(error.contains("HF_TOKEN"));
            assert!(!error.contains("secret"));
        }
    }
}

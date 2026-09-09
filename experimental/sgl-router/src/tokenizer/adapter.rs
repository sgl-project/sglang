// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use dynamo_tokenizers::{traits::DecodeResult, Tokenizer};
use std::env::VarError;
use std::path::Path;
use std::sync::Arc;

/// Load a tokenizer from `source`, which is either a local `tokenizer.json`
/// path or a HuggingFace repo id.
///
/// An existing local file (or anything with a filesystem-path shape) is
/// loaded directly via `Tokenizer::from_file`. Otherwise `source` is treated
/// as a HuggingFace repo id and its `tokenizer.json` is downloaded (once, at
/// startup) into the HF cache, honoring `HF_HOME`, `HF_ENDPOINT` and
/// `HF_TOKEN` — `hf-hub` reads the first two, `hf_token_override` below reads
/// the third. `hf-hub` 0.4 has no offline mode, so `HF_HUB_OFFLINE` is NOT
/// honored. `dynamo_tokenizers` itself has no HF-download path, so the
/// fetch is done here via `hf-hub`.
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
/// `ureq`, `from_env` for the `HF_HOME` / `HF_ENDPOINT` overrides plus an
/// explicit `HF_TOKEN` read) lives in [`download_repo_file`].
fn download_tokenizer_json(repo_id: &str) -> Result<std::path::PathBuf> {
    download_repo_file(repo_id, "tokenizer.json").with_context(|| {
        format!(
            "download tokenizer.json for HuggingFace repo {repo_id:?} \
             (pass --tokenizer-path with a local tokenizer.json, or set HF_TOKEN \
             to a valid token for a gated/private repo — a blank or malformed \
             HF_TOKEN is ignored with a startup warning)"
        )
    })
}

/// Download `file` from a HuggingFace repo id and return the cached local path.
/// Shared by `tokenizer.json` (required) and `tokenizer_config.json` (optional).
fn download_repo_file(repo_id: &str, file: &str) -> Result<std::path::PathBuf> {
    use hf_hub::api::sync::ApiBuilder;
    let mut builder = ApiBuilder::from_env();
    if let Some(token) = hf_token_override(std::env::var("HF_TOKEN")) {
        builder = builder.with_token(Some(token));
    }
    let api = builder
        .build()
        .context("initialize HuggingFace Hub client")?;
    api.model(repo_id.to_string())
        .get(file)
        .with_context(|| format!("download {file} for HuggingFace repo {repo_id:?}"))
}

/// Resolve `HF_TOKEN` into a token override, or `None` to leave alone whatever
/// token `hf-hub` already resolved.
///
/// hf-hub 0.4's `ApiBuilder::from_env()` reads `HF_HOME` and `HF_ENDPOINT` but
/// NOT `HF_TOKEN`; its only token source is the `$HF_HOME/token` file written
/// by `huggingface-cli login`, so a gated repo 401s with `HF_TOKEN` exported.
///
/// `None` means "leave the existing token alone", never "no token".
/// `with_token` OVERWRITES, so passing `None` through to it would clobber the
/// file-based token — which is why the caller must keep the `if let`.
fn hf_token_override(var: Result<String, VarError>) -> Option<String> {
    let ignored = |reason: &str| {
        tracing::warn!(
            "HF_TOKEN is set but {reason}; ignoring it and falling back to $HF_HOME/token \
             or anonymous access (gated/private repos will 401)"
        );
        None
    };
    match var {
        // Trimmed: a surrounding space rides through ureq's header validation
        // and 401s at HF indistinguishably from sending no token at all.
        Ok(token) => match token.trim() {
            "" => ignored("empty or whitespace-only"),
            // ureq validates header values before sending and puts the WHOLE
            // offending line — token included — in the error it returns, which
            // reaches stderr. Reject here so a malformed token can never be
            // echoed. Legal header bytes are ' ', '\t' and 0x21..=0x7E; after
            // trimming, anything outside 0x21..=0x7E is a broken secret mount.
            t => match t.bytes().position(|b| !(0x21..=0x7E).contains(&b)) {
                // Never interpolate the token itself: offset and byte value
                // are enough to find an embedded newline or a stray CR.
                Some(i) => ignored(&format!(
                    "not a valid HTTP header value (byte {:#04x} at offset {i})",
                    t.as_bytes()[i]
                )),
                None => Some(t.to_string()),
            },
        },
        Err(VarError::NotUnicode(_)) => ignored("not valid UTF-8"),
        // Genuinely unset is the legitimate token-file / public-repo path.
        Err(VarError::NotPresent) => None,
    }
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

#[cfg(test)]
mod tests {
    use super::hf_token_override;
    use std::env::VarError;

    #[test]
    fn unset_hf_token_yields_no_override() {
        // Load-bearing: `with_token` OVERWRITES, so an unset HF_TOKEN must
        // produce no override at all. Passing `None` through would clobber the
        // $HF_HOME/token value `from_env` already resolved.
        assert_eq!(hf_token_override(Err(VarError::NotPresent)), None);
    }

    #[test]
    fn set_hf_token_overrides() {
        assert_eq!(
            hf_token_override(Ok("hf_abc123".into())),
            Some("hf_abc123".to_string()),
        );
    }

    #[test]
    fn hf_token_is_trimmed() {
        // Secret mounts can leave trailing whitespace on the value. A trailing
        // space or tab is a LEGAL header byte, so untrimmed it rides through
        // ureq's validation and HF rejects it with the same 401 a missing
        // token gives. A trailing newline would instead trip ureq's local
        // header validation — see token_with_invalid_header_byte_yields_no_override.
        assert_eq!(
            hf_token_override(Ok("  hf_abc123\n".into())),
            Some("hf_abc123".to_string()),
        );
    }

    #[test]
    fn blank_hf_token_yields_no_override() {
        assert_eq!(hf_token_override(Ok(String::new())), None);
        assert_eq!(hf_token_override(Ok("  \n\t ".into())), None);
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_hf_token_yields_no_override() {
        use std::os::unix::ffi::OsStringExt;
        let bad = std::ffi::OsString::from_vec(vec![0xff, 0xfe]);
        assert_eq!(hf_token_override(Err(VarError::NotUnicode(bad))), None);
    }

    #[test]
    fn token_with_invalid_header_byte_yields_no_override() {
        // A newline INSIDE the value survives trimming and would reach ureq,
        // which rejects it and quotes the whole `authorization: Bearer <tok>`
        // line into an error that lands on stderr. Rejecting here is what
        // keeps the credential out of the logs.
        for bad in ["hf_abc\ndef", "hf_abc\rdef", "hf_abc\u{7f}def", "hf_é_abc"] {
            assert_eq!(
                hf_token_override(Ok(bad.to_string())),
                None,
                "must reject a token carrying a byte illegal in a header value: {bad:?}",
            );
        }
    }
}

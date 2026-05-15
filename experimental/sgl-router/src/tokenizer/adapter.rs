// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use dynamo_tokenizers::{traits::DecodeResult, Tokenizer};
use std::sync::Arc;

pub fn load(path: &str) -> Result<Arc<Tokenizer>> {
    Tokenizer::from_file(path)
        .map(Arc::new)
        .with_context(|| format!("load tokenizer from {path}"))
}

pub fn encode(t: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = t.encode(text).context("encode")?;
    Ok(enc.token_ids().to_vec())
}

/// Decode a complete sequence of token IDs to text. The dynamo-tokenizers
/// `DecodeResult::Partial` variant — used to signal "this ends mid-UTF8
/// codepoint, you may need more bytes" — is intentionally collapsed into
/// the same String here.
///
/// **Do NOT use this for incremental streaming detokenization.** A token
/// boundary that splits a multi-byte UTF-8 codepoint (CJK, emoji,
/// byte-fallback BPE) will appear here as a Partial string, and the
/// caller will lose the signal that the next token's prefix must be
/// concatenated before display. Streaming detokenization will get its
/// own API in M2 (likely returning `DecodeResult` directly).
pub fn decode_complete(t: &Tokenizer, ids: &[u32], skip_special: bool) -> Result<String> {
    let res = t.decode(ids, skip_special).context("decode")?;
    Ok(match res {
        DecodeResult::Complete(s) => s,
        DecodeResult::Partial(s) => s,
    })
}

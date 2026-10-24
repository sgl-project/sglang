// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Kimi-K3's tokenization: `sglang_processor` renders its prompts, and the
//! checkpoint's `tokenization_kimi.py` encodes them in chunks.

use anyhow::Result;
use dynamo_tokenizers::{EncodeSegment, Tokenizer};

/// `tokenization_kimi.py` encodes 400k-char windows, each split after 25k
/// consecutive (non-)whitespace chars, and BPE is not chunk-invariant.
pub(super) fn encode(tokenizer: &Tokenizer, segments: &[EncodeSegment<'_>]) -> Result<Vec<u32>> {
    let mut chunks = Vec::new();
    for segment in segments {
        let (mut start, mut run, mut was_space) = (0, 0, false);
        for (count, (offset, ch)) in segment.text.char_indices().enumerate() {
            // Python's `str.isspace` also covers U+001C..U+001F.
            let space = ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch);
            run = if space == was_space { run + 1 } else { 1 };
            if (count > 0 && count % 400_000 == 0) || run > 25_000 {
                chunks.push(EncodeSegment::new(
                    &segment.text[start..offset],
                    segment.allow_special,
                ));
                (start, run) = (offset, 1);
            }
            was_space = space;
        }
        chunks.push(EncodeSegment::new(
            &segment.text[start..],
            segment.allow_special,
        ));
    }
    Ok(tokenizer.encode_segments(&chunks)?.token_ids().to_vec())
}

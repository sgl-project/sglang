// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Kimi's segmented encoding preserves control tokens and Python's chunk boundaries.

use anyhow::{Context, Result};
use dynamo_tokenizers::{EncodeSegment, Tokenizer};
use std::{path::Path, sync::Arc};

pub fn load(source: &str) -> Result<Arc<Tokenizer>> {
    let files = super::adapter::ModelFiles::open(source);
    let path = if Path::new(source).is_file() && !source.ends_with(".json") {
        Path::new(source).to_path_buf()
    } else {
        files
            .path("tiktoken.model")
            .context("Kimi-K3 requires the model's tiktoken.model vocabulary")?
    };
    // The native tokenizer reads these siblings when loading its vocabulary.
    files.json("config.json")?;
    files.json("tokenizer_config.json")?;
    let tokenizer = Tokenizer::from_file(path.to_str().context("tokenizer path is not UTF-8")?)?;
    for marker in ["<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"] {
        anyhow::ensure!(
            tokenizer
                .encode_segments(&[EncodeSegment::new(marker, true)])?
                .token_ids()
                .len()
                == 1,
            "Kimi vocabulary is missing {marker}"
        );
    }
    Ok(Arc::new(tokenizer))
}

// The pinned renderer defaults a null effort to max; the engine omits its preamble.
pub fn without_effort_preamble(
    prompt: dynamo_renderer::RenderedPrompt,
) -> dynamo_renderer::RenderedPrompt {
    if prompt
        .as_str()
        .starts_with("<|open|>message role=\"system\" type=\"thinking-effort\"")
    {
        if let Some(segments) = prompt.segments() {
            if let Some(end) = segments
                .iter()
                .position(|s| s.allow_special && s.text == "<|end_of_msg|>")
            {
                return dynamo_renderer::RenderedPrompt::segmented(segments[end + 1..].to_vec());
            }
        }
    }
    prompt
}

pub fn encode_segments(tokenizer: &Tokenizer, segments: &[EncodeSegment<'_>]) -> Result<Vec<u32>> {
    let mut chunks = Vec::new();
    for segment in segments {
        let mut outer_start = 0;
        for (count, (offset, _)) in segment.text.char_indices().enumerate() {
            if count > 0 && count % 400_000 == 0 {
                split_runs(
                    &segment.text[outer_start..offset],
                    segment.allow_special,
                    &mut chunks,
                );
                outer_start = offset;
            }
        }
        split_runs(
            &segment.text[outer_start..],
            segment.allow_special,
            &mut chunks,
        );
    }
    Ok(tokenizer.encode_segments(&chunks)?.token_ids().to_vec())
}

fn split_runs<'a>(text: &'a str, special: bool, chunks: &mut Vec<EncodeSegment<'a>>) {
    let (mut start, mut length, mut was_space) = (0, 0, false);
    for (offset, ch) in text.char_indices() {
        let space = ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch);
        length = if space == was_space { length + 1 } else { 1 };
        was_space = space;
        if length > 25_000 {
            chunks.push(EncodeSegment::new(&text[start..offset], special));
            start = offset;
            length = 1;
        }
    }
    if start < text.len() {
        chunks.push(EncodeSegment::new(&text[start..], special));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::chat_formatter::ChatFormatter;

    #[test]
    fn reference_prompt_token_ids() {
        let path = "tests/fixtures/kimi_k3/tiktoken.model";
        let tokenizer = load(path).unwrap();
        let formatter = ChatFormatter::load("served-alias", path).unwrap().unwrap();
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("../../tests/fixtures/kimi_k3/prompts.json"))
                .unwrap();
        for case in cases {
            let expected: Vec<u32> = serde_json::from_value(case["token_ids"].clone()).unwrap();
            assert_eq!(
                formatter.encode(&tokenizer, &case["request"]).unwrap(),
                expected,
                "{}",
                case["name"]
            );
        }
    }

    #[test]
    fn python_whitespace_and_unicode_chunk_boundaries() {
        let text = format!("{}\u{1c}{}", "x".repeat(20_000), "y".repeat(20_000));
        let mut chunks = Vec::new();
        split_runs(&text, false, &mut chunks);
        assert_eq!(chunks.len(), 1);
        let text = "界".repeat(25_001);
        chunks.clear();
        split_runs(&text, false, &mut chunks);
        assert_eq!(
            chunks
                .iter()
                .map(|c| c.text.chars().count())
                .collect::<Vec<_>>(),
            [25_000, 1]
        );
        assert!(chunks.iter().all(|c| !c.allow_special));
    }
}

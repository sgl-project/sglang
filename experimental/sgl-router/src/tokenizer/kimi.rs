// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Model-file loading and parity guards for Dynamo's native Kimi support.

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

/// Python splits long segments before BPE; the pinned native backend does not.
/// Leave those requests to the engine until Dynamo implements matching chunking.
/// This only checks eligibility; Dynamo owns all encoding and segment handling.
pub fn validate_native_segments(segments: &[EncodeSegment<'_>]) -> Result<()> {
    for segment in segments {
        let (mut run, mut was_space) = (0, false);
        for (count, ch) in segment.text.chars().enumerate() {
            anyhow::ensure!(
                count < 400_000,
                "Kimi segment requires engine-side chunking"
            );
            // Python str.isspace() also includes these four control characters.
            let space = ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch);
            run = if space == was_space { run + 1 } else { 1 };
            was_space = space;
            anyhow::ensure!(run <= 25_000, "Kimi text run requires engine-side chunking");
        }
    }
    Ok(())
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
            // Dynamo 5.1.2 defaults null effort to max; Python omits the preamble.
            // Keep the reference case, but reject encoding instead of rewriting output.
            if case["name"] == "no_effort" {
                assert!(formatter.encode(&tokenizer, &case["request"]).is_err());
                continue;
            }
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
    fn long_segments_fall_back_without_reimplementing_python_chunking() {
        let path = "tests/fixtures/kimi_k3/tiktoken.model";
        let tokenizer = load(path).unwrap();
        let formatter = ChatFormatter::load("served-alias", path).unwrap().unwrap();
        // Boundaries count Unicode characters, and Python treats U+001C as whitespace.
        for (text, supported) in [
            ("界".repeat(25_000), true),
            ("界".repeat(25_001), false),
            (" ".repeat(25_001), false),
            (
                format!("{}\u{1c}{}", "x".repeat(20_000), "y".repeat(20_000)),
                true,
            ),
            ("x ".repeat(200_000), true),
            (format!("{}x", "x ".repeat(200_000)), false),
        ] {
            let request = serde_json::json!({"messages": [{"role": "user", "content": text}]});
            assert_eq!(formatter.encode(&tokenizer, &request).is_ok(), supported);
        }
    }
}

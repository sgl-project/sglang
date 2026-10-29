// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use base64::{engine::general_purpose::STANDARD, Engine};

pub fn tokenizer() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let bytes = (0..=255u8).map(|byte| vec![byte]);
    let merges = include_str!("kimi_k3/merges.txt")
        .split_whitespace()
        .map(|s| s.as_bytes().to_vec());
    let vocab: String = bytes
        .chain(merges)
        .enumerate()
        .map(|(rank, token)| format!("{} {rank}\n", STANDARD.encode(token)))
        .collect();
    std::fs::write(dir.path().join("tiktoken.model"), vocab).unwrap();
    for (name, contents) in [
        ("config.json", r#"{"model_type":"kimi_k3"}"#),
        (
            "tokenizer_config.json",
            include_str!("kimi_k3/tokenizer_config.json"),
        ),
    ] {
        std::fs::write(dir.path().join(name), contents).unwrap();
    }
    dir
}

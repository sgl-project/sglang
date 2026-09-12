// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Load-time raw-prompt specials probe (`add_special_tokens` parity for
//! `/generate` routing): which loads store a delta and which degrade to
//! "no specials" without failing startup.

use sgl_router::config::Cli;
use sgl_router::tokenizer::TokenizerRegistry;

/// Build a registry the way a deployment does — through `Cli`, so the flag
/// spelling is what these tests pin.
fn registry(tokenizer_path: &str, extra_flags: &[&str]) -> TokenizerRegistry {
    let mut argv = vec![
        "sgl-router",
        "--model-id",
        "tiny",
        "--tokenizer-path",
        tokenizer_path,
        "--worker-urls",
        "http://placeholder:0",
    ];
    argv.extend_from_slice(extra_flags);
    let cfg = <Cli as clap::Parser>::parse_from(argv)
        .into_config()
        .expect("flags must parse");
    TokenizerRegistry::load_from_config(&cfg).expect("registry load must succeed")
}

/// A byte-level post-processor adds no specials: the probe runs (HF backend,
/// HF artifact) and stores nothing.
#[test]
fn byte_level_model_stores_no_specials() {
    let reg = registry("tests/fixtures/tiny_tokenizer.json", &[]);
    assert!(reg.raw_prompt_specials("tiny").is_none());
}

/// The `TemplateProcessing` fixture prepends `<|endoftext|>` (id 256) under
/// `add_special_tokens = true`: the probe stores exactly that prefix.
#[test]
fn template_model_stores_bos_prefix() {
    let reg = registry("tests/fixtures/tiny_bos_tokenizer.json", &[]);
    let specials = reg
        .raw_prompt_specials("tiny")
        .expect("BOS-adding tokenizer must store a probed delta");
    assert_eq!(specials.prefix, vec![256]);
    assert!(specials.suffix.is_empty());
}

/// The fast backend ignores `add_special_tokens`, so the probe must be
/// skipped — and crucially startup must still succeed. (Uses the BPE
/// fixture, the one fastokens can load.)
#[test]
fn fast_backend_skips_probe_and_still_starts() {
    let reg = registry(
        "tests/fixtures/tiny_bpe_tokenizer.json",
        &["--tokenizer-backend", "fast"],
    );
    assert!(reg.get("tiny").is_some(), "tokenizer must be served");
    assert!(reg.raw_prompt_specials("tiny").is_none());
}

/// A tiktoken-directory artifact (Kimi-K2 / Moonlight shape) has no
/// `tokenizer.json` for the HF construction option to dispatch on: probe
/// skipped, startup succeeds, the vocabulary still serves.
#[test]
fn tiktoken_dir_skips_probe_and_still_starts() {
    let dir = std::env::temp_dir().join("sgl_router_specials_tiktoken");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tokenizer/testdata");
    for f in ["tiktoken.model", "tokenizer_config.json"] {
        std::fs::copy(src.join("kimi_k3_tiny_vocab").join(f), dir.join(f)).unwrap();
    }
    // `from_file_auto` needs a `model_type` to pick the BPE pattern.
    std::fs::write(dir.join("config.json"), br#"{"model_type": "kimi_k2"}"#).unwrap();

    let reg = registry(dir.to_str().unwrap(), &[]);
    assert!(reg.get("tiny").is_some(), "tokenizer must be served");
    assert!(reg.raw_prompt_specials("tiny").is_none());
    std::fs::remove_dir_all(&dir).unwrap();
}

// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use base64::{engine::general_purpose::STANDARD, Engine};
use dynamo_tokenizers::{traits::Encoder as _, FastTikTokenTokenizer};
use serde_json::{json, Value};
use sgl_router::config::{TokenizerBackend, TokenizerConfig};
use sgl_router::tokenizer::{adapter, chat_formatter::ChatFormatter};
use sha2::{Digest, Sha256};

#[path = "../../fixtures/kimi_k3.rs"]
mod fixture;

#[test]
fn kimi_tokens_match_sglang() {
    for backend in [TokenizerBackend::Hf, TokenizerBackend::Fast] {
        kimi_tokens_match_sglang_on(backend);
    }
}

/// A vocabulary with one merge across an apostrophe (`q'`), so the pretoken fastokens
/// ends before `'ſ` changes the ids the way `I'` does in the real vocabulary.
fn long_s_vocab() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let vocab: String = (0..=255u8)
        .map(|byte| vec![byte])
        .chain([b"q'".to_vec()])
        .enumerate()
        .map(|(rank, token)| format!("{} {rank}\n", STANDARD.encode(token)))
        .collect();
    std::fs::write(dir.path().join("tiktoken.model"), vocab).unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"kimi_k3"}"#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{"added_tokens_decoder":{"257":{"content":"[BOS]","special":true},"258":{"content":"[EOS]","special":true}}}"#,
    )
    .unwrap();
    dir
}

/// tiktoken folds `'ſ` (U+017F) into the `(?i:'s)` contraction; fastokens alone does
/// not, so the `fast` backend must encode that input on tiktoken-rs.
#[test]
fn fast_backend_matches_tiktoken_on_long_s() {
    let dir = long_s_vocab();
    let path = dir.path().join("tiktoken.model");
    let path = path.to_str().unwrap();
    let reference = adapter::load(path).unwrap();
    let cfg = TokenizerConfig {
        backend: TokenizerBackend::Fast,
        l1_cache_mb: 0,
    };
    let (fast, _) = adapter::load_with(path, cfg).unwrap();
    let text = " q'\u{17f} and q's";
    let expected = adapter::encode(&reference, text).unwrap();
    assert_eq!(adapter::encode(&fast, text).unwrap(), expected);
    // Why the guard exists. When this stops holding, fastokens agrees with tiktoken
    // on the contraction and the tiktoken-rs fallback can go.
    let bare = FastTikTokenTokenizer::from_file_auto(path).unwrap();
    assert_ne!(bare.encode(text).unwrap().token_ids(), expected.as_slice());
}

fn kimi_tokens_match_sglang_on(backend: TokenizerBackend) {
    let fixture = fixture::tokenizer();
    let path = fixture.path().join("tiktoken.model");
    let path = path.to_str().unwrap();
    let cfg = TokenizerConfig {
        backend,
        l1_cache_mb: 0,
    };
    let (tokenizer, _) = adapter::load_with(path, cfg).unwrap();
    let formatter = ChatFormatter::load("served-alias", path).unwrap().unwrap();
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("../../fixtures/kimi_k3/prompts.json")).unwrap();
    for case in cases {
        let mut request = case["request"].clone();
        if let Some(repeat) = case["repeat"].as_u64() {
            request["messages"][0]["content"] = request["messages"][0]["content"]
                .as_str()
                .unwrap()
                .repeat(repeat as usize)
                .into();
        }
        let ids = formatter.encode(&tokenizer, &request).unwrap();
        let mut hash = Sha256::new();
        for id in &ids {
            hash.update(id.to_le_bytes());
        }
        assert_eq!(json!(ids.len()), case["token_count"], "{}", case["name"]);
        assert_eq!(
            format!("{:x}", hash.finalize()),
            case["sha256"],
            "{}",
            case["name"]
        );
    }
    let request = json!({"messages":[{"role":"user","content":"hi"}], "chat_template_kwargs":{"thinking_effort":null}});
    assert!(formatter.encode(&tokenizer, &request).is_err());
    // SGLang's `_validate_request` rejects a required tool_choice without tools.
    let request = json!({"messages":[{"role":"user","content":"hi"}], "tool_choice":"required"});
    assert!(formatter.encode(&tokenizer, &request).is_err());
    // Media renders on the engine; the router never forwards ids for it.
    let image = json!({"type":"image_url","image_url":{"url":"https://example.com/a.png"}});
    let request = json!({"messages":[{"role":"user","content":[image]}]});
    assert!(formatter.encode(&tokenizer, &request).is_err());
}

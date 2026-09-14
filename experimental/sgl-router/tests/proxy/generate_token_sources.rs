// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Where `/generate` routing tokens come from: client `input_ids` verbatim,
//! else the prompt `text`, and nothing else. `GenerateReqInput` ignores extra
//! keys, so a body carrying a stray `messages` or `prompt` is legal — routing
//! on either would key on a prompt the engine never sees. Also pins the
//! chat surface against the same confusion: a chat body carrying `input_ids`
//! still routes through the chat encoder.

use serde_json::{json, Value};
use sgl_router::config::Cli;
use sgl_router::discovery::ModelId;
use sgl_router::policies::{request_tokens_for, request_tokens_for_generate};
use sgl_router::tokenizer::{adapter, TokenizerRegistry};

fn registry_with(tokenizer_path: &str) -> TokenizerRegistry {
    let cfg = <Cli as clap::Parser>::parse_from([
        "sgl-router",
        "--model-id",
        "tiny",
        "--tokenizer-path",
        tokenizer_path,
        "--worker-urls",
        "http://placeholder:0",
    ])
    .into_config()
    .expect("flags must parse");
    TokenizerRegistry::load_from_config(&cfg).expect("registry load must succeed")
}

/// A registry whose model has a chat encoder: the tiny tokenizer plus a
/// sibling `tokenizer_config.json` carrying a chat template.
fn chat_encoder_registry() -> (tempfile::TempDir, TokenizerRegistry) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::copy(
        "tests/fixtures/tiny_tokenizer.json",
        dir.path().join("tokenizer.json"),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{"chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}",
            "bos_token": "<|endoftext|>"}"#,
    )
    .unwrap();
    let reg = registry_with(dir.path().to_str().unwrap());
    assert!(
        reg.has_chat_encoder("tiny"),
        "the fixture must attach a Jinja chat encoder"
    );
    (dir, reg)
}

/// A `text` body routes on exactly the registry's own encode of that text —
/// not a re-tokenization, not a decorated variant. This fixture is
/// `ByteLevel`, so it cannot tell `add_special_tokens` true from false; that
/// flag is pinned where a fixture CAN tell the difference, in
/// `adapter::tests::router_encodes_without_special_tokens`.
#[test]
fn generate_text_routes_on_the_plain_encoding() {
    let reg = registry_with("tests/fixtures/tiny_tokenizer.json");
    let model = ModelId("tiny".into());
    let tokens = request_tokens_for_generate(&reg, &model, &json!({"text": "hello"}))
        .expect("text body must tokenize");
    let want = adapter::encode(&reg.get("tiny").unwrap(), "hello").unwrap();
    assert_eq!(tokens.ids, want);
}

/// Client-supplied `input_ids` (flat integer array) are used verbatim for
/// routing — perfect parity by construction, no tokenizer needed.
#[test]
fn generate_client_input_ids_are_used_verbatim() {
    let reg = registry_with("tests/fixtures/tiny_tokenizer.json");
    let model = ModelId("tiny".into());
    let tokens = request_tokens_for_generate(
        &reg,
        &model,
        &json!({"input_ids": [450, 12, 99], "text": "ignored for routing"}),
    )
    .expect("input_ids body must produce routing tokens");
    assert_eq!(tokens.ids, vec![450, 12, 99]);
    assert!(
        !tokens.engine_equivalent,
        "verbatim client ids are never re-forwarded by the router"
    );
}

/// A `/generate` body carrying a stray `messages` (legal — the engine's
/// dataclass ignores it) routes on `text`, NOT through the chat encoder;
/// same for a stray `prompt`. Routing on either would key on a prompt the
/// engine never sees.
#[test]
fn generate_ignores_stray_messages_and_prompt() {
    let (_dir, reg) = chat_encoder_registry();
    let model = ModelId("tiny".into());
    let want = adapter::encode(&reg.get("tiny").unwrap(), "hello").unwrap();

    for body in [
        json!({"text": "hello", "messages": [{"role": "user", "content": "zzzz"}]}),
        json!({"text": "hello", "prompt": "zzzz"}),
    ] {
        let tokens =
            request_tokens_for_generate(&reg, &model, &body).expect("text body must tokenize");
        assert_eq!(
            tokens.ids, want,
            "routing tokens must come from `text`, not the stray key: {body}"
        );
        assert!(!tokens.engine_equivalent, "{body}");
    }
}

/// The isolation cuts the other way too: a `/v1/chat/completions` body
/// carrying client `input_ids` still routes via `encode_chat` — the
/// `input_ids` branch is `Generate`-only.
#[test]
fn chat_input_ids_does_not_bypass_the_chat_encoder() {
    let (_dir, reg) = chat_encoder_registry();
    let model = ModelId("tiny".into());
    let body: Value = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "input_ids": [1, 2, 3],
    });
    let tokens = request_tokens_for(&reg, &model, &body).expect("chat body must tokenize");
    assert!(
        tokens.engine_equivalent,
        "a chat body must route through the chat encoder even when input_ids is present"
    );
    assert_ne!(
        tokens.ids,
        vec![1, 2, 3],
        "the client's input_ids must not win on the chat surface"
    );
}

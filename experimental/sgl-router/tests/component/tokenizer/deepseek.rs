// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use serde_json::{json, Value};
use sgl_router::tokenizer::{adapter, chat_formatter::ChatFormatter};
use sha2::{Digest, Sha256};

fn check_fixture(fixture: &str, model_type: &str) {
    let fixture: Value = serde_json::from_str(fixture).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, "{}").unwrap();
    let hf_home = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|p| p.join(".cache/huggingface")));
    let cached = hf_home.map(|p| {
        p.join("hub")
            .join(format!(
                "models--{}",
                fixture["model"].as_str().unwrap().replace('/', "--")
            ))
            .join("snapshots")
            .join(fixture["revision"].as_str().unwrap())
            .join("tokenizer.json")
    });
    let tokenizer = cached
        .filter(|p| p.is_file())
        .map(|p| adapter::load(p.to_str().unwrap()).unwrap());
    if tokenizer.is_none() {
        eprintln!(
            "{}: checking rendered text; pinned tokenizer is not cached",
            fixture["model"]
        );
    }
    for case in fixture["cases"].as_array().unwrap() {
        std::fs::write(
            dir.path().join("config.json"),
            json!({"model_type":model_type, "dsv4_reasoning_effort_profile":case["profile"]})
                .to_string(),
        )
        .unwrap();
        let formatter = ChatFormatter::load("served-alias", path.to_str().unwrap())
            .unwrap()
            .unwrap();
        let text = formatter
            .render(&case["request"])
            .unwrap_or_else(|e| panic!("{}: {e:#}", case["name"]));
        assert_eq!(text, case["prompt"].as_str().unwrap(), "{}", case["name"]);
        if let Some(tokenizer) = &tokenizer {
            let ids = formatter.encode(tokenizer, &case["request"]).unwrap();
            let mut hash = Sha256::new();
            for id in &ids {
                hash.update(id.to_le_bytes());
            }
            assert_eq!(
                ids.len() as u64,
                case["token_count"].as_u64().unwrap(),
                "{}",
                case["name"]
            );
            assert_eq!(
                format!("{:x}", hash.finalize()),
                case["token_sha256"].as_str().unwrap(),
                "{}",
                case["name"]
            );
        }
    }
}

#[test]
fn v4_matches_sglang_serving() {
    check_fixture(
        include_str!("../../fixtures/deepseek/v4.json"),
        "deepseek_v4",
    );
}

#[test]
fn v41_matches_sglang_serving() {
    check_fixture(
        include_str!("../../fixtures/deepseek/v41.json"),
        "deepseek_v41",
    );
}

#[test]
fn v41_leaves_unsupported_shapes_to_the_worker() {
    let formatter = ChatFormatter::deepseek_native(Some("deepseek_v41"), "alias").unwrap();
    for request in [
        json!({"messages":[{"role":"developer","content":"rule"}]}),
        json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]}),
    ] {
        assert!(formatter.render(&request).is_err(), "{request}");
    }
}

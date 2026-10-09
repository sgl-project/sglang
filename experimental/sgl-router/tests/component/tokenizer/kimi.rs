// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use serde_json::{json, Value};
use sgl_router::tokenizer::{adapter, chat_formatter::ChatFormatter};
use sha2::{Digest, Sha256};

#[path = "../../fixtures/kimi_k3.rs"]
mod fixture;

#[test]
fn kimi_tokens_match_sglang() {
    let fixture = fixture::tokenizer();
    let path = fixture.path().join("tiktoken.model");
    let path = path.to_str().unwrap();
    let tokenizer = adapter::load(path).unwrap();
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
        // Dynamo rejects a system turn with both content and tools, which SGLang
        // renders; the router leaves that shape to the worker.
        if case["name"] == "developer_tools" {
            assert!(formatter.encode(&tokenizer, &request).is_err());
            continue;
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
}

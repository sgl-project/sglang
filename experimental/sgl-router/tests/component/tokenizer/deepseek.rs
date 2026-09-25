// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::sync::Arc;

use dynamo_tokenizers::Tokenizer;
use serde_json::{json, Value};
use sgl_router::config::{TokenizerBackend, TokenizerConfig};
use sgl_router::tokenizer::{adapter, chat_formatter::ChatFormatter, stats::EncodeBackend};
use sha2::{Digest, Sha256};

/// The fixture's pinned tokenizer.json, when it is in the local HF cache.
fn pinned_tokenizer(fixture: &Value) -> Option<PathBuf> {
    let hf_home = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|p| p.join(".cache/huggingface")))?;
    let path = hf_home
        .join("hub")
        .join(format!(
            "models--{}",
            fixture["model"].as_str().unwrap().replace('/', "--")
        ))
        .join("snapshots")
        .join(fixture["revision"].as_str().unwrap())
        .join("tokenizer.json");
    path.is_file().then_some(path)
}

/// Every encoder configuration the served token ids must hold under, with how many passes to
/// encode each case: the L1 cache is replayed so the second pass reads cached prefixes.
fn encoders(path: &Path) -> Vec<(&'static str, Arc<Tokenizer>, usize)> {
    let fast = |l1_cache_mb| TokenizerConfig {
        backend: TokenizerBackend::Fast,
        l1_cache_mb,
    };
    [
        ("hf", TokenizerConfig::default(), EncodeBackend::Hf, 1),
        ("fast", fast(0), EncodeBackend::Fast, 1),
        ("fast+l1", fast(64), EncodeBackend::Fast, 2),
    ]
    .into_iter()
    .map(|(label, cfg, backend, passes)| {
        let (tokenizer, stats) = adapter::load_with(path.to_str().unwrap(), cfg).unwrap();
        assert_eq!(stats.backend(), backend, "{label}");
        (label, tokenizer, passes)
    })
    .collect()
}

fn check_fixture(fixture: &str, model_type: &str) {
    let fixture: Value = serde_json::from_str(fixture).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, "{}").unwrap();
    let tokenizers = pinned_tokenizer(&fixture).map_or_else(Vec::new, |p| encoders(&p));
    if tokenizers.is_empty() {
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
        for (label, tokenizer, passes) in &tokenizers {
            for pass in 0..*passes {
                let ids = formatter.encode(tokenizer, &case["request"]).unwrap();
                let mut hash = Sha256::new();
                for id in &ids {
                    hash.update(id.to_le_bytes());
                }
                assert_eq!(
                    ids.len() as u64,
                    case["token_count"].as_u64().unwrap(),
                    "{} [{label} pass {pass}]",
                    case["name"]
                );
                assert_eq!(
                    format!("{:x}", hash.finalize()),
                    case["token_sha256"].as_str().unwrap(),
                    "{} [{label} pass {pass}]",
                    case["name"]
                );
            }
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

/// A conversation growing one turn at a time must encode identically through the L1 cache,
/// whose partial hits reuse earlier turns and extend the cache with the new suffix.
#[test]
fn v4_l1_cache_matches_uncached_across_turns() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/deepseek/v4.json")).unwrap();
    let Some(path) = pinned_tokenizer(&fixture) else {
        eprintln!(
            "{}: pinned tokenizer is not cached; skipping",
            fixture["model"]
        );
        return;
    };
    let path = path.to_str().unwrap();
    let plain = adapter::load(path).unwrap();
    let cached = [TokenizerBackend::Hf, TokenizerBackend::Fast].map(|backend| {
        adapter::load_with(
            path,
            TokenizerConfig {
                backend,
                l1_cache_mb: 64,
            },
        )
        .unwrap()
    });
    let formatter = ChatFormatter::deepseek_native(Some("deepseek_v4"), "alias").unwrap();
    let snippets = [
        "Summarize: fn main() { let x = vec![1, 2, 3]; println!(\"{x:?}\"); }",
        "中文测试：请解释一下缓存。数字 1234567 和 3.14159。",
        "  leading spaces, trailing tabs\t\t\nnew lines\n\n\nand emoji 😀🚀",
        "<｜end▁of▁sentence｜> looks special but arrives as user text",
        "Numbers 000111222333 and URLs https://example.com/a?b=c&d=e",
    ];
    let mut messages = vec![json!({"role": "system", "content": "You are terse."})];
    for turn in 0..12 {
        messages.push(json!({"role": "user", "content": format!("{} (turn {turn})", snippets[turn % snippets.len()])}));
        let request = json!({"model": "m", "messages": messages});
        let expected = formatter.encode(&plain, &request).unwrap();
        for (tokenizer, _) in &cached {
            assert_eq!(
                formatter.encode(tokenizer, &request).unwrap(),
                expected,
                "turn {turn}"
            );
        }
        messages.push(json!({"role": "assistant", "content": format!("Answer {turn}: {}", snippets[(turn + 2) % snippets.len()])}));
    }
    for (_, stats) in &cached {
        let (hits, misses) = stats.l1_lookups();
        assert!(
            hits >= 10,
            "later turns must reuse cached prefixes: {hits} hits, {misses} misses"
        );
    }
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

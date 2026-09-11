// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Compare Dynamo prompt IDs with SGLang-generated fixtures using cached tokenizers.
//! CI skips this matrix unless model snapshots are available.

use serde::Deserialize;
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::ModelId;
use sgl_router::policies::request_tokens_for;
use sgl_router::tokenizer::TokenizerRegistry;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Fixture {
    model_id: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    shape: String,
    request: serde_json::Value,
    expected_token_ids: Vec<u32>,
}

fn snapshot_tokenizer(model_id: &str) -> Option<PathBuf> {
    let hf_home = std::env::var("HF_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".cache/huggingface")))?;
    let snapshots = hf_home
        .join("hub")
        .join(format!("models--{}", model_id.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|e| e.path().join("tokenizer.json"))
        .find(|p| p.is_file())
}

fn registry(model_id: &str, tokenizer_path: PathBuf) -> TokenizerRegistry {
    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: model_id.into(),
            tokenizer_path: tokenizer_path.to_str().unwrap().into(),
            policy: PolicyKind::RoundRobin,
            decode_policy: Default::default(),
            bucket_config: None,
            circuit_breaker: None,
            cache_aware: None,
            affinity: None,
            sticky: None,
            fused: None,
            eligibility: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    };
    TokenizerRegistry::load_from_config(&cfg).unwrap()
}

#[test]
fn chat_render_parity_matrix() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chat_render_parity");
    let mut checked = 0;
    for entry in std::fs::read_dir(&root).unwrap().flatten() {
        let raw = std::fs::read_to_string(entry.path()).unwrap();
        let fixture: Fixture = serde_json::from_str(&raw).unwrap();
        let Some(tokenizer_path) = snapshot_tokenizer(&fixture.model_id) else {
            eprintln!("skip {}: snapshot not cached", fixture.model_id);
            continue;
        };
        let reg = registry(&fixture.model_id, tokenizer_path);
        assert!(
            reg.has_chat_encoder(&fixture.model_id),
            "{}: no chat encoder resolved",
            fixture.model_id
        );
        for case in &fixture.cases {
            let tokens =
                request_tokens_for(&reg, &ModelId(fixture.model_id.clone()), &case.request)
                    .unwrap_or_else(|| panic!("{}/{}: no tokens", fixture.model_id, case.shape));
            assert!(
                tokens.chat_rendered,
                "{}/{}: fell back to raw text",
                fixture.model_id, case.shape
            );
            assert_eq!(
                tokens.ids, case.expected_token_ids,
                "DRIFT on {}/{}",
                fixture.model_id, case.shape
            );
            checked += 1;
        }
    }
    assert!(
        checked > 0,
        "no cached model snapshots; parity was not checked"
    );
    eprintln!("chat render parity: {checked} cases checked");
}

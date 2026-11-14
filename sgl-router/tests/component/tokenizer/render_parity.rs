// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Compare Dynamo prompt IDs with SGLang-generated fixtures using cached tokenizers.
//! CI skips this matrix unless model snapshots are available.

use serde::Deserialize;
use sgl_router::config::{
    Config, DiscoveryBackend, InflightLoadConfig, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::ModelId;
use sgl_router::policies::request_tokens_for;
use sgl_router::tokenizer::chat_formatter::{ChatFormatter, ChatTemplateKwargs};
use sgl_router::tokenizer::{adapter, TokenizerRegistry};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
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

/// String-to-array conversion is a known parity gap, so these templates must
/// opt out of forwarding. This fixture needs no cached model files.
#[test]
fn array_only_template_content_parity() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/array_content_rendering.json")).unwrap();
    let formatter = ChatFormatter::from_tokenizer_config(
        serde_json::json!({"chat_template": fixture["chat_template"]}),
        None,
    )
    .unwrap()
    .unwrap();
    let tokenizer = adapter::load("tests/fixtures/tiny_tokenizer.json").unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let request =
            serde_json::json!({"messages": [{"role": "user", "content": case["content"]}]});
        let ids = formatter.encode(&tokenizer, &request).unwrap();
        assert_eq!(
            serde_json::json!(ids) == case["engine_token_ids"],
            case["content"].is_array(),
            "{case}"
        );
    }
}

/// Replace every `YYYY-MM-DD` with a placeholder.
///
/// Templates that call `strftime_now` render the day the prompt is built, so a
/// fixture captured earlier differs from today's render in the date alone. That
/// is not drift: the engine consumes forwarded IDs verbatim and never re-renders.
/// Masking keeps the rest of the prompt under exact comparison, and dates that
/// come from the request render the same on both sides, so masking them is a
/// no-op.
fn mask_dates(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if starts_with_iso_date(rest) {
            out.push_str("<DATE>");
            rest = &rest[10..];
        } else {
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    out
}

fn starts_with_iso_date(text: &str) -> bool {
    let b = text.as_bytes();
    b.len() >= 10
        && b[..4].iter().all(u8::is_ascii_digit)
        && b[4] == b'-'
        && b[5..7].iter().all(u8::is_ascii_digit)
        && b[7] == b'-'
        && b[8..10].iter().all(u8::is_ascii_digit)
}

/// The HF hub cache: `HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`.
fn hub_cache() -> Option<PathBuf> {
    std::env::var("HF_HUB_CACHE")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HF_HOME")
                .ok()
                .map(PathBuf::from)
                .or_else(|| dirs::home_dir().map(|h| h.join(".cache/huggingface")))
                .map(|home| home.join("hub"))
        })
}

fn snapshot_tokenizer(model_id: &str) -> Option<PathBuf> {
    let snapshots = hub_cache()?
        .join(format!("models--{}", model_id.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|e| e.path().join("tokenizer.json"))
        .find(|p| p.is_file())
}

fn registry(model_id: &str, tokenizer_path: PathBuf) -> TokenizerRegistry {
    registry_with_defaults(model_id, tokenizer_path, Default::default())
}

fn registry_with_defaults(
    model_id: &str,
    tokenizer_path: PathBuf,
    default_chat_template_kwargs: ChatTemplateKwargs,
) -> TokenizerRegistry {
    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            aliases: Vec::new(),
            id: model_id.into(),
            tokenizer_path: Some(tokenizer_path.to_str().unwrap().into()),
            disable_input_ids_forwarding: false,
            tokenizer: Default::default(),
            policy: PolicyKind::RoundRobin,
            decode_policy: Default::default(),
            dp_aware: false,
            dp_rank_policy: Default::default(),
            bucket_config: None,
            reorg_buckets: None,
            reorg_admission: Default::default(),
            circuit_breaker: None,
            cache_aware: None,
            affinity: None,
            sticky: None,
            fused: None,
            eligibility: None,
            sampling_overrides: Default::default(),
            default_chat_template_kwargs,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        router_inflight_load: InflightLoadConfig::default(),
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
            reg.has_chat_formatter(&fixture.model_id),
            "{}: no chat formatter resolved",
            fixture.model_id
        );
        for case in &fixture.cases {
            let tokens =
                request_tokens_for(&reg, &ModelId(fixture.model_id.clone()), &case.request)
                    .unwrap_or_else(|| panic!("{}/{}: no tokens", fixture.model_id, case.shape));
            assert!(
                tokens.rendered_from_chat,
                "{}/{}: fell back to raw text",
                fixture.model_id, case.shape
            );
            if tokens.ids != case.expected_token_ids {
                // Decode with special tokens kept, so a difference in them still fails.
                let tokenizer = reg.get(&fixture.model_id).unwrap();
                let rendered = adapter::decode_complete(&tokenizer, &tokens.ids, false).unwrap();
                let expected =
                    adapter::decode_complete(&tokenizer, &case.expected_token_ids, false).unwrap();
                assert_eq!(
                    mask_dates(&rendered),
                    mask_dates(&expected),
                    "DRIFT on {}/{}",
                    fixture.model_id,
                    case.shape
                );
                eprintln!(
                    "{}/{}: date drift only; fixture captured on another day",
                    fixture.model_id, case.shape
                );
            }
            checked += 1;
        }
    }
    assert!(
        checked > 0,
        "no cached model snapshots; parity was not checked"
    );
    eprintln!("chat render parity: {checked} cases checked");
}

/// The router's own render and tokenization (`encode_chat`, continuation prefixes
/// included) reproduce `rust/sglang-processor`'s parity fixtures, which SGLang's
/// Python serving path wrote: the ids of every case it renders, and no ids where it
/// fails. Checks only the fixtures whose pinned snapshot is cached.
#[test]
fn processor_fixture_parity() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../rust/sglang-processor/tests/fixtures/parity");
    for entry in std::fs::read_dir(&root).unwrap().flatten() {
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(entry.path()).unwrap()).unwrap();
        let model = fixture["model"].as_str().unwrap();
        let snapshot = hub_cache().map(|hub| {
            hub.join(format!("models--{}", model.replace('/', "--")))
                .join("snapshots")
                .join(fixture["revision"].as_str().unwrap())
        });
        let Some(tokenizer_path) = ["tokenizer.json", "tiktoken.model"]
            .iter()
            .filter_map(|file| Some(snapshot.as_ref()?.join(file)))
            .find(|path| path.is_file())
        else {
            eprintln!("skip {model}: pinned snapshot not cached");
            continue;
        };
        // One registry per server `--default-chat-template-kwargs`.
        let mut registries: HashMap<String, TokenizerRegistry> = HashMap::new();
        for case in fixture["cases"].as_array().unwrap() {
            // A server-side template override is the processor's to check.
            if case.get("chat_template").is_some() {
                continue;
            }
            let defaults = case["default_chat_template_kwargs"].clone();
            let reg = registries.entry(defaults.to_string()).or_insert_with(|| {
                let defaults = defaults
                    .as_object()
                    .map(|d| d.clone().into_iter().collect())
                    .unwrap_or_default();
                registry_with_defaults(model, tokenizer_path.clone(), defaults)
            });
            let name = format!("{model} {}", case["name"]);
            let ids = reg.encode_chat(model, &case["request"]);
            if !case["error"].is_null() {
                assert!(ids.is_none(), "{name}: SGLang rejects this request");
                continue;
            }
            if case.get("known_gap").is_some() {
                continue;
            }
            let ids = ids.unwrap_or_else(|| panic!("{name}: not rendered"));
            let digest = ids.iter().fold(Sha256::new(), |hash, id| {
                hash.chain_update(id.to_le_bytes())
            });
            assert_eq!(ids.len() as u64, case["token_count"], "{name}");
            assert_eq!(
                format!("{:x}", digest.finalize()),
                case["token_sha256"],
                "{name}"
            );
        }
    }
}

//! Every `tests/fixtures/parity/*.json` renders and tokenizes as SGLang's serving
//! path does. Fixtures come from `tests/scripts/generate_parity.py`; the
//! `processor-model-parity` skill covers adding a model.

#![cfg(all(feature = "render", feature = "tokenizer"))]

use std::path::Path;

use serde_json::Value;
use sglang_processor::dynamo_tokenizers::Tokenizer;
use sglang_processor::{
    ChatFormatter, ChatFormatterOptions, load_tokenizer, resolve_model_file, select_chat_formatter,
};
use sha2::{Digest, Sha256};

#[test]
fn fixtures_match_sglang() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parity");
    for entry in std::fs::read_dir(dir).unwrap() {
        let fixture: Value =
            serde_json::from_str(&std::fs::read_to_string(entry.unwrap().path()).unwrap()).unwrap();
        check(&fixture);
    }
}

fn check(fixture: &Value) {
    let model = fixture["model"].as_str().unwrap();
    let formatter = formatter(model, &fixture["config"]);
    let tokenizer = cached_tokenizer(model, fixture["revision"].as_str().unwrap());
    for case in fixture["cases"].as_array().unwrap() {
        let name = format!("{model} {}", case["name"]);
        let rendered = formatter.render_request(&case["request"]);
        if !case["error"].is_null() {
            assert!(rendered.is_err(), "{name}: SGLang rejects this request");
            continue;
        }
        let (prompt, prefix) = rendered.unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(prompt.clone() + &prefix, case["prompt"], "{name}");
        let Some(tokenizer) = &tokenizer else {
            continue;
        };
        let ids = encode(tokenizer, &prompt, &prefix, &fixture["bos_token_id"]);
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

/// What `select_chat_formatter` picks for the fixture's `config.json`.
fn formatter(model: &str, config: &Value) -> ChatFormatter {
    let dir = std::env::temp_dir().join(format!(
        "sglang-processor-parity-{}-{}",
        std::process::id(),
        model.replace('/', "--")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let (formatter, error) = select_chat_formatter(&ChatFormatterOptions {
        tokenizer_path: dir.to_string_lossy().into_owned(),
        ..Default::default()
    });
    std::fs::remove_dir_all(&dir).unwrap();
    formatter.unwrap_or_else(|| panic!("{model}: {error:?}"))
}

/// The pinned tokenizer when it is in the HF cache. hf-hub resolves refs, not
/// commit hashes, so resolve `main` and require the pinned snapshot.
fn cached_tokenizer(model: &str, revision: &str) -> Option<Tokenizer> {
    let file =
        resolve_model_file(model, None, "tokenizer.json").filter(|file| file.contains(revision))?;
    Some(load_tokenizer(Some(&file), None, true).unwrap())
}

/// `serving_chat._append_assistant_prefix_to_prompt_ids`: the continuation is
/// encoded on its own, minus a leading BOS.
fn encode(tokenizer: &Tokenizer, prompt: &str, prefix: &str, bos: &Value) -> Vec<u32> {
    let ids = |text: &str| tokenizer.encode(text).unwrap().token_ids().to_vec();
    let mut out = ids(prompt);
    let mut suffix = ids(prefix);
    if suffix.first().is_some_and(|&id| *bos == id) {
        suffix.remove(0);
    }
    out.extend(suffix);
    out
}

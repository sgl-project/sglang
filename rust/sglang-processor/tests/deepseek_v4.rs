//! DeepSeek-V4-Flash-0731 prompts match SGLang's serving path
//! (fixture from `tests/scripts/generate_deepseek_v4_parity.py`).

#![cfg(all(feature = "render", feature = "tokenizer"))]

use serde_json::{Value, json};
use sglang_processor::dynamo_tokenizers::Tokenizer;
use sglang_processor::{
    ChatFormatterOptions, load_tokenizer, resolve_model_file, select_chat_formatter,
};
use sha2::{Digest, Sha256};

fn encode(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(text).unwrap().token_ids().to_vec()
}

#[test]
fn deepseek_v4_0731_matches_sglang() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/deepseek_v4_0731.json")).unwrap();
    let dir = std::env::temp_dir().join(format!("sglang-processor-dsv4-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Offline stand-in for the checkpoint's encoder, which detects as "official".
    let config = json!({"model_type": "deepseek_v4", "dsv4_reasoning_effort_profile": "official"});
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let (formatter, error) = select_chat_formatter(&ChatFormatterOptions {
        tokenizer_path: dir.to_string_lossy().into_owned(),
        ..Default::default()
    });
    let formatter = formatter.unwrap_or_else(|| panic!("{error:?}"));
    std::fs::remove_dir_all(&dir).unwrap();

    // Token ids are checked only when the pinned tokenizer is in the HF cache;
    // hf-hub resolves refs, not commit hashes, so match the snapshot path.
    let revision = fixture["revision"].as_str().unwrap();
    let tokenizer = resolve_model_file(fixture["model"].as_str().unwrap(), None, "tokenizer.json")
        .filter(|file| file.contains(revision))
        .map(|file| load_tokenizer(Some(&file), None, false).unwrap());
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (prompt, prefix) = formatter
            .render_request(&case["request"])
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(
            prompt.clone() + &prefix,
            case["prompt"].as_str().unwrap(),
            "{name}"
        );
        let Some(tokenizer) = &tokenizer else {
            continue;
        };
        // SGLang encodes the continuation prefix separately, without its BOS.
        let mut ids = encode(tokenizer, &prompt);
        let mut suffix = encode(tokenizer, &prefix);
        if suffix.first() == encode(tokenizer, "<｜begin▁of▁sentence｜>").first() {
            suffix.remove(0);
        }
        ids.extend(suffix);
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

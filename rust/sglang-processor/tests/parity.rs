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
        let outcome = check_case(
            &formatter,
            tokenizer.as_ref(),
            &fixture["bos_token_id"],
            case,
        );
        // `known_gap` marks a divergence the processor cannot fix itself, such as one
        // inside Dynamo; the test fails once it matches so the marker gets dropped.
        match (outcome, case["known_gap"].as_str()) {
            (Ok(()), None) => {}
            (Err(error), None) => panic!("{name}: {error}"),
            (Err(_), Some(gap)) => eprintln!("{name}: known gap: {gap}"),
            (Ok(()), Some(_)) => panic!("{name} now matches SGLang; drop its known_gap"),
        }
    }
}

fn check_case(
    formatter: &ChatFormatter,
    tokenizer: Option<&Tokenizer>,
    bos: &Value,
    case: &Value,
) -> Result<(), String> {
    let rendered = formatter.render_request(&case["request"]);
    if !case["error"].is_null() {
        return match rendered {
            Ok(_) => Err("SGLang rejects this request".into()),
            Err(_) => Ok(()),
        };
    }
    let (prompt, prefix) = rendered.map_err(|error| error.to_string())?;
    if prompt.clone() + &prefix != case["prompt"] {
        return Err(format!("prompt differs: {prompt}{prefix}"));
    }
    let Some(tokenizer) = tokenizer else {
        return Ok(());
    };
    let ids = encode(tokenizer, &prompt, &prefix, bos);
    let digest = ids.iter().fold(Sha256::new(), |hash, id| {
        hash.chain_update(id.to_le_bytes())
    });
    if ids.len() as u64 != case["token_count"]
        || format!("{:x}", digest.finalize()) != case["token_sha256"]
    {
        return Err(format!("token ids differ ({} tokens)", ids.len()));
    }
    Ok(())
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

/// The pinned tokenizer when it is in the HF cache; otherwise say token ids are skipped.
fn cached_tokenizer(model: &str, revision: &str) -> Option<Tokenizer> {
    let Some(file) = resolve_model_file(model, Some(revision), "tokenizer.json") else {
        eprintln!("{model}@{revision} is not in the HF cache: token ids not checked");
        return None;
    };
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

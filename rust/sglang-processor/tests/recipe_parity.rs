//! Every `tests/fixtures/recipe_parity/*.json` renders as the Python bridge
//! (`deepseek_recipe_bridge.py`) does: same prompt, same special-token ranges,
//! same thinking mode and output constraint, and, with the pinned tokenizer in
//! the HF cache, the same token ids. Fixtures come from
//! `tests/scripts/generate_recipe_parity.py`.
//!
//! Unlike `tests/parity.rs`, whose fixtures record SGLang's builtin serving path,
//! these record the Python recipe bridge: both hosts hand the same request body to
//! the same recipe library, so what needs pinning is the shared normalization
//! (`pre_normalize`) and the tokenization, not the host's request validation.

#![cfg(feature = "recipe")]

use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;

use serde_json::Value;
use sglang_processor::{ChatFormatter, RecipeFormatter, RecipeModel, resolve_model_file};
use sha2::{Digest, Sha256};

#[test]
fn fixtures_match_the_python_bridge() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recipe_parity");
    let mut seen = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|ext| ext != "json") {
            continue;
        }
        let fixture: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        check(&fixture);
        seen += 1;
    }
    assert!(seen > 0, "no recipe parity fixtures found");
}

fn check(fixture: &Value) {
    let model = fixture["model"].as_str().unwrap();
    let revision = fixture["revision"].as_str().unwrap();
    let recipe_model = match fixture["model_version"].as_str().unwrap() {
        "v4" => RecipeModel::V4,
        "v41" => RecipeModel::V41,
        other => panic!("{model}: unknown model_version {other}"),
    };
    // The pinned tokenizer when it is in the HF cache; otherwise token ids are skipped.
    let formatter = match resolve_model_file(model, Some(revision), "tokenizer.json") {
        Some(file) => RecipeFormatter::from_tokenizer_file(recipe_model, &file).unwrap(),
        None => {
            eprintln!("{model}@{revision} is not in the HF cache: token ids not checked");
            RecipeFormatter::new(recipe_model)
        }
    };
    let formatter = ChatFormatter::DeepSeekRecipe(formatter);
    let defaults: HashMap<String, Value> = fixture["default_kwargs"]
        .as_object()
        .map(|kwargs| kwargs.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();
    for case in fixture["cases"].as_array().unwrap() {
        let name = format!("{model} {}", case["name"]);
        if let Err(error) = check_case(&formatter, &defaults, case) {
            panic!("{name}: {error}");
        }
    }
}

fn check_case(
    formatter: &ChatFormatter,
    defaults: &HashMap<String, Value>,
    case: &Value,
) -> Result<(), String> {
    let rendered = formatter.render_recipe(case["request"].clone(), defaults);
    if let Some(expected) = case["error"].as_str() {
        return match rendered {
            Ok(render) => Err(format!(
                "the bridge rejects this request ({expected}) but the processor rendered {:?}",
                render.prompt
            )),
            Err(_) => Ok(()),
        };
    }
    let render = rendered.map_err(|error| error.to_string())?;
    let prompt = case["prompt"]
        .as_str()
        .ok_or("fixture case has no prompt; run tests/scripts/generate_recipe_parity.py")?;
    if render.prompt != prompt {
        return Err(format!("prompt differs: {}", render.prompt));
    }
    let ranges: Vec<(usize, usize)> = serde_json::from_value(case["special_ranges"].clone())
        .map_err(|error| format!("bad special_ranges in fixture: {error}"))?;
    let expected_ranges = char_ranges_to_bytes(prompt, &ranges);
    if render.special_ranges != expected_ranges {
        return Err(format!(
            "special ranges differ: {:?} vs fixture {:?}",
            render.special_ranges, expected_ranges
        ));
    }
    if render.thinking_mode
        != case["thinking_mode"]
            .as_bool()
            .unwrap_or(render.thinking_mode)
    {
        return Err("thinking mode differs".into());
    }
    if let Some(expected) = case.get("output_constraint") {
        let got = render
            .output_constraint
            .as_ref()
            .map(|(kind, _)| kind.as_str());
        if got != expected.as_str() {
            return Err(format!("output constraint differs: {got:?} vs {expected}"));
        }
    }
    let Some(ids) = render.token_ids else {
        return Ok(());
    };
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

/// The fixture records code-point offsets (the Python binding's); the processor
/// reports byte ranges.
fn char_ranges_to_bytes(prompt: &str, ranges: &[(usize, usize)]) -> Vec<Range<usize>> {
    let offsets: Vec<usize> = prompt
        .char_indices()
        .map(|(offset, _)| offset)
        .chain(std::iter::once(prompt.len()))
        .collect();
    ranges
        .iter()
        .map(|&(start, end)| offsets[start]..offsets[end])
        .collect()
}

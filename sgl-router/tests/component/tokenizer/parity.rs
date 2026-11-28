// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Bit-parity check: dynamo-tokenizers must produce the same token_ids as
//! SGLang's reference (transformers.AutoTokenizer) for every (model, shape)
//! fixture. Any drift is a regression.
//!
//! ## Running
//!
//! `cargo test --release --test component tokenizer::parity` runs the test.
//!
//! Each fixture cell needs the model's `tokenizer.json` on disk; the test
//! looks in the local HuggingFace cache (`HF_HOME` or `~/.cache/huggingface`).
//! Cells whose snapshot isn't cached are skipped (with a warning); cells
//! whose snapshot IS cached are asserted bit-identical.
//!
//! Locally, when no fixtures can be checked (fresh cache) the test emits a
//! warning and passes — useful for contributors without the model snapshots.
//! In CI (`SGLANG_IS_IN_CI=true`) the same condition is a hard failure: a
//! parity matrix that validates nothing is worse than no test at all, since
//! it gives a false sense of coverage. The e2e HTTP tokenize test remains
//! the authoritative live-model parity gate, but this matrix must actually
//! run against cached snapshots when present in CI.
//!
//! ## Regenerating fixtures
//!
//! Run `tests/scripts/generate_parity_fixtures.py` after changing a prompt
//! shape or adding a model, then commit the new JSON.

use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Fixture {
    model_id: String,
    shape: String,
    prompt_text: String,
    expected_token_ids: Vec<u32>,
    #[allow(dead_code)]
    skip_special_tokens: bool,
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tokenizer_parity")
}

/// Resolve a model's tokenizer.json file from the local HF cache.
///
/// Strategy:
/// 1. Check HF_HOME env var, or default to ~/.cache/huggingface
/// 2. Look for models--<safe-name>/snapshots/<hash>/tokenizer.json
/// 3. Return None if not found — the test cell is skipped.
fn resolve_tokenizer_path(model_id: &str) -> Option<PathBuf> {
    let hf_home = std::env::var("HF_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".cache/huggingface")))?;
    let safe = model_id.replace('/', "--");
    let candidate = hf_home.join("hub").join(format!("models--{safe}"));
    if !candidate.exists() {
        return None;
    }
    let snapshots = candidate.join("snapshots");
    let snap = std::fs::read_dir(&snapshots).ok()?.next()?.ok()?.path();
    let tj = snap.join("tokenizer.json");
    tj.exists().then_some(tj)
}

/// Parity matrix: dynamo-tokenizers vs. transformers.AutoTokenizer.
///
/// Skips cells whose tokenizer.json isn't in the local HF cache. See
/// module-level docs.
#[test]
fn parity_matrix() {
    let mut checked = 0;
    let mut skipped = vec![];
    for model_dir in std::fs::read_dir(fixture_root()).unwrap() {
        let model_dir = model_dir.unwrap().path();
        if !model_dir.is_dir() {
            continue;
        }
        for shape_file in std::fs::read_dir(&model_dir).unwrap() {
            let p = shape_file.unwrap().path();
            if p.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let raw = std::fs::read_to_string(&p).unwrap();
            let f: Fixture =
                serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()));
            let Some(tp) = resolve_tokenizer_path(&f.model_id) else {
                skipped.push((f.model_id.clone(), f.shape.clone()));
                continue;
            };
            let tok = sgl_router::tokenizer::adapter::load(tp.to_str().unwrap()).unwrap();
            let ids = sgl_router::tokenizer::adapter::encode(&tok, &f.prompt_text).unwrap();
            assert_eq!(
                ids, f.expected_token_ids,
                "DRIFT on {}/{}",
                f.model_id, f.shape
            );
            checked += 1;
        }
    }
    let expected = std::fs::read_dir(fixture_root())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| {
            std::fs::read_dir(e.path())
                .unwrap()
                .filter_map(|f| f.ok())
                .filter(|f| f.path().extension().and_then(|s| s.to_str()) == Some("json"))
                .count()
        })
        .sum::<usize>();
    assert_eq!(
        checked + skipped.len(),
        expected,
        "expected {expected} fixtures, found {}",
        checked + skipped.len()
    );
    if checked == 0 {
        let families: Vec<String> = skipped
            .iter()
            .map(|(m, _)| m.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let msg = format!(
            "parity_matrix: no fixtures could be checked — HF cache empty? skipped {} cells \
             across model families: [{}]. The e2e HTTP tokenize test remains the \
             authoritative live-model parity gate.",
            skipped.len(),
            families.join(", "),
        );
        if std::env::var("SGLANG_IS_IN_CI").as_deref() == Ok("true") {
            panic!(
                "{msg}\n\nThis is a hard failure in CI: a parity test that validates zero \
                 cells provides no coverage. Either pre-populate the HF cache for these \
                 model families on the runner, or remove the parity test."
            );
        }
        eprintln!("{msg}");
    } else {
        eprintln!(
            "parity: {checked} cells passed, {} skipped (no HF snapshot)",
            skipped.len()
        );
    }
}

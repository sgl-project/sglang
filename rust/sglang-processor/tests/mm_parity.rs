//! Every `tests/fixtures/mm_parity/*.json` preprocesses as SGLang's Python CPU
//! image path does, bit for bit. Fixtures come from
//! `tests/scripts/generate_mm_parity.py`.

#![cfg(feature = "mm")]

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use sglang_processor::dynamo_multimodal::processor::{PositionOutput, TensorData};
use sglang_processor::dynamo_multimodal::registry::ProcessorSpec;
use sglang_processor::{MmOutput, MmProcessor};
use sha2::{Digest, Sha256};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mm_parity")
}

#[test]
fn fixtures_match_sglang() {
    let mut failures = Vec::new();
    for entry in std::fs::read_dir(fixtures()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|ext| ext == "json") {
            let fixture = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            failures.extend(check(&fixture));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

fn check(fixture: &Value) -> Vec<String> {
    let model = fixture["model"].as_str().unwrap();
    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let name = format!("{model} {}", case["name"]);
        // Same contract as tests/parity.rs: a known gap must keep failing.
        match (check_case(case), case["known_gap"].as_str()) {
            (Ok(()), None) => {}
            (Err(error), None) => failures.push(format!("{name}: {error}")),
            (Err(_), Some(gap)) => eprintln!("{name}: known gap: {gap}"),
            (Ok(()), Some(_)) => {
                failures.push(format!("{name} now matches SGLang; drop its known_gap"))
            }
        }
    }
    failures
}

fn check_case(case: &Value) -> Result<(), String> {
    let spec: ProcessorSpec = serde_json::from_value(case["spec"].clone()).unwrap();
    let images: Vec<Vec<u8>> = (case["images"].as_array().unwrap().iter())
        .map(|name| std::fs::read(fixtures().join("images").join(name.as_str().unwrap())).unwrap())
        .collect();
    let images: Vec<&[u8]> = images.iter().map(Vec::as_slice).collect();
    let input_ids: Vec<i32> = (case["input_ids"].as_array().unwrap().iter())
        .map(|id| id.as_i64().unwrap() as i32)
        .collect();

    let output =
        MmProcessor::from_spec(spec).and_then(|processor| processor.process(&input_ids, &images));
    match (output, case.get("error")) {
        (Ok(output), None) => compare(&summary(&output), &case["expected"]),
        (Err(_), Some(_)) => Ok(()),
        (Ok(_), Some(error)) => Err(format!("SGLang rejects this case: {error}")),
        (Err(error), None) => Err(error.to_string()),
    }
}

/// The fields `generate_mm_parity.py` records, computed from the Rust output.
fn summary(output: &MmOutput) -> Value {
    let (mrope, delta) = match &output.positions {
        PositionOutput::MRope { positions, delta } => (positions.clone(), *delta),
        _ => panic!("Qwen-VL uses M-RoPE"),
    };
    let grids: Vec<Value> = (output.items.iter())
        .map(|item| {
            let (_, grid) = (item.aux.iter())
                .find(|(name, _)| name == "image_grid_thw")
                .expect("missing image_grid_thw");
            let TensorData::I64(grid) = &grid.data else {
                panic!("image_grid_thw is i64");
            };
            json!(grid)
        })
        .collect();
    let features = output.items.iter().map(|item| match &item.feature.data {
        TensorData::F32(values) => values,
        _ => panic!("Qwen-VL features are f32"),
    });
    json!({
        "input_ids_sha256": sha256(output.input_ids.iter().flat_map(|id| id.to_le_bytes())),
        "num_input_ids": output.input_ids.len(),
        "offsets": output.offsets,
        "image_grid_thw": grids,
        "feature_shapes": output.items.iter().map(|item| &item.feature.shape).collect::<Vec<_>>(),
        "feature_sha256": features
            .map(|values| sha256(values.iter().flat_map(|v| v.to_le_bytes())))
            .collect::<Vec<_>>(),
        "mrope_sha256": sha256(mrope.iter().flat_map(|p| p.to_le_bytes())),
        "mrope_delta": delta,
    })
}

fn compare(actual: &Value, expected: &Value) -> Result<(), String> {
    let mismatches: Vec<String> = (expected.as_object().unwrap().iter())
        .filter(|(field, value)| actual[field.as_str()] != **value)
        .map(|(field, value)| format!("{field}: rust {} != python {value}", actual[field.as_str()]))
        .collect();
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(mismatches.join("; "))
    }
}

fn sha256(bytes: impl Iterator<Item = u8>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes.collect::<Vec<u8>>());
    format!("{:x}", hasher.finalize())
}

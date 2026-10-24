//! Every `tests/fixtures/mm_parity/*.json` preprocesses as SGLang's Python CPU
//! image path does, bit for bit. Fixtures come from
//! `tests/scripts/generate_mm_parity.py`.

#![cfg(feature = "mm")]

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use sglang_processor::dynamo_multimodal::processor::{PositionOutput, Tensor, TensorData};
use sglang_processor::{MmOutput, MmProcessorOptions, select_mm_processor};
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
    let model_dir = write_model_files(fixture);
    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let name = format!("{model} {}", case["name"]);
        // Same contract as tests/parity.rs: a known gap must keep failing.
        match (check_case(&model_dir, case), case["known_gap"].as_str()) {
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

/// The checkpoint's config files, as the fixture recorded them.
fn write_model_files(fixture: &Value) -> PathBuf {
    let name = fixture["model"].as_str().unwrap().replace('/', "--");
    let dir = std::env::temp_dir().join(format!("sglang-mm-parity-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), fixture["config"].as_str().unwrap()).unwrap();
    let processor_file = fixture["processor_config_file"].as_str().unwrap();
    std::fs::write(
        dir.join(processor_file),
        fixture["processor_config"].as_str().unwrap(),
    )
    .unwrap();
    dir
}

fn check_case(model_dir: &Path, case: &Value) -> Result<(), String> {
    let options = MmProcessorOptions {
        tokenizer_path: model_dir.to_string_lossy().into_owned(),
        image_processor_backend: case["image_processor_backend"].as_str().map(str::to_owned),
        image_overrides: Some(case["image_overrides"].clone()).filter(|v| !v.is_null()),
        ..MmProcessorOptions::default()
    };
    let images: Vec<Vec<u8>> = (case["images"].as_array().unwrap().iter())
        .map(|name| std::fs::read(fixtures().join("images").join(name.as_str().unwrap())).unwrap())
        .collect();
    let images: Vec<&[u8]> = images.iter().map(Vec::as_slice).collect();
    let input_ids: Vec<i32> = (case["input_ids"].as_array().unwrap().iter())
        .map(|id| id.as_i64().unwrap() as i32)
        .collect();

    let output =
        select_mm_processor(&options).and_then(|processor| processor.process(&input_ids, &images));
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
        .map(
            |item| match &item.aux.iter().find(|(name, _)| name == "image_grid_thw") {
                Some((
                    _,
                    Tensor {
                        data: TensorData::I64(grid),
                        ..
                    },
                )) => json!(grid),
                _ => panic!("missing image_grid_thw"),
            },
        )
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

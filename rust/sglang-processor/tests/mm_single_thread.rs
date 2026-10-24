//! `MmProcessor::process` runs inline on the caller's thread. Hosts such as
//! `sglang-server` supply concurrency across requests and pin their own cores,
//! so a pool spawned behind their back would fight them. Checked from the
//! outside (the process thread count), so it holds however Dynamo fans out.

#![cfg(feature = "mm")]

use std::path::Path;

use serde_json::Value;
use sglang_processor::{MmProcessorOptions, select_mm_processor};

fn thread_count() -> usize {
    std::fs::read_dir("/proc/self/task")
        .expect("procfs")
        .count()
}

#[test]
fn processing_a_request_spawns_no_threads() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mm_parity");
    let fixture: Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures.join("qwen2.5-vl-3b-instruct.json")).unwrap(),
    )
    .unwrap();
    let model_dir = std::env::temp_dir().join(format!("sglang-mm-threads-{}", std::process::id()));
    std::fs::create_dir_all(&model_dir).unwrap();
    for (file, key) in [
        ("config.json", "config"),
        ("preprocessor_config.json", "processor_config"),
    ] {
        std::fs::write(model_dir.join(file), fixture[key].as_str().unwrap()).unwrap();
    }
    let images: Vec<Vec<u8>> = ["rgb_96x80.png", "rgb_333x251.jpg"]
        .map(|name| std::fs::read(fixtures.join("images").join(name)).unwrap())
        .into();
    let image_token_id = serde_json::from_str::<Value>(fixture["config"].as_str().unwrap()).unwrap()
        ["image_token_id"]
        .as_i64()
        .unwrap() as i32;

    let before = thread_count();
    let processor = select_mm_processor(&MmProcessorOptions {
        tokenizer_path: model_dir.to_string_lossy().into_owned(),
        ..MmProcessorOptions::default()
    })
    .unwrap();
    // Two images, so any per-item fan-out is exercised.
    let output = processor
        .process(
            &[7, image_token_id, 8, image_token_id, 9],
            &[&images[0], &images[1]],
        )
        .unwrap();
    assert_eq!(output.items.len(), 2);
    assert_eq!(thread_count(), before, "MmProcessor spawned threads");
}

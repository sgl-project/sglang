#![cfg(feature = "python-mm")]

use std::{
    env,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use base64::{engine::general_purpose::STANDARD as BASE64_ENGINE, Engine};
use image::{ImageBuffer, Rgb};
use serde_json::json;
use sglang_router_rs::multimodal::{
    EncodedImage, MmProcessRequest, MultiModalInputs, PythonMmBridge,
};

fn ensure_hf_cache(model: &str) -> bool {
    if let Ok(home) = env::var("HOME") {
        let cache_dir: PathBuf = [
            home.as_str(),
            ".cache",
            "huggingface",
            "hub",
            &format!("models--{}", model.replace('/', "--")),
        ]
        .iter()
        .collect();
        return cache_dir.exists();
    }
    false
}

#[test]
fn python_bridge_llava_smoke() {
    let model_id = "llava-hf/llava-1.5-7b-hf";
    if !ensure_hf_cache(model_id) {
        eprintln!(
            "Skipping python_bridge_llava_smoke: HF cache for {model_id} not found in ~/.cache"
        );
        return;
    }

    env::set_var("HF_HUB_OFFLINE", "1");

    let mut request = MmProcessRequest {
        model_id: model_id.to_string(),
        prompt: "<image>\nDescribe the picture.".into(),
        ..Default::default()
    };

    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(8, 8, Rgb([255, 0, 0]));
    request.images.push(EncodedImage::from_rgb_bytes(
        img.width(),
        img.height(),
        img.into_raw(),
    ));

    let bridge = PythonMmBridge::new().expect("python bridge init");
    let inputs = bridge.process(&request).expect("bridge process");

    assert!(!inputs.prompt_token_ids.is_empty());
    assert!(inputs.mm_kwargs.contains_key("pixel_values"));

    run_vllm_parity_test(&request, &inputs);
}

fn run_vllm_parity_test(request: &MmProcessRequest, ours: &MultiModalInputs) {
    let script_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/scripts/vllm_parity.py");
    if !script_path.exists() {
        eprintln!(
            "Skipping vLLM parity: script missing at {}",
            script_path.display()
        );
        return;
    }

    let Some(image) = request.images.first() else {
        eprintln!("Skipping vLLM parity: request has no image payload");
        return;
    };

    let payload = json!({
        "model_id": request.model_id,
        "prompt": request.prompt,
        "image": {
            "width": image.width,
            "height": image.height,
            "channels": image.channels,
            "data": BASE64_ENGINE.encode(&image.bytes),
        }
    });

    let mut child = match Command::new("python3")
        .arg(script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            eprintln!("Skipping vLLM parity: failed to spawn python3 ({err})");
            return;
        }
    };

    if let Some(stdin) = child.stdin.as_mut() {
        if serde_json::to_writer(stdin, &payload).is_err() {
            eprintln!("Skipping vLLM parity: failed to send payload to python");
            return;
        }
    }

    let output = match child.wait_with_output() {
        Ok(out) => out,
        Err(err) => {
            eprintln!("Skipping vLLM parity: failed to read python output ({err})");
            return;
        }
    };

    if !output.status.success() {
        if output.status.code() == Some(2) {
            eprintln!(
                "Skipping vLLM parity: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        panic!(
            "vLLM parity script failed: {}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let theirs: MultiModalInputs =
        serde_json::from_slice(&output.stdout).expect("deserialize vllm output");

    assert_eq!(ours.prompt_token_ids, theirs.prompt_token_ids);

    let ours_json = serde_json::to_value(ours).expect("serialize ours");
    let theirs_json = serde_json::to_value(theirs).expect("serialize theirs");
    assert_eq!(ours_json["mm_kwargs"], theirs_json["mm_kwargs"]);
}

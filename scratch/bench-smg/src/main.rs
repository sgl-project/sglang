//! Bench smg's Qwen2-VL preprocessor on real image files (JPEG / PNG bytes).
//!
//! Reports decode_us (bytes -> DynamicImage) and process_us (smg preprocess()) per image.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use image::DynamicImage;
use llm_multimodal::vision::{
    image_processor::ImagePreProcessor,
    preprocessor_config::PreProcessorConfig,
    processors::Qwen2VLProcessor,
};

const ITERS: u32 = 20;

fn qwen2_vl_config() -> PreProcessorConfig {
    PreProcessorConfig::from_json(
        r#"{
            "do_resize": true,
            "do_rescale": true,
            "do_normalize": true,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "min_pixels": 12544,
            "max_pixels": 1003520,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std":  [0.26862954, 0.26130258, 0.27577711],
            "size": {"shortest_edge": 12544, "longest_edge": 1003520}
        }"#,
    )
    .expect("config")
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let fixtures = PathBuf::from(args.get(1).cloned().unwrap_or_else(|| "fixtures".into()));
    eprintln!("# smg image preprocessor bench (Qwen2-VL)");
    eprintln!("# fixtures dir: {:?}", fixtures);

    let mut files: Vec<PathBuf> = fs::read_dir(&fixtures)
        .expect("read fixtures")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
            matches!(ext, "jpg" | "jpeg" | "png")
        })
        .collect();
    files.sort();

    let cfg = qwen2_vl_config();
    let proc = Qwen2VLProcessor::new();

    // Warmup
    if let Some(f) = files.first() {
        let bytes = fs::read(f).unwrap();
        let img = image::load_from_memory(&bytes).unwrap();
        let _ = proc.preprocess(&[img], &cfg).unwrap();
    }

    println!("# smg Qwen2-VL preprocessor on real image bytes (no scaled iDCT)");
    println!(
        "{:<28} {:>8} {:>10} {:>11} {:>10}",
        "fixture", "size_KB", "decode_us", "process_us", "total_us"
    );

    for path in &files {
        let bytes = fs::read(path).unwrap();
        let size_kb = bytes.len() / 1024;
        let mut decode_ns_total = 0u64;
        let mut process_ns_total = 0u64;
        let mut total_ns_total = 0u64;
        for _ in 0..ITERS {
            let t0 = Instant::now();
            let img: DynamicImage = image::load_from_memory(&bytes).expect("decode");
            let t1 = Instant::now();
            let _ = proc.preprocess(&[img], &cfg).expect("preprocess");
            let t2 = Instant::now();
            decode_ns_total += (t1 - t0).as_nanos() as u64;
            process_ns_total += (t2 - t1).as_nanos() as u64;
            total_ns_total += (t2 - t0).as_nanos() as u64;
        }
        let to_us = |ns: u64| (ns / 1000) as f64 / ITERS as f64;
        println!(
            "{:<28} {:>8} {:>10.1} {:>11.1} {:>10.1}",
            path.file_name().unwrap().to_string_lossy(),
            size_kb,
            to_us(decode_ns_total),
            to_us(process_ns_total),
            to_us(total_ns_total),
        );
    }
}

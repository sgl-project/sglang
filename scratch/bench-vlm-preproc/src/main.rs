mod smart_resize;
mod preprocess;

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;

use preprocess::{
    preprocess_batch_into, preprocess_image, preprocess_image_fused_into,
    preprocess_image_into, QwenCfg,
};

#[derive(Parser, Debug)]
struct Args {
    /// Path to a fixtures directory containing JPEG/PNG images.
    #[arg(long, default_value = "fixtures")]
    fixtures: PathBuf,
    /// Iterations per image for the per-image timing.
    #[arg(long, default_value_t = 10)]
    iters: u32,
    /// Run batch (rayon) timing too.
    #[arg(long, default_value_t = true)]
    batch: bool,
    /// Iterations for the batch timing.
    #[arg(long, default_value_t = 5)]
    batch_iters: u32,
}

fn load_dir(dir: &Path) -> anyhow::Result<Vec<(String, Vec<u8>)>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let p = entry?.path();
        let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
        if matches!(ext, "jpg" | "jpeg" | "png") {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            let bytes = fs::read(&p)?;
            out.push((name, bytes));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let cfg = QwenCfg::default();
    let images = load_dir(&args.fixtures)?;
    if images.is_empty() {
        anyhow::bail!("no fixtures in {:?}", args.fixtures);
    }

    println!("# Rust VLM preprocessor bench");
    println!("# fixtures: {} images from {:?}", images.len(), args.fixtures);
    println!("# config: Qwen2-VL (patch=14, merge=2, tp=2, factor=28, max_pixels=1,003,520)");
    println!();

    // Warm up (load thread-local resizer, JIT cpu paths)
    let mut warm_out = Vec::new();
    for (_, b) in &images[..1.min(images.len())] {
        let _ = preprocess_image_into(b, &cfg, &mut warm_out)?;
    }

    println!("## Per-image timing ({} iterations each)", args.iters);
    println!(
        "{:<28} {:>8} {:>10} {:>10} {:>10} {:>10}   decoded          target           patches",
        "fixture", "size_KB", "decode_us", "resize_us", "patch_us", "total_us"
    );
    // Reuse one Vec<f32> across iterations — measures the realistic pooled case.
    let mut out_buf: Vec<f32> = Vec::new();
    for (name, bytes) in &images {
        let size_kb = bytes.len() / 1024;
        let mut totals = [0u64; 4];
        let mut last_t = None;
        for _ in 0..args.iters {
            let (info, t) = preprocess_image_into(bytes, &cfg, &mut out_buf)?;
            totals[0] += t.decode_ns;
            totals[1] += t.resize_ns;
            totals[2] += t.normpack_ns;
            totals[3] += t.total_ns;
            last_t = Some((t, info.num_patches));
        }
        let (t, np) = last_t.unwrap();
        let n = args.iters as u64;
        let to_us = |ns: u64| (ns / 1000) as f64 / n as f64;
        println!(
            "{:<28} {:>8} {:>10.1} {:>10.1} {:>10.1} {:>10.1}   {:>5}x{:<5}     {:>5}x{:<5}     {:>6}",
            name,
            size_kb,
            to_us(totals[0]),
            to_us(totals[1]),
            to_us(totals[2]),
            to_us(totals[3]),
            t.decoded_h,
            t.decoded_w,
            t.target_h,
            t.target_w,
            np,
        );
    }

    // --- Fused (single-pass) variant: u8 RGB -> bilinear+normalize+patch fused,
    //     rayon-parallelized across merge-block-rows.
    println!();
    println!("## Per-image timing — FUSED single-pass + rayon ({} iters)", args.iters);
    println!(
        "{:<28} {:>8} {:>10} {:>10}   decoded          target           patches",
        "fixture", "size_KB", "decode_us", "fused_us"
    );
    let mut fused_buf: Vec<f32> = Vec::new();
    // warmup
    for (_, b) in &images[..1.min(images.len())] {
        let _ = preprocess_image_fused_into(b, &cfg, &mut fused_buf)?;
    }
    for (name, bytes) in &images {
        let size_kb = bytes.len() / 1024;
        let mut totals = [0u64; 3];
        let mut last_t = None;
        for _ in 0..args.iters {
            let (info, t) = preprocess_image_fused_into(bytes, &cfg, &mut fused_buf)?;
            totals[0] += t.decode_ns;
            totals[1] += t.normpack_ns; // fused step (resize+norm+patch combined)
            totals[2] += t.total_ns;
            last_t = Some((t, info.num_patches));
        }
        let (t, np) = last_t.unwrap();
        let n = args.iters as u64;
        let to_us = |ns: u64| (ns / 1000) as f64 / n as f64;
        println!(
            "{:<28} {:>8} {:>10.1} {:>10.1}   {:>5}x{:<5}     {:>5}x{:<5}     {:>6}",
            name,
            size_kb,
            to_us(totals[0]),
            to_us(totals[1]),
            t.decoded_h,
            t.decoded_w,
            t.target_h,
            t.target_w,
            np,
        );
    }

    if args.batch {
        println!();
        println!("## Batch timing (rayon, {} iters)", args.batch_iters);
        // Batch = all fixtures as a single batch.
        // preprocess_batch_into: each image has a caller-owned Vec<f32> we reuse
        // across iterations, so the "no allocation after warm-up" contract holds.
        let bytes_only: Vec<Vec<u8>> = images.iter().map(|(_, b)| b.clone()).collect();
        let mut out_buffers: Vec<Vec<f32>> = (0..bytes_only.len()).map(|_| Vec::new()).collect();

        // warm
        let _ = preprocess_batch_into(&bytes_only, &cfg, &mut out_buffers);

        let mut total_ns: u64 = 0;
        for _ in 0..args.batch_iters {
            let start = Instant::now();
            let results = preprocess_batch_into(&bytes_only, &cfg, &mut out_buffers);
            let dur = start.elapsed().as_nanos() as u64;
            total_ns += dur;
            for r in &results {
                if r.is_err() { eprintln!("batch err: {:?}", r.as_ref().err()); }
            }
        }
        let avg_ms = (total_ns as f64) / 1e6 / args.batch_iters as f64;
        println!(
            "batch of {} images: avg wall time = {:.2} ms ({:.2} ms/image)",
            images.len(),
            avg_ms,
            avg_ms / images.len() as f64,
        );
    }

    Ok(())
}

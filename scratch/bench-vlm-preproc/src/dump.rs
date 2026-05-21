//! Dump preprocessor output for cross-validation against Python HF.
//!
//! Reads one image, runs preprocess, writes pixel_values as raw f32 little-endian
//! plus a small JSON sidecar with shape info.

mod smart_resize;
mod preprocess;

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use preprocess::{preprocess_image, QwenCfg};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    image: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let bytes = fs::read(&args.image)?;
    let cfg = QwenCfg::default();
    let (pixel_values, info, _t) = preprocess_image(&bytes, &cfg)?;

    let bin_path = args.out.with_extension("f32");
    let json_path = args.out.with_extension("json");

    let mut f = fs::File::create(&bin_path)?;
    let bytes: &[u8] = bytemuck::cast_slice(&pixel_values);
    f.write_all(bytes)?;

    let meta = format!(
        "{{\"num_patches\":{},\"patch_features\":{},\"grid_thw\":[{},{},{}]}}",
        info.num_patches,
        info.patch_features,
        info.grid_thw[0], info.grid_thw[1], info.grid_thw[2],
    );
    fs::write(&json_path, meta)?;
    println!("wrote {} ({} f32 values) and {}", bin_path.display(), pixel_values.len(), json_path.display());
    Ok(())
}

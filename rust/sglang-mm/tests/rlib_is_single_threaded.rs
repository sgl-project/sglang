//! The pure-Rust `rlib` that `sglang-server` links must not own worker
//! threads: the server supplies concurrency across requests and pins its own
//! cores, so a library spawning pools behind its back would fight it.
//!
//! Guarding this from the outside (thread count of the process) rather than by
//! inspecting the code, so it stays true no matter how the fan-out seam in
//! `common::par` is refactored. Runs only in the default (rayon-less) build;
//! under `--features parallel` the pools are expected.

#![cfg(all(
    not(feature = "parallel"),
    any(target_os = "linux", target_os = "macos")
))]

use sglang_mm_core::driver::{ImageSource, MmInput, process};
use sglang_mm_core::registry::pipeline_from_spec;

const SPEC: &str = r#"{"family":"qwen_vl","image_token_id":1,"patch_size":14,
    "merge_size":2,"temporal_patch_size":2,"min_pixels":3136,
    "max_pixels":12845056,"image_mean":[0.0,0.0,0.0],"image_std":[1.0,1.0,1.0]}"#;

/// The process's thread count, with the threads' names where the OS lets a
/// process read them (Linux); the names only sharpen a failure's message.
fn threads() -> (usize, Vec<String>) {
    #[cfg(target_os = "linux")]
    {
        let names: Vec<String> = std::fs::read_dir("/proc/self/task")
            .expect("procfs")
            .filter_map(|entry| {
                let comm = entry.ok()?.path().join("comm");
                Some(std::fs::read_to_string(comm).ok()?.trim().to_string())
            })
            .collect();
        (names.len(), names)
    }
    #[cfg(target_os = "macos")]
    {
        // SAFETY: proc_taskinfo is plain data, so all-zero is a valid value;
        // libproc overwrites at most `size` bytes of it.
        let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
        let written = unsafe {
            libc::proc_pidinfo(
                std::process::id() as libc::c_int,
                libc::PROC_PIDTASKINFO,
                0,
                (&mut info as *mut libc::proc_taskinfo).cast(),
                size,
            )
        };
        assert_eq!(written, size, "proc_pidinfo(PROC_PIDTASKINFO)");
        (
            usize::try_from(info.pti_threadnum).expect("thread count"),
            Vec::new(),
        )
    }
}

fn png(w: u32, h: u32) -> Vec<u8> {
    let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([x as u8, y as u8, 7]));
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
    buf.into_inner()
}

#[test]
fn processing_a_request_spawns_no_worker_threads() {
    let (before, _) = threads();
    let family = pipeline_from_spec(SPEC).unwrap();

    // Two images, so the per-item fan-out seam is exercised, not bypassed.
    let out = process(
        family.as_ref(),
        MmInput {
            input_ids: vec![7, 1, 8, 1, 9],
            images: vec![
                ImageSource::Bytes(png(112, 112)),
                ImageSource::Bytes(png(84, 140)),
            ],
        },
    )
    .expect("request should succeed");
    assert_eq!(out.items.len(), 2);

    let (after, names) = threads();
    let spawned: Vec<&String> = names.iter().filter(|t| t.starts_with("sgl-mm")).collect();
    assert!(
        spawned.is_empty(),
        "rlib build spawned crate-owned worker threads: {spawned:?}"
    );
    assert_eq!(
        after, before,
        "rlib build changed the process thread count: {names:?}"
    );
}

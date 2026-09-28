//! The generator: `proto/sglang/` -> `sglang-api-types/src/generated/`.
//!
//! Pass 1 (prost-build + tonic-prost-build) emits the prost structs and tonic
//! service code, plus a `FileDescriptorSet` with the `sglang.json.v1` options
//! intact. Pass 2 (`emit`) reads those options through prost-reflect and
//! emits the serde implementations — proto3's canonical JSON mapping is NOT
//! used; the options are the JSON contract.
//!
//! Regeneration is manual. From anywhere in the `rust/` workspace:
//!
//! ```text
//! cargo run -p sglang-api-codegen
//! ```
//!
//! `proto/sglang/api/v1/service.proto` is the single root: every API must be
//! defined (or imported) there, and protoc's import closure supplies the rest
//! of the package — a file the service does not reach is not part of the API
//! and gets no generated code. The output under
//! `sglang-api-types/src/generated/` is checked in, so commit it together
//! with the .proto change. The vendored protoc (pinned by Cargo.lock) is used
//! unless `PROTOC` is set, so the output is byte-stable across machines.

mod emit;
mod model;

use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let proto_root = manifest_dir.join("../../proto");
    let out_dir = manifest_dir.join("../sglang-api-types/src/generated");
    std::fs::create_dir_all(&out_dir).expect("create generated dir");

    // Vendored protoc first (same policy as sglang-grpc/build.rs): the
    // checked-in output must not depend on whichever protoc a machine has.
    // `PROTOC` still overrides for platforms the vendored crate lacks.
    if std::env::var_os("PROTOC").is_none()
        && let Ok(vendored) = protoc_bin_vendored::protoc_bin_path()
    {
        // SAFETY: nothing else is running yet; the generator is single-threaded.
        unsafe { std::env::set_var("PROTOC", vendored) };
    }

    // The descriptor set is only pass 2's input, not part of the crate; it is
    // a protoc-version-dependent binary, so keep it out of the tree.
    let scratch_dir = std::env::temp_dir().join("sglang-api-codegen");
    std::fs::create_dir_all(&scratch_dir).expect("create scratch dir");
    let descriptor_path = scratch_dir.join("sglang.api.v1.descriptor.bin");
    // The service file is the API surface; everything it imports (the rest of
    // sglang.api.v1 plus the sglang.json.v1 options) comes in transitively
    // via the include path, in topological order, so the output is stable.
    let file_paths = [proto_root.join("sglang/api/v1/service.proto")];

    // Pass 1: structs + tonic service (+ the descriptor set pass 2 reads).
    // Services land in their own file so lib.rs can gate them if ever needed.
    let mut config = prost_build::Config::new();
    config
        .out_dir(&out_dir)
        .file_descriptor_set_path(&descriptor_path)
        .service_generator(tonic_prost_build::configure().service_generator())
        // The abort payload is the widest and rarest finish variant; boxed so
        // it does not set the size of every ChunkEvent (the hand-written type
        // boxed it for the same reason; response.rs pins the frame size).
        .boxed(".sglang.api.v1.FinishReason.kind.abort")
        .protoc_arg("--experimental_allow_proto3_optional");
    config
        .compile_protos(&file_paths, std::slice::from_ref(&proto_root))
        .expect("pass 1: prost compile");

    // prost writes structs + services into one file; split the tonic modules
    // out so the include! layout stays stable.
    split_tonic(&out_dir.join("sglang.api.v1.rs"), &out_dir);

    // Pass 2: schema-driven serde from the options in the descriptor set.
    let bytes = std::fs::read(&descriptor_path).expect("descriptor set");
    let serde_src = emit::emit_serde(&bytes, "sglang.api.v1");
    std::fs::write(out_dir.join("sglang.api.v1.serde.rs"), serde_src).expect("write serde");

    // The options package itself needs no Rust types (it exists for the
    // generator); drop pass-1 output for it if prost emitted one.
    let _ = std::fs::remove_file(out_dir.join("sglang.json.v1.rs"));

    rustfmt(&out_dir);
    println!("generated into {}", out_dir.display());
}

/// Move the tonic `pub mod sglang_api_client/_server` blocks from the prost
/// output into `sglang.api.v1.tonic.rs`.
fn split_tonic(prost_file: &Path, out_dir: &Path) {
    let src = std::fs::read_to_string(prost_file).expect("read pass-1 output");
    let marker = "/// Generated client implementations.";
    let (structs, tonic) = match src.find(marker) {
        Some(pos) => src.split_at(pos),
        None => (src.as_str(), ""),
    };
    std::fs::write(prost_file, structs.trim_end().to_string() + "\n").expect("write structs");
    std::fs::write(out_dir.join("sglang.api.v1.tonic.rs"), tonic.trim_start())
        .expect("write tonic");
}

fn rustfmt(out_dir: &Path) {
    for entry in std::fs::read_dir(out_dir).expect("read generated dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().is_some_and(|e| e == "rs") {
            // Best effort: the output is already well-formed; rustfmt only
            // normalizes.
            let _ = std::process::Command::new("rustfmt")
                .arg("--edition=2024")
                .arg(&path)
                .status();
        }
    }
}

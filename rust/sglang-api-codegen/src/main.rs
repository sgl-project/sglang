//! The generator: every checked-in binding of `proto/sglang/`, from one protoc.
//!
//! Targets (see [`TARGETS`]):
//! Both targets land in `rust/sglang-api-types/src/generated/`:
//! - `sglang.api.v1`: prost structs, tonic services, schema-driven serde, plus
//!   `python/sglang/api/v1/types.py`.
//! - `sglang.runtime.v1`: prost structs and tonic services only.
//!
//! Pass 1 (prost-build + tonic-prost-build) emits the prost structs and tonic
//! service code, plus a `FileDescriptorSet`. For targets with a JSON contract
//! pass 2 reads the `sglang.json.v1` options from that set through
//! prost-reflect and emits, from one model (`model`), the Rust serde
//! implementations (`emit`) and the Python interface definition (`emit_py`);
//! proto3's canonical JSON mapping is NOT used, the options are the contract.
//!
//! Regeneration is manual. From anywhere in the `rust/` workspace:
//!
//! ```text
//! cargo gen-api
//! ```
//!
//! Each target has a single root .proto: every API must be defined (or
//! imported) there, and protoc's import closure supplies the rest of the
//! package; a file the root does not reach gets no generated code. All output
//! is checked in, so commit it together with the .proto change. The vendored
//! protoc (pinned by Cargo.lock) is used unless `PROTOC` is set, so the output
//! is byte-stable across machines.

mod emit;
mod emit_py;
mod model;

use std::path::{Path, PathBuf};

/// One generated binding set. Paths are relative to the repository root.
struct Target {
    /// The root .proto under `proto/`; its import closure is the API surface.
    root: &'static str,
    package: &'static str,
    rust_out: &'static str,
    /// Set when the schema carries `sglang.json.v1` options: emits the serde
    /// implementations next to the structs and the Python module here.
    python_out: Option<&'static str>,
    /// prost `boxed` paths (message fields stored as `Box<T>`).
    boxed: &'static [&'static str],
}

const TARGETS: &[Target] = &[
    Target {
        root: "sglang/api/v1/service.proto",
        package: "sglang.api.v1",
        rust_out: "rust/sglang-api-types/src/generated",
        python_out: Some("python/sglang/api/v1/types.py"),
        // Boxed abort payload won't set the size of every ChunkEvent.
        boxed: &[".sglang.api.v1.FinishReason.kind.abort"],
    },
    Target {
        root: "sglang/runtime/v1/sglang.proto",
        package: "sglang.runtime.v1",
        rust_out: "rust/sglang-api-types/src/generated",
        python_out: None,
        boxed: &[],
    },
];

fn main() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let proto_root = repo_root.join("proto");

    // Vendored protoc first: the checked-in output must not depend on
    // whichever protoc a machine has. `PROTOC` still overrides for platforms
    // the vendored crate lacks.
    if std::env::var_os("PROTOC").is_none()
        && let Ok(vendored) = protoc_bin_vendored::protoc_bin_path()
    {
        // SAFETY: nothing else is running yet; the generator is single-threaded.
        unsafe { std::env::set_var("PROTOC", vendored) };
    }

    // The descriptor set is only pass 2's input, not part of any crate; it is
    // a protoc-version-dependent binary, so keep it out of the tree.
    let scratch_dir = std::env::temp_dir().join("sglang-api-codegen");
    std::fs::create_dir_all(&scratch_dir).expect("create scratch dir");

    for target in TARGETS {
        generate(target, &repo_root, &proto_root, &scratch_dir);
    }
}

fn generate(target: &Target, repo_root: &Path, proto_root: &Path, scratch_dir: &Path) {
    let package = target.package;
    let out_dir = repo_root.join(target.rust_out);
    std::fs::create_dir_all(&out_dir).expect("create generated dir");
    let descriptor_path = scratch_dir.join(format!("{package}.descriptor.bin"));

    // Pass 1: structs + tonic services (+ the descriptor set pass 2 reads).
    // Everything the root imports comes in transitively via the include path,
    // in topological order, so the output is stable.
    let mut config = prost_build::Config::new();
    config
        .out_dir(&out_dir)
        .file_descriptor_set_path(&descriptor_path)
        .service_generator(tonic_prost_build::configure().service_generator())
        .protoc_arg("--experimental_allow_proto3_optional");
    for path in target.boxed {
        config.boxed(path);
    }
    config
        .compile_protos(&[proto_root.join(target.root)], &[proto_root])
        .unwrap_or_else(|e| panic!("pass 1 ({package}): {e}"));

    // prost writes structs + services into one file; split the tonic modules
    // out so the include! layout stays stable.
    split_tonic(&out_dir, package);

    if let Some(python_out) = target.python_out {
        let bytes = std::fs::read(&descriptor_path).expect("descriptor set");

        // Pass 2: schema-driven serde from the options in the descriptor set.
        let serde_src = emit::emit_serde(&bytes, package);
        std::fs::write(out_dir.join(format!("{package}.serde.rs")), serde_src)
            .expect("write serde");

        // Pass 2 (Python): the same model rendered as Structs + codecs.
        let py_out = repo_root.join(python_out);
        std::fs::create_dir_all(py_out.parent().expect("py out dir")).expect("create py dir");
        std::fs::write(&py_out, emit_py::emit_python(&bytes, package)).expect("write python types");
        println!("generated {}", py_out.display());

        // The options package itself needs no Rust types (it exists for the
        // generator); drop pass-1 output for it if prost emitted one.
        let _ = std::fs::remove_file(out_dir.join("sglang.json.v1.rs"));
    }

    rustfmt(&out_dir);
    println!("generated into {}", out_dir.display());
}

/// Move the tonic `pub mod *_client/_server` blocks from the prost output
/// `<package>.rs` into `<package>.tonic.rs`.
fn split_tonic(out_dir: &Path, package: &str) {
    let prost_file = out_dir.join(format!("{package}.rs"));
    let src = std::fs::read_to_string(&prost_file).expect("read pass-1 output");
    let marker = "/// Generated client implementations.";
    let (structs, tonic) = match src.find(marker) {
        Some(pos) => src.split_at(pos),
        None => (src.as_str(), ""),
    };
    std::fs::write(&prost_file, structs.trim_end().to_string() + "\n").expect("write structs");
    std::fs::write(
        out_dir.join(format!("{package}.tonic.rs")),
        tonic.trim_start(),
    )
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

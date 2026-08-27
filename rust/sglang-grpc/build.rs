fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Prefer an explicitly configured protoc; otherwise use the vendored
    // binary so builds (including `cargo clippy` and the pre-commit hook on
    // machines/CI runners without protobuf installed) are self-contained.
    // protoc_bin_path() errs on platforms the vendored crate doesn't cover;
    // those fall back to prost-build's own lookup of `protoc` on PATH.
    if std::env::var_os("PROTOC").is_none()
        && let Ok(vendored) = protoc_bin_vendored::protoc_bin_path()
    {
        // SAFETY: build scripts are single-threaded at this point.
        unsafe { std::env::set_var("PROTOC", vendored) };
    }

    let runtime_proto = "../../proto/sglang/runtime/v1/sglang.proto";
    let renderer_proto = "../../proto/sglang/renderer/v1/renderer.proto";

    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .protoc_arg("--experimental_allow_proto3_optional")
        .file_descriptor_set_path(
            std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap())
                .join("sglang_descriptor.bin"),
        )
        .compile_protos(&[runtime_proto, renderer_proto], &["../../proto"])?;

    println!("cargo:rerun-if-changed={}", runtime_proto);
    println!("cargo:rerun-if-changed={}", renderer_proto);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Prefer an explicitly configured protoc; otherwise use the vendored
    // binary so builds remain self-contained on machines without protobuf.
    if std::env::var_os("PROTOC").is_none()
        && let Ok(vendored) = protoc_bin_vendored::protoc_bin_path()
    {
        // SAFETY: build scripts are single-threaded at this point.
        unsafe { std::env::set_var("PROTOC", vendored) };
    }

    let proto_root = "../../proto";
    let proto_path = "../../proto/sglang/runtime/v1/sglang.proto";
    let descriptor_path =
        std::path::PathBuf::from(std::env::var("OUT_DIR")?).join("sglang_descriptor.bin");

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .protoc_arg("--experimental_allow_proto3_optional")
        .file_descriptor_set_path(descriptor_path)
        .compile_protos(&[proto_path], &[proto_root])?;

    println!("cargo:rerun-if-changed={proto_path}");
    Ok(())
}

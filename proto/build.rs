fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Prefer an explicitly configured protoc; otherwise use the vendored
    // binary so builds remain self-contained on machines without protobuf.
    if std::env::var_os("PROTOC").is_none()
        && let Ok(vendored) = protoc_bin_vendored::protoc_bin_path()
    {
        // SAFETY: build scripts are single-threaded at this point.
        unsafe { std::env::set_var("PROTOC", vendored) };
    }

    let proto_path = "sglang/runtime/v1/sglang.proto";

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&[proto_path], &["."])?;

    println!("cargo:rerun-if-changed={proto_path}");
    Ok(())
}

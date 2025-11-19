fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Only regenerate if proto files change
    println!("cargo:rerun-if-changed=src/proto/sglang_scheduler.proto");
    println!("cargo:rerun-if-changed=src/proto/vllm_engine.proto");

    // Configure tonic-prost-build for gRPC code generation
    tonic_prost_build::configure()
        // Generate both client and server code
        .build_server(true)
        .build_client(true)
        // Add serde Serialize for model info messages (we only need to serialize to labels)
        .type_attribute("GetModelInfoResponse", "#[derive(serde::Serialize)]")
        // Allow proto3 optional fields
        .protoc_arg("--experimental_allow_proto3_optional")
        // Compile both proto files
        .compile_protos(
            &[
                "src/proto/sglang_scheduler.proto",
                "src/proto/vllm_engine.proto",
            ],
            &["src/proto"],
        )?;

    println!("cargo:info=Protobuf compilation completed successfully");

    Ok(())
}

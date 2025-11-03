fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Only regenerate if the proto file changes
    println!("cargo:rerun-if-changed=src/proto/sglang_scheduler.proto");

    // Configure tonic-prost-build for gRPC code generation
    tonic_prost_build::configure()
        // Generate both client and server code
        .build_server(true)
        .build_client(true)
        // Allow proto3 optional fields
        .protoc_arg("--experimental_allow_proto3_optional")
        // Compile the proto file
        .compile_protos(&["src/proto/sglang_scheduler.proto"], &["src/proto"])?;

    println!("cargo:warning=Protobuf compilation completed successfully");

    Ok(())
}

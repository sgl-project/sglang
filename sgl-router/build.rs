fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Only regenerate if the proto file changes
    println!("cargo:rerun-if-changed=src/proto/sglang_scheduler.proto");

    // Configure protobuf compilation with custom settings
    let config = prost_build::Config::new();

    // Skip serde for types that use prost_types::Struct
    // These cause conflicts and we don't need serde for all generated types

    // Configure tonic-build for gRPC code generation
    tonic_build::configure()
        // Generate both client and server code
        .build_server(true)
        .build_client(true)
        // Add protoc arguments for proto3 optional support
        .protoc_arg("--experimental_allow_proto3_optional")
        // Add a module-level attribute for documentation and clippy warnings
        .server_mod_attribute(
            "sglang.grpc.scheduler",
            "#[allow(unused, unused_qualifications, clippy::mixed_attributes_style)]",
        )
        .client_mod_attribute(
            "sglang.grpc.scheduler",
            "#[allow(unused, unused_qualifications, clippy::mixed_attributes_style)]",
        )
        // Compile the proto file with the custom config
        .compile_protos_with_config(
            config,
            &["src/proto/sglang_scheduler.proto"],
            &["src/proto"],
        )?;

    println!("cargo:warning=Protobuf compilation completed successfully");

    Ok(())
}

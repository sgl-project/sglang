use prost_types::field_descriptor_proto::{Label, Type};
use std::process::Command;

/// Derive serde attributes for every generated message so the proto messages
/// can be (de)serialized as msgpack (see `src/msgpack.rs`) for the ZMQ path,
/// using the SAME types tonic uses for gRPC. The wire convention is a msgpack
/// map keyed by proto field name, with default-valued fields omitted — matching
/// the Python `msgspec` codec generated from the same proto.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_path = "../../proto/sglang/runtime/v1/sglang.proto";
    let proto_root = "../../proto";
    let out_dir = std::env::var("OUT_DIR")?;
    let protoc = std::env::var("PROTOC").unwrap_or_else(|_| "protoc".to_string());

    // Pass 1: dump a FileDescriptorSet so we can compute per-field serde
    // attributes (skip_serializing_if) before generating code.
    let descriptor = format!("{out_dir}/serde_descriptor.bin");
    let status = Command::new(&protoc)
        .args([
            "--experimental_allow_proto3_optional",
            &format!("--proto_path={proto_root}"),
            &format!("--descriptor_set_out={descriptor}"),
            "--include_imports",
            proto_path,
        ])
        .status()?;
    if !status.success() {
        return Err(format!("protoc failed to produce descriptor set (status {status})").into());
    }
    let fds: prost_types::FileDescriptorSet =
        prost::Message::decode(&*std::fs::read(&descriptor)?)?;

    let mut builder = tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .protoc_arg("--experimental_allow_proto3_optional")
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        // Tolerant decode: any field absent from the msgpack map falls back to its
        // default, so adding fields to the proto never breaks older encoders.
        .type_attribute(".", "#[serde(default)]")
        .file_descriptor_set_path(
            std::path::PathBuf::from(&out_dir).join("sglang_descriptor.bin"),
        );

    // Pass 2: omit default-valued fields on the wire, mirroring msgspec's
    // `omit_defaults=True`, so unset optionals / empty collections cost no bytes.
    for file in &fds.file {
        let package = file.package();
        for msg in &file.message_type {
            if msg.options.as_ref().and_then(|o| o.map_entry).unwrap_or(false) {
                continue;
            }
            for field in &msg.field {
                let path = format!(".{}.{}.{}", package, msg.name(), field.name());
                let is_map = field.r#type() == Type::Message
                    && msg.nested_type.iter().any(|n| {
                        n.options.as_ref().and_then(|o| o.map_entry).unwrap_or(false)
                            && field.type_name().ends_with(n.name())
                    });
                let attr = if is_map {
                    "#[serde(skip_serializing_if = \"std::collections::HashMap::is_empty\")]"
                } else if field.label() == Label::Repeated {
                    "#[serde(skip_serializing_if = \"Vec::is_empty\")]"
                } else if field.proto3_optional() || field.r#type() == Type::Message {
                    "#[serde(skip_serializing_if = \"Option::is_none\")]"
                } else {
                    // Implicit-presence proto3 scalar: always present on the wire.
                    continue;
                };
                builder = builder.field_attribute(&path, attr);
            }
        }
    }

    builder.compile_protos(&[proto_path], &[proto_root])?;

    println!("cargo:rerun-if-changed={}", proto_path);
    Ok(())
}

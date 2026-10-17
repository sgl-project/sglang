// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parent hashes need proto3 presence semantics to distinguish roots from
    // valid hash values. Use the bundled compiler to keep codegen stable.
    let mut config = tonic_prost_build::Config::new();
    config.protoc_executable(protoc_bin_vendored::protoc_bin_path()?);

    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_with_config(config, &["proto/kv_indexer.proto"], &["proto"])?;
    Ok(())
}

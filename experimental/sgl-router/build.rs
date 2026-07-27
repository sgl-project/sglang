// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;

/// Generates Rust gRPC bindings from the Router-local load-monitor protocol.
///
/// The build uses a vendored `protoc`, so contributors and CI do not need a
/// system protobuf compiler. The generated code is written to Cargo's normal
/// `OUT_DIR` and included by `src/load_monitor/proto.rs`.
///
/// # Errors
///
/// Returns an error when the vendored compiler cannot be located or when the
/// protobuf schema cannot be compiled.
fn main() -> Result<(), Box<dyn Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    std::env::set_var("PROTOC", protoc);
    println!("cargo:rerun-if-changed=proto/load_monitor.proto");
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/load_monitor.proto"], &["proto"])?;
    Ok(())
}

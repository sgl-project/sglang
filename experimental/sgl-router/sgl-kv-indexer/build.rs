// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 父哈希需要 proto3 optional 的 presence 语义：根节点与哈希值本身不能混淆。
    // 固定使用随 crate 发布的 protoc，避免宿主机旧版编译器改变生成契约。
    let mut config = tonic_prost_build::Config::new();
    config.protoc_executable(protoc_bin_vendored::protoc_bin_path()?);

    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_with_config(config, &["proto/kv_indexer.proto"], &["proto"])?;
    Ok(())
}

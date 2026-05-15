// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "sgl-router", version = sgl_router::VERSION)]
struct Cli {}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = Cli::parse();
    tracing_subscriber::fmt::init();
    tracing::info!(version = sgl_router::VERSION, "sgl-router scaffold");
    Ok(())
}

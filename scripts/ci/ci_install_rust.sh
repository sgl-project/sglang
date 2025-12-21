#!/bin/bash
set -euxo pipefail

# Check if sudo is available
if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y libssl-dev pkg-config protobuf-compiler
else
    apt-get update
    apt-get install -y libssl-dev pkg-config protobuf-compiler
fi

# Install rustup (Rust installer and version manager)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.90


# Follow the installation prompts, then reload your shell
. "$HOME/.cargo/env"
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
protoc --version

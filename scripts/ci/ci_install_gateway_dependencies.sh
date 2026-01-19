#!/bin/bash
set -euxo pipefail

SUDO=$(command -v sudo || true)
$SUDO apt-get update
$SUDO apt-get install -y libssl-dev pkg-config protobuf-compiler redis-server

# Install rustup (Rust installer and version manager)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.90


# Follow the installation prompts, then reload your shell
. "$HOME/.cargo/env"
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
protoc --version

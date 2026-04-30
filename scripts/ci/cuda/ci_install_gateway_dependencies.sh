#!/bin/bash
# Install dependencies for the sgl-model-gateway CI jobs.
#
# Gateway-specific apt deps are installed here; protoc and the Rust toolchain
# are delegated to the shared installer (the toolchain version is pinned by
# sgl-model-gateway/rust-toolchain.toml, picked up automatically on first
# `cargo` invocation).
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GATEWAY_APT_PACKAGES=(libssl-dev pkg-config redis-server)
if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y "${GATEWAY_APT_PACKAGES[@]}"
else
    apt-get update
    apt-get install -y "${GATEWAY_APT_PACKAGES[@]}"
fi

bash "${SCRIPT_DIR}/../utils/install_rust_protoc.sh"

# Make cargo/rustc/protoc visible in this shell.
. "$HOME/.cargo/env"

rustc --version
cargo --version
protoc --version

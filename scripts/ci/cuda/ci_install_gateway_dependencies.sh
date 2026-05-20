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
APT_OPTS=(
    -y
    -o "Acquire::Retries=5"
    -o "Acquire::http::Timeout=30"
    -o "Acquire::https::Timeout=30"
)
SUDO=""
command -v sudo >/dev/null 2>&1 && SUDO="sudo"

# GH-hosted runners' Azure Ubuntu mirrors flake periodically. Retry the
# whole install with backoff so we don't fail the whole CI on a 1-min
# DNS hiccup at apt-mirrors.txt → azure.archive.ubuntu.com.
for attempt in 1 2 3 4 5; do
    if $SUDO apt-get update "${APT_OPTS[@]}" \
       && $SUDO apt-get install "${APT_OPTS[@]}" "${GATEWAY_APT_PACKAGES[@]}"; then
        break
    fi
    if [ "$attempt" = 5 ]; then
        echo "apt-get install failed after 5 attempts; giving up." >&2
        exit 1
    fi
    sleep $((attempt * 15))
done

bash "${SCRIPT_DIR}/../utils/install_rust_protoc.sh"

# Make cargo/rustc/protoc visible in this shell.
. "$HOME/.cargo/env"

rustc --version
cargo --version
protoc --version

#!/bin/bash
# Ensure a Rust toolchain (rustc/cargo) is installed for crates built from
# source, e.g. the native gRPC extension bundled into the sglang wheel via
# setuptools-rust. Minimum supported version is 1.85 (edition 2024).
set -euxo pipefail

# Pick up cargo if rustup was installed in a previous CI step.
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:${PATH}"

if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
    echo "rust already installed: $(rustc --version), $(cargo --version)"
    exit 0
fi

echo "rust not found, installing via rustup..."

# rustup.rs requires curl — make sure it's present.
if ! command -v curl >/dev/null 2>&1; then
    if command -v apt-get &> /dev/null; then
        apt-get update || true
        apt-get install -y --no-install-recommends curl ca-certificates
    elif command -v yum &> /dev/null; then
        yum install -y curl ca-certificates
    else
        echo "ERROR: curl is required to install rustup, but no supported package manager was found"
        exit 1
    fi
fi

curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path

# Make cargo/rustc visible to the rest of this shell and to subsequent
# GitHub Actions steps in the same job.
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:${PATH}"
if [ -n "${GITHUB_PATH:-}" ]; then
    echo "${CARGO_HOME:-$HOME/.cargo}/bin" >> "${GITHUB_PATH}"
fi

rustc --version
cargo --version

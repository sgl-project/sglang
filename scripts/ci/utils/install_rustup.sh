#!/bin/bash
# Ensure a Rust toolchain (rustc/cargo) is installed for crates built from
# source, e.g. the native gRPC extension bundled into the sglang wheel via
# setuptools-rust. Minimum supported version is 1.85 (edition 2024).
set -euxo pipefail

# Make cargo/rustc visible to the rest of this shell and to subsequent
# GitHub Actions steps in the same job.
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:${PATH}"
if [ -n "${GITHUB_PATH:-}" ]; then
    echo "${CARGO_HOME:-$HOME/.cargo}/bin" >> "${GITHUB_PATH}"
fi

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

if [ -n "${RUSTUP_CACHE_URL:-}" ]; then
    # An in-cluster HTTP mirror is available (e.g. on NPU runners).
    export RUSTUP_DIST_SERVER="${RUSTUP_CACHE_URL}/rustup"
    export RUSTUP_UPDATE_ROOT="${RUSTUP_CACHE_URL}/rustup/rustup"
    case "$(uname -m)" in
        x86_64)  RUSTUP_ARCH="x86_64-unknown-linux-gnu" ;;
        aarch64) RUSTUP_ARCH="aarch64-unknown-linux-gnu" ;;
        *) echo "ERROR: unsupported arch $(uname -m)"; exit 1 ;;
    esac
    RUSTUP_TMP="$(mktemp -d)"
    trap 'rm -rf "${RUSTUP_TMP}"' EXIT
    curl --retry 3 --retry-delay 2 -sSfL \
        "${RUSTUP_UPDATE_ROOT}/dist/${RUSTUP_ARCH}/rustup-init" \
        -o "${RUSTUP_TMP}/rustup-init"
    chmod +x "${RUSTUP_TMP}/rustup-init"
    "${RUSTUP_TMP}/rustup-init" -y --no-modify-path
else
    curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path
fi

rustc --version
cargo --version

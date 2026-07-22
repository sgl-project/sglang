#!/bin/bash
# Ensure a Rust toolchain (rustc/cargo) is installed for crates built from
# source, e.g. the native gRPC extension bundled into the sglang wheel via
# setuptools-rust. Minimum supported version is 1.85 (edition 2024).
#
# Also pre-installs the workspace-pinned toolchain from rust/rust-toolchain.toml
# (best-effort) so cargo commands run inside rust/ don't pay the rustup
# auto-install on first use.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_WORKSPACE_DIR="${SCRIPT_DIR}/../../../rust"

# Make cargo/rustc visible to the rest of this shell and to subsequent
# GitHub Actions steps in the same job.
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:${PATH}"
if [ -n "${GITHUB_PATH:-}" ]; then
    # Self-heal if _runner_file_commands/ disappears mid-job on some self-hosted
    # runners; the runner reads this file by its registered UUID at step end, so
    # recreating the path keeps PATH propagation working for subsequent steps.
    mkdir -p "$(dirname "${GITHUB_PATH}")" 2>/dev/null || true
    echo "${CARGO_HOME:-$HOME/.cargo}/bin" >> "${GITHUB_PATH}" || true
fi

# An in-cluster HTTP mirror may be available (e.g. on NPU runners); export it
# up front so every rustup invocation below (self-heal and pinned-toolchain
# install included) goes through the mirror.
if [ -n "${RUSTUP_CACHE_URL:-}" ]; then
    export RUSTUP_DIST_SERVER="${RUSTUP_CACHE_URL}/rustup"
    export RUSTUP_UPDATE_ROOT="${RUSTUP_CACHE_URL}/rustup/rustup"
fi

install_workspace_pinned_toolchain() {
    # Pre-install the toolchain pinned by rust/rust-toolchain.toml: with no
    # arguments and cwd inside rust/, `rustup toolchain install` (rustup >=
    # 1.28) resolves channel/profile from the toolchain file; older rustups
    # fall back to parsing the channel out of the file. Best-effort: the pin
    # only governs cargo runs with cwd inside rust/ — setuptools-rust wheel
    # builds run cargo from python/ and use the default toolchain, so a failure
    # here must not fail the build (rustup auto-installs the pin on first use
    # anyway).
    if ! command -v rustup >/dev/null 2>&1; then
        return 0
    fi
    local toolchain_file="${RUST_WORKSPACE_DIR}/rust-toolchain.toml"
    if [ ! -f "${toolchain_file}" ]; then
        return 0
    fi
    if (cd "${RUST_WORKSPACE_DIR}" && rustup toolchain install); then
        return 0
    fi
    local channel
    channel="$(sed -n 's/^channel *= *"\([^"]*\)".*/\1/p' "${toolchain_file}")"
    if [ -n "${channel}" ] && rustup toolchain install --profile minimal "${channel}"; then
        return 0
    fi
    echo "WARNING: could not pre-install the toolchain pinned by ${toolchain_file}; rustup will auto-install it on first cargo use inside rust/"
}

if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
    # `command -v` only proves the rustup shims exist. A runner image can ship
    # them with no default toolchain configured (rustup-init --default-toolchain
    # none, or a removed toolchain), in which case every cargo/rustc call fails
    # with "rustup could not choose a version of rustc to run". Check
    # functionally and self-heal instead of trusting shim presence.
    if rustc --version >/dev/null 2>&1; then
        echo "rust already installed: $(rustc --version), $(cargo --version)"
    elif command -v rustup >/dev/null 2>&1; then
        echo "rustup shims present but no usable toolchain; installing stable as default..."
        rustup default stable
    else
        echo "ERROR: cargo/rustc on PATH but non-functional and rustup is missing; remove the stale binaries and re-run"
        exit 1
    fi
else
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
        # Mirror env vars (RUSTUP_DIST_SERVER/RUSTUP_UPDATE_ROOT) were exported
        # at the top of this script.
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
fi

install_workspace_pinned_toolchain

rustc --version
cargo --version

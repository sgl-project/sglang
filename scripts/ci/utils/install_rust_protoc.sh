#!/bin/bash
# Install protoc and a Rust toolchain (rustup/cargo). Required by setuptools-rust
# to build the bundled native gRPC extension (rust/sglang-grpc) when installing
# the main `sglang` wheel from source. Idempotent — both helpers no-op if
# already installed.
#
# protoc installs system-wide (/usr/local) and apt deps, so it needs root.
# rustup installs per-user under $HOME/.cargo, so it must run as the calling
# user (running it under sudo would put cargo in /root/.cargo and the rest of
# the job wouldn't find it).
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$(id -u)" = "0" ]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    SUDO=""
fi

${SUDO} bash "${SCRIPT_DIR}/install_protoc.sh"
bash "${SCRIPT_DIR}/install_rustup.sh"

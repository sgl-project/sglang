#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for SGLang.
#
# This VM is CPU-only, so it targets the parts of SGLang that build and run
# without a GPU: the contributor lint/CI toolchain (pre-commit) and the Rust
# workspaces (rust/ services and the sgl-model-gateway router). The full
# CUDA serving stack (torch==2.13.0, flashinfer, sgl-kernel, ...) requires an
# NVIDIA GPU and the lmsysorg/sglang image and is intentionally not installed
# here.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# pip --user installs console scripts here.
export PATH="$HOME/.local/bin:$PATH"
# Make the system rust toolchain manager available to every phase.
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"

echo "==> [1/5] System build dependencies"
# The rust/ workspace compiles C++ deps (e.g. esaxx-rs, pulled in by the
# HuggingFace tokenizers crate) through clang, which targets the gcc-14
# libstdc++ headers. The base image ships only libstdc++-13-dev, so <cstdint>
# and friends are missing without this. libssl-dev/pkg-config cover any
# openssl-sys build. apt-get is a no-op when the packages are already present.
if command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq
  # protobuf-compiler + libprotobuf-dev: sgl-model-gateway's smg-grpc-client
  # builds .proto files via prost-build, which needs a system protoc (no
  # vendored fallback) plus the well-known types (google/protobuf/*.proto)
  # that ship in libprotobuf-dev.
  # redis-server: the gateway's Redis history-backend integration tests spawn a
  # local redis-server on a random port (tests/common/redis_test_server.rs).
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    libstdc++-14-dev libssl-dev pkg-config protobuf-compiler libprotobuf-dev redis-server
fi

echo "==> [2/5] Python lint/CI tooling (pre-commit)"
python3 -m pip install --user --quiet --upgrade pre-commit
# Warm every hook environment (isort, ruff, black, codespell, clang-format,
# nbstripout, ...) so the first real `pre-commit run` is fast. Safe to re-run.
pre-commit install-hooks
# Install the git hook so commits are linted locally.
pre-commit install >/dev/null 2>&1 || true

echo "==> [3/5] Rust toolchains"
# rust/ pins 1.92 and sgl-model-gateway pins 1.90 via rust-toolchain.toml, so
# rustup fetches them on first cargo invocation. The fmt pre-commit hooks use a
# nightly rustfmt; install it up front so linting works offline afterwards.
rustup toolchain install nightly --profile minimal --component rustfmt 2>/dev/null || \
  rustup component add --toolchain nightly rustfmt 2>/dev/null || true

echo "==> [4/5] Build rust/ workspace (gRPC, mm, server)"
( cd rust && cargo build --workspace )

echo "==> [5/5] Build sgl-model-gateway (router/control plane)"
( cd sgl-model-gateway && cargo build )

echo "==> SGLang Cloud Agent environment ready."
echo "    Lint:            pre-commit run --all-files"
echo "    Rust tests:      (cd rust && cargo test --workspace)"
echo "    Gateway tests:   (cd sgl-model-gateway && cargo test)"
echo "    Run the router:  (cd sgl-model-gateway && cargo run --bin sgl-model-gateway -- --help)"

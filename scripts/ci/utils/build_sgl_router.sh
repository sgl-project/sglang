#!/bin/bash
# Build the sgl-router binary (release) and copy it to <out_dir>/sgl-router.
# Expects cargo from scripts/ci/utils/install_rustup.sh.
set -euxo pipefail

out_dir="$(realpath -m "$1")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:${PATH}"
# install_rustup.sh exports rust/'s pin, which would outrank sgl-router's own
# rust-toolchain.toml; cargo below runs from sgl-router/ so that file applies.
unset RUSTUP_TOOLCHAIN

# Per-job path: a self-hosted runner is persistent and shared.
target_dir="${RUNNER_TEMP:-/tmp}/sgl-router-target-${GITHUB_RUN_ID:-norun}-$$"
trap 'rm -rf "${target_dir}" || true' EXIT

cd "${repo_root}/sgl-router"
CARGO_TARGET_DIR="${target_dir}" cargo build --locked --release --bin sgl-router
mkdir -p "${out_dir}"
cp "${target_dir}/release/sgl-router" "${out_dir}/sgl-router"

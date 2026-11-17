#!/bin/bash
# Build the release sgl-router binary from this checkout and install it at <dest>.
# Images that git-clone their source (npu, rocm, xpu) build inline:
# the clone may predate this script.
#
# Usage: build.sh <dest> [--cleanup]
#   --cleanup  remove the rustup, toolchain, and crate cache this build added
set -euo pipefail

dest=""
cleanup=0
for arg in "$@"; do
    case "${arg}" in
        --cleanup) cleanup=1 ;;
        -*) echo "build.sh: unknown flag ${arg}" >&2; exit 2 ;;
        *)
            if [ -n "${dest}" ]; then
                echo "build.sh: more than one <dest>" >&2
                exit 2
            fi
            dest="${arg}"
            ;;
    esac
done
if [ -z "${dest}" ]; then
    echo "usage: build.sh <dest> [--cleanup]" >&2
    exit 2
fi
dest="$(realpath -m "${dest}")"

router_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cargo_home="${CARGO_HOME:-${HOME}/.cargo}"
rustup_home="${RUSTUP_HOME:-${HOME}/.rustup}"
export PATH="${cargo_home}/bin:${PATH}"

installed_rustup=0
if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal --default-toolchain none
    installed_rustup=1
fi

# A caller's RUSTUP_TOOLCHAIN (install_rustup.sh exports rust/'s pin) would
# outrank sgl-router/rust-toolchain.toml; the router builds at its own pin.
unset RUSTUP_TOOLCHAIN
cd "${router_dir}"
pinned="$(sed -n 's/^channel *= *"\([^"]*\)".*/\1/p' rust-toolchain.toml)"
# Toolchain names are `<channel>-<host triple>`; 1.92.0 is a different one.
if rustup toolchain list | grep -q "^${pinned//./\\.}-"; then
    had_pinned=1
else
    had_pinned=0
    rustup toolchain install
fi

# pcre2-sys links the system libpcre2-8 when pkg-config finds one; the vendored
# copy lets the binary run where it is absent.
export PCRE2_SYS_STATIC=1

# A per-build target dir: a self-hosted CI runner is persistent and shared.
target_dir="$(mktemp -d "${RUNNER_TEMP:-/tmp}/sgl-router-target.XXXXXX")"
trap 'rm -rf "${target_dir}"' EXIT

CARGO_TARGET_DIR="${target_dir}" cargo build --locked --release --bin sgl-router
install -D -m 0755 "${target_dir}/release/sgl-router" "${dest}"
"${dest}" --version

if [ "${cleanup}" = 1 ]; then
    if [ "${installed_rustup}" = 1 ]; then
        rm -rf "${cargo_home}" "${rustup_home}"
    else
        if [ "${had_pinned}" = 0 ]; then
            rustup toolchain uninstall "${pinned}"
        fi
        rm -rf "${cargo_home}/registry" "${cargo_home}/git"
    fi
fi

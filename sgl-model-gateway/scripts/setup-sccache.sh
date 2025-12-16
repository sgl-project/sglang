#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

echo "Setting up sccache for faster Rust compilation..."

has_cmd() { command -v "$1" >/dev/null 2>&1; }

install_sccache() {
  echo "sccache not found."
  if [[ "${AUTO_INSTALL:-0}" != "1" ]]; then
    read -r -p "Install sccache now? [y/N] " response
    response=${response:-N}
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
      echo "Skipping installation. Please install sccache manually:"
      echo "  cargo install sccache"
      echo "  or"
      echo "  brew install sccache (macOS)"
      echo "  or"
      echo "  sudo apt-get install -y sccache (Debian/Ubuntu)"
      echo "  or"
      echo "  sudo dnf install -y sccache (RHEL/Fedora)"
      echo "  or"
      echo "  sudo pacman -S sccache (Arch)"
      exit 0
    fi
  fi

  if has_cmd cargo; then
    echo "Installing via cargo..."
    cargo install sccache --locked
  elif has_cmd brew; then
    echo "Installing via Homebrew..."
    brew install sccache
  elif has_cmd apt-get; then
    echo "Installing via apt-get..."
    sudo apt-get update -y && sudo apt-get install -y sccache
  elif has_cmd dnf; then
    echo "Installing via dnf..."
    sudo dnf install -y sccache
  elif has_cmd pacman; then
    echo "Installing via pacman..."
    sudo pacman -S --noconfirm sccache
  else
    echo "No supported package manager detected. Install manually:"
    echo "  cargo install sccache"
    exit 1
  fi
}

if ! has_cmd sccache; then
  install_sccache
fi

echo "Configuring sccache..."

export SCCACHE_CACHE_SIZE="${SCCACHE_CACHE_SIZE:-10G}"
export SCCACHE_STATS="${SCCACHE_STATS:-1}"

# Set RUSTC_WRAPPER to sccache for this shell session.
SCCACHE_BIN="$(command -v sccache)"
if [[ -z "${SCCACHE_BIN}" ]]; then
  echo "Unexpected: sccache still not on PATH after install. Check your environment."
  exit 1
fi
export RUSTC_WRAPPER="${SCCACHE_BIN}"

echo "sccache version: $(sccache --version || echo 'unknown')"
echo "Current cache stats:"
sccache -s || true

# If script not sourced, remind user about persistence.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo
  echo "Environment variables exported for this process only."
  echo "To persist, add to your shell profile (e.g., ~/.bashrc or ~/.zshrc):"
  echo '  export RUSTC_WRAPPER="$(command -v sccache 2>/dev/null || echo "")"'
  echo '  export SCCACHE_CACHE_SIZE="10G"'
  # echo '  export SCCACHE_DIR="$HOME/.cache/sccache"'
  echo '  export SCCACHE_STATS="1"'
fi

echo "sccache is configured."

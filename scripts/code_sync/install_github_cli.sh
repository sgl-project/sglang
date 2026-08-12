#!/bin/bash
set -euo pipefail

# Check if gh is installed before attempting to install it
if ! command -v gh >/dev/null 2>&1; then
    echo "GitHub CLI not found. Installing now..."

    if ! command -v wget >/dev/null 2>&1; then
        apt-get update
        apt-get install -y wget
    fi

    install -d -m 755 /etc/apt/keyrings /etc/apt/sources.list.d
    keyring_tmp=$(mktemp)
    trap 'rm -f "$keyring_tmp"' EXIT

    wget -nv -O "$keyring_tmp" \
        https://cli.github.com/packages/githubcli-archive-keyring.gpg
    install -m 0644 "$keyring_tmp" \
        /etc/apt/keyrings/githubcli-archive-keyring.gpg
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg

    printf '%s\n' \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        >/etc/apt/sources.list.d/github-cli.list

    apt-get update
    apt-get install -y gh
else
    echo "GitHub CLI is already installed. Skipping installation."
fi

#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# sccache management script
# This script handles sccache installation, environment setup, and statistics display

SCCACHE_VERSION="v0.8.2"


usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    install         Install sccache binary (requires ARCH_ALT environment variable)
    show-stats      Display sccache statistics with optional build name
    help            Show this help message

Environment variables:
    USE_SCCACHE             Set to 'true' to enable sccache
    SCCACHE_BUCKET          S3 bucket name (fallback if not passed as parameter)
    SCCACHE_REGION          S3 region (fallback if not passed as parameter)
    ARCH                    Architecture for S3 key prefix (fallback if not passed as parameter)
    ARCH_ALT                Alternative architecture name for downloads (e.g., x86_64, aarch64)

Examples:
    # Install sccache (requires ARCH_ALT to be set)
    ARCH_ALT=x86_64 $0 install
    # Show stats with build name
    $0 show-stats "UCX"
EOF
}

install_sccache() {
    if [ -z "${ARCH_ALT:-}" ]; then
        echo "Error: ARCH_ALT environment variable is required for sccache installation"
        exit 1
    fi
    echo "Installing sccache ${SCCACHE_VERSION} for architecture ${ARCH_ALT}..."
    # Download and install sccache
    wget --tries=3 --waitretry=5 \
        "https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/sccache-${SCCACHE_VERSION}-${ARCH_ALT}-unknown-linux-musl.tar.gz"
    tar -xzf "sccache-${SCCACHE_VERSION}-${ARCH_ALT}-unknown-linux-musl.tar.gz"
    mv "sccache-${SCCACHE_VERSION}-${ARCH_ALT}-unknown-linux-musl/sccache" /usr/local/bin/
    # Cleanup
    rm -rf sccache*
    echo "sccache installed successfully"
}

show_stats() {
    if command -v sccache >/dev/null 2>&1; then
        echo "=== sccache statistics AFTER $1 ==="
        sccache --show-stats
    else
        echo "sccache is not available"
    fi
}

main() {
    case "${1:-help}" in
        install)
            install_sccache
            ;;
        generate-env)
            shift  # Remove the command from arguments
            generate_env_file "$@"  # Pass all remaining arguments
            ;;
        show-stats)
            shift  # Remove the command from arguments
            show_stats "$@"  # Pass all remaining arguments
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            echo "Unknown command: $1"
            usage
            exit 1
            ;;
    esac
}

main "$@"

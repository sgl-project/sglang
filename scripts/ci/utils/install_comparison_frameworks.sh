#!/usr/bin/env bash
set -euo pipefail

FRAMEWORK="${1:-}"

if [[ -z "$FRAMEWORK" ]]; then
    echo "Usage: $0 <framework>"
    echo "  Supported: vllm-omni, lightx2v"
    exit 1
fi

case "$FRAMEWORK" in
    vllm-omni)
        echo "Installing vllm-omni ..."
        pip install vllm --no-build-isolation 2>&1
        echo "vllm-omni installed successfully."
        ;;
    lightx2v)
        echo "Installing lightx2v ..."
        pip install lightx2v 2>&1
        echo "lightx2v installed successfully."
        ;;
    *)
        echo "ERROR: Unknown framework '$FRAMEWORK'"
        echo "  Supported: vllm-omni, lightx2v"
        exit 1
        ;;
esac

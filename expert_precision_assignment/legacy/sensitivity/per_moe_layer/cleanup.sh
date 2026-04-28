#!/bin/bash
# Clean up intermediate sweep artifacts after summary.json is verified.
# Keeps: summary.json, bf16_baseline.json
# Removes: per-layer JSONs, server logs, generated configs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results}"

if [[ ! -f "${OUT_DIR}/summary.json" ]]; then
    echo "No summary.json found in ${OUT_DIR} — nothing to clean."
    exit 1
fi

echo "Cleaning sweep artifacts in ${OUT_DIR}..."

# Per-layer result JSONs
count=$(ls "${OUT_DIR}"/layer*.json 2>/dev/null | wc -l)
rm -f "${OUT_DIR}"/layer*.json
echo "  Removed ${count} per-layer JSONs"

# Server logs
count=$(ls "${OUT_DIR}"/*.server.log 2>/dev/null | wc -l)
rm -f "${OUT_DIR}"/*.server.log
echo "  Removed ${count} server logs"

# Per-GPU logs
count=$(ls "${OUT_DIR}"/gpu*.log 2>/dev/null | wc -l)
rm -f "${OUT_DIR}"/gpu*.log
echo "  Removed ${count} GPU logs"

# Generated configs
if [[ -d "${OUT_DIR}/configs" ]]; then
    count=$(find "${OUT_DIR}/configs" -type f | wc -l)
    rm -rf "${OUT_DIR}/configs"
    echo "  Removed configs/ (${count} files)"
fi

echo "Done. Kept: summary.json, bf16_baseline.json"

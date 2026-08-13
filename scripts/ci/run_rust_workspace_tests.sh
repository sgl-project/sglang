#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../rust"
# The tests themselves take ~1s; the ceiling covers a cold cargo cache, which
# codegens the whole dependency graph first (clippy only leaves metadata).
timeout 900 cargo test --workspace

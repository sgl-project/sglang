#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../rust"
# Tests take ~1s; the ceiling is for a cold cache codegenning the dep graph.
timeout 900 cargo test --workspace

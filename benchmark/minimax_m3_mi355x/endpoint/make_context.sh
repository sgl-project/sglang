#!/bin/bash
# Assemble the docker build context for the M3 engine image: pure-Python overlay of this branch on the pinned ROCm base.
# Usage: bash make_context.sh <ctxdir> [git ref, default HEAD]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd); BC=${1:?build context dir}; REF=${2:-HEAD}
rm -rf "$BC"; mkdir -p "$BC"
git -C "$REPO" archive "$REF" python/sglang | gzip > "$BC/sglang_python.tar.gz"
git -C "$REPO" archive "$REF" benchmark/minimax_m3_mi355x | tar -x -C "$BC"
cp "$HERE/../aiter_flydsl_xcd_swizzle_fix.patch" "$BC/aiter.patch"
cp "$HERE/../tuned_fmoe_m3_gfx950.csv" "$BC/minimax_m3_gfx950_perf_tuned_fmoe.csv"
cp "$HERE/Dockerfile" "$BC/Dockerfile"; cp "$HERE/build_on_cpu_vm.sh" "$BC/build_on_cpu_vm.sh"
git -C "$REPO" rev-parse "$REF" > "$BC/SGLANG_COMMIT"
du -sh "$BC"; echo "context: $BC (sglang $(cat "$BC/SGLANG_COMMIT"))"

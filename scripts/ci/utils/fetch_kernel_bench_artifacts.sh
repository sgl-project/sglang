#!/usr/bin/env bash
# Download JIT kernel benchmark artifacts from recent completed main-branch runs
# of a workflow into <out>/<run_id>/. Runs without a matching artifact are skipped.
#
#   fetch_kernel_bench_artifacts.sh --workflow kernel-bench-nightly.yml \
#       (--name NAME | --pattern GLOB) --limit N --out DIR [--exclude-run ID]
set -euo pipefail

workflow=""
name=""
pattern=""
limit=1
out=""
exclude_run=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workflow) workflow="$2"; shift 2 ;;
    --name) name="$2"; shift 2 ;;
    --pattern) pattern="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --exclude-run) exclude_run="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$workflow" && -n "$out" && ( -n "$name" || -n "$pattern" ) ]] || {
  echo "usage: $0 --workflow W (--name N | --pattern P) --limit N --out DIR" >&2
  exit 2
}

repo="${GITHUB_REPOSITORY:-sgl-project/sglang}"
mkdir -p "$out"
select=(-p "$pattern")
[[ -n "$name" ]] && select=(-n "$name")

# Over-fetch: failed or cancelled runs may have no artifact.
run_ids=$(gh run list -R "$repo" --workflow "$workflow" --branch main --status completed \
  --limit "$((limit * 3))" --json databaseId --jq '.[].databaseId')

found=0
for run_id in $run_ids; do
  [[ $found -ge $limit ]] && break
  [[ "$run_id" == "$exclude_run" ]] && continue
  if gh run download "$run_id" -R "$repo" "${select[@]}" -D "$out/$run_id" 2>/dev/null; then
    found=$((found + 1))
  else
    rm -rf "${out:?}/$run_id"
  fi
done
echo "downloaded from $found run(s) of $workflow into $out"

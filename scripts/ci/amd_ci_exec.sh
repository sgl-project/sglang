#!/bin/bash
set -euo pipefail

WORKDIR="/sglang-checkout/test/srt"
declare -A ENV_MAP=(
  [SGLANG_AMD_CI]=1
  [SGLANG_IS_IN_CI]=1
  [SGLANG_USE_AITER]=1
)

# Parse -w/--workdir and -e ENV=VAL
while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--workdir)
      WORKDIR="$2"
      shift 2
      ;;
    -e)
      IFS="=" read -r key val <<< "$2"
      ENV_MAP["$key"]="$val"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

# Build final ENV_ARGS
ENV_ARGS=()
for key in "${!ENV_MAP[@]}"; do
  ENV_ARGS+=("-e" "$key=${ENV_MAP[$key]}")
done

# Run docker exec
docker exec \
  -w "$WORKDIR" \
  "${ENV_ARGS[@]}" \
  ci_sglang "$@"

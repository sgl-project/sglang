#!/bin/bash
set -euo pipefail

# Default working directory
WORKDIR="/sglang-checkout/test/srt"
ENV_ARGS=(
  -e SGLANG_AMD_CI=1
  -e SGLANG_IS_IN_CI=1
  -e SGLANG_AITER_MOE=1
)

# Parse optional -w/--workdir and -e ENV=VAL flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--workdir)
      WORKDIR="$2"
      shift 2
      ;;
    -e)
      ENV_ARGS+=("-e" "$2")
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

# Run docker exec
docker exec \
  -w "$WORKDIR" \
  "${ENV_ARGS[@]}" \
  ci_sglang "$@"

#!/bin/bash
set -euo pipefail

# Detect GPU family from hostname (e.g., linux-mi35x-gpu-1-xxxxx-runner-zzzzz)
HOSTNAME_VALUE=$(hostname)
GPU_FAMILY=""

# Host names look like: linux-mi35x-gpu-1-xxxxx-runner-zzzzz
if [[ "${HOSTNAME_VALUE}" =~ ^linux-(mi[0-9]+[a-z]*)-gpu-[0-9]+ ]]; then
  GPU_FAMILY="${BASH_REMATCH[1]}"
  echo "Detected GPU family from hostname: ${GPU_FAMILY}"
else
  echo "Warning: could not parse GPU family from '${HOSTNAME_VALUE}'"
fi

WORKDIR="/sglang-checkout/test/srt"
declare -A ENV_MAP=(
  [SGLANG_IS_IN_CI_AMD]=1
  [SGLANG_IS_IN_CI]=1
  [SGLANG_USE_AITER]=1
)

# Conditionally add GPU_ARCHS only for mi35x
if [[ "${GPU_FAMILY}" == "mi35x" ]]; then
  ENV_MAP[GPU_ARCHS]="gfx950"
fi

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

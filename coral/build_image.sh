#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-coral/sglang:v0.5.22-q6-coral}

docker build -f "$ROOT/coral/Dockerfile" -t "$IMAGE" "$ROOT/coral"
docker image inspect "$IMAGE" >/dev/null
echo "built $IMAGE"

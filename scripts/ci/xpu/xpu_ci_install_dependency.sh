#!/bin/bash
# Install the checked-out sglang into the running ci_sglang_xpu container.
# Usage: xpu_ci_install_dependency.sh [extra pip packages...]
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-ci_sglang_xpu}"
PY=/opt/venv/bin/python3

# Bridge fix: install libssl-dev in the running container so the
# JIT build of hicache_hash_cpp (needs <openssl/sha.h>) succeeds.
# Can be dropped once intel/sglang-dev:latest is rebuilt from a
# docker/xpu.Dockerfile that already ships libssl-dev.
docker exec -e DEBIAN_FRONTEND=noninteractive "${CONTAINER_NAME}" apt-get update
docker exec -e DEBIAN_FRONTEND=noninteractive "${CONTAINER_NAME}" apt-get install -y libssl-dev
# Pin the image's torch so test deps (e.g. lmcache) cannot swap in a CUDA build.
docker exec "${CONTAINER_NAME}" bash -c "uv pip freeze --python ${PY} | grep -iE '^(torch|torchvision|torchaudio|triton-xpu)==' > /tmp/image-pins.txt"
docker exec "${CONTAINER_NAME}" uv pip install --python "${PY}" -c /tmp/image-pins.txt pytest expecttest ray huggingface_hub tabulate "lmcache>=0.3.9" "$@"
docker exec "${CONTAINER_NAME}" uv pip uninstall --python "${PY}" flashinfer-python sgl-kernel sglang
docker exec "${CONTAINER_NAME}" cp /sglang-checkout/python/pyproject_xpu.toml /sglang-checkout/python/pyproject.toml
# Fetch tags so setuptools_scm resolves a real version instead of
# falling back to 0.0.0 on a shallow/tag-less checkout.
docker exec -w /sglang-checkout "${CONTAINER_NAME}" git fetch origin '+refs/tags/*:refs/tags/*' --force
docker exec -w /sglang-checkout/python "${CONTAINER_NAME}" uv pip install --python "${PY}" --no-cache .
docker exec "${CONTAINER_NAME}" uv pip install --python "${PY}" --no-cache --no-deps xgrammar==0.1.33
# HF rate-limits /whoami-v2 hard; HF_TOKEN in the container env still authenticates downloads.
for attempt in 1 2 3; do
  docker exec "${CONTAINER_NAME}" /bin/bash -c '/opt/venv/bin/hf auth login --token "${HF_TOKEN}"' && break
  echo "::warning::hf auth login failed (attempt ${attempt}/3)"
  if [[ ${attempt} -lt 3 ]]; then sleep $(( 20 * attempt + RANDOM % 15 )); fi
done

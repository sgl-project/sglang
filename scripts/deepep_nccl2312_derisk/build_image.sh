#!/bin/bash
# Build sglang PR 48678 (EFA-in-main-image) docker image, amd64, on EC2 build VM.
# Covers R4 (removed apt packages break build) and produces the image for R5/R6.
set -euxo pipefail

PR_SHA=831272f3f02da9f3a4433769aeed9c1d88b570c0

cd "$HOME"
if [ ! -d sglang-pr ]; then
  git clone --filter=blob:none https://github.com/sgl-project/sglang.git sglang-pr
fi
cd sglang-pr
git fetch origin "pull/48678/head:pr48678" || git fetch https://github.com/sgl-project/sglang.git "pull/48678/head:pr48678"
git checkout "$PR_SHA"

# Same invocation as .github/workflows/_docker-build-and-publish.yml, plus
# BRANCH_TYPE=local so the image installs THIS checkout of sglang.
docker buildx build \
  --target framework_final \
  --build-arg CUDA_VERSION=13.0.3 \
  --build-arg BUILD_TYPE=all \
  --build-arg INSTALL_FLASHINFER_JIT_CACHE=1 \
  --build-arg BRANCH_TYPE=local \
  --load \
  -f docker/Dockerfile \
  -t sglang-pr48678:latest \
  . 2>&1 | tee /tmp/build_pr48678.log

echo "BUILD_OK"
docker images | grep sglang-pr48678

# Save for transfer to the GPU cluster
docker save sglang-pr48678:latest | gzip > /tmp/sglang-pr48678.tar.gz
ls -lh /tmp/sglang-pr48678.tar.gz
echo "SAVE_OK"

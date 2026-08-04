#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?set AWS_REGION}"
: "${ECR_REPOSITORY:?set ECR_REPOSITORY, for example leap-world/minwm-realtime}"
: "${PYTHON_IMAGE_DIGEST:?set PYTHON_IMAGE_DIGEST to an immutable image reference}"
: "${GPU_IMAGE_DIGEST:?set GPU_IMAGE_DIGEST to an immutable image reference}"

for IMAGE in "${PYTHON_IMAGE_DIGEST}" "${GPU_IMAGE_DIGEST}"; do
  if ! [[ "${IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
    echo "base image must be pinned by sha256 digest: ${IMAGE}" >&2
    exit 1
  fi
done

ROOT="$(git rev-parse --show-toplevel)"
GIT_SHA="$(git -C "${ROOT}" rev-parse HEAD)"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
REPOSITORY_URI="${REGISTRY}/${ECR_REPOSITORY}"

aws ecr describe-repositories \
  --region "${AWS_REGION}" \
  --repository-names "${ECR_REPOSITORY}" >/dev/null

aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${REGISTRY}"

for ROLE in gateway coordinator denoiser vae; do
  TAG="${ROLE}-${GIT_SHA}"
  docker buildx build \
    --platform linux/amd64 \
    --file "${ROOT}/benchmark/minwm_realtime_async_vae/docker/Dockerfile" \
    --target "${ROLE}" \
    --build-arg "PYTHON_IMAGE=${PYTHON_IMAGE_DIGEST}" \
    --build-arg "GPU_IMAGE=${GPU_IMAGE_DIGEST}" \
    --label "org.opencontainers.image.revision=${GIT_SHA}" \
    --tag "${REPOSITORY_URI}:${TAG}" \
    --push \
    "${ROOT}"
done

OUTPUT="${ROOT}/benchmark/minwm_realtime_async_vae/.env.images"
: >"${OUTPUT}"
for ROLE in gateway coordinator denoiser vae; do
  TAG="${ROLE}-${GIT_SHA}"
  DIGEST="$(aws ecr describe-images \
    --region "${AWS_REGION}" \
    --repository-name "${ECR_REPOSITORY}" \
    --image-ids imageTag="${TAG}" \
    --query 'imageDetails[0].imageDigest' \
    --output text)"
  printf '%s_IMAGE_DIGEST=%s@%s\n' \
    "$(printf '%s' "${ROLE}" | tr '[:lower:]' '[:upper:]')" \
    "${REPOSITORY_URI}" "${DIGEST}" >>"${OUTPUT}"
done
printf 'Wrote immutable image references to %s\n' "${OUTPUT}"

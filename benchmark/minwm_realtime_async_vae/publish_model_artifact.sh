#!/usr/bin/env bash
set -euo pipefail

: "${DENOISER_IMAGE_DIGEST:?set DENOISER_IMAGE_DIGEST}"
: "${MODEL_ARTIFACT_PUBLISHER_ROLE_ARN:?set MODEL_ARTIFACT_PUBLISHER_ROLE_ARN}"
: "${MODEL_ID:?set MODEL_ID}"
: "${MODEL_ARTIFACT_REVISION:?set MODEL_ARTIFACT_REVISION}"
: "${SOURCE_CHECKPOINT_PATH:?set SOURCE_CHECKPOINT_PATH}"
: "${SOURCE_CHECKPOINT_URI:?set SOURCE_CHECKPOINT_URI}"
: "${SOURCE_CHECKPOINT_VERSION_ID:?set SOURCE_CHECKPOINT_VERSION_ID}"

if ! [[ "${DENOISER_IMAGE_DIGEST}" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "denoiser image must be pinned by sha256 digest" >&2
  exit 1
fi
if ! [[ "${MODEL_ARTIFACT_REVISION}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "MODEL_ARTIFACT_REVISION is not immutable-path safe" >&2
  exit 1
fi
if ! [[ "${MODEL_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "MODEL_ID is not immutable-path safe" >&2
  exit 1
fi
if [[ "${SOURCE_CHECKPOINT_PATH}" != /checkpoint-east/* ]]; then
  echo "SOURCE_CHECKPOINT_PATH must use the read-only east S3 mount" >&2
  exit 1
fi

ROOT="$(git rev-parse --show-toplevel)"
K8S_DIR="${ROOT}/benchmark/minwm_realtime_async_vae/k8s"
CONVERTER_GIT_SHA="$(git -C "${ROOT}" rev-parse HEAD)"
RENDERED="$(mktemp)"
trap 'rm -f "${RENDERED}" "${RENDERED}.bak"' EXIT

kubectl get job/minwm-model-artifact-publisher -n minwm-realtime >/dev/null 2>&1 && {
  echo "publisher Job already exists; delete it explicitly after inspecting its state" >&2
  exit 1
}

{
  cat "${K8S_DIR}/namespace.yaml"
  printf '%s\n' '---'
  cat "${K8S_DIR}/west-s3-volume.yaml"
  printf '%s\n' '---'
  cat "${K8S_DIR}/east-s3-source-volume.yaml"
  printf '%s\n' '---'
  cat "${K8S_DIR}/model-artifact-publisher.yaml"
} >"${RENDERED}"
sed -i.bak \
  -e "s|REPLACE_WITH_DENOISER_IMAGE_DIGEST|${DENOISER_IMAGE_DIGEST}|g" \
  -e "s|REPLACE_WITH_MODEL_ARTIFACT_PUBLISHER_ROLE_ARN|${MODEL_ARTIFACT_PUBLISHER_ROLE_ARN}|g" \
  -e "s|REPLACE_WITH_MODEL_ID|${MODEL_ID}|g" \
  -e "s|REPLACE_WITH_MODEL_ARTIFACT_REVISION|${MODEL_ARTIFACT_REVISION}|g" \
  -e "s|REPLACE_WITH_SOURCE_CHECKPOINT_PATH|${SOURCE_CHECKPOINT_PATH}|g" \
  -e "s|REPLACE_WITH_SOURCE_CHECKPOINT_URI|${SOURCE_CHECKPOINT_URI}|g" \
  -e "s|REPLACE_WITH_SOURCE_CHECKPOINT_VERSION_ID|${SOURCE_CHECKPOINT_VERSION_ID}|g" \
  -e "s|REPLACE_WITH_CONVERTER_GIT_SHA|${CONVERTER_GIT_SHA}|g" \
  "${RENDERED}"
rm -f "${RENDERED}.bak"

kubectl apply --server-side --field-manager=minwm-model-publisher -f "${RENDERED}"
kubectl wait --for=condition=complete --timeout=10800s \
  job/minwm-model-artifact-publisher -n minwm-realtime
aws s3api head-object \
  --region us-west-2 \
  --bucket leap-world-us-west-2 \
  --key "world-model/minwm/serving-artifacts/${MODEL_ID}/${MODEL_ARTIFACT_REVISION}/model/_READY" \
  >/dev/null

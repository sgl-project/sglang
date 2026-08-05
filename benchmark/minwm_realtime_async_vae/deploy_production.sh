#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?set AWS_REGION}"
: "${COORDINATOR_TABLE:?set COORDINATOR_TABLE}"
: "${GATEWAY_IMAGE_DIGEST:?set GATEWAY_IMAGE_DIGEST}"
: "${COORDINATOR_IMAGE_DIGEST:?set COORDINATOR_IMAGE_DIGEST}"
: "${DENOISER_IMAGE_DIGEST:?set DENOISER_IMAGE_DIGEST}"
: "${VAE_IMAGE_DIGEST:?set VAE_IMAGE_DIGEST}"
: "${ADOT_IMAGE_DIGEST:?set ADOT_IMAGE_DIGEST}"
: "${GATEWAY_ROLE_ARN:?set GATEWAY_ROLE_ARN}"
: "${COORDINATOR_ROLE_ARN:?set COORDINATOR_ROLE_ARN}"
: "${ADOT_ROLE_ARN:?set ADOT_ROLE_ARN}"
: "${MODEL_ARTIFACT_REVISION:?set MODEL_ARTIFACT_REVISION}"

TRACE_LOG_GROUP="${TRACE_LOG_GROUP:-/aws/eks/minwm/realtime-traces}"
GPU_SCALE_UP_SCHEDULE="${GPU_SCALE_UP_SCHEDULE:-0 9 * * *}"
GPU_SCALE_DOWN_SCHEDULE="${GPU_SCALE_DOWN_SCHEDULE:-0 23 * * *}"
GPU_SCALE_TIME_ZONE="${GPU_SCALE_TIME_ZONE:-Asia/Shanghai}"
GPU_SCALE_UP_SUSPEND="${GPU_SCALE_UP_SUSPEND:-false}"
GPU_SCALE_DOWN_SUSPEND="${GPU_SCALE_DOWN_SUSPEND:-false}"
DENOISER_BASE_REPLICAS="${DENOISER_BASE_REPLICAS:-1}"
VAE_BASE_REPLICAS="${VAE_BASE_REPLICAS:-1}"
DENOISER_PEAK_REPLICAS="${DENOISER_PEAK_REPLICAS:-1}"
VAE_PEAK_REPLICAS="${VAE_PEAK_REPLICAS:-1}"

if ! [[ "${MODEL_ARTIFACT_REVISION}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "MODEL_ARTIFACT_REVISION is not immutable-path safe" >&2
  exit 1
fi

aws s3api head-object \
  --region us-west-2 \
  --bucket leap-world-us-west-2 \
  --key "world-model/minwm/serving-artifacts/wan22-5b-stage3-dmd-30-gs1800/${MODEL_ARTIFACT_REVISION}/model/_READY" \
  >/dev/null

for REPLICAS in \
  "${DENOISER_BASE_REPLICAS}" \
  "${VAE_BASE_REPLICAS}" \
  "${DENOISER_PEAK_REPLICAS}" \
  "${VAE_PEAK_REPLICAS}"; do
  if ! [[ "${REPLICAS}" =~ ^[1-8]$ ]]; then
    echo "GPU peak replicas must be between 1 and 8" >&2
    exit 1
  fi
done
if (( DENOISER_BASE_REPLICAS > DENOISER_PEAK_REPLICAS )); then
  echo "Denoiser base replicas cannot exceed peak replicas" >&2
  exit 1
fi
if (( VAE_BASE_REPLICAS > VAE_PEAK_REPLICAS )); then
  echo "VAE base replicas cannot exceed peak replicas" >&2
  exit 1
fi
for SUSPEND in "${GPU_SCALE_UP_SUSPEND}" "${GPU_SCALE_DOWN_SUSPEND}"; do
  if [[ "${SUSPEND}" != "true" && "${SUSPEND}" != "false" ]]; then
    echo "GPU schedule suspend flags must be true or false" >&2
    exit 1
  fi
done
ROOT="$(git rev-parse --show-toplevel)"
K8S_DIR="${ROOT}/benchmark/minwm_realtime_async_vae/k8s"
RENDERED="$(mktemp)"
trap 'rm -f "${RENDERED}"' EXIT

aws dynamodb describe-table \
  --region "${AWS_REGION}" \
  --table-name "${COORDINATOR_TABLE}" >/dev/null
RETENTION="$(aws logs describe-log-groups \
  --region "${AWS_REGION}" \
  --log-group-name-prefix "${TRACE_LOG_GROUP}" \
  --query "logGroups[?logGroupName=='${TRACE_LOG_GROUP}'].retentionInDays | [0]" \
  --output text)"
if [[ "${RETENTION}" != "5" ]]; then
  echo "Trace log group must exist with retentionInDays=5" >&2
  exit 1
fi

for IMAGE in \
  "${GATEWAY_IMAGE_DIGEST}" \
  "${COORDINATOR_IMAGE_DIGEST}" \
  "${DENOISER_IMAGE_DIGEST}" \
  "${VAE_IMAGE_DIGEST}" \
  "${ADOT_IMAGE_DIGEST}"; do
  if ! [[ "${IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
    echo "all production images must be pinned by sha256 digest: ${IMAGE}" >&2
    exit 1
  fi
done

kubectl kustomize "${K8S_DIR}" >"${RENDERED}"
sed -i.bak \
  -e "s|REPLACE_WITH_GATEWAY_IMAGE_DIGEST|${GATEWAY_IMAGE_DIGEST}|g" \
  -e "s|REPLACE_WITH_COORDINATOR_IMAGE_DIGEST|${COORDINATOR_IMAGE_DIGEST}|g" \
  -e "s|REPLACE_WITH_DENOISER_IMAGE_DIGEST|${DENOISER_IMAGE_DIGEST}|g" \
  -e "s|REPLACE_WITH_VAE_IMAGE_DIGEST|${VAE_IMAGE_DIGEST}|g" \
  -e "s|REPLACE_WITH_ADOT_IMAGE_DIGEST|${ADOT_IMAGE_DIGEST}|g" \
  -e "s|REPLACE_WITH_GATEWAY_ROLE_ARN|${GATEWAY_ROLE_ARN}|g" \
  -e "s|REPLACE_WITH_COORDINATOR_ROLE_ARN|${COORDINATOR_ROLE_ARN}|g" \
  -e "s|REPLACE_WITH_ADOT_ROLE_ARN|${ADOT_ROLE_ARN}|g" \
  -e "s|REPLACE_WITH_COORDINATOR_TABLE|${COORDINATOR_TABLE}|g" \
  -e "s|REPLACE_WITH_AWS_REGION|${AWS_REGION}|g" \
  -e "s|REPLACE_WITH_TRACE_LOG_GROUP|${TRACE_LOG_GROUP}|g" \
  -e "s|REPLACE_WITH_GPU_SCALE_UP_SCHEDULE|${GPU_SCALE_UP_SCHEDULE}|g" \
  -e "s|REPLACE_WITH_GPU_SCALE_DOWN_SCHEDULE|${GPU_SCALE_DOWN_SCHEDULE}|g" \
  -e "s|REPLACE_WITH_GPU_SCALE_UP_SUSPEND|${GPU_SCALE_UP_SUSPEND}|g" \
  -e "s|REPLACE_WITH_GPU_SCALE_DOWN_SUSPEND|${GPU_SCALE_DOWN_SUSPEND}|g" \
  -e "s|REPLACE_WITH_DENOISER_BASE_REPLICAS|${DENOISER_BASE_REPLICAS}|g" \
  -e "s|REPLACE_WITH_VAE_BASE_REPLICAS|${VAE_BASE_REPLICAS}|g" \
  -e "s|REPLACE_WITH_GPU_SCALE_TIME_ZONE|${GPU_SCALE_TIME_ZONE}|g" \
  -e "s|REPLACE_WITH_DENOISER_PEAK_REPLICAS|${DENOISER_PEAK_REPLICAS}|g" \
  -e "s|REPLACE_WITH_VAE_PEAK_REPLICAS|${VAE_PEAK_REPLICAS}|g" \
  -e "s|REPLACE_WITH_MODEL_ARTIFACT_REVISION|${MODEL_ARTIFACT_REVISION}|g" \
  "${RENDERED}"
rm -f "${RENDERED}.bak"

if rg -n 'REPLACE_WITH_' "${RENDERED}"; then
  echo "unresolved production manifest placeholders" >&2
  exit 1
fi
kubectl apply --server-side --field-manager=minwm-production -f "${RENDERED}"

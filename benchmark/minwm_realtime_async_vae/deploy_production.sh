#!/usr/bin/env bash
set -Eeuo pipefail

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
: "${MODEL_ID:?set MODEL_ID}"
: "${MODEL_ARTIFACT_REVISION:?set MODEL_ARTIFACT_REVISION}"

TRACE_LOG_GROUP="${TRACE_LOG_GROUP:-/aws/eks/minwm/realtime-traces}"
GPU_SCALE_UP_SCHEDULE="${GPU_SCALE_UP_SCHEDULE:-0 9 * * *}"
GPU_SCALE_DOWN_SCHEDULE="${GPU_SCALE_DOWN_SCHEDULE:-0 23 * * *}"
GPU_SCALE_TIME_ZONE="${GPU_SCALE_TIME_ZONE:-Asia/Shanghai}"
GPU_SCALE_UP_SUSPEND="${GPU_SCALE_UP_SUSPEND:-false}"
GPU_SCALE_DOWN_SUSPEND="${GPU_SCALE_DOWN_SUSPEND:-false}"
GPU_EVENT_SCALER_SUSPEND="${GPU_EVENT_SCALER_SUSPEND:-false}"
DENOISER_BASE_REPLICAS="${DENOISER_BASE_REPLICAS:-1}"
VAE_BASE_REPLICAS="${VAE_BASE_REPLICAS:-1}"
DENOISER_PEAK_REPLICAS="${DENOISER_PEAK_REPLICAS:-1}"
VAE_PEAK_REPLICAS="${VAE_PEAK_REPLICAS:-1}"
DENOISER_RESTART_BATCH_SIZE="${DENOISER_RESTART_BATCH_SIZE:-2}"
DENOISER_NODEPOOL="${DENOISER_NODEPOOL:-}"
NAMESPACE="minwm-realtime"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-45m}"

if ! [[ "${MODEL_ARTIFACT_REVISION}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "MODEL_ARTIFACT_REVISION is not immutable-path safe" >&2
  exit 1
fi
if ! [[ "${MODEL_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "MODEL_ID is not immutable-path safe" >&2
  exit 1
fi

aws s3api head-object \
  --region us-west-2 \
  --bucket leap-world-us-west-2 \
  --key "world-model/minwm/serving-artifacts/${MODEL_ID}/${MODEL_ARTIFACT_REVISION}/model/_READY" \
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
if ! [[ "${DENOISER_RESTART_BATCH_SIZE}" =~ ^[1-8]$ ]]; then
  echo "DENOISER_RESTART_BATCH_SIZE must be between 1 and 8" >&2
  exit 1
fi
if (( DENOISER_BASE_REPLICAS > DENOISER_PEAK_REPLICAS )); then
  echo "Denoiser base replicas cannot exceed peak replicas" >&2
  exit 1
fi
if (( VAE_BASE_REPLICAS > VAE_PEAK_REPLICAS )); then
  echo "VAE base replicas cannot exceed peak replicas" >&2
  exit 1
fi
if [[ -z "${DENOISER_NODEPOOL}" ]]; then
  if (( DENOISER_PEAK_REPLICAS == 1 )); then
    DENOISER_NODEPOOL=minwm-async-denoiser-h100
  else
    DENOISER_NODEPOOL=minwm-async-denoiser-h100-8x
  fi
fi
if [[ "${DENOISER_NODEPOOL}" != "minwm-async-denoiser-h100" && \
      "${DENOISER_NODEPOOL}" != "minwm-async-denoiser-h100-8x" ]]; then
  echo "DENOISER_NODEPOOL must select an approved H100 Spot pool" >&2
  exit 1
fi
for SUSPEND in \
  "${GPU_SCALE_UP_SUSPEND}" \
  "${GPU_SCALE_DOWN_SUSPEND}" \
  "${GPU_EVENT_SCALER_SUSPEND}"; do
  if [[ "${SUSPEND}" != "true" && "${SUSPEND}" != "false" ]]; then
    echo "GPU schedule suspend flags must be true or false" >&2
    exit 1
  fi
done
if [[ "${GPU_EVENT_SCALER_SUSPEND}" == "true" ]]; then
  GPU_EVENT_SCALER_REPLICAS=0
else
  GPU_EVENT_SCALER_REPLICAS=1
fi
ROOT="$(git rev-parse --show-toplevel)"
K8S_DIR="${ROOT}/benchmark/minwm_realtime_async_vae/k8s"
RELEASE_STATE_DIR="$(mktemp -d)"
RENDERED="${RELEASE_STATE_DIR}/rendered.yaml"
DRY_RUN_RENDERED="${RELEASE_STATE_DIR}/rendered-dry-run.yaml"
KUSTOMIZED="${RELEASE_STATE_DIR}/kustomized.yaml"
TABLE_STATE="${RELEASE_STATE_DIR}/coordinator-table.json"
TTL_STATE="${RELEASE_STATE_DIR}/coordinator-ttl.json"
RELEASE_APPLIED=0
LEGACY_DENOISER_REPLICAS=""
LEGACY_DENOISER_SCALED_DOWN=0
DENOISER_CONTROLLER_RECREATED=0
DENOISER_TEMPLATE_CHANGED=0
DENOISER_PROTECTED_NODES=""
SNAPSHOT_WORKLOADS=(
  deployment/minwm-realtime-adot
  deployment/minwm-realtime-coordinator
  deployment/minwm-realtime-gateway
  deployment/minwm-realtime-gpu-capacity-scaler
  statefulset/minwm-async-denoiser
  deployment/minwm-async-vae
)

cleanup() {
  unprotect_denoiser_nodes || true
  rm -rf "${RELEASE_STATE_DIR}"
}

wait_for_ondelete_statefulset() {
  local workload="$1"
  local timeout_seconds
  local deadline
  timeout_seconds="$(python3 - "${ROLLOUT_TIMEOUT}" <<'PY'
import re
import sys

match = re.fullmatch(r"([1-9][0-9]*)([smh])", sys.argv[1])
if match is None:
    raise SystemExit("ROLLOUT_TIMEOUT must use s, m, or h")
value = int(match.group(1))
multiplier = {"s": 1, "m": 60, "h": 3600}[match.group(2)]
print(value * multiplier)
PY
)"
  deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS <= deadline )); do
    if kubectl get "${workload}" --namespace "${NAMESPACE}" --output json | \
      python3 -c '
import json
import sys

state = json.load(sys.stdin)
spec = state.get("spec", {})
status = state.get("status", {})
desired_replicas = int(spec.get("replicas") or 0)
ready_replicas = int(status.get("readyReplicas") or 0)
updated_replicas = int(status.get("updatedReplicas") or 0)
generation = int(state.get("metadata", {}).get("generation") or 0)
observed_generation = int(status.get("observedGeneration") or 0)
if (
    observed_generation >= generation
    and updated_replicas >= desired_replicas
    and ready_replicas >= desired_replicas
):
    raise SystemExit(0)
raise SystemExit(1)
'; then
      return 0
    fi
    sleep 5
  done
  kubectl get "${workload}" --namespace "${NAMESPACE}" --output wide >&2
  echo "timed out waiting for parallel OnDelete rollout: ${workload}" >&2
  return 1
}

wait_for_rollout() {
  local workload="$1"
  local strategy=""
  if [[ "${workload}" == statefulset/* ]]; then
    strategy="$(kubectl get "${workload}" --namespace "${NAMESPACE}" \
      --output jsonpath='{.spec.updateStrategy.type}')"
  fi
  if [[ "${strategy}" == "OnDelete" ]]; then
    wait_for_ondelete_statefulset "${workload}"
  else
    kubectl rollout status "${workload}" \
      --namespace "${NAMESPACE}" \
      --timeout "${ROLLOUT_TIMEOUT}"
  fi
}

statefulset_template_hash() {
  python3 -c '
import hashlib, json, sys
document = json.load(sys.stdin)
payload = json.dumps(document["spec"]["template"], sort_keys=True, separators=(",", ":"))
print(hashlib.sha256(payload.encode()).hexdigest())
'
}

protect_denoiser_nodes() {
  local selector="${1:-app.kubernetes.io/name=minwm-async-denoiser}"
  local node
  local nodes
  if [[ -n "${DENOISER_PROTECTED_NODES}" ]]; then
    return 0
  fi
  nodes="$(kubectl get pods --namespace "${NAMESPACE}" \
    --selector "${selector}" \
    --output jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' | \
    sort -u | sed '/^$/d')"
  for node in ${nodes}; do
    kubectl annotate node "${node}" \
      karpenter.sh/do-not-disrupt=true --overwrite >/dev/null
    DENOISER_PROTECTED_NODES+=" ${node}"
  done
  DENOISER_PROTECTED_NODES="${DENOISER_PROTECTED_NODES# }"
}

wait_for_denoiser_batch() {
  local resources=("$@")
  local timeout_seconds
  local deadline
  local resource
  local all_created
  timeout_seconds="$(python3 -c '
import re
import sys

match = re.fullmatch(r"([1-9][0-9]*)([smh])", sys.argv[1])
if match is None:
    raise SystemExit("ROLLOUT_TIMEOUT must use s, m, or h")
value = int(match.group(1))
multiplier = {"s": 1, "m": 60, "h": 3600}[match.group(2)]
print(value * multiplier)
' "${ROLLOUT_TIMEOUT}")"
  deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS <= deadline )); do
    all_created=1
    for resource in "${resources[@]}"; do
      if ! kubectl get "${resource}" --namespace "${NAMESPACE}" >/dev/null 2>&1; then
        all_created=0
        break
      fi
    done
    if (( all_created == 1 )); then
      kubectl wait --namespace "${NAMESPACE}" --for=condition=Ready \
        --timeout "${ROLLOUT_TIMEOUT}" "${resources[@]}"
      return
    fi
    sleep 2
  done
  echo "timed out waiting for Denoiser batch creation: ${resources[*]}" >&2
  return 1
}

restart_statefulset_in_batches() {
  local name="$1"
  local selector="$2"
  local pod
  local pods=()
  local batch=()
  while IFS= read -r pod; do
    [[ -n "${pod}" ]] && pods+=("${pod}")
  done < <(
    kubectl get pods --namespace "${NAMESPACE}" --selector "${selector}" \
      --output name | sort
  )
  if (( ${#pods[@]} == 0 )); then
    return 0
  fi
  for pod in "${pods[@]}"; do
    batch+=("${pod}")
    if (( ${#batch[@]} < DENOISER_RESTART_BATCH_SIZE )); then
      continue
    fi
    kubectl delete --namespace "${NAMESPACE}" --wait=true "${batch[@]}"
    wait_for_denoiser_batch "${batch[@]}"
    batch=()
  done
  if (( ${#batch[@]} > 0 )); then
    kubectl delete --namespace "${NAMESPACE}" --wait=true "${batch[@]}"
    wait_for_denoiser_batch "${batch[@]}"
  fi
}

unprotect_denoiser_nodes() {
  local node
  for node in ${DENOISER_PROTECTED_NODES}; do
    kubectl annotate node "${node}" \
      karpenter.sh/do-not-disrupt- >/dev/null 2>&1 || true
  done
  DENOISER_PROTECTED_NODES=""
}

snapshot_path() {
  local workload="$1"
  echo "${RELEASE_STATE_DIR}/${workload//\//__}.json"
}

snapshot_absent_path() {
  local workload="$1"
  echo "$(snapshot_path "${workload}").absent"
}

snapshot_workload() {
  local workload="$1"
  local target
  local pending
  local raw
  local error
  target="$(snapshot_path "${workload}")"
  pending="${target}.pending"
  raw="${target}.raw"
  error="${target}.error"
  if kubectl get "${workload}" --namespace "${NAMESPACE}" \
    --ignore-not-found --output json \
    >"${raw}" 2>"${error}"; then
    if [[ -s "${raw}" ]]; then
      python3 "${ROOT}/benchmark/minwm_realtime_async_vae/prepare_kubernetes_snapshot.py" \
        <"${raw}" >"${pending}"
      mv "${pending}" "${target}"
      rm -f "$(snapshot_absent_path "${workload}")"
    else
      touch "$(snapshot_absent_path "${workload}")"
    fi
  else
    cat "${error}" >&2
    echo "failed to snapshot ${workload}; refusing to mutate the cluster" >&2
    rm -f "${pending}" "${raw}" "${error}"
    return 1
  fi
  rm -f "${raw}" "${error}"
}

restore_release_snapshot() {
  local status=$?
  if (( $# > 0 )); then
    status="$1"
  fi
  local rollback_failed=0
  local workload
  local snapshot
  trap - ERR INT TERM
  set +e
  echo "production rollout failed; restoring the exact pre-release workload specs" >&2

  if (( RELEASE_APPLIED == 1 )); then
    if (( DENOISER_CONTROLLER_RECREATED == 1 )) && \
       [[ -s "$(snapshot_path statefulset/minwm-async-denoiser)" ]]; then
      kubectl delete statefulset/minwm-async-denoiser \
        --namespace "${NAMESPACE}" --cascade=orphan --wait=true >/dev/null \
        || rollback_failed=1
    fi
    for workload in "${SNAPSHOT_WORKLOADS[@]}"; do
      snapshot="$(snapshot_path "${workload}")"
      if [[ -s "${snapshot}" ]]; then
        kubectl apply --server-side --force-conflicts \
          --field-manager=minwm-production -f "${snapshot}" >/dev/null \
          || rollback_failed=1
      elif [[ -f "$(snapshot_absent_path "${workload}")" ]]; then
        kubectl delete "${workload}" --namespace "${NAMESPACE}" \
          --ignore-not-found --cascade=foreground --wait=true >/dev/null \
          || rollback_failed=1
      else
        echo "missing rollback state for ${workload}" >&2
        rollback_failed=1
      fi
    done

    snapshot="$(snapshot_path deployment/minwm-async-denoiser)"
    if [[ -s "${snapshot}" ]]; then
      kubectl apply --server-side --force-conflicts \
        --field-manager=minwm-production -f "${snapshot}" >/dev/null \
        || rollback_failed=1
    fi

    if [[ -s "$(snapshot_path statefulset/minwm-async-denoiser)" ]]; then
      protect_denoiser_nodes
      restart_statefulset_in_batches minwm-async-denoiser \
        app.kubernetes.io/name=minwm-async-denoiser >/dev/null \
        || rollback_failed=1
    fi

    for workload in "${SNAPSHOT_WORKLOADS[@]}"; do
      snapshot="$(snapshot_path "${workload}")"
      if [[ -s "${snapshot}" ]]; then
        wait_for_rollout "${workload}" >/dev/null || rollback_failed=1
      fi
    done
    if [[ -s "$(snapshot_path deployment/minwm-async-denoiser)" ]]; then
      wait_for_rollout deployment/minwm-async-denoiser >/dev/null \
        || rollback_failed=1
    fi

  fi
  set -e
  if (( rollback_failed == 1 )); then
    echo "automatic rollback did not converge; inspect ${NAMESPACE} immediately" >&2
    exit 70
  fi
  exit "${status}"
}

trap cleanup EXIT
trap restore_release_snapshot ERR
trap 'restore_release_snapshot 130' INT
trap 'restore_release_snapshot 143' TERM

aws dynamodb describe-table \
  --region "${AWS_REGION}" \
  --table-name "${COORDINATOR_TABLE}" >"${TABLE_STATE}"
aws dynamodb describe-time-to-live \
  --region "${AWS_REGION}" \
  --table-name "${COORDINATOR_TABLE}" >"${TTL_STATE}"
python3 "${ROOT}/benchmark/minwm_realtime_async_vae/validate_coordinator_table.py" \
  --table-file "${TABLE_STATE}" --ttl-file "${TTL_STATE}"
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

kubectl kustomize "${K8S_DIR}" >"${KUSTOMIZED}"
python3 "${ROOT}/benchmark/minwm_realtime_async_vae/render_production.py" \
  <"${KUSTOMIZED}" >"${RENDERED}"
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
  -e "s|REPLACE_WITH_GPU_EVENT_SCALER_REPLICAS|${GPU_EVENT_SCALER_REPLICAS}|g" \
  -e "s|REPLACE_WITH_DENOISER_BASE_REPLICAS|${DENOISER_BASE_REPLICAS}|g" \
  -e "s|REPLACE_WITH_VAE_BASE_REPLICAS|${VAE_BASE_REPLICAS}|g" \
  -e "s|REPLACE_WITH_DENOISER_NODEPOOL|${DENOISER_NODEPOOL}|g" \
  -e "s|REPLACE_WITH_GPU_SCALE_TIME_ZONE|${GPU_SCALE_TIME_ZONE}|g" \
  -e "s|REPLACE_WITH_DENOISER_PEAK_REPLICAS|${DENOISER_PEAK_REPLICAS}|g" \
  -e "s|REPLACE_WITH_VAE_PEAK_REPLICAS|${VAE_PEAK_REPLICAS}|g" \
  -e "s|REPLACE_WITH_MODEL_ID|${MODEL_ID}|g" \
  -e "s|REPLACE_WITH_MODEL_ARTIFACT_REVISION|${MODEL_ARTIFACT_REVISION}|g" \
  "${RENDERED}"
rm -f "${RENDERED}.bak"

if rg -n 'REPLACE_WITH_' "${RENDERED}"; then
  echo "unresolved production manifest placeholders" >&2
  exit 1
fi

cp "${RENDERED}" "${DRY_RUN_RENDERED}"
CURRENT_DENOISER_POLICY="$(kubectl get statefulset/minwm-async-denoiser \
  --namespace "${NAMESPACE}" --ignore-not-found \
  --output jsonpath='{.spec.podManagementPolicy}')"
if [[ -n "${CURRENT_DENOISER_POLICY}" && \
      "${CURRENT_DENOISER_POLICY}" != "Parallel" ]]; then
  # podManagementPolicy is immutable. Preflight the remaining release against
  # the live policy, then recreate only this controller after snapshots exist.
  sed -i.bak \
    "s/podManagementPolicy: Parallel/podManagementPolicy: ${CURRENT_DENOISER_POLICY}/" \
    "${DRY_RUN_RENDERED}"
  rm -f "${DRY_RUN_RENDERED}.bak"
fi

kubectl apply --server-side --force-conflicts --dry-run=server \
  --field-manager=minwm-production -f "${DRY_RUN_RENDERED}" >/dev/null

for workload in "${SNAPSHOT_WORKLOADS[@]}"; do
  snapshot_workload "${workload}"
done
snapshot_workload deployment/minwm-async-denoiser
if [[ -s "$(snapshot_path statefulset/minwm-async-denoiser)" ]]; then
  OLD_DENOISER_TEMPLATE_HASH="$(
    statefulset_template_hash <"$(snapshot_path statefulset/minwm-async-denoiser)"
  )"
else
  OLD_DENOISER_TEMPLATE_HASH=""
fi
RELEASE_APPLIED=1

# Kubernetes cannot change a workload's kind in place. Keep the legacy
# Deployment as a rollback target, but scale it to zero before creating the
# StatefulSet so both controllers never compete for the same H100s.
if [[ -s "$(snapshot_path deployment/minwm-async-denoiser)" ]]; then
  LEGACY_DENOISER_REPLICAS="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["spec"].get("replicas", 0))' \
    "$(snapshot_path deployment/minwm-async-denoiser)")"
  LEGACY_DENOISER_REPLICAS="${LEGACY_DENOISER_REPLICAS:-0}"
  kubectl scale deployment/minwm-async-denoiser \
    --namespace "${NAMESPACE}" --replicas 0
  LEGACY_DENOISER_SCALED_DOWN=1
  wait_for_rollout "deployment/minwm-async-denoiser"
fi

if [[ -n "${CURRENT_DENOISER_POLICY}" && \
      "${CURRENT_DENOISER_POLICY}" != "Parallel" ]]; then
  kubectl delete statefulset/minwm-async-denoiser \
    --namespace "${NAMESPACE}" --cascade=orphan --wait=true
  DENOISER_CONTROLLER_RECREATED=1
fi

kubectl apply --server-side --force-conflicts \
  --field-manager=minwm-production -f "${RENDERED}"

NEW_DENOISER_TEMPLATE_HASH="$(
  kubectl get statefulset/minwm-async-denoiser --namespace "${NAMESPACE}" \
    --output json | statefulset_template_hash
)"
if [[ -n "${OLD_DENOISER_TEMPLATE_HASH}" && \
      "${OLD_DENOISER_TEMPLATE_HASH}" != "${NEW_DENOISER_TEMPLATE_HASH}" ]]; then
  DENOISER_TEMPLATE_CHANGED=1
fi
if (( DENOISER_CONTROLLER_RECREATED == 1 || DENOISER_TEMPLATE_CHANGED == 1 )); then
  protect_denoiser_nodes
  restart_statefulset_in_batches minwm-async-denoiser \
    app.kubernetes.io/name=minwm-async-denoiser
fi

wait_for_rollout "deployment/minwm-realtime-adot"
wait_for_rollout "deployment/minwm-realtime-coordinator"
wait_for_rollout "deployment/minwm-realtime-gateway"
wait_for_rollout "deployment/minwm-realtime-gpu-capacity-scaler"
wait_for_rollout "statefulset/minwm-async-denoiser"
unprotect_denoiser_nodes
wait_for_rollout "deployment/minwm-async-vae"

if (( LEGACY_DENOISER_SCALED_DOWN == 1 )); then
  kubectl delete deployment/minwm-async-denoiser \
    --namespace "${NAMESPACE}" --cascade=foreground --wait=true
fi

RELEASE_APPLIED=0
trap - ERR
echo "production rollout completed successfully"

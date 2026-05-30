#!/usr/bin/env bash
# Bootstrap a kind cluster for sgl-router K8s integration E2E tests.
#
# Prerequisites: Docker, kind, kubectl
#
# Usage:
#   ./tests/e2e/k8s_integration/setup.sh           # full setup
#   ./tests/e2e/k8s_integration/setup.sh teardown  # delete the cluster

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"  # repo root (above experimental/)
SGL_ROUTER_DIR="${REPO_ROOT}/experimental/sgl-router"
CLUSTER_NAME="${CLUSTER:-sgl-router-kind}"
NAMESPACE="${NAMESPACE:-sgl-router-test}"
CONTEXT="kind-${CLUSTER_NAME}"
MANIFESTS_DIR="${SCRIPT_DIR}/manifests"

log() { echo "==> $*"; }

teardown() {
    log "Tearing down cluster '${CLUSTER_NAME}'..."
    if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
        kind delete cluster --name "${CLUSTER_NAME}"
    else
        log "Cluster '${CLUSTER_NAME}' not found, nothing to tear down."
    fi
    log "Done."
}

if [[ "${1:-}" == "teardown" ]]; then
    teardown
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 1: Create kind cluster (idempotent)
# ---------------------------------------------------------------------------
if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    log "Kind cluster '${CLUSTER_NAME}' already exists — reusing."
else
    log "Creating kind cluster '${CLUSTER_NAME}'..."
    kind create cluster --name "${CLUSTER_NAME}" --wait 60s
fi

kubectl config use-context "${CONTEXT}"

# ---------------------------------------------------------------------------
# Step 2: Build Docker images (unless SKIP_DOCKER_BUILD=1)
# ---------------------------------------------------------------------------
if [[ "${SKIP_DOCKER_BUILD:-}" == "1" ]]; then
    log "SKIP_DOCKER_BUILD=1 — skipping docker build; expecting images to exist locally."
    for img in sgl-router:e2e sgl-router-fake-worker:e2e; do
        if ! docker image inspect "${img}" >/dev/null 2>&1; then
            log "ERROR: ${img} not found locally; cannot continue without building."
            exit 1
        fi
    done
else
    log "Building sgl-router:e2e from ${REPO_ROOT} ..."
    docker build \
        -f "${SCRIPT_DIR}/Dockerfile.router" \
        -t sgl-router:e2e \
        "${REPO_ROOT}"

    log "Building sgl-router-fake-worker:e2e ..."
    docker build \
        -f "${SCRIPT_DIR}/Dockerfile.fake_worker" \
        -t sgl-router-fake-worker:e2e \
        "${SCRIPT_DIR}"
fi

# ---------------------------------------------------------------------------
# Step 3: Load images into kind
# ---------------------------------------------------------------------------
log "Loading images into kind cluster '${CLUSTER_NAME}'..."
kind load docker-image sgl-router:e2e --name "${CLUSTER_NAME}"
kind load docker-image sgl-router-fake-worker:e2e --name "${CLUSTER_NAME}"

# ---------------------------------------------------------------------------
# Step 4: Apply namespace and RBAC
# ---------------------------------------------------------------------------
log "Applying namespace and RBAC..."
kubectl --context "${CONTEXT}" apply -f "${MANIFESTS_DIR}/namespace.yaml"
kubectl --context "${CONTEXT}" apply -f "${MANIFESTS_DIR}/rbac.yaml"

# ---------------------------------------------------------------------------
# Step 5: Deploy 3 fake-worker replicas behind a Service
#         The Service causes K8s to auto-create an EndpointSlice, which
#         the sgl-router K8s discovery backend watches.
# ---------------------------------------------------------------------------
log "Deploying fake-worker Deployment + Service (3 replicas, app=sglang)..."
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fake-worker
  namespace: ${NAMESPACE}
  labels:
    app: sglang
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sglang
  template:
    metadata:
      labels:
        app: sglang
    spec:
      containers:
        - name: worker
          image: sgl-router-fake-worker:e2e
          imagePullPolicy: Never
          ports:
            - containerPort: 30000
          readinessProbe:
            httpGet:
              path: /health
              port: 30000
            initialDelaySeconds: 2
            periodSeconds: 3
---
apiVersion: v1
kind: Service
metadata:
  name: fake-worker
  namespace: ${NAMESPACE}
  labels:
    app: sglang
spec:
  selector:
    app: sglang
  ports:
    - port: 30000
      targetPort: 30000
EOF

log "Waiting for fake-worker rollout..."
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" rollout status deployment/fake-worker --timeout=120s

# ---------------------------------------------------------------------------
# Step 6: Create sgl-router ConfigMap with k8s discovery pointing at the
#         namespace where fake-worker pods live.
# ---------------------------------------------------------------------------
log "Creating sgl-router-config ConfigMap..."
ROUTER_CONFIG="[server]
host = \"0.0.0.0\"
port = 8090

[[models]]
id = \"tiny\"
tokenizer_path = \"/etc/tokenizer/tiny.json\"
policy = \"round_robin\"
# Aggressive breaker so a terminating pod's connection-refused
# immediately excludes it from the next request's candidate set —
# the reconciliation tests scale workers rapidly and depend on
# fast worker eviction to absorb the churn.
circuit_breaker = { threshold = 1, cool_down_secs = 5 }

[discovery]
backend = \"k8s\"

[discovery.k8s]
namespace = \"${NAMESPACE}\"
label_selector = \"app=sglang\""

kubectl --context "${CONTEXT}" -n "${NAMESPACE}" create configmap sgl-router-config \
    --from-literal=router.toml="${ROUTER_CONFIG}" \
    --dry-run=client -o yaml \
    | kubectl --context "${CONTEXT}" apply -f -

# ---------------------------------------------------------------------------
# Step 7: Deploy sgl-router
# ---------------------------------------------------------------------------
log "Deploying sgl-router..."
kubectl --context "${CONTEXT}" apply -f "${MANIFESTS_DIR}/router.yaml"

log "Waiting for sgl-router rollout..."
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" rollout status deployment/sgl-router --timeout=300s

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log ""
log "Setup complete! Run the integration tests with:"
log "  pytest tests/e2e/k8s_integration/ -v -s"
log ""
log "To tear down:"
log "  ./tests/e2e/k8s_integration/setup.sh teardown"

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "benchmark/minwm_realtime_async_vae/deploy_production.sh"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_ondelete_statefulset_waits_for_the_whole_batched_rollout():
    source = DEPLOY.read_text(encoding="utf-8")

    assert "set -Eeuo pipefail" in source
    assert "wait_for_ondelete_statefulset" in source
    assert 'updated_replicas >= desired_replicas' in source
    assert 'ready_replicas >= desired_replicas' in source
    assert 'DENOISER_RESTART_BATCH_SIZE="${DENOISER_RESTART_BATCH_SIZE:-2}"' in source
    assert "restart_statefulset_in_batches()" in source
    assert "restart_statefulset_in_parallel()" not in source


def test_parallel_gpu_restart_temporarily_protects_karpenter_nodes():
    source = DEPLOY.read_text(encoding="utf-8")

    protect = source.index("protect_denoiser_nodes()")
    restart = source.index("restart_statefulset_in_batches()")
    unprotect = source.index("unprotect_denoiser_nodes()")

    assert protect < restart < unprotect
    assert "karpenter.sh/do-not-disrupt=true" in source
    assert "karpenter.sh/do-not-disrupt-" in source


def test_rollback_recreates_ondelete_gpu_pods_before_waiting():
    source = DEPLOY.read_text(encoding="utf-8")
    rollback = source[source.index("restore_release_snapshot()") :]
    restart = rollback.index("restart_statefulset_in_batches")
    wait = rollback.index('wait_for_rollout "${workload}"')

    assert restart < wait


def test_partial_live_apply_restores_the_exact_pre_release_snapshot(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "kubectl.log"
    _write_executable(
        bin_dir / "aws",
        """#!/usr/bin/env bash
set -eu
if [[ "$1 $2" == "dynamodb describe-table" ]]; then
  cat <<'JSON'
{"Table":{"TableStatus":"ACTIVE","AttributeDefinitions":[{"AttributeName":"pk","AttributeType":"S"},{"AttributeName":"sk","AttributeType":"S"},{"AttributeName":"allocation_key","AttributeType":"S"},{"AttributeName":"allocation_sort","AttributeType":"S"}],"KeySchema":[{"AttributeName":"pk","KeyType":"HASH"},{"AttributeName":"sk","KeyType":"RANGE"}],"GlobalSecondaryIndexes":[{"IndexName":"allocation-index","IndexStatus":"ACTIVE","Projection":{"ProjectionType":"ALL"},"KeySchema":[{"AttributeName":"allocation_key","KeyType":"HASH"},{"AttributeName":"allocation_sort","KeyType":"RANGE"}]}]}}
JSON
elif [[ "$1 $2" == "dynamodb describe-time-to-live" ]]; then
  echo '{"TimeToLiveDescription":{"TimeToLiveStatus":"ENABLED","AttributeName":"ttl"}}'
elif [[ "$1 $2" == "logs describe-log-groups" ]]; then
  echo 5
else
  exit 0
fi
""",
    )
    _write_executable(
        bin_dir / "kubectl",
        """#!/usr/bin/env bash
set -eu
echo "$*" >>"${KUBECTL_LOG}"
if [[ "$1" == "kustomize" ]]; then
  cat <<'YAML'
apiVersion: v1
kind: Namespace
metadata:
  name: minwm-realtime
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minwm-realtime-gateway
  namespace: minwm-realtime
spec:
  replicas: 2
  selector:
    matchLabels:
      app: gateway
  template:
    metadata:
      labels:
        app: gateway
    spec:
      containers:
      - name: gateway
        image: REPLACE_WITH_GATEWAY_IMAGE_DIGEST
YAML
  exit 0
fi
if [[ "$1" == "get" ]]; then
  if [[ "$2" == "deployment/minwm-realtime-gateway" ]]; then
    if [[ "${SNAPSHOT_ERROR:-0}" == "1" ]]; then
      echo "Error from server (NotFound): the server could not find the requested resource" >&2
      exit 1
    fi
    cat <<'JSON'
{"apiVersion":"apps/v1","kind":"Deployment","metadata":{"name":"minwm-realtime-gateway","namespace":"minwm-realtime","resourceVersion":"9","uid":"old"},"spec":{"replicas":2,"selector":{"matchLabels":{"app":"gateway"}},"template":{"metadata":{"labels":{"app":"gateway"}},"spec":{"containers":[{"name":"gateway","image":"old.example/gateway@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}}},"status":{"readyReplicas":2}}
JSON
    exit 0
  fi
  exit 0
fi
if [[ "$1" == "apply" ]]; then
  if [[ "$*" == *"--dry-run=server"* ]]; then
    exit 0
  fi
  if [[ "$*" == *"/rendered.yaml"* ]]; then
    exit 42
  fi
  if [[ "$*" == *"--force-conflicts"* ]]; then
    exit 0
  fi
  exit 42
fi
exit 0
""",
    )

    digest = "example.invalid/minwm@sha256:" + "1" * 64
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "KUBECTL_LOG": str(log),
            "AWS_REGION": "us-east-2",
            "COORDINATOR_TABLE": "unit-test-table",
            "GATEWAY_IMAGE_DIGEST": digest,
            "COORDINATOR_IMAGE_DIGEST": digest,
            "DENOISER_IMAGE_DIGEST": digest,
            "VAE_IMAGE_DIGEST": digest,
            "ADOT_IMAGE_DIGEST": digest,
            "GATEWAY_ROLE_ARN": "arn:aws:iam::123456789012:role/gateway",
            "COORDINATOR_ROLE_ARN": "arn:aws:iam::123456789012:role/coordinator",
            "ADOT_ROLE_ARN": "arn:aws:iam::123456789012:role/adot",
            "MODEL_ID": "unit-test-model",
            "MODEL_ARTIFACT_REVISION": "unit-test-revision",
        }
    )

    completed = subprocess.run(
        ["bash", str(DEPLOY)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 42, completed.stderr
    calls = log.read_text(encoding="utf-8")
    assert (
        "apply --server-side --force-conflicts --field-manager=minwm-production"
        in calls
    )
    assert "replace --force" not in calls
    assert (
        "apply --server-side --force-conflicts --field-manager=minwm-production"
        in calls
    )
    assert "deployment__minwm-realtime-gateway.json" in calls
    assert "delete deployment/minwm-realtime-coordinator" in calls
    assert "delete deployment/minwm-realtime-adot" in calls
    assert "rollout undo" not in calls

    log.unlink()
    env["SNAPSHOT_ERROR"] = "1"
    refused = subprocess.run(
        ["bash", str(DEPLOY)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert refused.returncode != 0
    assert "refusing to mutate the cluster" in refused.stderr
    refused_calls = log.read_text(encoding="utf-8").splitlines()
    assert not any(
        line.startswith(
            "apply --server-side --force-conflicts --field-manager=minwm-production"
        )
        for line in refused_calls
    )
    assert not any(line.startswith("delete ") for line in refused_calls)

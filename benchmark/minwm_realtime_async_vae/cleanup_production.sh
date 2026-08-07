#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?set AWS_REGION}"

STACK_NAME="${STACK_NAME:-minwm-realtime-benchmark}"
DELETE_CONTROL_PLANE="${DELETE_CONTROL_PLANE:-1}"
LABEL="seedleap.ai/test-run=minwm-async-vae-benchmark"
NODEPOOLS=(
  minwm-realtime-cpu
  minwm-async-denoiser-h100
  minwm-async-denoiser-h100-8x
  minwm-async-vae-l4
  minwm-async-vae-l40s
  minwm-model-artifact-publisher
)

if [[ "${1:-}" != "--execute" ]]; then
  cat <<EOF
Dry run only. The following bounded resources would be deleted:
- namespace/minwm-realtime (Deployments, Pods, Services, NLB, CronJobs, PVC)
- NodePools: ${NODEPOOLS[*]}
- EC2NodeClass/minwm-model-artifact-publisher
- PersistentVolume/minwm-async-west-s3-pv
- PersistentVolume/minwm-async-east-s3-source-pv
- CloudFormation stack ${STACK_NAME} when DELETE_CONTROL_PLANE=1

Retained intentionally:
- immutable ECR repository and images
- versioned S3 serving artifact

Re-run with --execute only after the exact cleanup operation is approved.
EOF
  exit 0
fi

LB_HOST="$(kubectl get service/minwm-realtime-public \
  -n minwm-realtime \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || true)"
LB_ARN=""
if [[ -n "${LB_HOST}" ]]; then
  LB_ARN="$(aws elbv2 describe-load-balancers \
    --region "${AWS_REGION}" \
    --query "LoadBalancers[?DNSName=='${LB_HOST}'].LoadBalancerArn | [0]" \
    --output text)"
  [[ "${LB_ARN}" == "None" ]] && LB_ARN=""
fi

kubectl delete namespace/minwm-realtime --ignore-not-found --wait=true --timeout=15m
for NODEPOOL in "${NODEPOOLS[@]}"; do
  kubectl delete nodepool.karpenter.sh/"${NODEPOOL}" --ignore-not-found --wait=true
done
kubectl delete ec2nodeclass.karpenter.k8s.aws/minwm-model-artifact-publisher \
  --ignore-not-found --wait=true
kubectl delete persistentvolume/minwm-async-west-s3-pv --ignore-not-found --wait=true
kubectl delete persistentvolume/minwm-async-east-s3-source-pv --ignore-not-found --wait=true

for _ in $(seq 1 60); do
  NODE_COUNT="$(kubectl get nodes -l "${LABEL}" --no-headers 2>/dev/null | wc -l | tr -d ' ')"
  [[ "${NODE_COUNT}" == "0" ]] && break
  sleep 10
done
if [[ "$(kubectl get nodes -l "${LABEL}" --no-headers 2>/dev/null | wc -l | tr -d ' ')" != "0" ]]; then
  echo "labeled benchmark nodes are still present" >&2
  exit 1
fi

if [[ -n "${LB_ARN}" ]]; then
  for _ in $(seq 1 60); do
    if ! aws elbv2 describe-load-balancers \
      --region "${AWS_REGION}" \
      --load-balancer-arns "${LB_ARN}" >/dev/null 2>&1; then
      LB_ARN=""
      break
    fi
    sleep 10
  done
  if [[ -n "${LB_ARN}" ]]; then
    echo "NLB still exists after cleanup timeout: ${LB_ARN}" >&2
    exit 1
  fi
fi

if [[ "${DELETE_CONTROL_PLANE}" == "1" ]]; then
  aws cloudformation delete-stack \
    --region "${AWS_REGION}" \
    --stack-name "${STACK_NAME}"
  aws cloudformation wait stack-delete-complete \
    --region "${AWS_REGION}" \
    --stack-name "${STACK_NAME}"
fi

if kubectl get all -A -l "${LABEL}" --no-headers 2>/dev/null | grep -q .; then
  echo "labeled namespaced benchmark resources remain" >&2
  exit 1
fi
printf 'Cleanup complete; ECR images and the immutable S3 model artifact were retained.\n'

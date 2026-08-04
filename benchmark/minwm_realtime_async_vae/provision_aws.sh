#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?set AWS_REGION}"
: "${EKS_CLUSTER:?set EKS_CLUSTER}"

STACK_NAME="${STACK_NAME:-minwm-realtime-benchmark}"
ENVIRONMENT="${ENVIRONMENT:-benchmark}"
COORDINATOR_TABLE="${COORDINATOR_TABLE:-minwm-realtime-benchmark}"
TRACE_LOG_GROUP="${TRACE_LOG_GROUP:-/aws/eks/minwm/realtime-traces-benchmark}"
ECR_REPOSITORY="${ECR_REPOSITORY:-leap-world/minwm-realtime}"
ARTIFACT_BUCKET="${ARTIFACT_BUCKET:-leap-world-us-west-2}"
ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-world-model/minwm/serving-artifacts/wan22-5b-stage3-dmd-30-gs1800}"

ROOT="$(git rev-parse --show-toplevel)"
TEMPLATE="${ROOT}/benchmark/minwm_realtime_async_vae/aws/stack.yaml"
RENDERED="$(mktemp)"
trap 'rm -f "${RENDERED}" "${RENDERED}.bak"' EXIT

OIDC_ISSUER="$(aws eks describe-cluster \
  --region "${AWS_REGION}" \
  --name "${EKS_CLUSTER}" \
  --query 'cluster.identity.oidc.issuer' \
  --output text)"
OIDC_HOSTPATH="${OIDC_ISSUER#https://}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
OIDC_PROVIDER_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/${OIDC_HOSTPATH}"
aws iam get-open-id-connect-provider \
  --open-id-connect-provider-arn "${OIDC_PROVIDER_ARN}" >/dev/null

cp "${TEMPLATE}" "${RENDERED}"
sed -i.bak \
  "s|REPLACE_WITH_OIDC_PROVIDER_HOSTPATH|${OIDC_HOSTPATH}|g" \
  "${RENDERED}"
rm -f "${RENDERED}.bak"

aws cloudformation deploy \
  --region "${AWS_REGION}" \
  --stack-name "${STACK_NAME}" \
  --template-file "${RENDERED}" \
  --capabilities CAPABILITY_NAMED_IAM \
  --no-fail-on-empty-changeset \
  --parameter-overrides \
    "Environment=${ENVIRONMENT}" \
    "Namespace=minwm-realtime" \
    "CoordinatorTableName=${COORDINATOR_TABLE}" \
    "TraceLogGroupName=${TRACE_LOG_GROUP}" \
    "EcrRepositoryName=${ECR_REPOSITORY}" \
    "OidcProviderArn=${OIDC_PROVIDER_ARN}" \
    "ArtifactBucket=${ARTIFACT_BUCKET}" \
    "ArtifactPrefix=${ARTIFACT_PREFIX}"

OUTPUTS="$(aws cloudformation describe-stacks \
  --region "${AWS_REGION}" \
  --stack-name "${STACK_NAME}" \
  --query 'Stacks[0].Outputs' \
  --output json)"
output() {
  printf '%s' "${OUTPUTS}" | python3 -c \
    'import json,sys; key=sys.argv[1]; print(next(x["OutputValue"] for x in json.load(sys.stdin) if x["OutputKey"] == key))' \
    "$1"
}

ENV_FILE="${ROOT}/benchmark/minwm_realtime_async_vae/.env.aws"
cat >"${ENV_FILE}" <<EOF
AWS_REGION=${AWS_REGION}
EKS_CLUSTER=${EKS_CLUSTER}
STACK_NAME=${STACK_NAME}
COORDINATOR_TABLE=$(output CoordinatorTableName)
TRACE_LOG_GROUP=$(output TraceLogGroupName)
ECR_REPOSITORY=$(output EcrRepositoryName)
GATEWAY_ROLE_ARN=$(output GatewayRoleArn)
COORDINATOR_ROLE_ARN=$(output CoordinatorRoleArn)
ADOT_ROLE_ARN=$(output AdotRoleArn)
MODEL_ARTIFACT_PUBLISHER_ROLE_ARN=$(output ModelArtifactPublisherRoleArn)
EOF
printf 'Wrote production control-plane outputs to %s\n' "${ENV_FILE}"

from pathlib import Path


ROOT = Path(__file__).parent


def test_cloudformation_declares_the_minimum_production_control_plane():
    template = (ROOT / "aws" / "stack.yaml").read_text()

    assert "AWS::DynamoDB::Table" in template
    assert "PAY_PER_REQUEST" in template
    assert "allocation-index" in template
    assert "AttributeName: ttl" in template
    assert "RetentionInDays: 5" in template
    assert "AWS::ECR::Repository" in template
    assert "ImageTagMutability: IMMUTABLE" in template
    for service_account in (
        "minwm-realtime-gateway",
        "minwm-realtime-coordinator",
        "minwm-realtime-adot",
        "minwm-model-artifact-publisher",
    ):
        assert f"system:serviceaccount:${{Namespace}}:{service_account}" in template


def test_iam_policies_are_scoped_to_each_data_plane():
    template = (ROOT / "aws" / "stack.yaml").read_text()

    assert "dynamodb:Scan" not in template
    assert "dynamodb:DeleteTable" not in template
    assert "logs:DeleteLogGroup" not in template
    assert "s3:DeleteObject" not in template
    assert "dynamodb:TransactWriteItems" in template
    assert "dynamodb:ConditionCheckItem" in template
    assert "dynamodb:DeleteItem" in template
    assert "logs:StartQuery" in template
    assert "xray:PutTraceSegments" in template
    assert "s3:PutObject" in template
    assert "${ArtifactPrefix}/*" in template


def test_provisioner_is_declarative_and_exports_exact_runtime_inputs():
    script = (ROOT / "provision_aws.sh").read_text()

    assert "aws cloudformation deploy" in script
    assert "CAPABILITY_NAMED_IAM" in script
    assert "aws eks describe-cluster" in script
    assert "aws cloudformation describe-stacks" in script
    for variable in (
        "COORDINATOR_TABLE",
        "TRACE_LOG_GROUP",
        "ECR_REPOSITORY",
        "GATEWAY_ROLE_ARN",
        "COORDINATOR_ROLE_ARN",
        "ADOT_ROLE_ARN",
        "MODEL_ARTIFACT_PUBLISHER_ROLE_ARN",
    ):
        assert variable in script


def test_cleanup_is_explicit_bounded_and_verifies_billed_resources_are_gone():
    script = (ROOT / "cleanup_production.sh").read_text()

    assert '[[ "${1:-}" != "--execute" ]]' in script
    assert "namespace/minwm-realtime" in script
    for nodepool in (
        "minwm-async-denoiser-h100",
        "minwm-async-denoiser-h100-8x",
        "minwm-async-vae-l4",
        "minwm-model-artifact-publisher",
    ):
        assert nodepool in script
    assert "minwm-async-west-s3-pv" in script
    assert "minwm-async-east-s3-source-pv" in script
    assert "aws cloudformation delete-stack" in script
    assert "aws elbv2 describe-load-balancers" in script
    assert "seedleap.ai/test-run=minwm-async-vae-benchmark" in script

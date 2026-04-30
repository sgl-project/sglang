#!/usr/bin/env python3
"""Upload CI metrics to S3 for long-term storage.

Uploads consolidated benchmark JSON and JSONL test metrics from dump_metric().
Gracefully no-ops when AWS credentials are not configured.

S3 path structure:
    s3://{bucket}/{prefix}/{date}/{run_id}/consolidated-metrics.json
    s3://{bucket}/{prefix}/{date}/{run_id}/test-metrics.jsonl

Usage:
    python3 scripts/ci/utils/upload_metrics_to_s3.py \
        --consolidated-json consolidated-metrics-12345.json \
        --test-metrics-jsonl test-metrics-12345.jsonl \
        --run-id 12345 \
        --commit-sha abc123 \
        --bucket rdxa-eng-json-logs-05354378 \
        --prefix ci-metrics/nightly
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


def aws_credentials_available():
    return bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))


def upload_file(local_path, s3_uri):
    if not os.path.exists(local_path):
        print(f"  Skip (not found): {local_path}")
        return False
    if os.path.getsize(local_path) == 0:
        print(f"  Skip (empty): {local_path}")
        return False

    try:
        result = subprocess.run(
            ["aws", "s3", "cp", local_path, s3_uri],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            print(f"  Uploaded: {local_path} -> {s3_uri}")
            return True
        else:
            print(f"  Failed: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload CI metrics to S3")
    parser.add_argument("--consolidated-json", help="Path to consolidated JSON")
    parser.add_argument("--test-metrics-jsonl", help="Path to JSONL test metrics")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit-sha", default="unknown")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="ci-metrics/nightly")
    args = parser.parse_args()

    if not aws_credentials_available():
        print("AWS credentials not configured, skipping S3 upload.")
        print(
            "To enable, add CI_METRICS_AWS_ACCESS_KEY_ID and "
            "CI_METRICS_AWS_SECRET_ACCESS_KEY as GitHub repo secrets."
        )
        return

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    s3_base = f"s3://{args.bucket}/{args.prefix}/{date_str}/{args.run_id}"

    print(f"Uploading metrics to {s3_base}/")

    # Upload a small metadata file for easy querying
    metadata = {
        "run_id": args.run_id,
        "commit_sha": args.commit_sha,
        "date": date_str,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = "/tmp/run-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    uploaded = 0
    if upload_file(metadata_path, f"{s3_base}/run-metadata.json"):
        uploaded += 1
    if args.consolidated_json:
        if upload_file(args.consolidated_json, f"{s3_base}/consolidated-metrics.json"):
            uploaded += 1
    if args.test_metrics_jsonl:
        if upload_file(args.test_metrics_jsonl, f"{s3_base}/test-metrics.jsonl"):
            uploaded += 1

    print(f"Done: {uploaded} file(s) uploaded to S3.")


if __name__ == "__main__":
    main()

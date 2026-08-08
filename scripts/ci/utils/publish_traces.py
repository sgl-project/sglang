"""
Publish performance traces to S3

Keys keep the layout the nightly report links expect:
    s3://{bucket}/traces/{run_id}/{model}/{timestamp}/{file}.trace.json.gz

so TRACE_BASE_URL only needs to point at the bucket instead of a raw
githubusercontent URL. A lifecycle rule on the bucket expires traces after a
month; nothing here deletes anything.

Credentials come from the GitHub OIDC role assumed by
aws-actions/configure-aws-credentials, so there are no static secrets.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

UPLOAD_CONCURRENCY = 8


def collect_trace_files(source_dir, target_base_path):
    """Return (key, local_path) pairs for the traces under source_dir.

    Only collects traces from TP rank 0 to avoid duplicated data across tensor parallel ranks.
    """
    files_to_upload = []

    if not os.path.exists(source_dir):
        print(f"Warning: Traces directory {source_dir} does not exist")
        return files_to_upload

    # Walk through source directory and find .json.gz files
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".json.gz"):

                # Only upload TP rank 0 traces to avoid duplicates across tensor parallel ranks
                if "TP-" in file and "TP-0" not in file:
                    continue

                source_file = os.path.join(root, file)
                # Calculate relative path from source_dir
                rel_path = os.path.relpath(source_file, source_dir)
                target_path = f"{target_base_path}/{rel_path}"

                files_to_upload.append((target_path, source_file))

    return files_to_upload


def upload_traces(files_to_upload, bucket):
    """Upload the collected traces to S3, returning the number that succeeded."""
    import boto3

    s3 = boto3.client("s3")

    def put(item):
        key, local_path = item
        # Perfetto fetches the gzipped trace and inflates it itself, so serve the
        # bytes as-is. Setting ContentEncoding here would make browsers
        # transparently decompress and hand Perfetto a trace it cannot read.
        with open(local_path, "rb") as f:
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=f,
                ContentType="application/gzip",
            )
        return key

    uploaded = 0
    with ThreadPoolExecutor(max_workers=UPLOAD_CONCURRENCY) as pool:
        futures = {pool.submit(put, item): item for item in files_to_upload}
        for future in as_completed(futures):
            key, local_path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Failed to upload {local_path} -> s3://{bucket}/{key}: {e}")
                continue
            uploaded += 1
            if uploaded % 10 == 0 or uploaded == len(files_to_upload):
                print(f"Uploaded {uploaded}/{len(files_to_upload)} traces...")

    return uploaded


def main():
    parser = argparse.ArgumentParser(
        description="Publish performance traces to S3",
    )
    parser.add_argument(
        "--traces-dir",
        type=str,
        action="append",
        dest="traces_dirs",
        required=True,
        help="Traces directory to publish (can be specified multiple times)",
    )
    args = parser.parse_args()

    bucket = os.getenv("SGLANG_CI_TRACES_S3_BUCKET")
    if not bucket:
        print("Error: SGLANG_CI_TRACES_S3_BUCKET environment variable not set")
        sys.exit(1)

    run_id = os.getenv("GITHUB_RUN_ID", "test")

    # Collect trace files from all directories
    target_base_path = f"traces/{run_id}"
    all_files = []
    for traces_dir in args.traces_dirs:
        print(f"Processing traces from directory: {traces_dir}")
        all_files.extend(collect_trace_files(traces_dir, target_base_path))

    if not all_files:
        print("No trace files found to upload across all directories")
        return

    print(f"Found {len(all_files)} total files to upload")

    uploaded = upload_traces(all_files, bucket)
    print(
        f"Published {uploaded}/{len(all_files)} traces to s3://{bucket}/{target_base_path}"
    )

    if uploaded != len(all_files):
        sys.exit(1)


if __name__ == "__main__":
    main()

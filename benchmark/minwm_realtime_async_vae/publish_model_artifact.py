#!/usr/bin/env python3
"""Publish an immutable, fully materialized MinWM serving artifact to S3."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError


REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(model_dir: Path, args: argparse.Namespace) -> dict:
    files = []
    for path in sorted(item for item in model_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(model_dir).as_posix()
        files.append(
            {"path": relative, "size": path.stat().st_size, "sha256": sha256(path)}
        )
    return {
        "schema_version": 1,
        "revision": args.revision,
        "source_uri": args.source_uri,
        "source_version_id": args.source_version_id,
        "converter_git_sha": args.converter_git_sha,
        "files": files,
    }


def object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
            return False
        raise
    return True


def publish(model_dir: Path, manifest: dict, args: argparse.Namespace) -> None:
    client = boto3.client("s3", region_name=args.output_region)
    model_prefix = f"{args.output_prefix.strip('/')}/{args.revision}/model"
    ready_key = f"{model_prefix}/_READY"
    if object_exists(client, args.output_bucket, ready_key):
        raise RuntimeError(f"immutable artifact already exists: s3://{args.output_bucket}/{ready_key}")

    transfer = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=args.upload_concurrency,
        use_threads=True,
    )
    for entry in manifest["files"]:
        client.upload_file(
            str(model_dir / entry["path"]),
            args.output_bucket,
            f"{model_prefix}/{entry['path']}",
            Config=transfer,
        )

    manifest_body = json.dumps(
        manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    client.put_object(
        Bucket=args.output_bucket,
        Key=f"{model_prefix}/artifact-manifest.json",
        Body=manifest_body,
        ContentType="application/json",
    )
    client.put_object(
        Bucket=args.output_bucket,
        Key=ready_key,
        Body=json.dumps(
            {
                "revision": args.revision,
                "manifest_sha256": hashlib.sha256(manifest_body).hexdigest(),
            },
            sort_keys=True,
        ).encode("utf-8"),
        ContentType="application/json",
        IfNoneMatch="*",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--source-version-id", required=True)
    parser.add_argument("--donor-dir", type=Path, required=True)
    parser.add_argument("--converter-script", type=Path, required=True)
    parser.add_argument("--converter-git-sha", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--output-region", default="us-west-2")
    parser.add_argument(
        "--output-prefix", default="world-model/minwm/serving-artifacts/wan22-5b-stage3-dmd-30-gs1800"
    )
    parser.add_argument("--upload-concurrency", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not REVISION_RE.fullmatch(args.revision):
        raise ValueError("revision must be an immutable filesystem-safe identifier")
    if args.upload_concurrency < 1 or args.upload_concurrency > 32:
        raise ValueError("upload concurrency must be between 1 and 32")
    if not args.source_checkpoint.is_file():
        raise FileNotFoundError(args.source_checkpoint)
    if not (args.donor_dir / "transformer").is_dir():
        raise FileNotFoundError(args.donor_dir / "transformer")

    with tempfile.TemporaryDirectory(prefix="minwm-serving-artifact-") as temp:
        model_dir = Path(temp) / "model"
        subprocess.run(
            [
                "python3",
                str(args.converter_script),
                "--minwm-checkpoint",
                str(args.source_checkpoint),
                "--donor-diffusers-dir",
                str(args.donor_dir),
                "--output-dir",
                str(model_dir),
                "--source-uri",
                args.source_uri,
                "--action-type",
                "auto",
                "--local-attn-size",
                "32",
                "--sink-size",
                "8",
                "--sliding-window-num-frames",
                "32",
                "--rope-position-mode",
                "block_relative",
                "--rope-max-frame-gap",
                "12",
                "--prompt-first-frame-pin-enabled",
            ],
            check=True,
        )
        for required in ("model_index.json", "transformer"):
            if not (model_dir / required).exists():
                raise RuntimeError(f"converted artifact is missing {required}")
        manifest = build_manifest(model_dir, args)
        publish(model_dir, manifest, args)


if __name__ == "__main__":
    main()

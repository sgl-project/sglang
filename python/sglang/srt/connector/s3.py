# SPDX-License-Identifier: Apache-2.0

import fnmatch
import os
from pathlib import Path
from typing import Generator, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import torch

from sglang.srt.connector import BaseFileConnector


def _parse_s3_kwargs(url: str) -> tuple[str, dict]:
    """Strip query string from s3 URL and return (clean_url, boto3_kwargs).

    Recognizes ``endpoint_url``, ``region`` (alias ``region_name``),
    ``aws_access_key_id``, ``aws_secret_access_key`` from the query string.
    Falls back to ``AWS_ENDPOINT_URL`` / ``AWS_DEFAULT_REGION`` env vars.
    """
    parsed = urlparse(url)
    qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    kwargs = {}
    endpoint = qs.get("endpoint_url") or os.environ.get("AWS_ENDPOINT_URL")
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    region = qs.get("region_name") or qs.get("region") or os.environ.get(
        "AWS_DEFAULT_REGION"
    )
    if region:
        kwargs["region_name"] = region
    for key in ("aws_access_key_id", "aws_secret_access_key"):
        if key in qs:
            kwargs[key] = qs[key]
    clean = parsed._replace(query="").geturl()
    return clean, kwargs


def _filter_allow(paths: list[str], patterns: list[str]) -> list[str]:
    return [
        path
        for path in paths
        if any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
    ]


def _filter_ignore(paths: list[str], patterns: list[str]) -> list[str]:
    return [
        path
        for path in paths
        if not any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
    ]


def list_files(
    s3,
    path: str,
    allow_pattern: Optional[list[str]] = None,
    ignore_pattern: Optional[list[str]] = None,
) -> tuple[str, str, list[str]]:
    """
    List files from S3 path and filter by pattern.

    Args:
        s3: S3 client to use.
        path: The S3 path to list from.
        allow_pattern: A list of patterns of which files to pull.
        ignore_pattern: A list of patterns of which files not to pull.

    Returns:
        tuple[str, str, list[str]]: A tuple where:
            - The first element is the bucket name
            - The second element is string represent the bucket
              and the prefix as a dir like string
            - The third element is a list of files allowed or
              disallowed by pattern
    """
    parts = path.removeprefix("s3://").split("/")
    prefix = "/".join(parts[1:])
    bucket_name = parts[0]

    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    paths = [obj["Key"] for obj in objects.get("Contents", [])]

    paths = _filter_ignore(paths, ["*/"])
    if allow_pattern is not None:
        paths = _filter_allow(paths, allow_pattern)

    if ignore_pattern is not None:
        paths = _filter_ignore(paths, ignore_pattern)

    return bucket_name, prefix, paths


class S3Connector(BaseFileConnector):

    def __init__(self, url: str, **client_kwargs) -> None:
        import boto3

        clean_url, parsed_kwargs = _parse_s3_kwargs(url)
        # Explicit kwargs win over query-string / env values.
        parsed_kwargs.update({k: v for k, v in client_kwargs.items() if v is not None})
        super().__init__(clean_url)
        self.client = boto3.client("s3", **parsed_kwargs)

    def glob(self, allow_pattern: Optional[list[str]] = None) -> list[str]:
        bucket_name, _, paths = list_files(
            self.client, path=self.url, allow_pattern=allow_pattern
        )
        return [f"s3://{bucket_name}/{path}" for path in paths]

    def pull_files(
        self,
        allow_pattern: Optional[list[str]] = None,
        ignore_pattern: Optional[list[str]] = None,
    ) -> None:
        """
        Pull files from S3 storage into the temporary directory.

        Args:
            s3_model_path: The S3 path of the model.
            allow_pattern: A list of patterns of which files to pull.
            ignore_pattern: A list of patterns of which files not to pull.

        """
        bucket_name, base_dir, files = list_files(
            self.client, self.url, allow_pattern, ignore_pattern
        )
        if len(files) == 0:
            return

        for file in files:
            destination_file = os.path.join(self.local_dir, file.removeprefix(base_dir))
            local_dir = Path(destination_file).parent
            os.makedirs(local_dir, exist_ok=True)
            self.client.download_file(bucket_name, file, destination_file)

    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        from sglang.srt.model_loader.weight_utils import (
            runai_safetensors_weights_iterator,
        )

        # only support safetensor files now
        hf_weights_files = self.glob(allow_pattern=["*.safetensors"])
        return runai_safetensors_weights_iterator(hf_weights_files)

    def close(self):
        self.client.close()
        super().close()

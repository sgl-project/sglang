# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Generator, Optional, Tuple

import torch

from sglang.srt.connector import BaseFileConnector
from sglang.srt.connector.utils import filter_allow, filter_ignore


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
    # Match the prefix as a directory so that e.g. "model" does not also
    # match sibling keys such as "model-other/...".
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    paths = [obj["Key"] for obj in objects.get("Contents", [])]

    paths = filter_ignore(paths, ["*/"])
    if allow_pattern is not None:
        paths = filter_allow(paths, allow_pattern)

    if ignore_pattern is not None:
        paths = filter_ignore(paths, ignore_pattern)

    return bucket_name, prefix, paths


class S3Connector(BaseFileConnector):
    def __init__(self, url: str) -> None:
        import boto3

        super().__init__(url)
        self.client = boto3.client("s3")

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
            # S3 keys are arbitrary strings: refuse keys whose destination
            # would resolve outside the local model directory (e.g. keys
            # containing "..", or keys that become absolute paths once the
            # prefix is stripped).
            if not os.path.realpath(destination_file).startswith(
                os.path.realpath(self.local_dir) + os.sep
            ):
                raise ValueError(
                    f"S3 key {file!r} resolves outside the local model directory"
                )
            os.makedirs(Path(destination_file).parent, exist_ok=True)
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

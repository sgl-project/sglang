# SPDX-License-Identifier: Apache-2.0

import fnmatch
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from sglang.srt.connector import BaseKVConnector


def filter_allow(paths: list[str], patterns: list[str]) -> list[str]:
    return [
        path
        for path in paths
        if any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
    ]


def filter_ignore(paths: list[str], patterns: list[str]) -> list[str]:
    return [
        path
        for path in paths
        if not any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
    ]


def parse_model_name(url: str) -> str:
    """
    Parse the model name from the url.
    Only used for db connector
    """
    parsed_url = urlparse(url)
    return parsed_url.path.lstrip("/")


def pull_files_from_db(
    connector: BaseKVConnector,
    model_name: str,
    allow_pattern: Optional[list[str]] = None,
    ignore_pattern: Optional[list[str]] = None,
) -> None:
    prefix = f"{model_name}/files/"
    local_dir = connector.get_local_dir()
    files = connector.list(prefix)

    if allow_pattern is not None:
        files = filter_allow(files, allow_pattern)
    if ignore_pattern is not None:
        files = filter_ignore(files, ignore_pattern)

    for file in files:
        destination_file = os.path.join(local_dir, file.removeprefix(prefix))
        # KV keys are arbitrary strings: refuse keys whose destination
        # would resolve outside the local model directory (e.g. a key
        # containing "..", or a key that becomes absolute once the
        # prefix is stripped).
        if not os.path.realpath(destination_file).startswith(
            os.path.realpath(local_dir) + os.sep
        ):
            raise ValueError(f"Key {file!r} resolves outside the local model directory")
        os.makedirs(Path(destination_file).parent, exist_ok=True)
        with open(destination_file, "wb") as f:
            f.write(connector.getstr(file).encode("utf-8"))

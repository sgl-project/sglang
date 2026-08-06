# SPDX-License-Identifier: Apache-2.0

import fnmatch
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from sglang.srt.connector.base_connector import BaseKVConnector


def filter_file_paths(
    paths: list[str],
    allow_pattern: Optional[list[str]] = None,
    ignore_pattern: Optional[list[str]] = None,
) -> list[str]:
    """Apply allow patterns first, then exclude paths matching ignore patterns."""
    if allow_pattern is not None:
        paths = [
            path
            for path in paths
            if any(fnmatch.fnmatch(path, pattern) for pattern in allow_pattern)
        ]
    if ignore_pattern is not None:
        paths = [
            path
            for path in paths
            if not any(fnmatch.fnmatch(path, pattern) for pattern in ignore_pattern)
        ]
    return paths


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
    """Download database-backed model files into the connector's local directory."""
    prefix = f"{model_name}/files/"
    download_root = connector.get_local_dir()
    files = filter_file_paths(connector.list(prefix), allow_pattern, ignore_pattern)

    for file in files:
        # Anchor every destination to the connector root, regardless of key order.
        destination_file = os.path.join(download_root, file.removeprefix(prefix))
        parent_dir = Path(destination_file).parent
        os.makedirs(parent_dir, exist_ok=True)
        with open(destination_file, "wb") as f:
            f.write(connector.getstr(file).encode("utf-8"))

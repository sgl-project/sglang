# SPDX-License-Identifier: Apache-2.0

import fnmatch
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from sglang.srt.connector import BaseConnector


def parse_model_name(url: str) -> str:
    """
    Parse the model name from the url.
    Only used for db connector
    """
    parsed_url = urlparse(url)
    return parsed_url.path.lstrip("/")


def pull_files_from_db(
    connector: BaseConnector,
    model_name: str,
    allow_pattern: Optional[list[str]] = None,
    ignore_pattern: Optional[list[str]] = None,
) -> None:
    prefix = f"{model_name}/files/"
    download_root = connector.get_local_dir()
    files = connector.list(prefix)

    if allow_pattern is not None:
        files = [
            file
            for file in files
            if any(fnmatch.fnmatch(file, pattern) for pattern in allow_pattern)
        ]
    if ignore_pattern is not None:
        files = [
            file
            for file in files
            if not any(fnmatch.fnmatch(file, pattern) for pattern in ignore_pattern)
        ]

    for file in files:
        destination_file = os.path.join(download_root, file.removeprefix(prefix))
        parent_dir = Path(destination_file).parent
        os.makedirs(parent_dir, exist_ok=True)
        with open(destination_file, "wb") as f:
            f.write(connector.getstr(file).encode("utf-8"))

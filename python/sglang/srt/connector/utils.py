# SPDX-License-Identifier: Apache-2.0

from fnmatch import fnmatch
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from sglang.srt.connector.base_connector import BaseConnector


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
    local_dir = connector.get_local_dir()
    files = connector.list(prefix)

    for file in files:
        rel_path = file.removeprefix(prefix).lstrip("/")
        if not rel_path:
            continue
        if allow_pattern is not None and not any(
            fnmatch(rel_path, pattern) for pattern in allow_pattern
        ):
            continue
        if ignore_pattern is not None and any(
            fnmatch(rel_path, pattern) for pattern in ignore_pattern
        ):
            continue
        destination_file = os.path.join(local_dir, rel_path)
        os.makedirs(Path(destination_file).parent, exist_ok=True)
        with open(destination_file, "wb") as f:
            f.write(connector.getstr(file).encode("utf-8"))

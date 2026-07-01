# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from sglang.srt.connector import BaseConnector

COMMON_REMOTE_MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
]


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
    files = set(connector.list(prefix))
    key_exists = getattr(connector, "_key_exists", None)

    for file_name in COMMON_REMOTE_MODEL_FILES:
        remote_file = f"{prefix}{file_name}"
        if callable(key_exists) and not key_exists(remote_file):
            continue
        content = connector.getstr(remote_file)
        if content is not None:
            files.add(remote_file)

    for file in sorted(files):
        destination_file = os.path.join(local_dir, file.removeprefix(prefix))
        local_dir = Path(destination_file).parent
        os.makedirs(local_dir, exist_ok=True)
        with open(destination_file, "wb") as f:
            f.write(connector.getstr(file).encode("utf-8"))

# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Discovery and validation for a model-native trie output head."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

import msgspec
from safetensors import safe_open

TRIE_OUTPUT_HEAD_FILENAME = "trie_output_head.safetensors"
TRIE_OUTPUT_HEAD_FORMAT_VERSION = 1


class TrieOutputHeadConfig(msgspec.Struct, frozen=True):
    tensor_path: str
    token_start: int
    codebook_size: int
    num_codebooks: int

    @property
    def token_end(self) -> int:
        return self.token_start + self.num_codebooks * self.codebook_size

    def token_start_for_depth(self, depth: int) -> int:
        if not 0 <= depth < self.num_codebooks:
            raise ValueError(f"SID depth {depth} is outside [0, {self.num_codebooks})")
        return self.token_start + depth * self.codebook_size


def _metadata_int(metadata: dict[str, str], name: str) -> int:
    try:
        return int(metadata[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{TRIE_OUTPUT_HEAD_FILENAME} metadata requires integer {name!r}"
        ) from exc


@functools.lru_cache(maxsize=8)
def load_trie_output_head_config(tensor_path: str) -> TrieOutputHeadConfig:
    path = Path(tensor_path).expanduser()
    if not path.is_file():
        raise ValueError(f"Trie output-head tensors do not exist: {path}")
    path = path.absolute()

    with safe_open(path, framework="pt", device="cpu") as tensors:
        metadata = tensors.metadata() or {}

    format_version = _metadata_int(metadata, "format_version")
    if format_version != TRIE_OUTPUT_HEAD_FORMAT_VERSION:
        raise ValueError(
            "Unsupported trie output-head format version: "
            f"{format_version}; expected {TRIE_OUTPUT_HEAD_FORMAT_VERSION}"
        )
    token_start = _metadata_int(metadata, "token_start")
    codebook_size = _metadata_int(metadata, "codebook_size")
    num_codebooks = _metadata_int(metadata, "num_codebooks")
    if token_start < 0 or codebook_size <= 0 or num_codebooks != 3:
        raise ValueError(
            "Trie output head requires a non-negative token_start, "
            "a positive codebook_size, and exactly three codebooks"
        )

    return TrieOutputHeadConfig(
        tensor_path=str(path),
        token_start=token_start,
        codebook_size=codebook_size,
        num_codebooks=num_codebooks,
    )


def discover_trie_output_head_config(
    model_path: str, revision: Optional[str] = None
) -> Optional[TrieOutputHeadConfig]:
    """Discover the optional ``trie_output_head.safetensors`` model artifact."""

    path = Path(model_path).expanduser()
    if path.is_dir():
        tensor_path = path / TRIE_OUTPUT_HEAD_FILENAME
        if not tensor_path.is_file():
            return None
        return load_trie_output_head_config(str(tensor_path))
    if path.is_absolute() or model_path.startswith(("./", "../", "~")):
        return None

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        tensor_path = hf_hub_download(
            model_path,
            TRIE_OUTPUT_HEAD_FILENAME,
            revision=revision,
        )
    except (EntryNotFoundError, LocalEntryNotFoundError):
        return None
    return load_trie_output_head_config(tensor_path)

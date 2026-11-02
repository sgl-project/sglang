# Copyright 2023-2024 SGLang Team
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
"""Lightweight, reusable validators for hot-path API fields.

These are intended to be paired with ``pydantic.PlainValidator`` on
dataclass fields whose JSON shape is large or homogeneously typed, where
pydantic's default per-element walk has been measured to dominate
request latency.

Usage::

    from typing import Annotated, List, Optional, Union
    from pydantic import PlainValidator
    from sglang.srt.utils.field_validators import validate_optional_list_i64_1d_2d

    @dataclass
    class MyReq:
        input_ids: Annotated[
            Optional[Union[List[List[int]], List[int]]],
            PlainValidator(validate_optional_list_i64_1d_2d),
        ] = None
"""

from __future__ import annotations

import os
from array import array
from pathlib import Path
from typing import Any


def validate_list_i64_1d(v: Any) -> list[int]:
    """Validates type: list[int]"""
    if v is None:
        raise ValueError("must not be None")
    if not isinstance(v, list):
        raise ValueError(f"must be list; got {type(v).__name__}")
    if not v:
        return v
    if not isinstance(v[0], int):
        raise ValueError(f"elements must be int; got {type(v[0]).__name__}")
    try:
        array("q", v)
    except (TypeError, OverflowError) as e:
        raise ValueError(f"contains non-int64 element: {e}") from None
    return v


def validate_optional_list_i64_1d_2d(
    v: Any,
) -> list[int] | list[list[int]] | None:
    """Validates type: list[int] | list[list[int]] | None"""
    if v is None:
        # Accept None
        return v
    if not isinstance(v, list):
        raise ValueError(f"must be list or null; got {type(v).__name__}")
    if not v:
        # Accept empty list
        return v
    if isinstance(v[0], int):
        # Accept list[int]
        return validate_list_i64_1d(v)
    if isinstance(v[0], list):
        # Accept list[list[int]]
        for i, row in enumerate(v):
            try:
                validate_list_i64_1d(row)
            except ValueError as e:
                raise ValueError(f"row {i}: {e}") from None
        return v
    raise ValueError(f"elements must be int or list; got {type(v[0]).__name__}")


def validate_filename_component(
    name: Any, field_name: str = "filename component"
) -> None:
    """Validate that a filename component (e.g. profile_id, profile_prefix) is safe
    and cannot escape its destination directory via path separators or traversal."""
    if name is None or name == "":
        return
    if not isinstance(name, str):
        raise ValueError(f"{field_name} must be a string; got {type(name).__name__}")
    if (
        os.path.isabs(name)
        or "/" in name
        or "\\" in name
        or ".." in name
        or "\0" in name
    ):
        raise ValueError(
            f"Invalid {field_name}: '{name}'. Must be a safe filename component without path separators or traversal."
        )


def get_contained_path(base_dir: str | Path, filename: str) -> str:
    """Join base_dir and filename and ensure the resolved path remains strictly
    within base_dir, preventing path traversal."""
    base_path = Path(base_dir).resolve()
    target_path = (base_path / filename).resolve()
    try:
        target_path.relative_to(base_path)
    except ValueError:
        raise ValueError(
            f"Target path '{target_path}' escapes output directory '{base_path}'"
        )
    return str(target_path)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Shared K3 source locators that do not import GPU implementations."""

import ast
import json
from functools import cache
from pathlib import Path

import sglang

SGLANG_ROOT = Path(sglang.__file__).resolve().parent
RUNTIME_ROOT = SGLANG_ROOT / "srt/kernel_plugins/nvidia/kimi_k3"


@cache
def _source_mapping():
    manifest = json.loads((RUNTIME_ROOT / "SOURCE_MAPPING.json").read_text())
    prefix = "sgl_nvidia_mega/kimi_k3/"
    return {
        entry["source"].removeprefix(prefix): entry for entry in manifest["sources"]
    }


def source_path(relative, *qualname):
    """Resolve an original symbol to its canonical in-tree source file."""
    entry = _source_mapping()[relative]
    if qualname:
        symbol = next(item for item in entry["symbols"] if item["name"] == qualname[0])
        target = next(
            item
            for item in entry["targets"]
            if item["module"] == symbol["target_module"]
        )
    else:
        assert len(entry["targets"]) == 1, (relative, "qualname required after split")
        target = entry["targets"][0]
    path = SGLANG_ROOT / target["path"]
    assert path.is_file(), path
    return path


def source_definition(relative, *qualname):
    body = ast.parse(source_path(relative, *qualname).read_text()).body
    for name in qualname:
        node = next(
            item
            for item in body
            if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name == name
        )
        body = node.body
    return node

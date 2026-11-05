# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Explicit source identity of the in-tree K3 runtime and kernel closure.

Reading the manifest imports no CUDA backend and performs no compilation. Every
mapped file must exist; inventory never silently shrinks to the available files.
"""

import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_MAPPING = Path(__file__).resolve().with_name("SOURCE_MAPPING.json")


def payload_manifest() -> dict[str, dict[str, object]]:
    """Actual bytes keyed by the original source and its canonical destination."""
    mapping = json.loads(_MAPPING.read_text())
    entries = []
    for source in mapping["sources"]:
        for target in source["targets"]:
            entries.append((source["source"] + "::" + target["module"], target))
    for target in mapping["generated_modules"]:
        entries.append(("@migration/" + target["path"], target))
    for name in ("SOURCE_MAPPING.json", "NOTICE"):
        entries.append(
            (
                "@migration/" + name,
                {"path": str(_MAPPING.with_name(name).relative_to(_ROOT))},
            )
        )
    result = {}
    for key, target in sorted(entries):
        path = _ROOT / target["path"]
        content = path.read_bytes()
        result[key] = {
            "module": target.get("module"),
            "path": str(path),
            "nbytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    return result


def payload_digest() -> str:
    """Deterministic identity over original keys and all mapped actual bytes."""
    digest = hashlib.sha256()
    for key, record in payload_manifest().items():
        digest.update(key.encode() + b"\0")
        digest.update(str(record["nbytes"]).encode() + b"\0")
        digest.update(bytes.fromhex(record["sha256"]))
    return digest.hexdigest()


def source_paths() -> dict[str, str]:
    """The complete runtime/ops/JIT module inventory without importing kernels."""
    return {
        record["module"]: record["path"]
        for record in payload_manifest().values()
        if record["module"] is not None
    }
